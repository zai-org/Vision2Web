"""Inference engine for running agent tasks in Vision2Web"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from vision2web.core.config import Config
from vision2web.core.logger import setup_logger, get_logger
from vision2web.core.dataset import DatasetManager, Project
from vision2web.core.sandbox import SandboxManager
from vision2web.inference.adapters import get_adapter, BaseAdapter
from vision2web.inference.prompts import get_prompt_for_task


class InferenceEngine:
    """Main inference engine for running agent tasks"""

    def __init__(self, config: Config):
        """
        Initialize inference engine.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.logger = get_logger('vision2web.inference')

        # Initialize dataset manager
        self.dataset_manager = DatasetManager(config.datasets_dir)

        # Initialize sandbox manager
        self.sandbox_manager = SandboxManager(
            image_name=config.sandbox.image,
            workspace_dir=config.sandbox.workspace_dir,
            user=config.sandbox.user,
            logger=self.logger
        )

        # Initialize adapter
        self.adapter = self._create_adapter()

    def _create_adapter(self) -> BaseAdapter:
        """Create adapter based on configuration"""
        return get_adapter(
            framework=self.config.inference.framework,
            api_key=self.config.inference.api_key,
            model=self.config.inference.model,
            base_url=self.config.inference.base_url,
            sandbox_manager=self.sandbox_manager,
            logger=self.logger,
            timeout=self.config.inference.timeout
        )

    def _get_project_workspace(self, project: Project) -> Path:
        """Get workspace path for a project"""
        return (
            self.config.results_dir /
            project.task_type /
            self.config.inference.framework /
            self.config.inference.model /
            project.name
        )

    def _copy_project_files(self, project: Project, workspace: Path):
        """
        Copy project files to workspace.

        Args:
            project: Project instance
            workspace: Target workspace directory
        """
        workspace.mkdir(parents=True, exist_ok=True)

        # Copy prototypes directory
        prototypes_dest = workspace / 'prototypes'
        if prototypes_dest.exists():
            shutil.rmtree(prototypes_dest)
        shutil.copytree(project.prototypes_dir, prototypes_dest)

        # Copy resources directory if exists
        if project.resources_dir and project.resources_dir.exists():
            resources_dest = workspace / 'resources'
            if resources_dest.exists():
                shutil.rmtree(resources_dest)
            shutil.copytree(project.resources_dir, resources_dest)

        # Copy task-specific files
        if project.task_type == "website" and project.prd_path:
            shutil.copy2(project.prd_path, workspace / 'prd.md')
        
        if project.task_type == "frontend" and project.prompt_path:
            shutil.copy2(project.prompt_path, workspace / 'prompt.txt')

        self.logger.debug(f"Copied project files to {workspace}")

    def _save_logs(self, workspace: Path, logs: List[str]):
        """Save execution logs"""
        logs_dir = workspace / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"{workspace.name}.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(logs))

    def _save_result(self, workspace: Path, result: Dict[str, Any]):
        """Save execution result"""
        results_dir = workspace / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        result_file = results_dir / f"{workspace.name}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def _generate_container_name(self, project: Project) -> str:
        """Generate container name based on task, project, framework, and model"""
        return f"{project.task_type}_{project.name}_{self.config.inference.framework}_{self.config.inference.model.replace('/', '_').replace(':', '_')}"

    async def run_single_project(self, project: Project) -> Dict[str, Any]:
        """
        Run inference on a single project with retry mechanism.

        Args:
            project: Project to run

        Returns:
            Result dictionary
        """
        # Skip if already completed
        workspace = self._get_project_workspace(project)
        if (workspace / 'start.sh').exists():
            self.logger.info(f"Skipping {project.name}: already completed")
            return {
                'project': project.name,
                'task_type': project.task_type,
                'name': project.name,
                'status': 'skipped',
            }

        self.logger.info(f"Starting inference for project: {project.name} (task: {project.task_type})")
        start_time = datetime.now()

        result = {
            'project': project.name,
            'task_type': project.task_type,
            'name': project.name,
            'start_time': start_time.isoformat(),
        }

        workspace = None
        max_attempts = 1 + self.config.inference.max_retries
        attempt_results = []

        try:
            # Get workspace path
            workspace = self._get_project_workspace(project)

            # Copy project files to workspace
            self.logger.info(f"Copying project files to {workspace}")
            self._copy_project_files(project, workspace)

            # Generate container name
            container_name = self._generate_container_name(project)

            # Create and start container (once before retry loop)
            self.logger.info(f"Creating container: {container_name}")
            container_id = await self.sandbox_manager.create_container(
                workspace,
                container_name=container_name
            )
            if container_id:
                await self.sandbox_manager.start_container(workspace)

            # Get appropriate prompt for task type
            prompt = get_prompt_for_task(project.task_type)

            # Retry loop
            for attempt in range(1, max_attempts + 1):
                self.logger.info(f"Attempt {attempt}/{max_attempts} for {project.name}")

                # Run inference
                task_result = await self.adapter.run_task(
                    workspace=workspace,
                    prompt=prompt,
                    project_info=project.to_dict()
                )

                # Record attempt result
                attempt_results.append({
                    'attempt': attempt,
                    'status': task_result['status'],
                    'error': task_result.get('error')
                })

                if task_result['status'] == 'success':
                    # Success - exit loop
                    self.logger.info(f"Task succeeded on attempt {attempt}")
                    # Merge task result into main result
                    result.update({
                        'status': task_result['status'],
                        'logs': task_result.get('logs', []),
                        'conversation': task_result.get('conversation', []),
                        'error': task_result.get('error'),
                        'framework': task_result.get('framework'),
                        'model': task_result.get('model'),
                        'project_info': task_result.get('project_info')
                    })
                    # Save logs
                    if task_result.get('logs'):
                        self._save_logs(workspace, task_result['logs'])
                    break
                elif task_result['status'] == 'timeout':
                    # Timed out - terminal, do not retry (a hung run is unlikely
                    # to succeed on retry and would burn another full timeout).
                    self.logger.error(
                        f"Task timed out on attempt {attempt} for {project.name}; "
                        f"not retrying."
                    )
                    result.update({
                        'status': task_result['status'],
                        'logs': task_result.get('logs', []),
                        'conversation': task_result.get('conversation', []),
                        'error': task_result.get('error'),
                        'framework': task_result.get('framework'),
                        'model': task_result.get('model'),
                        'project_info': task_result.get('project_info')
                    })
                    if task_result.get('logs'):
                        self._save_logs(workspace, task_result['logs'])
                    break
                elif task_result['status'] == 'failed':
                    # Agent returned failed status
                    if attempt < max_attempts:
                        # Still have retries left
                        self.logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed: {task_result.get('error')}. Retrying..."
                        )
                        # Don't stop container, keep state for retry
                    else:
                        # Last attempt also failed
                        self.logger.error(f"All {max_attempts} attempts failed for {project.name}")
                        # Merge task result into main result
                        result.update({
                            'status': task_result['status'],
                            'logs': task_result.get('logs', []),
                            'conversation': task_result.get('conversation', []),
                            'error': task_result.get('error'),
                            'framework': task_result.get('framework'),
                            'model': task_result.get('model'),
                            'project_info': task_result.get('project_info')
                        })
                        # Save logs
                        if task_result.get('logs'):
                            self._save_logs(workspace, task_result['logs'])

        except Exception as e:
            self.logger.error(f"Error processing project {project.name}: {e}", exc_info=True)
            result['status'] = 'error'
            result['error'] = str(e)
            # Record exception in attempt history
            attempt_results.append({
                'attempt': len(attempt_results) + 1,
                'status': 'error',
                'error': str(e)
            })

        finally:
            # Add attempt tracking to result
            result['attempt_count'] = len(attempt_results)
            result['retry_count'] = len(attempt_results) - 1
            result['attempts_history'] = attempt_results

            # Copy workspace from container to host and stop container
            if workspace:
                container_id = self.sandbox_manager.get_container_id(workspace)
                if container_id:
                    try:
                        self.logger.info(f"Copying workspace from container for {project.name}...")
                        # Copy entire workspace directory from container to host
                        await self.sandbox_manager.copy_from_container(
                            workspace=workspace,
                            container_id=container_id,
                            container_path="/workspace",
                            host_path=workspace
                        )
                        self.logger.info(f"Workspace copied successfully.")

                        # Stop and remove container
                        self.logger.info(f"Stopping container for {project.name}...")
                        await self.sandbox_manager.stop_container(workspace)
                        self.logger.info(f"Container stopped and removed.")
                    except Exception as e:
                        self.logger.warning(f"Failed to copy workspace or stop container: {e}")

        end_time = datetime.now()
        result['end_time'] = end_time.isoformat()
        result['duration'] = (end_time - start_time).total_seconds()

        # Save result
        if workspace:
            self._save_result(workspace, result)

        duration_str = f"{result['duration']:.2f}s"
        self.logger.info(
            f"Completed project: {project.name} [{result['status']}] in {duration_str}"
        )

        return result

    async def run_all_projects(
        self,
        project_names: Optional[List[str]] = None,
        task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run inference on all projects (or specified subset).

        Args:
            project_names: Optional list of project names to run
                          (if None, run all projects)
            task_type: Optional filter for specific task type

        Returns:
            List of result dictionaries
        """
        # Discover projects
        self.logger.info(f"Discovering projects in {self.config.datasets_dir}")
        all_projects = self.dataset_manager.discover_projects(task_type=task_type)

        # Filter projects if specified
        if project_names:
            projects = [p for p in all_projects if p.name in project_names]
            if len(projects) < len(project_names):
                found_names = {p.name for p in projects}
                missing = set(project_names) - found_names
                self.logger.warning(f"Projects not found: {missing}")
        else:
            projects = all_projects

        if not projects:
            self.logger.warning("No projects found!")
            return []

        self.logger.info(f"Found {len(projects)} projects to process")

        # Check sandbox image
        self.logger.info(f"Checking sandbox image: {self.sandbox_manager.image_name}")
        if not await self.sandbox_manager.check_image():
            self.logger.error("Sandbox image not found. Please build it first.")
            return []

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.inference.max_workers)

        async def run_with_semaphore(project: Project):
            """Run project with semaphore control"""
            async with semaphore:
                return await self.run_single_project(project)

        # Create tasks
        tasks = [run_with_semaphore(project) for project in projects]

        # Run tasks in parallel
        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Process results
        valid_results = [r for r in results if isinstance(r, dict)]
        successful = sum(1 for r in valid_results if r.get('status') == 'success')
        failed = len(valid_results) - successful

        self.logger.info(
            f"Completed all projects in {total_duration:.2f}s\n"
            f"  Success: {successful}\n"
            f"  Failed: {failed}\n"
            f"  Total: {len(valid_results)}"
        )

        # Save summary
        summary = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': total_duration,
            'total_projects': len(valid_results),
            'successful': successful,
            'failed': failed,
            'config': {
                'framework': self.config.inference.framework,
                'model': self.config.inference.model,
                'max_workers': self.config.inference.max_workers,
                'task_type': task_type,
            },
            'results': valid_results
        }

        # Group by task type
        task_summary = {}
        for result in valid_results:
            task = result.get('task_type', 'unknown')
            if task not in task_summary:
                task_summary[task] = {'total': 0, 'success': 0, 'failed': 0}
            task_summary[task]['total'] += 1
            if result.get('status') == 'success':
                task_summary[task]['success'] += 1
            else:
                task_summary[task]['failed'] += 1
        
        summary['task_summary'] = task_summary

        summary_file = (
            self.config.results_dir /
            'summary.json'
        )
        summary_file.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Summary saved to {summary_file}")

        return valid_results
