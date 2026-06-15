"""Evaluation engine for functional testing and visual scoring in Vision2Web"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from vision2web.core.config import Config
from vision2web.core.logger import get_logger
from vision2web.core.dataset import DatasetManager, Project
from vision2web.core.sandbox import SandboxManager
from vision2web.evaluation.functional_tester import FunctionalTester
from vision2web.evaluation.visual_scorer import VisualScorer


class EvaluationEngine:
    """Main evaluation engine using Claude Code + playwright-cli for
    functional testing and VLM for visual scoring."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger('vision2web.evaluation')

        self.dataset_manager = DatasetManager(config.datasets_dir)

        self.sandbox_manager = SandboxManager(
            image_name=config.sandbox.image,
            workspace_dir=config.sandbox.workspace_dir,
            user=config.sandbox.user,
            logger=self.logger,
        )

        self.functional_tester = FunctionalTester(
            model=config.evaluation.functional_model,
            api_key=config.evaluation.functional_api_key,
            base_url=config.evaluation.functional_base_url,
            window_width=config.evaluation.window_width,
            window_height=config.evaluation.window_height,
        )

        self.visual_scorer = VisualScorer(
            api_key=config.evaluation.api_key,
            base_url=config.evaluation.base_url,
            visual_model=config.evaluation.visual_model,
        )

    def discover_result_projects(
        self,
        results_dir: Path,
        task_type: Optional[str] = None,
        framework: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        projects = []
        task_types = [task_type] if task_type else ['webpage', 'frontend', 'website']

        for task in task_types:
            task_dir = results_dir / task
            if not task_dir.exists() or not task_dir.is_dir():
                continue

            for framework_dir in task_dir.iterdir():
                if not framework_dir.is_dir() or framework_dir.name.startswith('.'):
                    continue
                if framework and framework_dir.name != framework:
                    continue

                for model_dir in framework_dir.iterdir():
                    if not model_dir.is_dir() or model_dir.name.startswith('.'):
                        continue
                    if model and model_dir.name != model:
                        continue

                    for project_dir in model_dir.iterdir():
                        if not project_dir.is_dir() or project_dir.name.startswith('.'):
                            continue

                        result_file = project_dir / 'evaluation_result.json'
                        if result_file.exists():
                            self.logger.info(
                                f"Skipping {project_dir.name}: evaluation_result.json exists"
                            )
                            continue
                        if not (project_dir / 'start.sh').exists():
                            self.logger.info(
                                f"Skipping {project_dir.name}: start.sh not found"
                            )
                            continue
                        if (project_dir / 'prototypes').exists():
                            projects.append({
                                'name': project_dir.name,
                                'task_type': task,
                                'framework': framework_dir.name,
                                'model': model_dir.name,
                                'path': project_dir,
                                'workspace': project_dir,
                            })

        self.logger.info(f"Found {len(projects)} inference results")
        return projects

    def _generate_container_name(self, result_project: Dict[str, Any]) -> str:
        task = result_project['task_type']
        project_name = result_project['name']
        framework = result_project['framework']
        model = result_project['model'].replace('/', '_').replace(':', '_')
        return f"{task}_{project_name}_{framework}_{model}"

    async def evaluate_single_project(
        self,
        result_project: Dict[str, Any],
        dataset_project: Project,
    ) -> Dict[str, Any]:
        project_name = result_project['name']
        self.logger.info(
            f"Evaluating project: {project_name} (task: {result_project['task_type']})"
        )

        start_time = datetime.now()
        result = {
            'project': project_name,
            'task_type': result_project['task_type'],
            'framework': result_project['framework'],
            'model': result_project['model'],
            'start_time': start_time.isoformat(),
        }

        workspace = result_project['workspace']
        container_id = None

        try:
            container_name = self._generate_container_name(result_project)

            self.logger.info(f"Creating container: {container_name}")
            container_id = await self.sandbox_manager.create_container(
                workspace,
                container_name=container_name,
            )
            if not container_id:
                result['status'] = 'error'
                result['error'] = 'Failed to create container'
                end_time = datetime.now()
                result['end_time'] = end_time.isoformat()
                result['duration'] = (end_time - start_time).total_seconds()
                self._save_evaluation_result(result)
                return result

            self.logger.info(f"Created container: {container_id[:12]}")

            await self.sandbox_manager.start_container(workspace)

            workflow_data = self.dataset_manager.load_workflow(dataset_project)

            await self._run_evaluation(
                container_id=container_id,
                workspace=workspace,
                dataset_project=dataset_project,
                workflow_data=workflow_data,
                result_project=result_project,
            )
            result['status'] = 'success'

        except Exception as e:
            self.logger.error(f"Error evaluating {project_name}: {e}", exc_info=True)
            result['status'] = 'error'
            result['error'] = str(e)

        finally:
            if container_id:
                try:
                    self.logger.info(f"Stopping container for {project_name}...")
                    await self.sandbox_manager.stop_container(workspace)
                    self.logger.info("Container stopped.")
                except Exception as e:
                    self.logger.warning(f"Failed to stop container: {e}")

        end_time = datetime.now()
        result['end_time'] = end_time.isoformat()
        result['duration'] = (end_time - start_time).total_seconds()

        # Persist the result for every outcome (success or error), so that
        # deploy timeouts / failures are recorded and skipped on re-runs
        # instead of being retried (each retry costs the full 600s port wait).
        self._save_evaluation_result(result)

        return result

    async def _wait_for_port(
        self,
        container_id: str,
        port: int = 3000,
        timeout: int = 600,
        interval: float = 5.0,
    ) -> bool:
        import time
        start_time = time.time()
        self.logger.info(f"Waiting for port {port} to become available...")

        while time.time() - start_time < timeout:
            try:
                check_cmd = [
                    "docker", "exec", container_id,
                    "bash", "-c", f"curl -s http://localhost:{port}",
                ]
                proc = await asyncio.create_subprocess_exec(
                    *check_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0 and stdout and stdout.strip():
                    elapsed = time.time() - start_time
                    self.logger.info(f"Port {port} available after {elapsed:.1f}s")
                    return True
            except Exception as e:
                self.logger.debug(f"Port check failed: {e}")

            await asyncio.sleep(interval)

        self.logger.error(f"Timeout waiting for port {port} after {timeout}s")
        return False

    async def _run_evaluation(
        self,
        container_id: str,
        workspace: Path,
        dataset_project: Project,
        workflow_data: List[Dict],
        result_project: Dict[str, Any],
    ):
        project_name = result_project['name']

        # Deploy project
        #
        # Redirect start.sh's stdout/stderr to a log file inside the container.
        # With "docker exec -d" the detached exec leaves stderr connected to a
        # pipe that nothing consumes; servers that log every request to stderr
        # (e.g. `python -m http.server`) eventually block on a full/broken pipe
        # mid-request, so the port listens but every request gets an empty
        # reply and _wait_for_port spins until its 600s timeout. Sending the
        # script's output to a file keeps the server unblocked and preserves
        # the deploy log for debugging on failure.
        self.logger.info("Deploying project with start.sh in background...")
        deploy_cmd = [
            "docker", "exec", "-d", "-w", "/workspace",
            container_id, "bash", "-c",
            "bash start.sh > /tmp/deploy.log 2>&1",
        ]
        deploy_proc = await asyncio.create_subprocess_exec(
            *deploy_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await deploy_proc.wait()

        if deploy_proc.returncode != 0:
            raise RuntimeError("Failed to start deployment in background")

        self.logger.info("Deployment started, waiting for service...")

        port_ready = await self._wait_for_port(container_id, port=3000, timeout=600)
        if not port_ready:
            logs_cmd = [
                "docker", "exec", container_id,
                "bash", "-c", "cat /tmp/deploy.log 2>/dev/null | tail -50",
            ]
            logs_proc = await asyncio.create_subprocess_exec(
                *logs_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            logs_stdout, logs_stderr = await logs_proc.communicate()
            self.logger.error(f"Deploy log (/tmp/deploy.log):\n{logs_stdout.decode()}")
            self.logger.error(f"Container errors:\n{logs_stderr.decode()}")
            raise RuntimeError("Service failed to start: port 3000 not available")

        self.logger.info("Service is ready")

        url = "http://localhost:3000"

        # Workflows run SERIALLY within a container, in the order they appear
        # in workflow.json. All playwright-cli commands share one default
        # browser instance per container, so concurrent workflows would clobber
        # each other's viewport size (resize), storage state, and even close
        # each other's browser. Serial execution keeps each workflow's browser
        # session clean. The dataset orders workflows so that any state set up
        # by an earlier workflow precedes the workflows that rely on it.
        failed_workflows = []
        successful_workflows = []

        for idx, workflow_item in enumerate(workflow_data):
            try:
                self.logger.info(
                    f"Running workflow {idx}: {workflow_item.get('summary', 'Unknown')}"
                )

                # Functional testing via Claude Code
                func_result = await self.functional_tester.run_workflow(
                    url=url,
                    workflow_item=workflow_item,
                    workflow_idx=idx,
                    output_dir=workspace,
                    container_id=container_id,
                    dataset_path=str(dataset_project.path),
                )

                # Visual scoring — temporarily disabled (VLM judge too slow).
                # Screenshots are still captured and copied to the host, so
                # scoring can be re-run later from the existing *_actual.png.
                # await self.visual_scorer.score_all_prototypes(
                #     workflow_item=workflow_item,
                #     workflow_idx=idx,
                #     dataset_path=str(dataset_project.path),
                #     output_dir=workspace,
                #     actual_pages=func_result.get('screenshots', {}),
                # )

                self.logger.info(f"Workflow {idx} completed")
                successful_workflows.append(idx)

            except Exception as e:
                self.logger.error(f"Error in workflow {idx}: {e}", exc_info=True)
                failed_workflows.append(idx)

        self.logger.info(
            f"Evaluation complete: {len(successful_workflows)} succeeded, "
            f"{len(failed_workflows)} failed"
        )

    def _save_evaluation_result(self, result: Dict[str, Any]):
        output_path = (
            self.config.results_dir
            / result['task_type']
            / result['framework']
            / result['model']
            / result['project']
        )
        output_path.mkdir(parents=True, exist_ok=True)

        result_file = output_path / 'evaluation_result.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved evaluation result to {result_file}")

    async def evaluate_all_projects(
        self,
        results_dir: Optional[Path] = None,
        task_type: Optional[str] = None,
        framework: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results_dir = results_dir or self.config.results_dir

        result_projects = self.discover_result_projects(
            results_dir,
            task_type=task_type,
            framework=framework,
            model=model,
        )

        if not result_projects:
            self.logger.warning("No inference results found")
            return []

        if not await self.sandbox_manager.check_image():
            self.logger.error("Sandbox image not found")
            return []

        self.logger.info(f"Found {len(result_projects)} projects to evaluate")

        semaphore = asyncio.Semaphore(self.config.evaluation.max_workers)

        async def evaluate_with_semaphore(result_project):
            async with semaphore:
                dataset_project = self.dataset_manager.get_project(
                    result_project['name'],
                    task_type=result_project['task_type'],
                )
                if not dataset_project:
                    self.logger.warning(
                        f"Dataset not found for {result_project['name']}"
                    )
                    return None
                return await self.evaluate_single_project(
                    result_project, dataset_project
                )

        tasks = [evaluate_with_semaphore(rp) for rp in result_projects]

        start_time = datetime.now()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        valid_results = [r for r in results if isinstance(r, dict) and r is not None]
        successful = sum(1 for r in valid_results if r.get('status') == 'success')
        failed = len(valid_results) - successful

        self.logger.info(
            f"Completed evaluation in {total_duration:.2f}s\n"
            f"  Success: {successful}\n"
            f"  Failed: {failed}\n"
            f"  Total: {len(valid_results)}"
        )

        return valid_results
