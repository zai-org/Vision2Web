"""Functional testing module using Claude Code + playwright-cli for Vision2Web"""

import asyncio
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from vision2web.evaluation.prompts import FUNCTIONAL_PROMPT
from vision2web.core.utils import build_claude_code_env, docker_env_flags


class FunctionalTester:
    """Runs structured test workflows using Claude Code + playwright-cli.

    A workflow is a sequence of verification nodes (functional and visual)
    arranged in execution order. Functional nodes execute actions and check
    validations. Visual nodes capture screenshots for later VLM comparison.

    Each workflow runs as a single Claude Code session inside a Docker
    container, maintaining browser state across all nodes.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        window_width: int = 1920,
        window_height: int = 1080,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.window_width = window_width
        self.window_height = window_height
        self.logger = logging.getLogger('FunctionalTester')

    def _build_execution_plan(
        self,
        workflow_item: Dict,
        output_dir: str,
        workflow_idx: int,
    ) -> str:
        """Build an interleaved execution plan from content (functional nodes)
        and prototype (visual nodes), ordered by idx.

        The prototype 'idx' field indicates the position in the execution
        sequence: idx 0 = before any test case (initial page load),
        idx N = after test case N-1. This maps visual nodes to their
        correct positions between functional nodes.
        """
        content = workflow_item.get('content', [])
        prototype_info = workflow_item.get('prototype', {})

        # Group visual nodes by their idx position
        # idx 0 → before test_case 0, idx N → after test_case N-1
        visual_at_position = {}
        for proto_name, proto_config in prototype_info.items():
            idx = proto_config.get('idx', 0)
            fullpage = proto_config.get('fullpage', True)
            if idx not in visual_at_position:
                visual_at_position[idx] = []
            visual_at_position[idx].append({
                'name': proto_name,
                'fullpage': fullpage,
            })

        steps = []
        step_num = 0

        # Position 0: visual nodes at the initial page load
        if 0 in visual_at_position:
            for vn in visual_at_position[0]:
                steps.append(self._format_visual_step(
                    step_num, vn['name'], output_dir, workflow_idx, vn['fullpage'],
                ))
                step_num += 1

        # Interleave functional and visual nodes
        for tc_idx, test_case in enumerate(content):
            # Functional node
            steps.append(self._format_functional_step(
                step_num, tc_idx, test_case, output_dir, workflow_idx,
            ))
            step_num += 1

            # Visual nodes after this test case (position = tc_idx + 1)
            pos = tc_idx + 1
            if pos in visual_at_position:
                for vn in visual_at_position[pos]:
                    steps.append(self._format_visual_step(
                        step_num, vn['name'], output_dir, workflow_idx, vn['fullpage'],
                    ))
                    step_num += 1

        # Any visual nodes at positions beyond the last test case
        max_pos = len(content)
        for pos in sorted(visual_at_position.keys()):
            if pos > max_pos:
                for vn in visual_at_position[pos]:
                    steps.append(self._format_visual_step(
                        step_num, vn['name'], output_dir, workflow_idx, vn['fullpage'],
                    ))
                    step_num += 1

        if not steps:
            return "(No verification nodes in this workflow.)"

        return "\n\n".join(steps)

    def _format_functional_step(
        self,
        step_num: int,
        test_case_idx: int,
        test_case: Dict,
        output_dir: str,
        workflow_idx: int,
    ) -> str:
        objective = test_case.get('objective', '')
        actions = test_case.get('actions', [])
        validations = test_case.get('validations', [])

        actions_text = "\n".join(
            f"   {j+1}. {a}" for j, a in enumerate(actions)
        )

        if validations:
            validations_text = "\n".join(
                f"   {j+1}. {v}" for j, v in enumerate(validations)
            )
        else:
            validations_text = "   (No validations — Pass if actions complete successfully)"

        result_dir = f"{output_dir}/test_results/workflow_{workflow_idx}/test_case_{test_case_idx}"

        return (
            f"---\n"
            f"### Step {step_num} — FUNCTIONAL NODE (test_case_{test_case_idx})\n"
            f"**Objective:** {objective}\n\n"
            f"**Actions:**\n{actions_text}\n\n"
            f"**Validations:**\n{validations_text}\n\n"
            f"**Save result to:** `{result_dir}/result.json`\n"
            f"(Create directory first: `mkdir -p {result_dir}`)"
        )

    def _format_visual_step(
        self,
        step_num: int,
        proto_name: str,
        output_dir: str,
        workflow_idx: int,
        fullpage: bool = True,
    ) -> str:
        screenshot_dir = f"{output_dir}/test_results/prototypes"
        screenshot_path = f"{screenshot_dir}/{proto_name}_actual.png"

        if fullpage:
            capture_desc = (
                "Take a **full-page** screenshot (the entire scrollable page) "
                "of the current page state."
            )
            screenshot_cmd = (
                f"playwright-cli screenshot --full-page --filename={screenshot_path}"
            )
        else:
            capture_desc = (
                "Take a **viewport** screenshot (only the currently visible area, "
                "do NOT capture the full scrollable page) of the current page state."
            )
            screenshot_cmd = (
                f"playwright-cli screenshot --filename={screenshot_path}"
            )

        return (
            f"---\n"
            f"### Step {step_num} — VISUAL NODE (prototype: {proto_name})\n"
            f"{capture_desc}\n"
            f"```bash\n"
            f"mkdir -p {screenshot_dir}\n"
            f"{screenshot_cmd}\n"
            f"```"
        )

    def _build_workflow_prompt(
        self,
        url: str,
        workflow_item: Dict,
        workflow_idx: int,
        output_dir: str,
    ) -> str:
        resolution = workflow_item.get('resolution', {})
        width = resolution.get('width', self.window_width)
        height = resolution.get('height', self.window_height)

        execution_plan = self._build_execution_plan(
            workflow_item, output_dir, workflow_idx,
        )

        return FUNCTIONAL_PROMPT.format(
            url=url,
            width=width,
            height=height,
            workflow_idx=workflow_idx,
            execution_plan=execution_plan,
        )

    async def _invoke_claude(
        self,
        prompt: str,
        container_id: str,
    ) -> int:
        env_flags = docker_env_flags(
            build_claude_code_env(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
            )
        )

        cmd = [
            "docker", "exec",
            "-w", "/workspace",
            *env_flags,
            container_id,
            "claude",
            "--print",
            "--model", self.model,
            "--dangerously-skip-permissions",
            "-p", prompt,
        ]

        self.logger.info(
            f"Invoking Claude Code inside container {container_id[:12]} "
            f"with model {self.model}..."
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if stdout:
            self.logger.info(
                f"Claude Code output:\n"
                f"{stdout.decode('utf-8', errors='replace')[:2000]}"
            )
        if stderr:
            self.logger.warning(
                f"Claude Code stderr:\n"
                f"{stderr.decode('utf-8', errors='replace')[:2000]}"
            )

        self.logger.info(f"Claude Code exited with code {proc.returncode}")
        return proc.returncode

    def _collect_results(
        self,
        output_dir: Path,
        workflow_idx: int,
        workflow_item: Dict,
    ) -> Dict[str, Any]:
        """Collect functional test results and visual node screenshots
        from disk after Claude Code execution."""
        content = workflow_item.get('content', [])
        prototype_info = workflow_item.get('prototype', {})

        results = []
        screenshots = {}

        test_results_base = output_dir / 'test_results'

        # Collect functional test results
        for i, test_case in enumerate(content):
            result_file = (
                test_results_base / f'workflow_{workflow_idx}'
                / f'test_case_{i}' / 'result.json'
            )

            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    # Normalize on-disk formatting: Claude Code writes these
                    # files with inconsistent indentation, so rewrite each one
                    # with indent=4 for a uniform result.json layout.
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, indent=4, ensure_ascii=False)
                    results.append(result_data)
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.error(f"Failed to read result for test_case_{i}: {e}")
                    results.append({
                        'test_case': test_case,
                        'result': 'Blocked',
                        'reasoning': f'Failed to read result file: {e}',
                        'process': [],
                    })
            else:
                self.logger.warning(
                    f"No result file for workflow_{workflow_idx}/test_case_{i}"
                )
                results.append({
                    'test_case': test_case,
                    'result': 'Blocked',
                    'reasoning': 'Result file not generated by Claude Code',
                    'process': [],
                })

        # Collect visual node screenshots (keyed by prototype name)
        for proto_name in prototype_info:
            screenshot_file = (
                test_results_base / 'prototypes' / f'{proto_name}_actual.png'
            )
            if screenshot_file.exists():
                screenshots[proto_name] = screenshot_file.read_bytes()
            else:
                self.logger.warning(
                    f"No screenshot for prototype '{proto_name}'"
                )

        return {
            'results': results,
            'screenshots': screenshots,
        }

    async def run_workflow(
        self,
        url: str,
        workflow_item: Dict,
        workflow_idx: int,
        output_dir: Path,
        container_id: str,
        dataset_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a complete workflow (functional + visual nodes) via a
        single Claude Code session inside the Docker container.

        Args:
            url: URL of the deployed web application (inside container)
            workflow_item: Workflow definition from workflow.json
            workflow_idx: Workflow index
            output_dir: Host output directory for collecting results
            container_id: Docker container ID to exec into
            dataset_path: Path to dataset (unused, kept for interface compat)

        Returns:
            Dict with:
              'results': list of functional test case results
              'screenshots': dict of prototype name → PNG bytes
        """
        content = workflow_item.get('content', [])
        prototype_info = workflow_item.get('prototype', {})

        self.logger.info(
            f"Running workflow {workflow_idx}: "
            f"{len(content)} functional nodes, "
            f"{len(prototype_info)} visual nodes"
        )

        # Fast path: a workflow with no functional nodes is a pure
        # screenshot task. There is nothing to reason about, so skip the
        # Claude Code session entirely and drive playwright-cli directly.
        if len(content) == 0 and prototype_info:
            self.logger.info(
                f"Workflow {workflow_idx} is screenshot-only; "
                f"taking screenshots directly without Claude Code"
            )
            await self._run_screenshot_only(
                url=url,
                workflow_item=workflow_item,
                container_id=container_id,
            )
            await self._copy_results_from_container(container_id, output_dir)
            return self._collect_results(output_dir, workflow_idx, workflow_item)

        prompt = self._build_workflow_prompt(
            url=url,
            workflow_item=workflow_item,
            workflow_idx=workflow_idx,
            output_dir="/workspace",
        )

        exit_code = await self._invoke_claude(
            prompt=prompt,
            container_id=container_id,
        )

        if exit_code != 0:
            self.logger.error(
                f"Claude Code exited with non-zero code {exit_code} "
                f"for workflow {workflow_idx}"
            )

        # Copy test_results from container to host
        await self._copy_results_from_container(container_id, output_dir)

        return self._collect_results(output_dir, workflow_idx, workflow_item)

    async def _run_screenshot_only(
        self,
        url: str,
        workflow_item: Dict,
        container_id: str,
    ) -> None:
        """Capture screenshots for a screenshot-only workflow by running
        playwright-cli directly inside the container (no Claude Code).

        Produces ``/workspace/test_results/prototypes/<name>_actual.png`` for
        each prototype, matching what the agent path writes, so the existing
        copy-back and collection logic can be reused unchanged.

        Failures are logged but not raised: a missing screenshot degrades to
        an unscored prototype downstream (same as the agent path), rather than
        falling back to a Claude Code session.
        """
        resolution = workflow_item.get('resolution', {})
        width = resolution.get('width', self.window_width)
        height = resolution.get('height', self.window_height)
        prototype_info = workflow_item.get('prototype', {})

        screenshot_dir = "/workspace/test_results/prototypes"

        cmds = [
            f"playwright-cli open {url}",
            f"playwright-cli resize {width} {height}",
            f"mkdir -p {screenshot_dir}",
        ]
        for proto_name, proto_config in prototype_info.items():
            fullpage = proto_config.get('fullpage', True)
            path = f"{screenshot_dir}/{proto_name}_actual.png"
            flag = "--full-page " if fullpage else ""
            cmds.append(f"playwright-cli screenshot {flag}--filename={path}")
        cmds.append("playwright-cli close")

        script = " && ".join(cmds)

        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-w", "/workspace", container_id,
            "bash", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            self.logger.warning(
                f"Screenshot-only run exited with code {proc.returncode}; "
                f"some screenshots may be missing.\n"
                f"stderr: {stderr.decode('utf-8', errors='replace')[:1000]}"
            )
        else:
            self.logger.info("Screenshot-only run completed")

    async def _copy_results_from_container(
        self,
        container_id: str,
        output_dir: Path,
    ) -> None:
        """Copy test_results directory from container to host."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy the directory itself (not its contents) into output_dir, so it
        # lands at output_dir/test_results. Using the trailing "/." form while
        # pre-creating output_dir/test_results nests it as
        # output_dir/test_results/test_results, which is where _collect_results
        # then fails to find result.json / prototype screenshots.
        cmd = [
            "docker", "cp",
            f"{container_id}:/workspace/test_results",
            str(output_dir),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            self.logger.warning(
                f"Failed to copy test_results from container: "
                f"{stderr.decode('utf-8', errors='replace')}"
            )
        else:
            self.logger.info("Copied test_results from container to host")
