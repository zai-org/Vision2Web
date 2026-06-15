"""OpenHands CLI adapter implementation for Vision2Web"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from vision2web.inference.adapters.base import BaseAdapter
from vision2web.core.utils import build_openhands_env, docker_env_flags


class OpenHandsAdapter(BaseAdapter):
    """Adapter that invokes the OpenHands CLI (`openhands --headless`) via docker exec."""

    @property
    def framework_name(self) -> str:
        return "openhands"

    async def run_task(
        self,
        workspace: Path,
        prompt: str,
        project_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.sandbox_manager:
            raise ValueError("Sandbox manager is required but not provided")

        start_time = datetime.now()
        logs = []
        status = 'failed'
        error_message = None
        conversation = []

        try:
            container_id = self.sandbox_manager.get_container_id(workspace)
            if container_id is None:
                container_id = await self.sandbox_manager.create_container(workspace)
                if container_id is None:
                    raise Exception("Failed to create sandbox container")
                await self.sandbox_manager.start_container(workspace)

            # The headless CLI reads LLM_MODEL / LLM_API_KEY / LLM_BASE_URL only
            # when invoked with --override-with-envs.
            env_flags = docker_env_flags(
                build_openhands_env(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    model=self.model,
                )
            )

            # `openhands --headless` runs a single task non-interactively
            # (always-approve; no confirmation prompts). --json streams JSONL
            # events on stdout. --override-with-envs applies the LLM_* env vars.
            cmd = [
                "docker", "exec",
                "-w", "/workspace",
                *env_flags,
                container_id,
                "openhands",
                "--headless",
                "--json",
                "--override-with-envs",
                "-t", prompt,
            ]

            self.logger.info(f"Running OpenHands CLI for {project_info['name']}...")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                if self.timeout:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self.timeout
                    )
                else:
                    stdout, stderr = await proc.communicate()
            except asyncio.TimeoutError:
                # OpenHands hung past the per-task limit. Kill the host-side
                # `docker exec` client, then reap the in-container openhands
                # process so it does not linger as an orphan keeping the
                # container busy with no API activity.
                self.logger.error(
                    f"OpenHands CLI timed out after {self.timeout}s for "
                    f"{project_info['name']}; killing process."
                )
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                await self._kill_container_process(container_id, "openhands")

                end_time = datetime.now()
                return {
                    'status': 'timeout',
                    'logs': logs + [f"Task timed out after {self.timeout}s"],
                    'conversation': [],
                    'error': f"OpenHands CLI timed out after {self.timeout}s",
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration': (end_time - start_time).total_seconds(),
                    'project_info': project_info,
                    'framework': self.framework_name,
                    'model': self.model,
                    'sandbox': True
                }

            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')

            # Parse the JSONL event stream into conversation messages
            for line in stdout_text.splitlines():
                line = line.strip()
                if line:
                    try:
                        conversation.append(json.loads(line))
                    except json.JSONDecodeError:
                        conversation.append({"type": "raw", "content": line})

            logs.extend(stderr_text.splitlines() if stderr_text else [])

            if proc.returncode == 0:
                check_code, check_stdout, _ = await self.sandbox_manager.exec_command(
                    workspace,
                    "test -f /workspace/start.sh && echo 'EXISTS' || echo 'NOT_FOUND'"
                )

                if 'EXISTS' in check_stdout:
                    status = 'success'
                    self.logger.info(f"Task completed successfully for {project_info['name']}")
                else:
                    status = 'failed'
                    error_message = "Agent completed but start.sh was not generated"
                    self.logger.error(error_message)
            else:
                error_message = f"OpenHands CLI exited with code {proc.returncode}"
                self.logger.error(error_message)

        except Exception as e:
            error_message = f"Error running OpenHands CLI: {e}"
            self.logger.error(error_message, exc_info=True)
            logs.append(error_message)

            import traceback
            logs.append(traceback.format_exc())

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            'status': status,
            'logs': logs,
            'conversation': conversation,
            'error': error_message,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration': duration,
            'project_info': project_info,
            'framework': self.framework_name,
            'model': self.model,
            'sandbox': True
        }

    async def _kill_container_process(self, container_id: str, name: str) -> None:
        """Kill any lingering CLI process inside the container.

        Killing the host-side `docker exec` client does not necessarily stop
        the process it spawned inside the container, which would otherwise keep
        running (and keep the container busy) with no API activity.
        """
        if not container_id:
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", container_id,
                "pkill", "-9", "-f", name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        except Exception as e:
            self.logger.warning(f"Failed to kill in-container {name} process: {e}")
