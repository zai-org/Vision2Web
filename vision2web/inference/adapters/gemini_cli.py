"""Gemini CLI adapter implementation for Vision2Web"""

import asyncio
import json
import shlex
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from vision2web.inference.adapters.base import BaseAdapter
from vision2web.core.utils import (
    build_gemini_cli_env,
    build_gemini_cli_settings,
    docker_env_flags,
)


class GeminiCliAdapter(BaseAdapter):
    """Adapter that invokes the Gemini CLI (`gemini`) via docker exec."""

    @property
    def framework_name(self) -> str:
        return "gemini_cli"

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

            env_flags = docker_env_flags(
                build_gemini_cli_env(
                    base_url=self.base_url,
                    api_key=self.api_key,
                )
            )

            # The Gemini CLI reads everything except its credentials from
            # ~/.gemini/settings.json: the auth type, folder trust (which
            # otherwise downgrades --yolo), model routing and compression.
            await self._write_settings_file(container_id)

            # The prompt is piped in from a file instead of being passed as an
            # argv element, for the same reason as in the Claude Code adapter:
            # its text contains `bash /workspace/start.sh` and
            # `http://localhost:3000`, so an argv-borne prompt puts those
            # strings in the agent process' own command line - the agent's
            # cleanup commands (`ps aux | grep start.sh | xargs kill`) and the
            # `pkill -f "localhost:3000"` inside generated start.sh scripts
            # then match the agent itself and kill the task mid-run.
            prompt_path = f"/tmp/v2w_prompt_{uuid.uuid4().hex}.txt"
            await self._write_prompt_file(container_id, prompt_path, prompt)

            # Piped stdin (a non-TTY) makes the CLI run headless and take the
            # piped text as its prompt; no -p/--prompt argument is needed.
            # --yolo auto-approves every tool call.
            gemini_cmd = (
                f"cat {shlex.quote(prompt_path)} | gemini"
                " --yolo"
                " --output-format stream-json"
                f" --model {shlex.quote(self.model)}"
            )

            cmd = [
                "docker", "exec",
                "-w", "/workspace",
                *env_flags,
                container_id,
                "sh", "-c", gemini_cmd,
            ]

            self.logger.info(f"Running Gemini CLI for {project_info['name']}...")

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
                # The Gemini CLI hung past the per-task limit. Kill the
                # host-side `docker exec` client, then reap the in-container
                # gemini process so it does not linger as an orphan keeping the
                # container busy with no API activity.
                self.logger.error(
                    f"Gemini CLI timed out after {self.timeout}s for "
                    f"{project_info['name']}; killing process."
                )
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                await self._kill_container_process(container_id, "gemini")

                end_time = datetime.now()
                return {
                    'status': 'timeout',
                    'logs': logs + [f"Task timed out after {self.timeout}s"],
                    'conversation': [],
                    'error': f"Gemini CLI timed out after {self.timeout}s",
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

            # Parse stream-json to extract conversation messages
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
                error_message = f"Gemini CLI exited with code {proc.returncode}"
                self.logger.error(error_message)

        except Exception as e:
            error_message = f"Error running Gemini CLI: {e}"
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

    async def _write_settings_file(self, container_id: str) -> None:
        """Write ~/.gemini/settings.json inside the container."""
        settings = json.dumps(
            build_gemini_cli_settings(model=self.model), indent=2
        ) + "\n"

        await self._write_container_file(
            container_id,
            "mkdir -p ~/.gemini && cat > ~/.gemini/settings.json",
            settings,
            "settings.json",
        )

    async def _write_prompt_file(
        self,
        container_id: str,
        path: str,
        prompt: str,
    ) -> None:
        """Write the prompt into the container, feeding it over stdin.

        Kept off the command line for the same reason the prompt is not passed
        to `gemini` as an argument: anything in a command line shows up in the
        container's `ps` output, where the agent's own process-cleanup commands
        can match it.
        """
        await self._write_container_file(
            container_id,
            f"cat > {shlex.quote(path)}",
            prompt,
            "prompt file",
        )

    async def _write_container_file(
        self,
        container_id: str,
        write_cmd: str,
        content: str,
        description: str,
    ) -> None:
        """Run a shell command in the container with ``content`` on its stdin.

        Piping the content avoids any quoting issues in the shell, and keeps it
        out of the container's `ps` output.
        """
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", container_id,
            "sh", "-c", write_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate(content.encode('utf-8'))

        if proc.returncode != 0:
            detail = stderr.decode('utf-8', errors='replace').strip()
            raise Exception(
                f"Failed to write {description} into container: {detail}"
            )

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
