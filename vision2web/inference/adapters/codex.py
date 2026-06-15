"""Codex CLI adapter implementation for Vision2Web"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from vision2web.inference.adapters.base import BaseAdapter


class CodexAdapter(BaseAdapter):
    """Adapter that invokes the Codex CLI (`codex exec`) via docker exec."""

    @property
    def framework_name(self) -> str:
        return "codex"

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

            # Codex 0.138 ignores OPENAI_BASE_URL; a custom (proxy/gateway)
            # endpoint must be registered as a model provider in
            # ~/.codex/config.toml. wire_api must be "responses" ("chat" was
            # removed). The bearer token is carried via experimental_bearer_token
            # so no env credential is needed.
            await self._write_codex_config(container_id)

            # `codex exec` runs non-interactively (approval policy Never).
            # --skip-git-repo-check: /workspace is not a git repo.
            # --dangerously-bypass-approvals-and-sandbox: the outer Docker
            #   container already provides isolation.
            cmd = [
                "docker", "exec",
                "-w", "/workspace",
                container_id,
                "codex", "exec",
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "--model", self.model,
                prompt,
            ]

            self.logger.info(f"Running Codex CLI for {project_info['name']}...")

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
                # Codex hung past the per-task limit. Kill the host-side
                # `docker exec` client, then reap the in-container codex
                # process so it does not linger as an orphan keeping the
                # container busy with no API activity.
                self.logger.error(
                    f"Codex CLI timed out after {self.timeout}s for "
                    f"{project_info['name']}; killing process."
                )
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
                await self._kill_container_process(container_id, "codex")

                end_time = datetime.now()
                return {
                    'status': 'timeout',
                    'logs': logs + [f"Task timed out after {self.timeout}s"],
                    'conversation': [],
                    'error': f"Codex CLI timed out after {self.timeout}s",
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

            logs.extend(stdout_text.splitlines() if stdout_text else [])
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
                error_message = f"Codex CLI exited with code {proc.returncode}"
                self.logger.error(error_message)

        except Exception as e:
            error_message = f"Error running Codex CLI: {e}"
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

    async def _write_codex_config(self, container_id: str) -> None:
        """Write ~/.codex/config.toml inside the container.

        Registers the configured base_url as a custom model provider so Codex
        routes through the gateway/proxy instead of api.openai.com. The bearer
        token is embedded directly (experimental_bearer_token) and
        requires_openai_auth is disabled so Codex does not attempt its own
        OpenAI auth flow.
        """
        # TOML basic strings: escape backslashes and double quotes.
        def toml_str(value: str) -> str:
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'

        lines = [
            f"model = {toml_str(self.model)}",
            'model_reasoning_effort = "high"',
            'model_provider = "vision2web-gateway"',
            "",
            "[model_providers.vision2web-gateway]",
            'name = "vision2web-gateway"',
            f"base_url = {toml_str(self.base_url)}",
            'wire_api = "responses"',
            "requires_openai_auth = false",
        ]
        if self.api_key:
            lines.append(f"experimental_bearer_token = {toml_str(self.api_key)}")
        config_toml = "\n".join(lines) + "\n"

        # Write atomically from stdin to avoid quoting issues in the shell.
        write_cmd = (
            "mkdir -p ~/.codex && cat > ~/.codex/config.toml"
        )
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", container_id,
            "sh", "-c", write_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate(input=config_toml.encode("utf-8"))
        if proc.returncode != 0:
            raise Exception(
                f"Failed to write codex config.toml: "
                f"{stderr.decode('utf-8', errors='replace')}"
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
