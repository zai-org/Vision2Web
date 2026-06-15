"""Docker sandbox management for isolated task execution"""

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from vision2web.core.constants import (
    CONTAINER_PREFIX,
    CONTAINER_WORKSPACE,
    CONTAINER_TMP_WORKSPACE,
    CONTAINER_USER,
    DOCKER_CREATE_TIMEOUT,
    DOCKER_START_TIMEOUT,
    DOCKER_STOP_TIMEOUT,
    DOCKER_EXEC_TIMEOUT,
    DOCKER_COPY_TIMEOUT,
)


class SandboxManager:
    """Manages Docker containers for running tasks in isolation"""

    def __init__(
        self,
        image_name: str,
        workspace_dir: str = CONTAINER_WORKSPACE,
        user: str = CONTAINER_USER,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize sandbox manager.

        Args:
            image_name: Docker image name to use
            workspace_dir: Working directory inside container
            user: User to run commands as
            logger: Optional logger instance
        """
        self.image_name = image_name
        self.workspace_dir = workspace_dir
        self.user = user
        self.logger = logger or logging.getLogger(__name__)
        self._containers: Dict[str, str] = {}  # workspace_path -> container_id

        # Load proxy configuration
        self.proxy_env = self._get_proxy_env()
        if self.proxy_env:
            self.logger.debug(f"Proxy config loaded: {list(self.proxy_env.keys())}")

    def _get_proxy_env(self) -> Dict[str, str]:
        """Get proxy environment variables from host"""
        proxy_vars = {}
        proxy_keys = [
            'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
            'no_proxy', 'NO_PROXY', 'all_proxy', 'ALL_PROXY'
        ]

        for key in proxy_keys:
            value = os.environ.get(key)
            if value:
                proxy_vars[key] = value

        return proxy_vars

    async def _exec_docker_command(
        self,
        cmd: List[str],
        error_msg: str,
        timeout: Optional[int] = None
    ) -> Tuple[int, str, str]:
        """
        Execute a Docker command with error handling.

        Args:
            cmd: Command arguments list
            error_msg: Error message prefix for logging
            timeout: Optional timeout in seconds

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            if timeout:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
            else:
                stdout, stderr = await proc.communicate()

            return (
                proc.returncode,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace")
            )

        except asyncio.TimeoutError:
            self.logger.error(f"{error_msg}: Command timed out after {timeout}s")
            return (-1, "", f"Command timed out after {timeout}s")
        except Exception as e:
            self.logger.error(f"{error_msg}: {e}")
            return (-1, "", str(e))

    async def check_image(self) -> bool:
        """
        Check if Docker image exists.

        Returns:
            True if image exists, False otherwise
        """
        cmd = ["docker", "images", "-q", self.image_name]
        returncode, stdout, stderr = await self._exec_docker_command(
            cmd,
            "Failed to check image",
            timeout=DOCKER_CREATE_TIMEOUT
        )

        if returncode != 0:
            self.logger.error(f"Failed to check image: {stderr}")
            return False

        if stdout.strip():
            self.logger.info(f"Docker image found: {self.image_name}")
            return True

        self.logger.error(f"Docker image not found: {self.image_name}")
        return False

    async def create_container(
        self,
        workspace: Path,
        container_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a Docker container for a workspace.

        Args:
            workspace: Host workspace directory
            container_name: Optional container name (auto-generated if not provided)
        Returns:
            Container ID if successful, None otherwise
        """
        if not workspace.exists():
            self.logger.error(f"Workspace does not exist: {workspace}")
            return None

        # Generate container name
        if container_name is None:
            workspace_hash = hashlib.md5(str(workspace).encode()).hexdigest()[:8]
            container_name = f"{CONTAINER_PREFIX}{workspace.name}-{workspace_hash}"

        # Build docker create command
        cmd = [
            "docker", "create",
            "--name", container_name,
            "--workdir", self.workspace_dir,
            "--rm",  # Auto-remove when stopped
        ]

        # Add proxy environment variables
        for key, value in self.proxy_env.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Add image and command
        cmd.extend([self.image_name, "sleep", "infinity"])

        returncode, stdout, stderr = await self._exec_docker_command(
            cmd,
            "Failed to create container",
            timeout=DOCKER_CREATE_TIMEOUT
        )

        if returncode != 0:
            self.logger.error(f"Failed to create container: {stderr}")
            return None

        container_id = stdout.strip()
        self._containers[str(workspace)] = container_id
        self.logger.info(f"Created container {container_name} ({container_id[:12]})")

        return container_id

    async def start_container(self, workspace: Path) -> bool:
        """
        Start a container and copy workspace contents to it.

        Args:
            workspace: Host workspace directory

        Returns:
            True if successful
        """
        workspace_str = str(workspace)
        if workspace_str not in self._containers:
            container_id = await self.create_container(workspace)
            if container_id is None:
                return False

        container_id = self._containers[workspace_str]

        try:
            # Start the container
            result = await asyncio.create_subprocess_exec(
                "docker", "start", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await result.communicate()

            if result.returncode != 0:
                self.logger.error(f"Failed to start container: {stderr.decode()}")
                return False

            self.logger.debug(f"Started container {container_id[:12]}")

            # Copy workspace contents to container
            await self.copy_to_container(workspace, container_id)

            return True

        except Exception as e:
            self.logger.error(f"Error starting container: {e}")
            return False

    async def copy_to_container(
        self,
        workspace: Path,
        container_id: Optional[str] = None
    ) -> bool:
        """
        Copy workspace contents to container.

        Args:
            workspace: Host workspace directory
            container_id: Container ID (looked up if not provided)

        Returns:
            True if successful
        """
        if container_id is None:
            workspace_str = str(workspace)
            if workspace_str not in self._containers:
                self.logger.error(f"No container found for workspace: {workspace}")
                return False
            container_id = self._containers[workspace_str]

        if not workspace.exists():
            self.logger.error(f"Workspace does not exist: {workspace}")
            return False

        try:
            self.logger.debug(f"Copying workspace to container {container_id[:12]}...")

            # Copy to temporary location
            result = await asyncio.create_subprocess_exec(
                "docker", "cp",
                str(workspace) + "/.", f"{container_id}:/tmp/workspace",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await result.communicate()

            if result.returncode != 0:
                self.logger.error(f"Failed to copy to container: {stderr.decode()}")
                return False

            # Move files to workspace and fix permissions
            move_cmd = (
                "rm -rf /workspace/* /workspace/.[!.]* /workspace/..?* 2>/dev/null || true; "
                "if [ -d /tmp/workspace ]; then "
                "  cp -r /tmp/workspace/* /workspace/ 2>/dev/null || true; "
                "  cp -r /tmp/workspace/.[!.]* /workspace/ 2>/dev/null || true; "
                "  cp -r /tmp/workspace/..?* /workspace/ 2>/dev/null || true; "
                "  rm -rf /tmp/workspace; "
                "fi"
            )

            result = await asyncio.create_subprocess_exec(
                "docker", "exec", container_id,
                "sh", "-c", move_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await result.communicate()

            if result.returncode != 0:
                self.logger.warning(f"Failed to move files: {stderr.decode()}")

            self.logger.debug(f"Copied workspace to container {container_id[:12]}")
            return True

        except Exception as e:
            self.logger.error(f"Error copying to container: {e}")
            return False

    async def copy_from_container(
        self,
        workspace: Path,
        container_id: Optional[str] = None,
        container_path: Optional[str] = None,
        host_path: Optional[Path] = None
    ) -> bool:
        """
        Copy contents from container to host.

        Args:
            workspace: Workspace directory (used to lookup container_id if not provided)
            container_id: Container ID (looked up if not provided)
            container_path: Path inside container to copy from (default: workspace_dir)
            host_path: Path on host to copy to (default: workspace)

        Returns:
            True if successful
        """
        if container_id is None:
            workspace_str = str(workspace)
            if workspace_str not in self._containers:
                self.logger.error(f"No container found for workspace: {workspace}")
                return False
            container_id = self._containers[workspace_str]

        # Use default paths if not provided
        if container_path is None:
            container_path = self.workspace_dir
        if host_path is None:
            host_path = workspace

        try:
            # Ensure target directory exists
            host_path.mkdir(parents=True, exist_ok=True)

            self.logger.debug(f"Copying {container_path} from container {container_id[:12]} to {host_path}...")

            # Stream the contents of container_path (not the directory itself) into
            # host_path via tar. `docker cp <id>:<path>/.` is unreliable across Docker
            # versions (some ignore the trailing `/.` and nest the copy under
            # host_path/<basename>), so we use tar to land the contents flat.
            copy_cmd = (
                f"docker exec {container_id} tar -C {container_path} -cf - . "
                f"| tar -C {str(host_path)} -xf -"
            )
            result = await asyncio.create_subprocess_exec(
                "bash", "-c", copy_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await result.communicate()

            if result.returncode != 0:
                self.logger.error(f"Failed to copy from container: {stderr.decode()}")
                return False

            self.logger.debug(f"Copied {container_path} from container {container_id[:12]} to {host_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error copying from container: {e}")
            return False

    async def exec_command(
        self,
        workspace: Path,
        command: str,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        user: Optional[str] = None
    ) -> Tuple[int, str, str]:
        """
        Execute a command in the container.

        Args:
            workspace: Workspace directory
            command: Command to execute
            env: Optional environment variables
            cwd: Optional working directory (relative to workspace_dir)
            user: Optional user to run as (default: agent)

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        workspace_str = str(workspace)
        if workspace_str not in self._containers:
            # Create and start container if it doesn't exist
            container_id = await self.create_container(workspace)
            if container_id is None:
                return (-1, "", "Failed to create container")
            await self.start_container(workspace)

        container_id = self._containers[workspace_str]

        # Build docker exec command
        cmd = ["docker", "exec"]

        # Set user
        if user:
            cmd.extend(["--user", user])
        else:
            cmd.extend(["--user", self.user])

        # Add proxy environment variables first
        for key, value in self.proxy_env.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Add user-provided environment variables
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        # Set working directory
        if cwd:
            cmd.extend(["-w", f"{self.workspace_dir}/{cwd}"])
        else:
            cmd.extend(["-w", self.workspace_dir])

        # Add container and command
        cmd.extend([container_id, "sh", "-c", command])

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            return (
                result.returncode,
                stdout.decode('utf-8', errors='replace'),
                stderr.decode('utf-8', errors='replace')
            )

        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return (-1, "", str(e))

    async def stop_container(self, workspace: Path) -> bool:
        """
        Stop the container.

        Args:
            workspace: Workspace directory

        Returns:
            True if successful
        """
        workspace_str = str(workspace)
        if workspace_str not in self._containers:
            self.logger.warning(f"No container found for workspace: {workspace}")
            return False

        container_id = self._containers[workspace_str]

        try:
            # Stop the container
            result = await asyncio.create_subprocess_exec(
                "docker", "stop", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await result.communicate()

            if result.returncode != 0:
                self.logger.warning(f"Failed to stop container: {stderr.decode()}")

            # Remove from tracking
            del self._containers[workspace_str]
            self.logger.debug(f"Stopped container {container_id[:12]}")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping container: {e}")
            return False

    async def cleanup_all(self):
        """Stop and remove all managed containers"""
        workspaces = list(self._containers.keys())
        for workspace_str in workspaces:
            workspace = Path(workspace_str)
            await self.stop_container(workspace)

    def get_container_id(self, workspace: Path) -> Optional[str]:
        """Get container ID for a workspace"""
        return self._containers.get(str(workspace))

    async def find_container_by_name(self, name_pattern: str) -> Optional[str]:
        """
        Find a container by name pattern.

        Args:
            name_pattern: Container name or pattern

        Returns:
            Container ID if found, None otherwise
        """
        try:
            result = await asyncio.create_subprocess_exec(
                "docker", "ps", "--filter", f"name={name_pattern}",
                "--format", "{{.ID}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()

            if stdout.strip():
                return stdout.decode().strip()
            return None

        except Exception as e:
            self.logger.error(f"Error finding container: {e}")
            return None
