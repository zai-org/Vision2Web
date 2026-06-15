"""Base adapter interface for agent frameworks"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional


class BaseAdapter(ABC):
    """Abstract base class for agent framework adapters"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        sandbox_manager = None,
        logger: Optional[logging.Logger] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize the adapter.

        Args:
            api_key: API key for the agent framework
            model: Model name to use
            base_url: Optional API base URL
            sandbox_manager: SandboxManager instance for containerized execution
            logger: Optional logger instance
            timeout: Optional max seconds for a single task run (None = no limit)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.sandbox_manager = sandbox_manager
        self.logger = logger or logging.getLogger(__name__)
        self.timeout = timeout

    @abstractmethod
    async def run_task(
        self,
        workspace: Path,
        prompt: str,
        project_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a task with the agent framework.

        Args:
            workspace: Working directory for the agent
            prompt: Task prompt
            project_info: Project metadata

        Returns:
            Result dictionary with status, logs, error, etc.
            {
                'status': 'success' | 'failed',
                'logs': List[str],
                'error': Optional[str],
                'start_time': str (ISO format),
                'end_time': str (ISO format),
                'duration': float (seconds),
                'project_info': Dict,
                'framework': str,
                'model': str
            }
        """
        pass

    @property
    def framework_name(self) -> str:
        """Get the framework name"""
        return self.__class__.__name__.replace('Adapter', '').lower()
