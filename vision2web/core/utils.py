"""Utility functions for Vision2Web."""

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def build_claude_code_env(
    base_url: Optional[str],
    api_key: Optional[str],
    model: str,
) -> Dict[str, str]:
    """Build the environment variables for running Claude Code CLI.

    Shared by the inference adapter and the functional-test runner so that
    Claude Code is invoked with an identical environment in both phases.

    Args:
        base_url: Anthropic-compatible API base URL (e.g. LiteLLM proxy).
        api_key: Auth token for the API.
        model: Model identifier; routed to every Claude tier so any internal
               model selection resolves to the target evaluation model.

    Returns:
        Mapping of environment variable name to value. Keys whose value is
        None are omitted.
    """
    env_vars = {
        'ANTHROPIC_BASE_URL': base_url,
        'ANTHROPIC_AUTH_TOKEN': api_key,
        'IS_SANDBOX': '1',
        'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        'ANTHROPIC_MODEL': model,
        'ANTHROPIC_DEFAULT_HAIKU_MODEL': model,
        'ANTHROPIC_DEFAULT_SONNET_MODEL': model,
        'ANTHROPIC_DEFAULT_OPUS_MODEL': model,
        'CLAUDE_CODE_EFFORT_LEVEL': 'high',
    }
    return {k: v for k, v in env_vars.items() if v is not None}


def build_codex_env(
    base_url: Optional[str],
    api_key: Optional[str],
) -> Dict[str, str]:
    """Build the environment variables for running the Codex CLI (`codex exec`).

    Routes Codex through an OpenAI-compatible endpoint (e.g. a LiteLLM proxy),
    mirroring how ``build_claude_code_env`` routes Claude Code. ``CODEX_API_KEY``
    is the credential ``codex exec`` reads in non-interactive runs; ``OPENAI_*``
    are set as a fallback for sub-flows that still read the standard names.

    Args:
        base_url: OpenAI-compatible API base URL (e.g. LiteLLM proxy).
        api_key: Auth token for the API.

    Returns:
        Mapping of environment variable name to value. Keys whose value is
        None are omitted.
    """
    env_vars = {
        'OPENAI_BASE_URL': base_url,
        'CODEX_API_KEY': api_key,
        'OPENAI_API_KEY': api_key,
    }
    return {k: v for k, v in env_vars.items() if v is not None}


def build_openhands_env(
    base_url: Optional[str],
    api_key: Optional[str],
    model: str,
) -> Dict[str, str]:
    """Build the environment variables for running the OpenHands CLI.

    The headless CLI (`openhands --headless`) reads its LLM configuration from
    these env vars only when invoked with ``--override-with-envs``. OpenHands
    routes through LiteLLM, so ``model`` should be a LiteLLM-style identifier
    (e.g. ``openai/<name>`` or ``anthropic/<name>``).

    Args:
        base_url: OpenAI-compatible API base URL (e.g. LiteLLM proxy).
        api_key: Auth token for the API.
        model: LiteLLM-style model identifier.

    Returns:
        Mapping of environment variable name to value. Keys whose value is
        None are omitted.
    """
    env_vars = {
        'LLM_MODEL': model,
        'LLM_API_KEY': api_key,
        'LLM_BASE_URL': base_url,
    }
    return {k: v for k, v in env_vars.items() if v is not None}


def docker_env_flags(env: Dict[str, str]) -> List[str]:
    """Expand an env mapping into ``docker exec``/``docker run`` -e flags."""
    flags: List[str] = []
    for key, value in env.items():
        if value is not None:
            flags.extend(["-e", f"{key}={value}"])
    return flags


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, default: Any = None) -> Any:
    """
    Load JSON data from file with error handling.

    Args:
        path: Path to JSON file
        default: Default value if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or default value

    Raises:
        ValueError: If file exists but contains invalid JSON
    """
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        if default is not None:
            return default
        raise ValueError(f"Invalid JSON in file {path}: {e}") from e


def save_json(
    data: Any,
    path: Path,
    ensure_dir: bool = True,
    indent: int = 2
) -> None:
    """
    Save data as JSON file with error handling.

    Args:
        data: Data to save
        path: Target file path
        ensure_dir: Create parent directory if it doesn't exist
        indent: JSON indentation level

    Raises:
        IOError: If file cannot be written
    """
    if ensure_dir:
        ensure_directory(path.parent)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except (IOError, TypeError) as e:
        raise IOError(f"Failed to save JSON to {path}: {e}") from e


def encode_image_to_base64(image_data: Union[bytes, str, Path]) -> str:
    """
    Encode image data to base64 string.

    Args:
        image_data: Image as bytes, base64 string, or file path

    Returns:
        Base64 encoded string

    Raises:
        ValueError: If image_data type is not supported
    """
    if isinstance(image_data, bytes):
        return base64.b64encode(image_data).decode("utf-8")

    if isinstance(image_data, str):
        # Check if already base64
        try:
            base64.b64decode(image_data)
            return image_data
        except Exception:
            # Assume it's a string that needs encoding
            return base64.b64encode(image_data.encode()).decode("utf-8")

    if isinstance(image_data, Path):
        with open(image_data, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    raise ValueError(f"Unsupported image data type: {type(image_data)}")


def decode_base64_to_bytes(b64_str: str) -> bytes:
    """
    Decode base64 string to bytes.

    Args:
        b64_str: Base64 encoded string

    Returns:
        Decoded bytes

    Raises:
        ValueError: If string is not valid base64
    """
    try:
        return base64.b64decode(b64_str)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}") from e


def save_image_from_base64(
    b64_data: str,
    output_path: Path,
    ensure_dir: bool = True
) -> None:
    """
    Save base64 encoded image to file.

    Args:
        b64_data: Base64 encoded image data
        output_path: Output file path
        ensure_dir: Create parent directory if it doesn't exist

    Raises:
        ValueError: If base64 data is invalid
        IOError: If file cannot be written
    """
    if ensure_dir:
        ensure_directory(output_path.parent)

    try:
        image_bytes = decode_base64_to_bytes(b64_data)
        with open(output_path, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        raise IOError(f"Failed to save image to {output_path}: {e}") from e


def build_output_path(
    base_dir: Path,
    framework: str,
    model: str,
    project: str,
    *additional_parts: str
) -> Path:
    """
    Build standardized output path.

    Args:
        base_dir: Base directory
        framework: Framework name
        model: Model name
        project: Project name
        *additional_parts: Additional path components

    Returns:
        Constructed path
    """
    path = base_dir / framework / model / project
    for part in additional_parts:
        path = path / part
    return path


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1m 30s" or "45.2s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m"


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.

    Args:
        text: Input string
        max_length: Maximum length
        suffix: Suffix to add when truncating

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_get_env(
    key: str,
    default: Optional[str] = None,
    required: bool = False,
    var_type: type = str
) -> Any:
    """
    Safely get environment variable with type conversion.

    Args:
        key: Environment variable key
        default: Default value if not found
        required: Raise error if not found and no default
        var_type: Type to convert to (str, int, float, bool)

    Returns:
        Environment variable value with proper type

    Raises:
        ValueError: If required variable is missing or type conversion fails
    """
    import os

    value = os.getenv(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable not set: {key}")
        return default

    try:
        if var_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        return var_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to convert environment variable {key}={value} to {var_type.__name__}: {e}"
        ) from e
