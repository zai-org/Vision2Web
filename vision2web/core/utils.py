"""Utility functions for Vision2Web."""

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


AUTO_COMPACT_WINDOW_MIN = 100_000
AUTO_COMPACT_WINDOW_MAX = 1_000_000


def build_claude_code_env(
    base_url: Optional[str],
    api_key: Optional[str],
    model: str,
    context_window: int = 200_000,
) -> Dict[str, str]:
    """Build the environment variables for running Claude Code CLI.

    Shared by the inference adapter and the functional-test runner so that
    Claude Code is invoked with an identical environment in both phases.

    Args:
        base_url: Anthropic-compatible API base URL (e.g. LiteLLM proxy).
        api_key: Auth token for the API.
        model: Model identifier; routed to every Claude tier so any internal
               model selection resolves to the target evaluation model.
        context_window: The evaluated model's real context window, in tokens.
               Pins auto-compact explicitly - see the note below. Must be
               within [100_000, 1_000_000]; Claude Code rejects values outside
               that range and would silently fall back to not compacting.

    Returns:
        Mapping of environment variable name to value. Keys whose value is
        None are omitted.
    """
    if not AUTO_COMPACT_WINDOW_MIN <= context_window <= AUTO_COMPACT_WINDOW_MAX:
        raise ValueError(
            f"context_window must be within "
            f"[{AUTO_COMPACT_WINDOW_MIN}, {AUTO_COMPACT_WINDOW_MAX}], "
            f"got {context_window}"
        )

    env_vars = {
        'ANTHROPIC_BASE_URL': base_url,
        'ANTHROPIC_AUTH_TOKEN': api_key,
        'IS_SANDBOX': '1',
        'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS': '1',
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        "CLAUDE_STREAM_IDLE_TIMEOUT_MS": 3600000,
        'ANTHROPIC_MODEL': model,
        'ANTHROPIC_DEFAULT_HAIKU_MODEL': model,
        'ANTHROPIC_DEFAULT_SONNET_MODEL': model,
        'ANTHROPIC_DEFAULT_OPUS_MODEL': model,
        'CLAUDE_CODE_EFFORT_LEVEL': 'high',
        'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '64000',

        # Auto-compact must be pinned explicitly for the models evaluated here.
        # Claude Code resolves its auto-compact window through a precedence
        # chain (env > settings > server-pushed clientdata > a hardcoded
        # known-model set) and only *enforces* the window when one of those
        # supplies it. An evaluation model name like `glm-5.3v-sft` matches no
        # catalog entry and no known-model set, and this harness has no
        # Anthropic login (plus CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1), so
        # no clientdata arrives either. Every source comes up empty, the window
        # resolves as "unenforced", and the proactive compaction check returns
        # early without ever reading the token count - sessions then grow until
        # the API itself rejects them. Observed on claude-code 2.1.221:
        # glm-5.3v-sft reached 260K input tokens with zero compact events.
        'CLAUDE_CODE_AUTO_COMPACT_WINDOW': str(context_window),

        # The window above is additionally clamped to the context window Claude
        # Code *assumes* for the model, which defaults to 200K for anything it
        # does not recognize. Raising that assumption is what lets a
        # context_window above 200K take effect. Only honoured for model names
        # that do not start with `claude-`, which is exactly the third-party
        # case here; it is ignored (harmlessly) for real Claude models, whose
        # window comes from the baked-in catalog instead.
        'CLAUDE_CODE_MAX_CONTEXT_TOKENS': str(context_window),
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


def build_gemini_cli_env(
    base_url: Optional[str],
    api_key: Optional[str],
) -> Dict[str, str]:
    """Build the environment variables for running the Gemini CLI (`gemini`).

    Routes the CLI through a Gemini-compatible endpoint (e.g. a LiteLLM proxy),
    mirroring how ``build_claude_code_env`` routes Claude Code.
    ``GOOGLE_GEMINI_BASE_URL`` replaces the generativelanguage.googleapis.com
    host, and ``GEMINI_API_KEY`` is sent as the ``x-goog-api-key`` header.

    Note that ``GOOGLE_GEMINI_BASE_URL`` alone also flips the CLI's env-derived
    auth type to ``gateway``, which its own ``validateAuthMethod`` then rejects
    ("Invalid auth method selected", observed on gemini-cli 0.57.0). The auth
    type is therefore pinned to ``gemini-api-key`` in the settings file - see
    ``build_gemini_cli_settings`` - which takes precedence over the env-derived
    one while still honouring the base URL override.

    Args:
        base_url: Gemini-compatible API base URL (e.g. LiteLLM proxy). The
               client appends ``/v1beta/models/<model>:<method>`` itself, so
               this must stop at the host or gateway prefix - an OpenAI-style
               ``/v1`` suffix would produce ``/v1/v1beta/...``.
        api_key: Auth token for the API.

    Returns:
        Mapping of environment variable name to value. Keys whose value is
        None are omitted.
    """
    env_vars = {
        'GOOGLE_GEMINI_BASE_URL': base_url,
        'GEMINI_API_KEY': api_key,

        # The outer Docker container already provides isolation; keep the CLI
        # from starting a nested sandbox of its own (also pinned in settings).
        'GEMINI_SANDBOX': 'false',
        'NO_COLOR': '1',
    }
    return {k: v for k, v in env_vars.items() if v is not None}


# Context window the Gemini CLI assumes for any model it does not recognize
# (`tokenLimit()` falls through to this for third-party names). Context
# compression is triggered at `model.compressionThreshold * this`, so the
# threshold has to be scaled down for models with a smaller real window.
GEMINI_CLI_TOKEN_LIMIT = 1_048_576

# Fraction of the *real* context window at which to compress. Roughly where
# Claude Code auto-compacts, so both frameworks compact at a comparable point:
# claude-code 2.1.197 reserves 20K tokens for the output and then compacts at
# 80% of the remainder, i.e. ~72% of a 200K window.
GEMINI_CLI_COMPRESSION_FRACTION = 0.83

# Models the CLI's built-in config points its internal helper calls at
# (classifiers, tool-output summarizers, edit correction, loop detection,
# chat compression, next-speaker checks, ...). A benchmark gateway only serves
# the model under evaluation, so every one of these is rerouted to it.
GEMINI_CLI_BUILTIN_MODELS = (
    'auto',
    'pro',
    'flash',
    'flash-lite',
    'gemini-3-pro-preview',
    'gemini-3-flash-preview',
    'gemini-3.5-flash',
    'gemini-3.1-pro-preview',
    'gemini-3.1-pro-preview-customtools',
    'gemini-3.1-flash-lite',
    'gemini-3.1-flash-lite-preview',
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
)


def build_gemini_cli_settings(
    model: str,
    context_window: int = 200_000,
    thinking_level: Optional[str] = 'HIGH',
) -> Dict[str, Any]:
    """Build the ``~/.gemini/settings.json`` contents for a benchmark run.

    The Gemini CLI takes most of its non-credential configuration from this
    file rather than from env vars, so this is the counterpart to
    ``build_claude_code_env`` / the Codex ``config.toml``.

    Args:
        model: Model identifier served by the configured endpoint.
        context_window: The evaluated model's real context window, in tokens.
               Converted into ``model.compressionThreshold`` - see below.
        thinking_level: Gemini 3 thinking level ("LOW"/"HIGH") to request for
               the evaluated model, matching Claude Code's
               ``CLAUDE_CODE_EFFORT_LEVEL=high`` and Codex's
               ``model_reasoning_effort = "high"``. Only applied to
               third-party model names: for first-party ``gemini-*`` /
               ``gemma-*`` names the CLI already picks the family-correct
               default (HIGH for gemini-3, a thinking *budget* for 2.5), and
               forcing a level on a 2.5 model would be rejected. Pass None to
               send no thinking config at all - the first thing to try if a
               gateway rejects the field.

    Returns:
        Settings mapping, ready to be serialized as JSON.
    """
    if not 0 < context_window <= GEMINI_CLI_TOKEN_LIMIT:
        raise ValueError(
            f"context_window must be within (0, {GEMINI_CLI_TOKEN_LIMIT}], "
            f"got {context_window}"
        )

    overrides: List[Dict[str, Any]] = [
        {'match': {'model': builtin}, 'modelConfig': {'model': model}}
        for builtin in GEMINI_CLI_BUILTIN_MODELS
        if builtin != model
    ]
    if thinking_level and not model.startswith(('gemini-', 'gemma-')):
        overrides.append({
            'match': {'model': model},
            'modelConfig': {
                'generateContentConfig': {
                    'thinkingConfig': {'thinkingLevel': thinking_level}
                }
            },
        })

    return {
        'security': {
            # Pinned so the env-derived `gateway` auth type is not used; see
            # build_gemini_cli_env.
            'auth': {'selectedType': 'gemini-api-key'},

            # Untrusted folders silently downgrade `--yolo` back to
            # "prompt for approval", which deadlocks a headless run on the
            # agent's first tool call.
            'folderTrust': {'enabled': False},
        },
        'privacy': {'usageStatisticsEnabled': False},
        'telemetry': {'enabled': False},
        'general': {
            # An auto-update mid-benchmark would change the agent under test.
            'enableAutoUpdate': False,
            'enableAutoUpdateNotification': False,
            'checkpointing': {'enabled': False},
        },
        'model': {
            'name': model,
            'maxSessionTurns': -1,  # unlimited, as for Claude Code / Codex

            # After every turn that ends without a tool call, the CLI otherwise
            # asks the model "who should speak next?" by replaying the history -
            # a request whose last entry is a *model* turn. The Gemini-compatible
            # upstream used here rejects those with
            # `400 Requests ending with a model turn are not supported`, which
            # aborts the whole task. Skipping the check also matches Claude Code
            # and Codex, neither of which has an equivalent auto-continue probe.
            'skipNextSpeakerCheck': True,

            # Scaled so compression fires at half of the model's *real* window
            # rather than half of the 1M window the CLI assumes for unknown
            # model names - without this, a 200K model would run until the API
            # itself rejects the request.
            'compressionThreshold': round(
                GEMINI_CLI_COMPRESSION_FRACTION * context_window
                / GEMINI_CLI_TOKEN_LIMIT,
                4,
            ),
        },
        'tools': {'sandbox': False},
        'experimental': {
            # Required for `model` to be honoured verbatim. The legacy
            # resolution path rewrites any name *ending in* "flash" to
            # gemini-3.5-flash (`isFlashModel()` matches on the suffix), so an
            # evaluation model such as `glm-5.3-flash` would silently be
            # replaced by a Google model. Verified on gemini-cli 0.57.0.
            'dynamicModelConfiguration': True,
        },
        'modelConfigs': {'customOverrides': overrides},
    }


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
