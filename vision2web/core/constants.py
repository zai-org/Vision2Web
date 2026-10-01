"""Constants for Vision2Web"""

# Container settings
CONTAINER_PREFIX = "vision2web-"
CONTAINER_WORKSPACE = "/workspace"
CONTAINER_TMP_WORKSPACE = "/tmp/workspace"
CONTAINER_USER = "root"

# Memory ceiling for a task container, as a `docker create --memory` value.
#
# This is not just a safety cap - the Gemini CLI reads it. Its launcher
# (bundle/gemini.js `getMemoryNodeArgs`) relaunches node with
# `--max-old-space-size = floor(os.totalmem() / 2)`, and os.totalmem() inside
# the container reports the cgroup limit, so whatever is set here halves into
# the agent's V8 heap. Left unset, containers inherited an 8 GiB default and
# the CLI capped its own heap at 4 GiB, which the accumulated base64
# screenshots in a long task's history then blew through while being
# JSON.stringify'd for each API request (`FATAL ERROR: Reached heap limit`).
CONTAINER_MEMORY_LIMIT = "64g"

# Environment applied to every task container.
#
# PLAYWRIGHT_MCP_SANDBOX=false maps to Playwright's `chromiumSandbox: false`,
# which is what makes it pass `--no-sandbox` to Chrome. Containers run as root,
# and Chrome refuses to start as root without that flag
# ("Running as root without --no-sandbox is not supported", crbug.com/638180).
# playwright-cli leaves chromiumSandbox unset by default, so every
# `playwright-cli open` died in the daemon with "Chromium sandboxing failed!"
# and the evaluation phase then reported "No screenshot for prototype '<name>'".
# Set on the container rather than per-exec so the agent's own playwright-cli
# calls inside a Claude Code session inherit it too.
CONTAINER_ENV = {
    "PLAYWRIGHT_MCP_SANDBOX": "false",
}

# Docker timeouts (seconds)
DOCKER_CREATE_TIMEOUT = 60
DOCKER_START_TIMEOUT = 120
DOCKER_STOP_TIMEOUT = 60
DOCKER_EXEC_TIMEOUT = 7200
DOCKER_COPY_TIMEOUT = 300

# Task types
TASK_TYPES = ["webpage", "frontend", "website"]
