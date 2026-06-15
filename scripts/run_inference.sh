#!/bin/bash

# Inference script for Vision2Web
# Usage: Set environment variables or pass them directly to the script

set -e

# Configuration from environment variables with defaults
# FRAMEWORK options: claude_code | openhands | codex
FRAMEWORK="${FRAMEWORK:-}"
MODEL="${MODEL:-}"
API_KEY="${API_KEY:-}"
BASE_URL="${BASE_URL:-}"
SANDBOX="${SANDBOX:-vision2web-sandbox:latest}"
DATASETS_DIR="${DATASETS_DIR:-./datasets}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
MAX_WORKERS="${MAX_WORKERS:-}"
TASK="${TASK:-}"
PROJECTS="${PROJECTS:-}"

# Build command
CMD="vision2web inference \
  --framework $FRAMEWORK \
  --model $MODEL \
  --api-key $API_KEY \
  --sandbox $SANDBOX \
  --datasets-dir $DATASETS_DIR \
  --results-dir $RESULTS_DIR \
  --max-workers $MAX_WORKERS"

# Add optional arguments
[ -n "$BASE_URL" ] && CMD="$CMD --base-url $BASE_URL"
[ -n "$TASK" ] && CMD="$CMD --task $TASK"

# Add projects if specified
if [ -n "$PROJECTS" ]; then
    for proj in $PROJECTS; do
        CMD="$CMD --projects $proj"
    done
fi

# Execute
echo "Running: $CMD"
exec $CMD
