#!/bin/bash

# Evaluation script for Vision2Web
# Functional testing runs via Claude Code + playwright-cli; visual scoring
# uses a VLM judge. The two phases can target different model endpoints.
# Usage: Set environment variables or pass them directly to the script

set -e

# Configuration from environment variables with defaults
# Functional testing (Claude Code) endpoint
FUNCTIONAL_MODEL="${FUNCTIONAL_MODEL:-}"
FUNCTIONAL_API_KEY="${FUNCTIONAL_API_KEY:-}"
FUNCTIONAL_BASE_URL="${FUNCTIONAL_BASE_URL:-}"
# Visual scoring (VLM judge) endpoint
VISUAL_API_KEY="${VISUAL_API_KEY:-}"
VISUAL_BASE_URL="${VISUAL_BASE_URL:-}"
VISUAL_MODEL="${VISUAL_MODEL:-}"
SANDBOX="${SANDBOX:-vision2web-sandbox:latest}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
DATASETS_DIR="${DATASETS_DIR:-./datasets}"
MAX_WORKERS="${MAX_WORKERS:-1}"
TASK="${TASK:-}"
FRAMEWORK="${FRAMEWORK:-}"
MODEL="${MODEL:-}"

# Build command
CMD="vision2web evaluate \
  --results-dir $RESULTS_DIR \
  --datasets-dir $DATASETS_DIR \
  --functional-model $FUNCTIONAL_MODEL \
  --functional-api-key $FUNCTIONAL_API_KEY \
  --functional-base-url $FUNCTIONAL_BASE_URL \
  --visual-api-key $VISUAL_API_KEY \
  --visual-base-url $VISUAL_BASE_URL \
  --visual-model $VISUAL_MODEL \
  --sandbox $SANDBOX \
  --max-workers $MAX_WORKERS"

# Add optional arguments
[ -n "$TASK" ] && CMD="$CMD --task $TASK"
[ -n "$FRAMEWORK" ] && CMD="$CMD --framework $FRAMEWORK"
[ -n "$MODEL" ] && CMD="$CMD --model $MODEL"

# Execute
echo "Running: $CMD"
exec $CMD
