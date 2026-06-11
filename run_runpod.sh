#!/bin/bash
# Launch a spotify-runpod experiment non-interactively.
# Usage: bash run_runpod.sh "<run-name>"
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/root/.uv-cache
export UV_PROJECT_ENVIRONMENT=/root/smr-venv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export $(grep -E '^WANDB_API_KEY=' /workspace/.env | xargs)
export WANDB_DIR=logger_runs/spotify-runpod

# main.py reads the run name from stdin (config dev:False), so pipe it in.
echo "$1" | uv run --no-sync python main.py config/spotify-runpod.yaml
