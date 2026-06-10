#!/bin/bash

set -e  # Exit immediately on error

# Install uv (https://docs.astral.sh/uv/) if it isn't already available.
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Make uv available in the current shell for the steps below.
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating virtual environment and installing dependencies from pyproject.toml..."
uv sync

echo "Setup complete!"
echo "Run commands with 'uv run', e.g.:"
echo "  uv run python main.py config/config_lfm.yml"
echo "Or activate the environment directly:"
echo "  source .venv/bin/activate"
