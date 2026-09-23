#!/usr/bin/env bash
# One-step server setup for TurboVLA on NexArm.
# Run on fresh GPU machine (Ubuntu/Debian)

set -e

echo "=== [1/4] Installing / Updating uv package manager ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "=== [2/4] Syncing locked Python environment with training extras ==="
uv sync --extra smolvla --extra training --extra groot

echo "=== [3/4] Verifying dependencies and health ==="
uv run python examples/nexarm/setup_turbovla.py --check-only

echo "=== [4/4] Downloading foundational backbones (DINOv3 and BERT) ==="
uv run python examples/nexarm/setup_turbovla.py --download-backbones

echo "=========================================================="
echo "Server setup complete! Ready to train TurboVLA on NexArm."
echo "=========================================================="
