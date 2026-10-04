#!/usr/bin/env bash
set -euo pipefail
collection_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$collection_repo"
exec uv run --no-sync python examples/nexarm/prepare_collection.py "$@"
