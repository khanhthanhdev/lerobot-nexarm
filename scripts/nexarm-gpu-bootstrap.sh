#!/usr/bin/env bash
# Prepare a fresh Linux GPU server (your own or a rented one) to train NexArm policies for the desktop app.
#
# It installs lerobot-nexarm and the NexArm engine under ~/.nexarm and writes ~/.nexarm/start-engine.sh,
# which the desktop runs over SSH to start the engine when it is not already running. The engine listens on
# 127.0.0.1 only; the desktop reaches it through an SSH tunnel, so no port is exposed on the machine.
#
# usage:
#   scripts/nexarm-gpu-bootstrap.sh                      # install from the checkout this script is in
#   scripts/nexarm-gpu-bootstrap.sh --repo URL [--ref BRANCH_OR_TAG]
#   NEXARM_EXTRAS="service training act" scripts/nexarm-gpu-bootstrap.sh
set -euo pipefail

HOME_DIR="${NEXARM_HOME_DIR:-$HOME/.nexarm}"
ENGINE_DIR="$HOME_DIR/engine"
STATE_DIR="$HOME_DIR/state"
PORT="${NEXARM_ENGINE_PORT:-8765}"
EXTRAS="${NEXARM_EXTRAS:-service training}"
REPO=""
REF=""

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="${2:?--repo needs a URL}"; shift 2 ;;
    --ref) REF="${2:?--ref needs a branch or tag}"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }

[ "$(uname -s)" = "Linux" ] || { echo "This script is for Linux servers." >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required." >&2; exit 1; }
if command -v nvidia-smi >/dev/null; then
  say "GPU"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "warning: nvidia-smi not found; training will run on the CPU unless the NVIDIA driver is installed." >&2
fi

say "uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

say "lerobot-nexarm"
mkdir -p "$HOME_DIR" "$STATE_DIR"
if [ -n "$REPO" ]; then
  command -v git >/dev/null || { echo "git is required for --repo." >&2; exit 1; }
  if [ -d "$ENGINE_DIR/.git" ]; then
    git -C "$ENGINE_DIR" fetch --all --tags --prune
  else
    git clone "$REPO" "$ENGINE_DIR"
  fi
  [ -z "$REF" ] || git -C "$ENGINE_DIR" checkout "$REF"
  git -C "$ENGINE_DIR" pull --ff-only 2>/dev/null || true
else
  SRC="$(cd "$(dirname "$0")/.." && pwd)"
  [ -f "$SRC/pyproject.toml" ] || { echo "Run this from a lerobot-nexarm checkout or pass --repo URL." >&2; exit 1; }
  mkdir -p "$ENGINE_DIR"
  # Copy the working tree (no git history, caches or virtualenvs) so the install is self-contained.
  tar -C "$SRC" --exclude=.git --exclude=.venv --exclude=__pycache__ --exclude=outputs -cf - . | tar -C "$ENGINE_DIR" -xf -
fi

EXTRA_FLAGS=""
for e in $EXTRAS; do EXTRA_FLAGS="$EXTRA_FLAGS --extra $e"; done
( cd "$ENGINE_DIR" && uv sync $EXTRA_FLAGS )

say "start-engine.sh"
cat > "$HOME_DIR/start-engine.sh" <<EOF
#!/usr/bin/env bash
# Starts the NexArm engine on 127.0.0.1:$PORT unless it already answers. Safe to run repeatedly.
export PATH="\$HOME/.local/bin:\$PATH"
if curl -fs "http://127.0.0.1:$PORT/api/v1/health" >/dev/null 2>&1; then exit 0; fi
cd "$ENGINE_DIR"
nohup uv run lerobot-nexarm-service --host 127.0.0.1 --port $PORT --home "$STATE_DIR" > "$HOME_DIR/engine.log" 2>&1 &
for _ in \$(seq 1 60); do
  if curl -fs "http://127.0.0.1:$PORT/api/v1/health" >/dev/null 2>&1; then exit 0; fi
  sleep 1
done
echo "the engine did not start; see $HOME_DIR/engine.log" >&2
exit 1
EOF
chmod +x "$HOME_DIR/start-engine.sh"

say "starting the engine once to check the install"
"$HOME_DIR/start-engine.sh"
curl -fs "http://127.0.0.1:$PORT/api/v1/health" && echo

cat <<EOF

Done. In the desktop app (Lab: Train), add this server with your SSH user and host, then pick it as the
runner. The app opens   ssh -L <local port>:127.0.0.1:$PORT user@host   itself; nothing needs to be exposed.
Datasets are uploaded and checkpoints come back automatically.
EOF
