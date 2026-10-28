# Shared by the Plan-Bind run scripts (D1 run_all.sh, D2 run_all.sh): environment, preflight, setup, Hub
# upload and once-per-output stage markers. Source it after setting REPO and cd-ing into it; stages use OUT,
# and publishing() uses PUBLISH.

export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_TELEMETRY=1
# Hub uploads (hf_xet): more parallel transfers, and patient timeouts and retries for slow links.
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1} HF_XET_CLIENT_CONNECT_TIMEOUT=${HF_XET_CLIENT_CONNECT_TIMEOUT:-2m}
export HF_XET_CLIENT_READ_TIMEOUT=${HF_XET_CLIENT_READ_TIMEOUT:-10m} HF_XET_CLIENT_RETRY_MAX_ATTEMPTS=${HF_XET_CLIENT_RETRY_MAX_ATTEMPTS:-20}
export HF_XET_CLIENT_RETRY_MAX_DURATION=${HF_XET_CLIENT_RETRY_MAX_DURATION:-1h}

say() { printf '\n== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# preflight <GB of free disk needed> <what needs it>
preflight() {
  [ "$(uname -s)" = "Linux" ] || die "run this on Linux (sim rendering needs EGL)"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: install the NVIDIA driver (>= 570 for the cu128 wheels)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  local missing=()
  command -v ffmpeg >/dev/null || missing+=(ffmpeg)
  command -v git >/dev/null || missing+=(git)
  git lfs version >/dev/null 2>&1 || missing+=(git-lfs)
  command -v curl >/dev/null || missing+=(curl)
  ldconfig -p | grep -q 'libEGL.so' || missing+=(libegl1)
  ldconfig -p | grep -q 'libGL.so' || missing+=(libgl1)
  if [ ${#missing[@]} -gt 0 ]; then
    die "missing system packages; install them with: sudo apt-get update && sudo apt-get install -y ${missing[*]}"
  fi
  local free_gb
  free_gb=$(df -Pk "$REPO" | awk 'NR==2 {print int($4 / 1048576)}')
  [ "$free_gb" -ge "$1" ] || echo "warning: only ${free_gb} GB free; $2 need about $1 GB" >&2
}

setup() {
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  # A system Python 3.12 avoids building native packages with uv's own interpreter.
  if [ -z "${UV_PYTHON:-}" ] && [ -x /usr/bin/python3.12 ]; then export UV_PYTHON=/usr/bin/python3.12; fi
  uv sync --locked --extra test --extra training --extra nexarm --extra diffusion
  # The sim meshes are Git LFS objects.
  git lfs install --local >/dev/null
  git lfs pull
  uv run python - <<'PY'
import mujoco
import torch

model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
renderer = mujoco.Renderer(model, 32, 32)
renderer.update_scene(mujoco.MjData(model))
assert renderer.render().shape == (32, 32, 3)
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
print("EGL rendering and CUDA OK:", torch.cuda.get_device_name(0))
PY
  if [ -n "${HF_REPO:-}${HF_DATASET_REPO:-}${HF_SOURCE_REPO:-}" ]; then
    uv run hf auth whoami >/dev/null || die "a Hugging Face repo is set but you are not logged in: run 'uv run hf auth login'"
  fi
}

# hub_upload <repo> <repo type> <local folder> <path in repo> [extra hf upload args...]
# Retries 5 times, 5 minutes apart: the Hub's storage backend fails now and then.
hub_upload() {
  local repo=$1 type=$2 folder=$3 path=$4 attempt
  shift 4
  for attempt in 1 2 3 4 5; do
    uv run hf upload "$repo" "$folder" "$path" --repo-type="$type" --private --exclude='*checkpoints/last/*' \
      --commit-message="Upload $path from $(basename "$0")" "$@" && return 0
    echo "warning: upload of $path failed (attempt $attempt of 5)" >&2
    sleep 300
  done
  return 1
}

# publishing <repo>: true in a full run when that repo is set.
publishing() { [ "$PUBLISH" = 1 ] && [ -n "$1" ]; }

# publish <stage> <command...>: uploads in the background so the GPU keeps working; a failed upload
# does not stop the run, and its stage reruns next time.
publish() {
  (run_stage "$@" || echo "warning: $1 failed; rerun this script later to upload it" >&2) &
}

# run_stage <name> <command...>: runs once per output root.
run_stage() {
  local name=$1; shift
  if [ -f "$OUT/.done_$name" ]; then say "skip $name (done)"; return; fi
  say "$name"
  "$@" || return 1
  touch "$OUT/.done_$name"
}
