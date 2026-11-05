#!/usr/bin/env bash
# Plan-Bind D4 end to end on a Linux GPU box: B-spline Policy (bsp10), the one-step control (onestep) and
# Plan-Bind on LIBERO (lerobot/libero, all four suites in one model), conditioned on the task instruction
# (`task_condition=hash`), scored by `lerobot-eval --env.type=libero`.
#   1. trains the three policies with SEEDS training seeds (1000, 1001, ...), last checkpoint only;
#   2. evaluates the last checkpoints: the main comparison, call rate and commit, latency 100 and 200 ms,
#      and bsp10 through the spline runtime.
# LeRobot's diffusion policy has no language input, so it is not in this matrix (it cannot solve libero_goal).
# Every stage leaves a .done marker, training resumes from its last checkpoint and finished eval cells are
# kept, so after a disconnect run the same command again. Run it under tmux. Use at most 2 GPUs on the
# shared server.
#
# usage (from anywhere):
#   experiments/261105-plan-bind-libero/run_all.sh smoke    # proves the pipeline (one suite, 200 steps)
#   experiments/261105-plan-bind-libero/run_all.sh full     # the experiment
# GPUS="0 1" spreads the trainings, then the eval cells, over those GPUs (one worker each; default one GPU).
# env overrides: SEEDS=2 STEPS=60000 BATCH=32 SAVE_FREQ=20000 NUM_WORKERS=2 N_EVAL=5 (episodes per task, 10
# tasks per suite, so 50 per suite) SUITES=libero_spatial,libero_object,libero_goal,libero_10
# HF_REPO=<user>/<repo> uploads each trained policy (then deletes its local checkpoints except the last),
# the eval cells and results.md to that repo (full run only; needs `uv run hf auth login`).set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
cd "$REPO"
# shellcheck source=../261027-plan-bind-two-step/common.sh
source "$REPO/experiments/261027-plan-bind-two-step/common.sh"
EXTRA_SYNC_ARGS="--extra libero"

EVAL_SEED=1000  # the same eval episodes in every cell, so cells compare episode by episode
POLICIES=(bsp10 onestep planbind)
# Full-run settings, read now: the smoke run reuses these names in `all` mode.
FULL_SEEDS=${SEEDS:-2} FULL_STEPS=${STEPS:-60000} FULL_BATCH=${BATCH:-32} FULL_SAVE_FREQ=${SAVE_FREQ:-20000}
FULL_NUM_WORKERS=${NUM_WORKERS:-2} FULL_N_EVAL=${N_EVAL:-5}
FULL_SUITES=${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}

seeds() { local i; for ((i = 0; i < SEEDS; i++)); do echo $((1000 + i)); done; }

# train <policy> <seed>: resumes from the last checkpoint.
train() {
  local name=$1 seed=$2 dir="$OUT/train_$1_s$2"
  local last="$dir/checkpoints/last/pretrained_model/train_config.json"
  if [ -f "$last" ]; then
    uv run lerobot-train --config_path="$last" --resume=true
  else
    uv run lerobot-train --config_path="$EXP_DIR/train_$name.yaml" --output_dir="$dir" --seed="$seed" \
      --job_name="plan-bind-d4-$name-s$seed" --steps="$STEPS" --batch_size="$BATCH" --num_workers="$NUM_WORKERS" \
      --save_freq="$SAVE_FREQ" --log_freq=100
  fi
}

# Uploads a trained policy, then keeps only the checkpoint `last` points to.
publish_train() {
  local dir="$OUT/train_$1" keep ckpt
  hub_upload "$HF_REPO" model "$dir" "train_$1" || return 1
  keep=$(basename "$(readlink -f "$dir/checkpoints/last")")
  for ckpt in "$dir"/checkpoints/[0-9]*; do
    [ "$(basename "$ckpt")" = "$keep" ] || rm -rf "$ckpt"
  done
}

publish_eval() {
  hub_upload "$HF_REPO" model "$OUT/eval" eval || return 1
  uv run hf upload "$HF_REPO" "$OUT/results.md" results.md --repo-type=model --private
}

# eval_cell <cell> <checkpoint dir> <episodes> [extra lerobot-eval args...]
eval_cell() {
  local cell=$1 path=$2 episodes=$3 dir="$OUT/eval/$1"
  shift 3
  # With several GPUs every worker walks the same cell list and takes each EVAL_N-th cell. The count comes
  # first, so a cell finished meanwhile by another worker cannot shift it.
  if [ -n "${EVAL_N:-}" ]; then
    EVAL_IDX=$((EVAL_IDX + 1))
    [ $((EVAL_IDX % EVAL_N)) = "$EVAL_W" ] || return 0
  fi
  if [ -f "$dir/eval_info.json" ]; then echo "skip eval $cell (done)"; return; fi
  rm -rf "$dir"
  # Sync envs, one task at a time: the shared server froze under many parallel simulator processes.
  say "eval $cell"
  uv run lerobot-eval --policy.path="$path" --policy.device=cuda --env.type=libero --env.task="$SUITES" --env.max_parallel_tasks=1 \
    --seed="$EVAL_SEED" --eval.n_episodes="$episodes" --eval.batch_size=1 --eval.use_async_envs=false --output_dir="$dir" "$@"
}

# The flags that run a policy at latency 0 the way the main comparison does.
main_args() {
  case $1 in
    onestep|planbind) echo --policy.planbind_latency=0 --policy.planbind_commit=latency ;;
  esac
}

# main_name <policy> <latency>: bsp10 and diffusion use their own action queue, the others the runtime.
main_name() { case $1 in onestep|planbind) echo "$1_L$2_latency" ;; *) echo "$1_L$2" ;; esac; }

# sweep: no checkpoint curve in D4 (last checkpoint only); the stage marker stays so the worker logic is shared.
sweep() { :; }

eval_matrix() {
  local seed p c every latency path
  for seed in $(seeds); do
    for p in "${POLICIES[@]}"; do
      path="$OUT/train_${p}_s$seed/checkpoints/last/pretrained_model"
      # shellcheck disable=SC2046
      eval_cell "$(main_name "$p" 0)_s$seed" "$path" "$N_EVAL" $(main_args "$p")
    done
    for p in onestep planbind; do
      path="$OUT/train_${p}_s$seed/checkpoints/last/pretrained_model"
      # Call rate: every 2 frames (the default); commit latency and none.
      for c in latency none; do
        for every in "${EVERY[@]}"; do
          local tag=""
          [ "$every" = 2 ] || tag="_every$every"
          eval_cell "${p}_L0_${c}${tag}_s$seed" "$path" "$N_EVAL" --policy.planbind_latency=0 \
            --policy.planbind_commit="$c" --policy.planbind_call_every="$every"
        done
      done
    done
    # Latency (frames at 10 fps: 100 and 200 ms), bsp10 through the same runtime.
    for p in bsp10 onestep planbind; do
      path="$OUT/train_${p}_s$seed/checkpoints/last/pretrained_model"
      local extra=()
      [ "$p" = bsp10 ] && extra=(--policy.bspline_runtime=true)
      for latency in "${LATENCIES[@]}"; do
        [ "$p" != bsp10 ] && [ "$latency" = 0 ] && continue  # the main cells
        eval_cell "${p}_L${latency}_latency_s$seed" "$path" "$N_EVAL" "${extra[@]}" \
          --policy.planbind_latency="$latency" --policy.planbind_commit=latency
      done
    done
  done
}

# worker <index> <gpu> <workers>: trains every <workers>-th (policy, seed) job on its GPU, waits for the other
# workers' trainings, then evaluates every <workers>-th eval cell.
worker() {
  local w=$1 gpu=$2 n=$3 job=0 seed p
  export CUDA_VISIBLE_DEVICES=$gpu
  for seed in $(seeds); do
    for p in "${POLICIES[@]}"; do
      if [ $((job % n)) = "$w" ]; then
        run_stage "train_${p}_s$seed" train "$p" "$seed"
        run_stage "sweep_${p}_s$seed" sweep "$p" "$seed"
        if publishing "${HF_REPO:-}"; then publish "hub_train_${p}_s$seed" publish_train "${p}_s$seed"; fi
      fi
      job=$((job + 1))
    done
  done
  for seed in $(seeds); do
    for p in "${POLICIES[@]}"; do
      while [ ! -f "$OUT/.done_sweep_${p}_s$seed" ]; do sleep 30; done
    done
  done
  EVAL_N=$n EVAL_W=$w EVAL_IDX=0
  run_stage "eval_w$w" eval_matrix
  # Wait for the uploads only: a bare `wait` would also wait for the log tee process substitutions.
  pending=$(jobs -p); [ -z "$pending" ] || wait $pending
}

run_all() {
  mkdir -p "$OUT/eval" "$OUT/logs"
  local gpus=(${GPUS:-0}) w pids=() failed=0
  for w in "${!gpus[@]}"; do
    # Each worker also logs to $OUT/logs/worker_<index>.log, which status.sh reads.
    worker "$w" "${gpus[$w]}" "${#gpus[@]}" > >(tee -a "$OUT/logs/worker_$w.log") 2>&1 &
    pids+=($!)
  done
  for w in "${pids[@]}"; do wait "$w" || failed=1; done
  [ "$failed" = 0 ] || die "a GPU worker failed; rerun this script to resume"
  uv run python "$EXP_DIR/summarize.py" "$OUT/eval" > "$OUT/results.md"
  if publishing "${HF_REPO:-}"; then publish hub_eval publish_eval; fi
  # Wait for the uploads only: a bare `wait` would also wait for the log tee process substitutions.
  pending=$(jobs -p); [ -z "$pending" ] || wait $pending
  say "results: $OUT/results.md"
  cat "$OUT/results.md"
}

run_smoke() {
  OUT="$EXP_DIR/outputs/smoke"
  # The full batch size, so a GPU that is too small fails here; the training log's step_s gives the full run time.
  SEEDS=1 STEPS=200 BATCH=$FULL_BATCH SAVE_FREQ=100 NUM_WORKERS=2 N_EVAL=1 SUITES=libero_spatial PUBLISH=0
  EVERY=(8) LATENCIES=(2)
  run_all
}

run_full() {
  OUT="$EXP_DIR/outputs/full"
  SEEDS=$FULL_SEEDS STEPS=$FULL_STEPS BATCH=$FULL_BATCH SAVE_FREQ=$FULL_SAVE_FREQ NUM_WORKERS=$FULL_NUM_WORKERS
  N_EVAL=$FULL_N_EVAL SUITES=$FULL_SUITES PUBLISH=1
  EVERY=(2) LATENCIES=(0 2)
  run_all
}

MODE=${1:-all}
case "$MODE" in
  smoke|full|all) ;;
  *) die "usage: $0 [smoke|full|all]" ;;
esac
say "preflight"
preflight 60 "the venv, the LIBERO videos and checkpoints (all but the last are deleted after an upload)"
say "setup"
setup
# A private LIBERO config that points at the installed package: ~/.libero may hold another checkout's paths,
# and a missing config makes LIBERO stop at an interactive question.
export LIBERO_CONFIG_PATH="$REPO/.libero_config"
mkdir -p "$LIBERO_CONFIG_PATH"
uv run python - <<'PY'
import importlib.util
import os

root = importlib.util.find_spec("libero.libero").submodule_search_locations[0]
paths = {
    "benchmark_root": root,
    "bddl_files": f"{root}/bddl_files",
    "init_states": f"{root}/init_files",
    "datasets": f"{root}/../datasets",
    "assets": f"{root}/assets",
}
with open(os.path.join(os.environ["LIBERO_CONFIG_PATH"], "config.yaml"), "w") as f:
    f.writelines(f"{k}: {v}\n" for k, v in paths.items())
PY
case "$MODE" in
  smoke) run_smoke ;;
  full) run_full ;;
  all) run_smoke; run_full ;;
esac
