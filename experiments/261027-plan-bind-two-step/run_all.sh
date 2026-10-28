#!/usr/bin/env bash
# Plan-Bind D1 (RQ1) end to end on a fresh Linux GPU box (tested target: one RTX 3090, 24 GB).
#
# Stages: preflight -> setup (uv sync) -> sim dataset -> train bsp10, onestep, planbind -> eval matrix
# (latency x commit mode) -> results.md. Every stage leaves a .done marker and is skipped when rerun, a
# training resumes from its last checkpoint, and finished eval cells are kept, so after a disconnect just
# run the same command again. Run it under tmux or nohup.
#
# usage (from anywhere):
#   experiments/261027-plan-bind-two-step/run_all.sh          # smoke run, then the full run
#   experiments/261027-plan-bind-two-step/run_all.sh smoke    # about 20-30 minutes: proves the pipeline
#   experiments/261027-plan-bind-two-step/run_all.sh full     # the experiment
# env overrides (full run): EPISODES=500 STEPS=60000 BATCH=32 N_EVAL=50 EVAL_BATCH=10 NUM_WORKERS=8 GEN_WORKERS=8
# Hub upload (full run only, private repos, needs `hf auth login`): HF_REPO=<user>/<model repo> gets each
# policy's checkpoints and logs as soon as it is trained, then the eval cells and results.md;
# HF_DATASET_REPO=<user>/<dataset repo> gets the sim dataset. After a policy is uploaded, its local
# checkpoints except the last one are deleted, so the run fits a 100 GB disk.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
cd "$REPO"
# shellcheck source=common.sh
source "$EXP_DIR/common.sh"

TASK=NexArmStackBowls-v0
SEED=1000
LATENCIES=(0 3 6 9)  # frames at 30 fps: 0, 100, 200, 300 ms
# Full-run settings, read now: the smoke run reuses these names in `all` mode.
FULL_EPISODES=${EPISODES:-500} FULL_STEPS=${STEPS:-60000} FULL_BATCH=${BATCH:-32}
FULL_SAVE_FREQ=${SAVE_FREQ:-10000} FULL_LOG_FREQ=${LOG_FREQ:-100} FULL_NUM_WORKERS=${NUM_WORKERS:-8}
FULL_N_EVAL=${N_EVAL:-50} FULL_EVAL_BATCH=${EVAL_BATCH:-10} FULL_GEN_WORKERS=${GEN_WORKERS:-8}
# One stacking order (bottom,middle,top) for the dataset and every eval episode.
STACK_ORDER=${STACK_ORDER:-red,blue,black}

# The dataset replaces the repo's content, so files of an earlier upload do not linger.
publish_dataset() { hub_upload "$HF_DATASET_REPO" dataset "$DATA" . --delete='*'; }

# Uploads a trained policy, then keeps only the checkpoint `last` points to.
publish_train() {
  local dir="$OUT/train_$1" keep
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

generate_data() {
  local info="$DATA/meta/info.json"
  if [ -f "$info" ]; then
    local have
    have=$(uv run python -c "import json, sys; print(json.load(open(sys.argv[1]))['total_episodes'])" "$info")
    if [ "$have" -ge "$EPISODES" ]; then echo "reusing $DATA ($have episodes)"; return; fi
    die "$DATA has $have of $EPISODES episodes (an interrupted run?); remove it and rerun"
  fi
  [ ! -e "$DATA" ] || die "$DATA exists but is not a LeRobot dataset; remove it and rerun"
  uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --repo-id local/nexarm_stack_bowls --root "$DATA" --episodes "$EPISODES" --order "$STACK_ORDER" --gpus 0 --workers "$GEN_WORKERS"
}

train() {
  local name=$1 dir="$OUT/train_$1"
  local last="$dir/checkpoints/last/pretrained_model/train_config.json"
  if [ -f "$last" ]; then
    uv run lerobot-train --config_path="$last" --resume=true
  else
    uv run lerobot-train --config_path="$EXP_DIR/train_$name.yaml" --dataset.root="$DATA" --output_dir="$dir" \
      --steps="$STEPS" --batch_size="$BATCH" --num_workers="$NUM_WORKERS" --save_freq="$SAVE_FREQ" \
      --log_freq="$LOG_FREQ"
  fi
}

# eval_cell <cell> <policy> [extra lerobot-eval args...]
eval_cell() {
  local cell=$1 name=$2; shift 2
  local dir="$OUT/eval/$cell"
  if [ -f "$dir/eval_info.json" ]; then echo "skip eval $cell (done)"; return; fi
  rm -rf "$dir"
  say "eval $cell"
  uv run lerobot-nexarm-eval --policy.path="$OUT/train_$name/checkpoints/last/pretrained_model" \
    --policy.device=cuda --env.type=nexarm --env.task="$TASK" --env.stack_order="$STACK_ORDER" --seed="$SEED" \
    --eval.n_episodes="$N_EVAL" --eval.batch_size="$EVAL_BATCH" --output_dir="$dir" "$@"
}

eval_matrix() {
  local name latency commit
  eval_cell bsp10_L0 bsp10
  for name in onestep planbind; do
    for latency in "${LATENCIES[@]}"; do
      for commit in latency none; do
        eval_cell "${name}_L${latency}_${commit}" "$name" \
          --policy.planbind_latency="$latency" --policy.planbind_commit="$commit"
      done
    done
  done
  # Plan-Bind with the residual trigger for global re-planning (RQ5 preview).
  eval_cell planbind_L6_latency_tau0.3 planbind \
    --policy.planbind_latency=6 --policy.planbind_commit=latency --policy.planbind_replan_threshold=0.3
}

run_all() {
  mkdir -p "$OUT"
  run_stage data generate_data
  if publishing "${HF_DATASET_REPO:-}"; then publish hub_data publish_dataset; fi
  for name in bsp10 onestep planbind; do
    run_stage "train_$name" train "$name"
    if publishing "${HF_REPO:-}"; then publish "hub_train_$name" publish_train "$name"; fi
  done
  run_stage eval eval_matrix
  uv run python "$EXP_DIR/summarize.py" "$OUT/eval" > "$OUT/results.md"
  if publishing "${HF_REPO:-}"; then publish hub_eval publish_eval; fi
  wait
  say "results: $OUT/results.md"
  cat "$OUT/results.md"
}

run_smoke() {
  OUT="$EXP_DIR/outputs/smoke" DATA="$EXP_DIR/outputs/smoke/data"
  EPISODES=4 STEPS=200 BATCH=8 SAVE_FREQ=200 LOG_FREQ=20 NUM_WORKERS=2 N_EVAL=2 EVAL_BATCH=2 PUBLISH=0 GEN_WORKERS=2
  LATENCIES=(0 6)
  run_all
}

run_full() {
  OUT="$EXP_DIR/outputs/full" DATA="$REPO/outputs/datasets/nexarm_stack_bowls"
  EPISODES=$FULL_EPISODES STEPS=$FULL_STEPS BATCH=$FULL_BATCH SAVE_FREQ=$FULL_SAVE_FREQ
  LOG_FREQ=$FULL_LOG_FREQ NUM_WORKERS=$FULL_NUM_WORKERS N_EVAL=$FULL_N_EVAL EVAL_BATCH=$FULL_EVAL_BATCH
  GEN_WORKERS=$FULL_GEN_WORKERS PUBLISH=1
  LATENCIES=(0 3 6 9)
  run_all
}

MODE=${1:-all}
case "$MODE" in
  smoke|full|all) ;;
  *) die "usage: $0 [smoke|full|all]" ;;
esac
say "preflight"
preflight 100 "the venv, dataset and checkpoints (3.5 GB each, 21 in all)"
say "setup"
setup
case "$MODE" in
  smoke) run_smoke ;;
  full) run_full ;;
  all) run_smoke; run_full ;;
esac
