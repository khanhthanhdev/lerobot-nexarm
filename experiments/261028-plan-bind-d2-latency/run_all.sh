#!/usr/bin/env bash
# Plan-Bind D2 end to end on a fresh Linux GPU box (tested target: one RTX 3090, 24 GB).
#
# Reuses D1's trained checkpoints from the Hub (no retraining), then:
#   1. times one inference call of bsp10, onestep and planbind on the GPU and on CPU (bench_inference.py);
#   2. evaluates each policy at its own measured latency (bsp10 through the same spline runtime);
#   3. separates latency from update rate (a call every 4, 8 or 12 frames at latency 0);
#   4. fills the gaps: commit "now", replan thresholds 0.5 and 1.0, bsp10 across latencies;
#   5. moves the bottom bowl when the middle bowl is lifted (reactivity);
#   6. optionally trains more seeds of onestep and planbind (SEEDS=2 or more; about 4 h each).
# Every stage leaves a .done marker and finished eval cells are kept, so after a disconnect run the same
# command again. Run it under tmux.
#
# usage (from anywhere):
#   experiments/261028-plan-bind-d2-latency/run_all.sh          # smoke run, then the full run
#   experiments/261028-plan-bind-d2-latency/run_all.sh smoke    # about 30 minutes: proves the pipeline
#   experiments/261028-plan-bind-d2-latency/run_all.sh full     # the experiment
# env overrides: N_EVAL=50 EVAL_BATCH=10 SEEDS=1 CPU_THREADS=4 SHIFT_M=0.03 STEPS=60000 BATCH=32 NUM_WORKERS=8
# Needs `uv run hf auth login` with read access to the D1 repos:
#   HF_SOURCE_REPO=thanhkt/plan-bind-d1 (checkpoints), HF_SOURCE_DATASET=thanhkt/nexarm-stack-bowls-sim-red-blue-black-500
#   (only downloaded when SEEDS > 1). HF_REPO=<user>/<repo> uploads the bench, eval cells, results.md and any
#   newly trained seeds (full run only, private).
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
D1_DIR="$REPO/experiments/261027-plan-bind-two-step"
cd "$REPO"
# shellcheck source=../261027-plan-bind-two-step/common.sh
source "$D1_DIR/common.sh"

TASK=NexArmStackBowls-v0
EVAL_SEED=1000  # the same 50 seeded episodes as D1, so every cell pairs with D1's
TRAIN_SEED=1000  # D1's training seed; extra seeds are 1001, 1002, ...
STACK_ORDER=${STACK_ORDER:-red,blue,black}
HF_SOURCE_REPO=${HF_SOURCE_REPO:-thanhkt/plan-bind-d1}
HF_SOURCE_DATASET=${HF_SOURCE_DATASET:-thanhkt/nexarm-stack-bowls-sim-red-blue-black-500}
CKPT_STEP=${CKPT_STEP:-060000}
D1_CKPTS="$EXP_DIR/outputs/d1"
DATA="$REPO/outputs/datasets/nexarm_stack_bowls"
POLICIES=(bsp10 onestep planbind)
# Full-run settings, read now: the smoke run reuses these names in `all` mode.
FULL_N_EVAL=${N_EVAL:-50} FULL_EVAL_BATCH=${EVAL_BATCH:-10} FULL_SEEDS=${SEEDS:-1}
FULL_STEPS=${STEPS:-60000} FULL_BATCH=${BATCH:-32} FULL_NUM_WORKERS=${NUM_WORKERS:-8}
CPU_THREADS=${CPU_THREADS:-4}
SHIFT_M=${SHIFT_M:-0.03}
SHIFT_TAG="shift$(awk -v m="$SHIFT_M" 'BEGIN { printf "%g", m * 100 }')cm"

# ckpt <policy> <training seed>: the pretrained_model directory to evaluate.
ckpt() {
  if [ "$2" = "$TRAIN_SEED" ]; then
    echo "$D1_CKPTS/train_$1/checkpoints/$CKPT_STEP/pretrained_model"
  else
    echo "$OUT/train_$1_s$2/checkpoints/last/pretrained_model"
  fi
}

fetch_d1() {
  uv run hf download "$HF_SOURCE_REPO" --include "train_*/checkpoints/$CKPT_STEP/pretrained_model/*" \
    --local-dir "$D1_CKPTS"
  local p
  for p in "${POLICIES[@]}"; do
    [ -f "$(ckpt "$p" "$TRAIN_SEED")/model.safetensors" ] || die "no $p checkpoint $CKPT_STEP in $HF_SOURCE_REPO"
  done
}

# bench <tag> [bench_inference.py args...]: writes $OUT/bench/<tag>.json
bench() {
  local tag=$1 p args=()
  shift
  for p in "${POLICIES[@]}"; do args+=(--policy "$p=$(ckpt "$p" "$TRAIN_SEED")"); done
  uv run python "$EXP_DIR/bench_inference.py" "${args[@]}" --out "$OUT/bench/$tag.json" "$@"
}

# measured <tag> <policy>: the latency in frames that the bench measured.
measured() {
  uv run python -c "import json, sys; print(json.load(open(sys.argv[1]))['policies'][sys.argv[2]]['latency_frames'])" \
    "$OUT/bench/$1.json" "$2"
}

fetch_data() {
  if [ -f "$DATA/meta/info.json" ]; then echo "reusing $DATA"; return; fi
  uv run hf download "$HF_SOURCE_DATASET" --repo-type dataset --local-dir "$DATA"
}

# train_seed <policy> <seed>: D1's recipe with another seed; resumes from its last checkpoint.
train_seed() {
  local name=$1 seed=$2 dir="$OUT/train_$1_s$2"
  local last="$dir/checkpoints/last/pretrained_model/train_config.json"
  if [ -f "$last" ]; then
    uv run lerobot-train --config_path="$last" --resume=true
  else
    uv run lerobot-train --config_path="$D1_DIR/train_$name.yaml" --dataset.root="$DATA" --output_dir="$dir" \
      --job_name="plan-bind-d2-$name-s$seed" --seed="$seed" --steps="$STEPS" --batch_size="$BATCH" \
      --num_workers="$NUM_WORKERS" --save_freq=10000 --log_freq=100
  fi
}

# Uploads a trained seed, then keeps only the checkpoint `last` points to.
publish_train() {
  local dir="$OUT/train_$1" keep ckpt
  hub_upload "$HF_REPO" model "$dir" "train_$1" || return 1
  keep=$(basename "$(readlink -f "$dir/checkpoints/last")")
  for ckpt in "$dir"/checkpoints/[0-9]*; do
    [ "$(basename "$ckpt")" = "$keep" ] || rm -rf "$ckpt"
  done
}

publish_results() {
  hub_upload "$HF_REPO" model "$OUT/bench" bench || return 1
  hub_upload "$HF_REPO" model "$OUT/eval" eval || return 1
  uv run hf upload "$HF_REPO" "$OUT/results.md" results.md --repo-type=model --private
}

# eval_cell <cell> <checkpoint dir> [extra lerobot-eval args...]
eval_cell() {
  local cell=$1 path=$2 dir="$OUT/eval/$1"
  shift 2
  if [ -f "$dir/eval_info.json" ]; then echo "skip eval $cell (done)"; return; fi
  rm -rf "$dir"
  say "eval $cell"
  uv run lerobot-nexarm-eval --policy.path="$path" --policy.device=cuda --env.type=nexarm --env.task="$TASK" \
    --env.stack_order="$STACK_ORDER" --seed="$EVAL_SEED" --eval.n_episodes="$N_EVAL" \
    --eval.batch_size="$EVAL_BATCH" --output_dir="$dir" "$@"
}

# runtime_cell <policy> <latency> <commit> <suffix ('' for none)> <training seed> [extra args...]
# Runs a policy through the spline runtime; bsp10 needs `bspline_runtime` for that.
runtime_cell() {
  local p=$1 latency=$2 commit=$3 suffix=$4 seed=$5 extra=()
  shift 5
  [ "$p" = bsp10 ] && extra+=(--policy.bspline_runtime=true)
  if [ $((latency + 4)) -ge 64 ]; then
    echo "warning: skip $p at latency $latency: above the 64-frame window" >&2
    return
  fi
  eval_cell "${p}_L${latency}_${commit}${suffix}" "$(ckpt "$p" "$seed")" "${extra[@]}" \
    --policy.planbind_latency="$latency" --policy.planbind_commit="$commit" "$@"
}

eval_matrix() {
  local p c every latency tau tag seed s=$TRAIN_SEED
  # 1. Each policy at its own measured latency (one call's p95, rounded up to frames).
  for tag in cuda cpu; do
    for p in "${POLICIES[@]}"; do runtime_cell "$p" "$(measured "$tag" "$p")" latency "" "$s"; done
  done
  # 2. Update rate without latency: a call every 4 (as in D1), 8 or 12 frames.
  for p in onestep planbind; do
    for c in latency none; do
      for every in "${EVERY[@]}"; do
        if [ "$every" = 4 ]; then tag=""; else tag="_every$every"; fi
        runtime_cell "$p" 0 "$c" "$tag" "$s" --policy.planbind_call_every="$every"
      done
    done
  done
  # 3. Design-space gaps: commit "now", larger replan thresholds, bsp10 across latencies.
  for p in onestep planbind; do
    for latency in "${NOW_LATENCIES[@]}"; do
      # latency and none repeat D1's cells at latency 6: the baselines, and a check that D2 reproduces D1.
      for c in now latency none; do runtime_cell "$p" "$latency" "$c" "" "$s"; done
    done
  done
  runtime_cell planbind 6 latency "" "$s"  # the replan thresholds' baseline
  for tau in "${TAUS[@]}"; do
    runtime_cell planbind 6 latency "_tau$tau" "$s" --policy.planbind_replan_threshold="$tau"
  done
  eval_cell bsp10_L0 "$(ckpt bsp10 "$s")"
  for latency in "${BSP_LATENCIES[@]}"; do runtime_cell bsp10 "$latency" latency "" "$s"; done
  runtime_cell bsp10 0 none "" "$s"
  # 4. The bottom bowl slides when the middle bowl is lifted.
  eval_cell "bsp10_L0_$SHIFT_TAG" "$(ckpt bsp10 "$s")" --env.bowl_shift_m="$SHIFT_M"
  for p in "${POLICIES[@]}"; do
    for latency in "${SHIFT_LATENCIES[@]}"; do
      runtime_cell "$p" "$latency" latency "_$SHIFT_TAG" "$s" --env.bowl_shift_m="$SHIFT_M"
    done
  done
  # 5. Extra training seeds of the two one-step-cost policies.
  for seed in $(extra_seeds); do
    for p in onestep planbind; do
      for latency in 0 6; do
        for c in latency none; do runtime_cell "$p" "$latency" "$c" "_s$seed" "$seed"; done
      done
    done
  done
}

extra_seeds() { local i; for ((i = 1; i < SEEDS; i++)); do echo $((TRAIN_SEED + i)); done; }

run_all() {
  mkdir -p "$OUT/bench" "$OUT/eval"
  run_stage fetch_d1 fetch_d1
  run_stage bench_cuda bench cuda --device cuda --n "$BENCH_N" --frames "$BENCH_FRAMES"
  run_stage bench_cuda_amp bench cuda_amp --device cuda --amp --n "$BENCH_N" --frames "$BENCH_FRAMES"
  run_stage bench_cpu bench cpu --device cpu --cpu-threads "$CPU_THREADS" --n "$BENCH_CPU_N" \
    --frames "$BENCH_FRAMES"
  if [ "$SEEDS" -gt 1 ]; then
    run_stage data fetch_data
    local seed p
    for seed in $(extra_seeds); do
      for p in onestep planbind; do
        run_stage "train_${p}_s$seed" train_seed "$p" "$seed"
        if publishing "${HF_REPO:-}"; then publish "hub_train_${p}_s$seed" publish_train "${p}_s$seed"; fi
      done
    done
  fi
  run_stage eval eval_matrix
  uv run python "$EXP_DIR/summarize.py" "$OUT" > "$OUT/results.md"
  if publishing "${HF_REPO:-}"; then publish hub_results publish_results; fi
  wait
  say "results: $OUT/results.md"
  cat "$OUT/results.md"
}

run_smoke() {
  OUT="$EXP_DIR/outputs/smoke"
  N_EVAL=2 EVAL_BATCH=2 SEEDS=$FULL_SEEDS STEPS=200 BATCH=8 NUM_WORKERS=2 PUBLISH=0
  BENCH_N=10 BENCH_CPU_N=5 BENCH_FRAMES=40
  EVERY=(8) NOW_LATENCIES=(6) TAUS=(0.5) BSP_LATENCIES=(6) SHIFT_LATENCIES=(0)
  run_all
}

run_full() {
  OUT="$EXP_DIR/outputs/full"
  N_EVAL=$FULL_N_EVAL EVAL_BATCH=$FULL_EVAL_BATCH SEEDS=$FULL_SEEDS STEPS=$FULL_STEPS BATCH=$FULL_BATCH
  NUM_WORKERS=$FULL_NUM_WORKERS PUBLISH=1
  BENCH_N=200 BENCH_CPU_N=50 BENCH_FRAMES=240
  EVERY=(4 8 12) NOW_LATENCIES=(0 6) TAUS=(0.5 1.0) BSP_LATENCIES=(0 3 6 9) SHIFT_LATENCIES=(0 6)
  run_all
}

MODE=${1:-all}
case "$MODE" in
  smoke|full|all) ;;
  *) die "usage: $0 [smoke|full|all]" ;;
esac
say "preflight"
if [ "${SEEDS:-1}" -gt 1 ]; then
  preflight 80 "the venv, the dataset and the extra seeds' checkpoints (3.5 GB each, 6 per seed and policy)"
else
  preflight 40 "the venv and the three D1 checkpoints"
fi
say "setup"
setup
case "$MODE" in
  smoke) run_smoke ;;
  full) run_full ;;
  all) run_smoke; run_full ;;
esac
