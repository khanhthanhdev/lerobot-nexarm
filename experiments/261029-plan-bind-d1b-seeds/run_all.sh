#!/usr/bin/env bash
# Plan-Bind D1b: D1 (stack bowls) repeated with training seeds, more eval episodes and
# stronger one-step controls, on one or several GPUs. Same data, configs and eval as D1
# (../261027-plan-bind-two-step); only the statistics change.
#   1. trains bsp10, onestep and planbind with SEEDS training seeds (1000, 1001, ...), plus two controls for
#      the one-step baseline (seed 1000 only): onestep_k2 (fine knots every 2 frames, so it calls twice as
#      often) and onestep_long (twice the steps);
#   2. after each training, evaluates its last checkpoint at latency 0 (the headline number);
#   3. when all are trained, evaluates the last checkpoints: latency 0, 200 and 300 ms, latency commit, and
#      naive swapping at 0 ms.
#   Only each training's last checkpoint is kept (the others are deleted after step 2), so there is no checkpoint curve.
# Every stage leaves a .done marker, training resumes from its last checkpoint and finished eval cells are
# kept, so after a disconnect run the same command again. Run it under tmux; status.sh shows progress.
#
# usage (from anywhere):
#   experiments/261029-plan-bind-d1b-seeds/run_all.sh          # smoke run, then the full run
#   experiments/261029-plan-bind-d1b-seeds/run_all.sh smoke    # proves the pipeline
#   experiments/261029-plan-bind-d1b-seeds/run_all.sh full     # the experiment
# GPUS="0 1 2" spreads the trainings, then the eval cells, over those GPUs (one worker each; default one GPU).
# env overrides: SEEDS=3 STEPS=60000 BATCH=32 SAVE_FREQ=10000 NUM_WORKERS=4 N_EVAL=100 EVAL_BATCH=10
# The dataset is D1's: with HF_DATASET_REPO=<user>/<repo> it is downloaded when missing, otherwise it is
# generated (500 episodes, about 1 h). HF_REPO=<user>/<repo> uploads each trained policy (then deletes its
# local checkpoints except the last), the eval cells and results.md to that private repo (full run only).
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$EXP_DIR/../.." && pwd)"
D1_DIR="$REPO/experiments/261027-plan-bind-two-step"
cd "$REPO"
# shellcheck source=../261027-plan-bind-two-step/common.sh
source "$D1_DIR/common.sh"

TASK=NexArmStackBowls-v0
EVAL_SEED=1000  # the same eval episodes in every cell, so cells compare episode by episode
STACK_ORDER=${STACK_ORDER:-red,blue,black}
# Full-run settings, read now: the smoke run reuses these names in `all` mode.
FULL_SEEDS=${SEEDS:-3} FULL_STEPS=${STEPS:-60000} FULL_BATCH=${BATCH:-32} FULL_SAVE_FREQ=${SAVE_FREQ:-10000}
FULL_NUM_WORKERS=${NUM_WORKERS:-4} FULL_N_EVAL=${N_EVAL:-100}
FULL_EVAL_BATCH=${EVAL_BATCH:-10}

seeds() { local i; for ((i = 0; i < SEEDS; i++)); do echo $((1000 + i)); done; }

# job_list (read through fd 3, so the commands in the loops cannot eat their stdin): "<variant>_s<seed>" lines, the slowest first so the GPUs finish together.
job_list() {
  local seed p
  echo onestep_long_s1000
  for p in bsp10 planbind onestep; do
    for seed in $(seeds); do echo "${p}_s$seed"; done
  done
  echo onestep_k2_s1000
}

# The D1 config a variant starts from, and its extra training flags.
base_config() { case $1 in onestep_*) echo onestep ;; *) echo "$1" ;; esac; }
variant_args() {
  case $1 in
    onestep_k2) echo --policy.planbind_fine_spacing=2 ;;
  esac
}

get_data() {
  local info="$DATA/meta/info.json"
  if [ -f "$info" ]; then
    local tasks
    tasks=$(uv run python -c "import json, sys; print(json.load(open(sys.argv[1])).get(\"total_tasks\", 1))" "$info")
    [ "$tasks" = 1 ] || die "$DATA has $tasks tasks (stacking orders); the policies need one. Set DATA to a single-order dataset"
    echo "reusing $DATA"
    return
  fi
  [ ! -e "$DATA" ] || die "$DATA exists but is not a LeRobot dataset; remove it and rerun"
  if [ "$PUBLISH" = 1 ] && [ -n "${HF_DATASET_REPO:-}" ]; then
    uv run hf download "$HF_DATASET_REPO" --repo-type=dataset --local-dir "$DATA"
  else
    uv run python examples/nexarm/generate_stack_bowls_dataset.py --repo-id local/nexarm_stack_bowls \
      --root "$DATA" --episodes "$EPISODES" --order "$STACK_ORDER" --gpus 0 --workers "$GEN_WORKERS"
  fi
}

# train <variant> <seed>: resumes from the last checkpoint.
train() {
  local name=$1 seed=$2 dir="$OUT/train_$1_s$2" steps=$STEPS
  [ "$name" = onestep_long ] && steps=$((STEPS * 2))  # twice the steps
  local last="$dir/checkpoints/last/pretrained_model/train_config.json"
  if [ -f "$last" ]; then
    uv run lerobot-train --config_path="$last" --resume=true
  else
    # shellcheck disable=SC2046  # variant_args prints a list of flags
    uv run lerobot-train --config_path="$D1_DIR/train_$(base_config "$name").yaml" --dataset.root="$DATA" \
      --output_dir="$dir" --seed="$seed" --job_name="plan-bind-d1b-$name-s$seed" --steps="$steps" \
      --batch_size="$BATCH" --num_workers="$NUM_WORKERS" --save_freq="$SAVE_FREQ" --log_freq=100 \
      $(variant_args "$name")
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

# eval_cell <cell> <checkpoint dir> <episodes> [extra lerobot-nexarm-eval args...]
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
  say "eval $cell"
  uv run lerobot-nexarm-eval --policy.path="$path" --policy.device=cuda --env.type=nexarm --env.task="$TASK" \
    --env.stack_order="$STACK_ORDER" --seed="$EVAL_SEED" --eval.n_episodes="$episodes" \
    --eval.batch_size="$EVAL_BATCH" --output_dir="$dir" "$@"
}

# Cell name stem: bsp10 runs its own action queue (no commit mode), the one-step modes the runtime.
stem() { case $1 in bsp10) echo "$1_L$2" ;; *) echo "$1_L$2_${3:-latency}" ;; esac; }
runtime_args() { case $1 in bsp10) ;; *) echo --policy.planbind_latency="$2" --policy.planbind_commit="${3:-latency}" ;; esac; }

# prune <job>: keeps only the checkpoint `last` points to, to save disk.
prune() {
  local dir="$OUT/train_$1" keep ckpt
  keep=$(basename "$(readlink -f "$dir/checkpoints/last")")
  for ckpt in "$dir"/checkpoints/[0-9]*; do
    [ "$(basename "$ckpt")" = "$keep" ] || rm -rf "$ckpt"
  done
}

# final0 <variant> <seed>: the last checkpoint at latency 0, the main comparison's cell (the matrix keeps it),
# right after training so the headline number of each policy comes first.
final0() {
  local name=$1 seed=$2
  # shellcheck disable=SC2046  # runtime_args prints a list of flags
  eval_cell "$(stem "$name" 0)_s$seed" "$OUT/train_${name}_s$seed/checkpoints/last/pretrained_model" "$N_EVAL" \
    $(runtime_args "$name" 0)
}

eval_matrix() {
  local line name seed path latency commit every
  while read -r line <&3; do
    name=${line%_s*} seed=${line##*_s}
    path="$OUT/train_$line/checkpoints/last/pretrained_model"
    if [ "$name" = bsp10 ]; then
      eval_cell "bsp10_L0_s$seed" "$path" "$N_EVAL"
      continue
    fi
    # shellcheck disable=SC2046
    for latency in 0 6 9; do  # frames at 30 fps: 0, 200 and 300 ms
      eval_cell "$(stem "$name" "$latency")_s$seed" "$path" "$N_EVAL" $(runtime_args "$name" "$latency")
    done
    eval_cell "$(stem "$name" 0 none)_s$seed" "$path" "$N_EVAL" $(runtime_args "$name" 0 none)
    # The control that calls twice as often is also run at D1's call rate (every 4 frames).
    if [ "$name" = onestep_k2 ]; then
      for latency in 0 6; do
        eval_cell "$(stem "$name" "$latency")_every4_s$seed" "$path" "$N_EVAL" \
          $(runtime_args "$name" "$latency") --policy.planbind_call_every=4
      done
    fi
  done 3< <(job_list)
}

# worker <index> <gpu> <workers>: takes the next unclaimed job (the list is slowest first), trains it on its GPU,
# evaluates its last checkpoint and deletes the others, and repeats; then waits for the other workers'
# trainings, evaluates every <workers>-th final cell, and uploads its policies.
# EXTERNAL_TRAIN="<job> ..." names jobs trained by another process, which touches $OUT/.done_train_<job>.
worker() {
  local w=$1 gpu=$2 n=$3 line mine=()
  export CUDA_VISIBLE_DEVICES=$gpu
  while read -r line <&3; do
    case " ${EXTERNAL_TRAIN:-} " in
      *" $line "*)
        if [ "$w" = 0 ]; then mine+=("$line"); fi
        continue
        ;;
    esac
    if mkdir "$OUT/.claim_$line" 2>/dev/null; then  # atomic: one worker per job
      mine+=("$line")
      run_stage "train_$line" train "${line%_s*}" "${line##*_s}"
      run_stage "final0_$line" final0 "${line%_s*}" "${line##*_s}"
      run_stage "prune_$line" prune "$line"
    fi
  done 3< <(job_list)
  while read -r line <&3; do
    while [ ! -f "$OUT/.done_train_$line" ]; do sleep 30; done
  done 3< <(job_list)
  EVAL_N=$n EVAL_W=$w EVAL_IDX=0
  run_stage "eval_w$w" eval_matrix
  if publishing "${HF_REPO:-}"; then
    for line in "${mine[@]}"; do publish "hub_train_$line" publish_train "$line"; done
  fi
  wait
}

run_all() {
  mkdir -p "$OUT/eval" "$OUT/logs"
  rm -rf "$OUT"/.claim_*  # claims last one launch; finished jobs are skipped by their .done markers
  run_stage data get_data
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
  wait
  say "results: $OUT/results.md"
  cat "$OUT/results.md"
}

run_smoke() {
  OUT="$EXP_DIR/outputs/smoke" DATA="$EXP_DIR/outputs/smoke/data"
  EPISODES=4 GEN_WORKERS=2 SEEDS=1 STEPS=200 BATCH=8 SAVE_FREQ=100 NUM_WORKERS=2 N_EVAL=2 EVAL_BATCH=2
  PUBLISH=0
  run_all
}

run_full() {
  # DATA: the single-order dataset (red, blue, black); a six-order dataset makes every policy fail.
  OUT="$EXP_DIR/outputs/full" DATA="${DATA:-$REPO/outputs/datasets/nexarm_stack_bowls_red_blue_black_1k}"
  EPISODES=500 GEN_WORKERS=8 SEEDS=$FULL_SEEDS STEPS=$FULL_STEPS BATCH=$FULL_BATCH SAVE_FREQ=$FULL_SAVE_FREQ
  NUM_WORKERS=$FULL_NUM_WORKERS N_EVAL=$FULL_N_EVAL EVAL_BATCH=$FULL_EVAL_BATCH PUBLISH=1
  run_all
}

MODE=${1:-all}
case "$MODE" in
  smoke|full|all) ;;
  *) die "usage: $0 [smoke|full|all]" ;;
esac
say "preflight"
preflight 120 "the venv, dataset and checkpoints (3.5 GB each; all but the last are deleted after an upload)"
say "setup"
setup
case "$MODE" in
  smoke) run_smoke ;;
  full) run_full ;;
  all) run_smoke; run_full ;;
esac
