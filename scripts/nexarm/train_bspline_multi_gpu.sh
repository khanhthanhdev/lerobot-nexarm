#!/usr/bin/env bash
# Multi-GPU training launcher for B-spline Policy on NexArm (accelerate launch lerobot-train --policy.type=bspline)
# Usage: train_bspline_multi_gpu.sh [DATASET_ROOT] [OUTPUT_DIR] [BATCH_SIZE] [MAX_STEPS] [SAVE_STEPS] [extra lerobot-train flags...]

set -e

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 2)}
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=2
fi

DATASET_ROOT=${1:-"outputs/datasets/nexarm_stack_bowls"}
OUTPUT_DIR=${2:-"outputs/train/nexarm_bspline"}
BATCH_SIZE=${3:-16}
MAX_STEPS=${4:-100000}
SAVE_STEPS=${5:-25000}
REPO_ID=${REPO_ID:-"local/$(basename "$DATASET_ROOT")"}
# Processes per GPU that fit the B-spline training targets on the CPU.
FIT_WORKERS=${FIT_WORKERS:-4}

echo "=== Training B-spline Policy on $NUM_GPUS GPUs (accelerate) ==="
echo "Dataset:     $REPO_ID ($DATASET_ROOT)"
echo "Output:      $OUTPUT_DIR"
echo "Batch size:  $BATCH_SIZE per GPU (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo "Max steps:   $MAX_STEPS"
echo "Fit workers: $FIT_WORKERS per GPU"

uv run accelerate launch \
    --multi_gpu \
    --num_processes="$NUM_GPUS" \
    --main_process_port=29500 \
    -m lerobot.scripts.lerobot_train \
    --policy.type=bspline \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.bspline_fit_workers="$FIT_WORKERS" \
    --batch_size="$BATCH_SIZE" \
    --steps="$MAX_STEPS" \
    --save_freq="$SAVE_STEPS" \
    --output_dir="$OUTPUT_DIR" \
    "${@:6}"
