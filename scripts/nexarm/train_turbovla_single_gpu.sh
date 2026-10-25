#!/usr/bin/env bash
# Single-GPU training launcher for TurboVLA on NexArm (lerobot-train --policy.type=turbovla)
# Usage: train_turbovla_single_gpu.sh [DATASET_ROOT] [OUTPUT_DIR] [BATCH_SIZE] [MAX_STEPS] [SAVE_STEPS] [extra lerobot-train flags...]

set -e

GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
DATASET_ROOT=${1:-"outputs/datasets/nexarm_stack_bowls"}
OUTPUT_DIR=${2:-"outputs/train/nexarm_turbovla"}
BATCH_SIZE=${3:-16}
MAX_STEPS=${4:-50000}
SAVE_STEPS=${5:-5000}
REPO_ID=${REPO_ID:-"local/$(basename "$DATASET_ROOT")"}

echo "=== Training TurboVLA on Single GPU ($GPU_ID) ==="
echo "Dataset:    $REPO_ID ($DATASET_ROOT)"
echo "Output:     $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max steps:  $MAX_STEPS"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run lerobot-train \
    --policy.type=turbovla \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET_ROOT" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.chunk_size=16 \
    --policy.optimizer_lr=5e-5 \
    --policy.optimizer_vision_lr=5e-5 \
    --policy.vision_compute_precision=bf16_autocast \
    --policy.interaction_compute_precision=bf16_autocast \
    --accelerator.mixed_precision=bf16 \
    --ema.enable=true \
    --ema.decay=0.999 \
    --batch_size="$BATCH_SIZE" \
    --steps="$MAX_STEPS" \
    --save_freq="$SAVE_STEPS" \
    --output_dir="$OUTPUT_DIR" \
    "${@:6}"
