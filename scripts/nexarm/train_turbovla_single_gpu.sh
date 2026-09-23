#!/usr/bin/env bash
# Single-GPU training launcher for TurboVLA on NexArm

set -e

GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
DATASET_ROOT=${1:-"outputs/datasets/nexarm_stack_bowls"}
OUTPUT_DIR=${2:-"outputs/train/nexarm_turbovla_single"}
BATCH_SIZE=${3:-16}
MAX_STEPS=${4:-50000}
SAVE_STEPS=${5:-5000}

echo "=== Training TurboVLA on Single GPU ($GPU_ID) ==="
echo "Dataset:    $DATASET_ROOT"
echo "Output:     $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max steps:  $MAX_STEPS"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run python examples/nexarm/train_turbovla.py \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --lr 5e-5 \
    --horizon 16 \
    --device cuda
