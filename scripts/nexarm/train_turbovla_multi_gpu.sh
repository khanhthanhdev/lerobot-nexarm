#!/usr/bin/env bash
# Multi-GPU training launcher for TurboVLA on NexArm using PyTorch DDP (torchrun)

set -e

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 2)}
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=2
fi

DATASET_ROOT=${1:-"outputs/datasets/nexarm_stack_bowls"}
OUTPUT_DIR=${2:-"outputs/train/nexarm_turbovla_ddp"}
BATCH_SIZE=${3:-16}
MAX_STEPS=${4:-50000}
SAVE_STEPS=${5:-5000}

echo "=== Training TurboVLA on $NUM_GPUS GPUs (torchrun) ==="
echo "Dataset:    $DATASET_ROOT"
echo "Output:     $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE per GPU (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo "Max steps:  $MAX_STEPS"

uv run torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=29500 \
    examples/nexarm/train_turbovla.py \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --lr 5e-5 \
    --horizon 16
