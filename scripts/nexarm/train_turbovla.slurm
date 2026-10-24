#!/usr/bin/env bash
#SBATCH --job-name=nexarm-turbovla
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/turbovla_%j.out
#SBATCH --error=logs/turbovla_%j.err

# Load system CUDA if required by cluster modules:
# module load cuda/12.4

echo "Job $SLURM_JOB_ID started on $(hostname) with $SLURM_GPUS_ON_NODE GPUs"

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

mkdir -p logs

uv run accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --main_process_port=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4)) \
    -m lerobot.scripts.lerobot_train \
    --policy.type=turbovla \
    --dataset.repo_id=local/nexarm_stack_bowls \
    --dataset.root=outputs/datasets/nexarm_stack_bowls \
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
    --batch_size=16 \
    --steps=50000 \
    --save_freq=5000 \
    --output_dir=outputs/train/nexarm_turbovla
