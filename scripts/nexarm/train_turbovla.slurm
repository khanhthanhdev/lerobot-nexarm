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

uv run torchrun \
    --nproc_per_node=4 \
    --master_port=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4)) \
    examples/nexarm/train_turbovla.py \
    --dataset-root outputs/datasets/nexarm_stack_bowls \
    --output-dir outputs/train/nexarm_turbovla_cluster \
    --batch-size 16 \
    --max-steps 50000 \
    --save-steps 5000 \
    --lr 5e-5 \
    --horizon 16
