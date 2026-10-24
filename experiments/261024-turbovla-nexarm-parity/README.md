# 261024-turbovla-nexarm-parity

**Status: pending (needs a Linux GPU machine; the sim eval also needs EGL).** Only the CPU-scale checks below have run. The GPU results tables are empty on purpose.

## Hypothesis

`lerobot-train --policy.type=turbovla` (plugin `policies/lerobot_policy_turbovla/`) trains TurboVLA the way `examples/nexarm/train_turbovla.py` does: same loss with episode-end padding masked, same parameter groups and frozen BERT, same AdamW and warmup-cosine schedule. On the same data, seed and effective batch, the two 2k-step loss curves overlap within run-to-run noise. A full run then reaches the same stack-bowls sim success rate as the old pipeline's checkpoint. Once that holds, the custom train and rollout scripts can be retired.

## Data

| Field             | Value                                                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset `repo_id` | parity: `thanhkt/nexarm_stack_bowls` (real teleop, `front` + `wrist`); full run: `local/nexarm_stack_bowls` (sim, root `outputs/datasets/nexarm_stack_bowls`) |
| Revision          | parity: `64ce8aeb1d3968293c0290c4110bcea7424325b1` (the Hub head on 2026-10-24); full run: record the sim generator commit                                    |
| Episodes used     | all                                                                                                                                                           |

`train_turbovla.py` has no revision flag and loads the Hub head. Before the parity run, check that the head is still the revision above (`uv run python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('thanhkt/nexarm_stack_bowls').sha)"`).

## Command

Both pipelines use one GPU, bf16, batch 16, seed 1000, 1000 warmup steps, lr 5e-5 for every group, EMA 0.999. Keep gradient accumulation at 1: lerobot-train steps the scheduler once per batch, the script once per optimizer update, so they only agree without accumulation.

2k-step parity, old pipeline:

```bash
uv run python examples/nexarm/train_turbovla.py \
    --no-sim --real-repo-id thanhkt/nexarm_stack_bowls \
    --cameras observation.images.front,observation.images.wrist \
    --output-dir experiments/261024-turbovla-nexarm-parity/outputs/script_parity \
    --batch-size 16 --max-steps 2000 --warmup-steps 1000 --save-steps 2000 --log-steps 20 \
    --lr 5e-5 --dinov3-lr 5e-5 --horizon 16 --precision bf16 --seed 1000 --num-workers 8 \
    --device cuda --wandb --wandb-project nexarm-turbovla
```

2k-step parity, LeRobot:

```bash
uv run lerobot-train --config_path experiments/261024-turbovla-nexarm-parity/train_parity.yaml
```

Compare `train/action_loss` (script) with `train/loss` (LeRobot) in W&B. The data order differs (the script shuffles with the global RNG, lerobot-train with a seeded generator), so compare the curves, not single steps.

Full run, both pipelines (script defaults: 50k steps, 1000 warmup):

```bash
uv run python examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --cameras observation.images.front,observation.images.wrist,observation.images.top \
    --output-dir experiments/261024-turbovla-nexarm-parity/outputs/script_full \
    --batch-size 16 --max-steps 50000 --save-steps 5000 --lr 5e-5 --horizon 16 \
    --precision bf16 --seed 1000 --num-workers 8 --device cuda --wandb --wandb-project nexarm-turbovla
uv run lerobot-train --config_path experiments/261024-turbovla-nexarm-parity/train_full.yaml
```

Convert the script's EMA checkpoint, then evaluate both in sim with 50 episodes each:

```bash
EXP=experiments/261024-turbovla-nexarm-parity/outputs
uv run python -m lerobot_policy_turbovla.convert_checkpoint \
    --checkpoint $EXP/script_full/final_ema_pytorch_model.pt --output-dir $EXP/script_full_lerobot
MUJOCO_GL=egl uv run lerobot-eval --policy.path=$EXP/script_full_lerobot \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=50 --output_dir=$EXP/eval_script
MUJOCO_GL=egl uv run lerobot-eval --policy.path=$EXP/lerobot_full/checkpoints/last/pretrained_model_ema \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=50 --output_dir=$EXP/eval_lerobot
```

Multi-GPU check (100 steps on 2 GPUs):

```bash
uv run accelerate launch --num_processes=2 .venv/bin/lerobot-train \
    --config_path experiments/261024-turbovla-nexarm-parity/train_parity.yaml \
    --steps=100 --output_dir=experiments/261024-turbovla-nexarm-parity/outputs/lerobot_2gpu
```

With 2 GPUs the effective batch is 32; compare it with `torchrun --nproc_per_node=2 examples/nexarm/train_turbovla.py ...` at the same per-GPU batch.

## CPU-scale checks (done, 2026-10-24)

Windows Server 2022, CPU, fp32, BERT-tiny text encoder with the bert-base tokenizer and `facebook/dinov2-with-registers-small` at 224 px, episodes 143-152 (3072 frames).

- `policies/lerobot_policy_turbovla/tests/test_training_parity.py`: on the same seeded weights and the same 6 frames of episode 143 (2 with padded chunks), the preprocessed LeRobot batch equals the script collator's output, and the loss from `TurboVLAPolicy.forward` equals the script's loss (difference 0.0, tolerance 1e-5). Two updates through lerobot-train's `update_policy` and through the script's optimizer, clipping and schedule give the same weights (max difference 0.0, tolerance 1e-6). Parameter groups, learning rates and frozen modules match.
- 30-step smoke, batch 8, lr 3e-4, 5 warmup steps, seed 1000, both pipelines: loss went from 0.687 / 0.707 (LeRobot / script) at step 1 to 0.479 / 0.351 at step 30; mean of steps 1-5 0.568 / 0.544, of steps 26-30 0.430 / 0.435. The learning-rate sequences are identical. The full sequences are in `plans/reports/turbovla-training-261024-1156-phase-06.md`.

## Checkpoint

Hub ids, steps and W&B run links: pending.

## Results

| Pipeline                  | Setting           | Steps | Episodes | Success rate | Notes |
| ------------------------- | ----------------- | ----- | -------- | ------------ | ----- |
| `train_turbovla.py`       | parity loss at 2k | 2000  | n/a      | n/a          |       |
| `lerobot-train`           | parity loss at 2k | 2000  | n/a      | n/a          |       |
| `train_turbovla.py` (EMA) | sim stack-bowls   | 50000 | 50       |              |       |
| `lerobot-train` (EMA)     | sim stack-bowls   | 50000 | 50       |              |       |
| `lerobot-train`, 2 GPUs   | 100 steps run     | 100   | n/a      | n/a          |       |

## Notes

- EMA: `--ema.enable=true --ema.decay=0.999` uses diffusers' `EMAModel`, whose first update copies the weights instead of blending them. This changes the EMA weights a little early in training, not the loss curve.
- Not available through lerobot-train: sim/real co-training with `--real-ratio`, and `--pretrained-checkpoint` (partial loading of TurboVLA release weights). Keep using the script for those until the plugin covers them.
- When a batch has no task, the script uses "Complete the robotic manipulation task." and the plugin uses `--policy.default_task` (empty by default). NexArm datasets always have a task.
