---
phase: 6
title: "TurboVLA training and LIBERO benchmark"
status: pending
priority: P2
effort: "3-5d + GPU time"
dependencies: [5]
---

# Phase 06: TurboVLA training and LIBERO benchmark

## Goal
Train TurboVLA with `lerobot-train` on NexArm datasets, benchmark it on LIBERO through LeRobot, and retire the custom TurboVLA train and rollout scripts.

## Context
- `examples/nexarm/train_turbovla.py` (1,128 lines) is a custom AdamW + LambdaLR loop with torchrun support, reading `LeRobotDataset`. The launchers are in `scripts/nexarm/` (single-GPU, multi-GPU, Slurm).
- Upstream LeRobot trains with `accelerate`, including multi-GPU and gradient accumulation.
- This fork has `src/lerobot/envs/libero.py`, so `lerobot-eval --env.type=libero` gives a benchmark without TurboVLA's RLDS loaders. TurboVLA's own mixed-suite RLDS training stays in the TurboVLA repo (non-goal).

## Files to Create / Modify
- Modify: `policies/lerobot_policy_turbovla/.../modeling_turbovla.py` (`forward`, `get_optim_params`), `configuration_turbovla.py` (optimizer and scheduler presets matching `train_turbovla.py`)
- Create: `experiments/<date>-turbovla-nexarm-parity/`, `experiments/<date>-turbovla-libero/`
- Modify: `scripts/nexarm/train_turbovla_*.sh`, `train_turbovla.slurm`: call `accelerate launch lerobot-train --policy.type=turbovla`
- Delete (after parity): `examples/nexarm/train_turbovla.py`, `rollout_turbovla.py`; keep or fold in `setup_turbovla.py` (weight download) and `diagnose_turbovla.py` if still useful

## Tasks & Steps
1. `forward` returns `(loss, logs)` with the same loss terms as `train_turbovla.py`, and handles `action_is_pad`.
2. `get_optim_params` reproduces the per-group learning rates and frozen modules; the scheduler preset matches the LambdaLR shape.
3. Parity run: same NexArm dataset, seed and effective batch, 2k steps in both pipelines. The loss curves should overlap within run-to-run noise.
4. Full NexArm run through `lerobot-train`, then `lerobot-eval` with 50 episodes on stack-bowls sim, compared with the old pipeline's checkpoint.
5. LIBERO: train on LeRobot-format LIBERO data from the Hub (pick the repo ids at execution time), then `lerobot-eval --env.type=libero` with 50 episodes per suite. Record the results in the experiment README.
6. Retire the old scripts and update the launchers.

## Verification
- The parity loss curves and the stack-bowls success rate are documented in the experiment folder.
- LIBERO results table with the exact `lerobot-eval` command and hardware.
- The multi-GPU launcher runs 100 steps on 2 GPUs.

## Risk / Rollback
If some TurboVLA training feature cannot be expressed through `lerobot-train`, keep that part as a documented exception rather than editing `lerobot_train.py`.
