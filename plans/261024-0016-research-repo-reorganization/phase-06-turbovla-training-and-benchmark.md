---
phase: 6
title: "TurboVLA training and LIBERO benchmark"
status: in-progress
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

## Execution notes (2026-10-24)

CPU-reachable part done on Windows Server 2022 (no GPU, no OpenGL); steps 1-2 and a CPU-scale stand-in for step 3. Report: [turbovla-training-261024-1156-phase-06.md](../reports/turbovla-training-261024-1156-phase-06.md).

- `TurboVLAPolicy.forward` returns `(loss, {"action_loss": ...})` with the script's loss: per-element L1 (or `mse` / `smooth_l1`), gripper weight 3.0, `action_is_pad` steps excluded, sum divided by valid steps times action dim. `get_optim_params` returns the script's three groups (action head; the rest except the vision encoder; the vision encoder, left out when frozen) with `optimizer_lr` / `optimizer_vision_lr`. `freeze_text` (default true), `freeze_vision`, `freeze_interaction` set `requires_grad` and keep those modules in eval mode across lerobot-train's per-step `train()` calls.
- Presets: `AdamWConfig(lr=5e-5, weight_decay=1e-10, grad_clip_norm=1.0)` and a new plugin scheduler `turbovla_cosine_with_warmup` (no LeRobot scheduler has the script's `(step + 1) / warmup` warmup and 5% cosine floor over `steps - warmup`). `action_delta_indices` = `range(chunk_size)`, so lerobot-train loads 16-step chunks with `action_is_pad`; checked against the script's `delta_timestamps`.
- Images: one behaviour, the plugin's rounding, in training and inference. For all 256 uint8 levels, `uint8 / 255` in float32 gives back the same byte under truncation and rounding, so the script and the plugin feed identical pixel values on dataset frames (the test asserts exact equality).
- `train_turbovla.py`: `build_param_groups`, `lr_multiplier` and `action_loss` extracted from `main()` unchanged so tests import them, and `--seed` added (default off). Launchers untouched; the script and `rollout_turbovla.py` stay until the GPU parity passes.
- Parity tests (`policies/lerobot_policy_turbovla/tests/test_training_parity.py`, BERT-tiny + DINOv2-small, 6 frames of episode 143 with 2 padded chunks): preprocessed batch equals the script collator's output; loss difference 0.0 (tolerance 1e-5) for two dropout seeds; after two updates through lerobot-train's `update_policy` vs the script's AdamW + clipping + LambdaLR, max weight difference 0.0 (tolerance 1e-6); groups, learning rates, frozen sets and schedule match. Mutations (no padding mask; frozen modules back in train mode) make the tests fail.
- 30-step CPU smoke, episodes 143-152, batch 8, lr 3e-4, warmup 5, seed 1000: `lerobot-train --policy.type=turbovla` loss 0.687 -> 0.479 (mean of first/last 5 steps 0.568 -> 0.430), checkpoint saved, reloaded with `make_policy`, `select_action` returns (8, 6) raw actions. `train_turbovla.py` with the same settings 0.707 -> 0.351 (0.544 -> 0.435); identical learning-rate sequence. Data order and initial weights differ between the two runs, so the curves agree in trend, not step by step.
- Experiments: `experiments/261024-turbovla-nexarm-parity/` (2k-step parity in both pipelines, full 50k runs, 50-episode `NexArmStackBowls-v0` eval, 2-GPU check) and `experiments/261024-turbovla-libero/` (`lerobot/libero` at `a1aaacb7`, verified on the Hub; 4 suites x 50 episodes), results pending GPU/Linux. Docs: the `lerobot-train` / `accelerate launch` commands sit next to the old ones in `SERVER_TRAINING_GUIDE.md` and `turbovla_guide.md`, marked as the replacement once parity is confirmed.
- Not expressible through lerobot-train (kept in the script, documented): sim/real co-training with `--real-ratio`, `--pretrained-checkpoint` partial loading; with gradient accumulation lerobot-train's scheduler counts batches, the script's counts updates. diffusers' EMA copies the weights on its first update where the script blends.
- Open: steps 3-6 on GPU (2k-step parity, full run + sim eval, LIBERO, 100-step 2-GPU run), then retire the scripts and switch the launchers.
