---
phase: 5
title: "TurboVLA policy plugin, inference"
status: completed
priority: P1
effort: "2-3d"
dependencies: [4]
---

# Phase 05: TurboVLA policy plugin, inference

## Goal

Package `lerobot_policy_turbovla` so an existing TurboVLA checkpoint runs through `lerobot-eval`, `lerobot-rollout` and the desktop app (`nexarm_service` rollout, eval and DAgger coaching), giving the same actions as `rollout_turbovla.py`.

## Context

- LeRobot finds plugins by the installed package name (prefix `lerobot_policy_`) and imports that name as a module (`utils/import_utils.py:227`). All seven `lerobot_*` CLI scripts call `register_third_party_plugins()`.
- The factory finds the policy class by renaming `configuration_x` to `modeling_x` and `XConfig` to `XPolicy` (`policies/factory.py:658`). The processor must be named `make_turbovla_pre_post_processors`.
- `turbovla.models` already has a class called `TurboVLAConfig`. Import it under an alias (`TurboVLAModelConfig`) so it does not clash with the LeRobot config class.
- `nexarm_service/rollout.py:59` loads any registered policy with `PreTrainedConfig.from_pretrained`, so the desktop app works once the plugin is installed.
- NexArm-specific TurboVLA code sits in `TurboVLA/experiments/nexarm` (configs, data registry). The parts needed for inference move into this package.

## Files to Create / Modify

- Create: `policies/lerobot_policy_turbovla/pyproject.toml` (`name = "lerobot_policy_turbovla"`, depends on `lerobot` and `turbovla`)
- Create: `policies/lerobot_policy_turbovla/src/lerobot_policy_turbovla/{__init__,configuration_turbovla,modeling_turbovla,processor_turbovla,convert_checkpoint}.py`
- Create: `policies/lerobot_policy_turbovla/tests/test_parity.py`, `test_plugin_discovery.py`
- Modify: root `pyproject.toml` (workspace member, `turbovla` extra pulls the plugin)

## Tasks & Steps

1. Config: `@PreTrainedConfig.register_subclass("turbovla")`, `TurboVLAConfig(PreTrainedConfig)` with the fields `rollout_turbovla.py` uses today (backbones, chunk size, n_action_steps, image keys, state/action dims, normalization mode).
2. Policy: `TurboVLAPolicy(PreTrainedPolicy)` wraps the turbovla model. Implement `reset`, `select_action` (action-chunk queue), `predict_action_chunk`. For now, `forward` raises `NotImplementedError` with a pointer to phase 06.
3. Processor: reproduce exactly what `rollout_turbovla.py` does: image resize and normalization, state/action normalization, language tokenization, action denormalization.
4. `convert_checkpoint.py`: an existing TurboVLA checkpoint becomes `config.json`, `model.safetensors` and the processor files, through `save_pretrained`.
5. After `uv sync`, check the installed name: `python -c "import importlib.metadata as m; print(m.metadata('lerobot_policy_turbovla')['Name'])"`, and that `turbovla` appears in `PreTrainedConfig.get_known_choices()`.

## Verification

- Parity test: for 20 recorded NexArm frames, the action from `TurboVLAPolicy` plus its processors matches the `rollout_turbovla.py` code path within `1e-4` in fp32. Both paths run in `eval()` mode (the action head is a deterministic ACT-style decoder; dropout must be off). If a sampled head such as the pi05 path is used, fix the noise seed in both paths first.
- `lerobot-eval --policy.path=<converted> --env.type=nexarm --env.task=stack_bowls --eval.n_episodes=50` on Linux: success rate within noise of the current `rollout_turbovla.py` sim numbers.
- Real arm: `lerobot-rollout --policy.path=<converted>`, 10 episodes; plus one eval session started from the desktop app.

## Risk / Rollback

A silent normalization mismatch is the main risk, and the parity test is the gate. The old scripts stay until phase 06 passes.

## Execution notes (2026-10-24)

- Package `policies/lerobot_policy_turbovla/` (deps `lerobot`, `turbovla`), a workspace member pulled only by the `turbovla` extra (`turbovla = ["lerobot[transformers-dep]", "turbovla", "lerobot_policy_turbovla"]`). Plain `uv sync --locked --extra test` (checked in a scratch venv) installs neither package, `register_third_party_plugins()` still works, and `pytest policies` skips the TurboVLA tests (14 passed, 2 skipped).
- `TurboVLAConfig` defaults reproduce `rollout_turbovla.py`: chunk 16, `n_action_steps` 8 (`--open-loop-steps`), BERT-base and DINOv3 ViT-B, image size 224, min-max state/action, gripper snapping at 1800 raw. `camera_keys` keeps the trained camera order. `TurboVLAPolicy.from_pretrained` loads strictly, as the rollout script does. `forward` raises `NotImplementedError` (training is the next phase).
- Processors: rename, batch, LeRobot `NormalizerProcessorStep` (same min-max formula as the script), `turbovla_image_processor` (float frames back to uint8 by rounding, then the backbone's `AutoImageProcessor` with the square resize), device; then `UnnormalizerProcessorStep`, `turbovla_gripper_snap`, cpu. The task stays text: TurboVLA tokenizes inside the model.
- `convert_checkpoint.py` (`python -m lerobot_policy_turbovla.convert_checkpoint`) reads the checkpoint's `config.json` and `stats_turbovla.json` and writes `config.json`, `model.safetensors` and the processor files; checked on the phase-04 smoke checkpoint.
- `rollout_turbovla.py` gained `--vision-path` (the vision backbone was hard-coded to DINOv3), needed for checkpoints trained with another `--dinov3-path`.
- Parity (`tests/test_parity.py`): a seeded random TurboVLA with BERT-tiny + DINOv2-small-with-registers, saved by `train_turbovla.save_checkpoint` with stats from the real dataset, run through the real `TurboVLAPolicyRunner` and through convert, `TurboVLAPolicy.from_pretrained` and `make_pre_post_processors(pretrained_path=...)`, on 20 frames of `thanhkt/nexarm_stack_bowls` episode 143 (inputs prepared by LeRobot's `prepare_observation_for_inference`). Max abs action difference 0.0 (tolerance 1e-4), snapped gripper equal, first 8 `select_action` outputs equal the chunk. Also 0.0 on the phase-04 20-step trained smoke checkpoint (`TURBOVLA_PARITY_CHECKPOINT`). No real trained TurboVLA checkpoint exists on this machine.
- Linux sim eval (2026-10-24, `marinelab`, RTX 5050, `MUJOCO_GL=egl`). The only trained checkpoint on the server is `~/code/lerobot-nexarm/outputs/models/nexarm_stack_bowls_turbovla/final_ema_pytorch_model.pt` (3 cameras: front, wrist, top), converted with `convert_checkpoint`. On `NexArmStackBowls-v0` with 900 steps per episode, `lerobot-eval --eval.n_episodes=50 --seed=0` gave 0/50 successes (no stage reached in any episode, avg max reward 0.047), and `rollout_turbovla.py --robot sim --task-type stack_bowls --max-steps 900` over seeds 0-49 gave 0/50 (every episode ran to 900 steps). The two pipelines agree, but this checkpoint does not solve the sim task, so the success rate cannot separate them; the action-parity test above stays the evidence. Rerun with a checkpoint that succeeds in sim (the phase-06 full run).
- `lerobot-eval --env.type=nexarm` crashed on every NexArm task unless `--env.episode_length` was set (`wrapper OrderEnforcing has no attribute '_max_episode_steps'`). This was already broken at `pre-reorg`; the env now sets `_max_episode_steps` (commit `77ab5035`, test in `tests/envs/test_nexarm_env.py`).
- The `[trained]` cases of `test_parity.py` cannot use that checkpoint: it needs a `top` camera, which `thanhkt/nexarm_stack_bowls` does not have.
- Open: 10 real-arm `lerobot-rollout` episodes and a desktop-app eval session.
