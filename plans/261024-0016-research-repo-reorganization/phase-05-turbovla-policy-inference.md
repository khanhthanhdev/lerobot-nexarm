---
phase: 5
title: "TurboVLA policy plugin, inference"
status: pending
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
