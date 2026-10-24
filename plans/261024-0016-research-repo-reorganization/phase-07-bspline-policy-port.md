---
phase: 7
title: "B-spline Policy port"
status: in-progress
priority: P2
effort: "3-5d + GPU time"
dependencies: [3]
---

# Phase 07: B-spline Policy port

## Goal

Port B-spline Policy into `lerobot_policy_bspline` so it trains, evaluates and runs on NexArm with LeRobot tools, next to the diffusion baseline.

## Context

- `khanhthanhdev/bspline-policy` is a fork of `B-spline-policy/bspline-policy` (paper: arXiv 2607.09648, Harvard/MIT/UT Austin). It contains `bspline_policy/` (config, dataset, policy, scripts), `diffusion_policy/`, `inference/` and `real_env/`. It is built on `diffusion_policy`, not on LeRobot.
- LeRobot's diffusion policy comes from the same `diffusion_policy` lineage. LeRobot's guide asks to subclass instead of copying modeling files.
- The submodule was never initialized here; this phase removes it.

## Files to Create / Modify

- Create: `policies/lerobot_policy_bspline/pyproject.toml`, `src/lerobot_policy_bspline/{__init__,configuration_bspline,modeling_bspline,processor_bspline,bspline}.py`, `tests/`
- Create: `policies/lerobot_policy_bspline/README.md`: paper, upstream repo and pinned commit used for the port, and license
- Delete: the `bspline-policy` entry in `.gitmodules` and its gitlink
- Create: `experiments/<date>-bspline-vs-diffusion-stack-bowls/`

## Tasks & Steps

1. Read the upstream code (the `ak:xia` skill fits here) and write down exactly what B-spline Policy changes compared with diffusion policy: action representation (control points), decoder, loss, inference-time sampling, and any real-env timing logic.
2. Check the upstream LICENSE is compatible with Apache-2.0 before copying any code.
3. If the change is limited to the action representation and head, subclass LeRobot's diffusion config and policy: `BsplineConfig(DiffusionConfig)` registered as `"bspline"`, and `BsplinePolicy(DiffusionPolicy)` in `modeling_bspline.py`. Otherwise write a standalone policy following the LeRobot template.
4. Put spline fitting and evaluation in `bspline.py`, with unit tests.
5. Processor: same normalization as diffusion, plus any conversion between actions and control points.
6. Train on the NexArm stack-bowls sim dataset and evaluate with 50 episodes against `--policy.type=diffusion` using the same data and steps. If feasible, reproduce one number from the paper on a benchmark LeRobot ships.

## Verification

- Unit tests: spline encode then decode reconstructs action chunks within tolerance; forward output shapes are right; `select_action` runs on a CPU batch.
- `lerobot-train --policy.type=bspline` runs 200 steps and saves a checkpoint that `lerobot-eval` loads.
- The experiment README has the bspline vs diffusion table.
- `lerobot-rollout` runs 10 episodes on the real arm.

## Risk / Rollback

Porting fidelity: compare against at least one upstream number before trusting NexArm results.

## Scouted facts (2026-10-24)

- License: MIT (Copyright (c) 2026 Haoyu Xiong) in both `khanhthanhdev/bspline-policy` and `B-spline-policy/bspline-policy`, which is compatible with Apache-2.0. Keep the MIT notice in the plugin README and in ported files.
- The recorded submodule commit `61ed5f42` ("hello bspline policy", 2026-07-22) exists on the fork. Use it as the pinned reference for the port.

## Execution notes (2026-10-24)

Status: steps 1-5 done; step 6 done for the CPU parts. The GPU/Linux parts are open.

- Upstream read at `61ed5f42`. The change against diffusion policy is limited to the action representation (a knot column plus control points, `(chunk_size + 2 * degree, 1 + action_dim)`), one extra model channel, and returning the whole chunk. Loss, network and scheduler are unchanged. So the port subclasses: `BsplineConfig(DiffusionConfig)` registered as `"bspline"`, and `BsplinePolicy(DiffusionPolicy)`. The analysis and the deliberate differences are in `policies/lerobot_policy_bspline/README.md`. The main differences: the adaptive-knot fit runs per training window in `forward` instead of per episode in the dataset, and upstream's asynchronous wall-clock controller is not ported (only `bspline_speed_up`).
- Packaging: workspace member `policies/lerobot_policy_bspline`. The root `pyproject.toml` installs it through a default dependency group `policies` with `[tool.uv.sources] lerobot_policy_bspline = { workspace = true }`. It is not in `project.dependencies`, so the published `lerobot` wheel does not depend on an unpublished package. `uv lock` resolves the `lerobot` <-> plugin cycle (346 packages).
- The `bspline-policy` submodule, its `.gitmodules` entry, its ruff exclude and the `AGENTS.md` (`CLAUDE.md`) submodule note are removed.
- Verification on Windows CPU: plugin tests 14 passed. `lerobot-train --policy.type=bspline` ran 200 steps on `thanhkt/nexarm_stack_bowls` (episodes 143-152), with loss going from 1.06 to 0.36, and saved a checkpoint. `make_policy` reloaded it and `select_action` ran. `lerobot-eval --env.type=pusht` loaded a 200-step `lerobot/pusht` checkpoint and ran 2 episodes (NexArm sim eval needs OpenGL). `pytest tests -k nexarm` gave 338 passed / 10 failed (baseline env failures).
- Open: 50-episode bspline vs diffusion in NexArm sim and 10 real-arm episodes (`experiments/261024-bspline-vs-diffusion-stack-bowls/`, pending GPU/Linux), the upstream-number fidelity check, and porting asynchronous replanning with time alignment if real-arm speed-ups stutter.
- Report: `plans/reports/bspline-port-261024-1156-phase-07.md`.
