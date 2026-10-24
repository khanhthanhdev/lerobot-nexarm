---
phase: 9
title: "Tools/examples split and docs sweep"
status: pending
priority: P2
effort: "1d"
dependencies: [6, 7, 8]
---

# Phase 09: Tools/examples split and docs sweep

## Goal
Separate tutorials from operational tools and experiment configs, and make every doc and script point at the new paths.

## Context
About 200 references to `examples/nexarm/` exist across `run.md` (22), `docs/source/nexarm/*.md`, `sim/README.md`, `README.md`, `SERVER_TRAINING_GUIDE.md`, `AGENT_GUIDE.md`, `DATA_COLLECTION_GUIDE.md`, `docs/8_Imitation_Learning_Tutorial.md`, `scripts/nexarm/*.sh`, the example scripts' own docstrings, and source messages in `motors/nexarm/mujoco_mapping.py`, `robots/nexarm_sim/calibration.py` and `envs/configs.py`.

## Files to Create / Modify
- Move to `tools/nexarm/`: `calibrate_arms.py`, `calibrate_camera_alignment.py`, `calibrate_sim.py`, `probe_joint_mapping.py`, `audit_real_dataset.py`, `servo_step_test.py`, `realsense_fovy.py`, `benchmark_sim.py`, `merge_collection.py`, `merge_hub_datasets.py`, `prepare_collection.py`, `upload_dataset.py`, `export_episode_video.py`, `overlay_recorded_frames.py`
- Move to `experiments/`: `train.yaml`, `stack_bowls_act.yaml` and `stack_bowls_act.md` (as an experiment README), `inference.yaml`
- Keep in `examples/nexarm/`: teleoperate, record, rollout, simulate, pick_place_sim, stack_bowls_sim, demo_gym_env, the sim dataset generators, vision_grasp, visualize_rerun_sim
- Modify: every file listed in Context; `CLAUDE.md` (module table: add `nexarm_service`, `firmware/`, `hardware/`, `sim/mobile_nexarm/`, `policies/`, `packages/`, `experiments/`, `tools/`; replace the submodule section; update "Scope of the fork")

## Tasks & Steps
1. `git mv` the files (one commit for the moves only, so history follows them).
2. Update references with a scripted search-and-replace, then read every changed doc section, because commands sometimes need new arguments, not just a new path.
3. Update `CLAUDE.md` and `AGENT_GUIDE.md` last, from the final tree.

## Verification
- `git grep -n "examples/nexarm/<moved file>"` returns nothing for every moved file.
- Each moved tool runs `--help`.
- Docs build: follow `docs-requirements.txt` and the existing docs workflow.

## Risk / Rollback
Low. Moves and text only.
