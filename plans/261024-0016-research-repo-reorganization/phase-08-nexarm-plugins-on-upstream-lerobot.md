---
phase: 8
title: "NexArm plugins on upstream LeRobot"
status: pending
priority: P2
effort: "3-5d"
dependencies: [2, 5]
---

# Phase 08: NexArm plugins on upstream LeRobot

## Goal
Move all NexArm code out of `src/lerobot/` into plugin packages, then depend on a pinned upstream `lerobot`, so upgrading LeRobot is a version bump. This is how Trossen Robotics moved off their LeRobot fork.

## Context
- LeRobot auto-imports installed packages named `lerobot_robot_*`, `lerobot_teleoperator_*` and `lerobot_env_*`.
- Registered type names stay identical (`nexarm_follower`, `nexarm_sim`, `mobile_bi_nexarm_sim`, `nexarm_leader`, `nexarm`). Datasets, checkpoints and the desktop app reference these strings, not Python paths.
- What to move: `motors/nexarm/`, `robots/nexarm_follower/`, `robots/nexarm_sim/`, `robots/mobile_bi_nexarm_sim/`, `teleoperators/nexarm_leader/`, `envs/nexarm.py` plus the `NexArmEnv` config, `datasets/mobile_nexarm_artifact.py`, `datasets/collection_merge.py`, `utils/recording_session.py`, `scripts/lerobot_nexarm_sim_benchmark.py`, `nexarm_service/`, and their tests.
- End state (user decision): `src/lerobot/` is removed and LeRobot comes from a pinned upstream version.

## Files to Create / Modify
- Create: `packages/lerobot_robot_nexarm/` (motors and the three robots), `packages/lerobot_teleoperator_nexarm/`, `packages/lerobot_env_nexarm/` (depends on the robot package for the sim backend), `packages/nexarm_service/` (keeps the `lerobot-nexarm-service` entry point and the `[service]` dependencies)
- Move: the matching tests into each package's `tests/`
- Modify: root `pyproject.toml`: depend on `lerobot==<version>` or a git pin; drop `src/lerobot` from the build; extras `nexarm`, `service` and `turbovla` point at the workspace packages
- Delete: `src/lerobot/`
- Modify: `.github/workflows/*`, `Makefile` (`test-nexarm-ete`), `docker/`

## Tasks & Steps
1. Move one package at a time, starting with the teleoperator because it is the smallest. After each move, delete the registration imports from the core scripts and check that the CLI still finds the type.
2. Check that `nexarm_service` calls `register_third_party_plugins()` (or imports the plugins) before it resolves robot types.
3. Rewrite imports (`lerobot.robots.nexarm_follower` becomes `lerobot_robot_nexarm.nexarm_follower`, and so on) across examples, sim, tools and tests.
4. Apply the phase-02 fallback for every carried patch whose upstream PR has not merged. When `git diff upstream/<tag> -- src/lerobot` is empty, replace `src/lerobot/` with the pinned dependency.
5. Pick the upstream release or commit to pin; document the upgrade step (bump the pin, `uv lock`, run the tests).

## Verification
- `uv run lerobot-record --help` lists `nexarm_follower` and `nexarm_leader`; `lerobot-eval --env.type=nexarm` resolves.
- All moved tests pass on Linux; `make test-nexarm-ete` passes.
- Desktop app: health, sessions, teleop and recording against the real arm.
- An old dataset and an old checkpoint (recorded before this phase) load and run.

## Risk / Rollback
Import breakage in downstream scripts and the desktop app. Move in small commits, one package each, so each can be reverted.
