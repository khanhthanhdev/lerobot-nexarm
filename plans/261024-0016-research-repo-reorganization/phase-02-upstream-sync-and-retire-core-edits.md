---
phase: 2
title: "Upstream sync and retire core edits"
status: in-progress
priority: P1
effort: "2-3d"
dependencies: [1]
---

# Phase 02: Upstream sync and retire core edits

## Goal

Merge current upstream LeRobot and shrink this fork's edits to upstream files down to what NexArm still needs before phase 08. New policy code is then written against the latest LeRobot API.

## Context

Measured against fork point `3e538352`: upstream has 332 newer commits touching 488 files. Outside the NexArm folders, this fork modifies 23 upstream files, deletes 2 and adds 5. The user does not intend to keep custom changes to train or record.

### What each edit is, and what happens to it

| Edit                                                                                              | Where                                                                                               | Disposition                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Registration imports of `nexarm_*` robots/teleops                                                 | `lerobot_record/rollout/teleoperate/calibrate/replay/find_*`, `envs/__init__.py`, `envs/configs.py` | Keep for now; they disappear in phase 08 when the NexArm code becomes plugins                                                                                                                                                            |
| Pruned robots, motors, `lerobot_setup_can.py`, `lerobot_setup_motors.py`                          | deletions                                                                                           | Accept upstream's files (user decision): stop pruning                                                                                                                                                                                    |
| `gradient_accumulation_steps`                                                                     | `configs/train.py`, `lerobot_train.py`                                                              | Take upstream's: upstream `d3793b7e` already has it                                                                                                                                                                                      |
| Per-stage success rates (`info["stage/<name>"]`)                                                  | `lerobot_eval.py`                                                                                   | Generic. Keep and open an upstream PR. Fallback if not merged when phase 08 starts: compute stage rates from the per-episode `eval_info.json` in a small NexArm tool, and drop the patch                                                 |
| `local_files_only` (no silent Hub fallback for incomplete local datasets)                         | `datasets/dataset_metadata.py`, `lerobot_dataset.py`                                                | Generic. Check upstream first; otherwise keep and open an upstream PR. Fallback if not merged when phase 08 starts: check that local files are complete in the record wrapper and in `nexarm_service` before loading, and drop the patch |
| PyAV fallback when torchcodec is missing                                                          | `datasets/streaming_dataset.py`, `utils/import_utils.py`                                            | Check upstream first; keep only if still needed                                                                                                                                                                                          |
| Lazy `groot` imports                                                                              | `policies/groot/__init__.py`                                                                        | Check upstream first; keep only if still needed                                                                                                                                                                                          |
| `--merge_root`, `--rerun_save_path`, `utils/recording_session.py`, `datasets/collection_merge.py` | `lerobot_record.py`                                                                                 | Move to a NexArm record wrapper with the same flags (user decision); `lerobot_record.py` returns to upstream                                                                                                                             |
| `datasets/mobile_nexarm_artifact.py`, `scripts/lerobot_nexarm_sim_benchmark.py`, `envs/nexarm.py` | new files                                                                                           | No conflict; they move in phase 08                                                                                                                                                                                                       |

## Files to Create / Modify

- Modify: the files in the table above, by merge resolution
- Modify: `pyproject.toml` and `uv.lock` (upstream dependency changes)
- Create: a NexArm record wrapper that calls `lerobot.scripts.lerobot_record.record` and then merges, keeping the same flags. It lives in `examples/nexarm/record.py` for now and moves in phase 08/09.
- Modify: `CLAUDE.md` "Scope of the fork": hardware layers are no longer pruned

## Tasks & Steps

1. `git switch -c chore/upstream-sync && git merge upstream/main`.
2. Resolve conflicts using the table. Where an edit is now upstream, take upstream's version.
3. Move the record extras into the wrapper. Keep the flag names and behaviour so `DATA_COLLECTION_GUIDE.md` commands only change the command name.
4. `uv lock`, then `uv sync --locked --extra all --extra test --extra dev`.
5. Open upstream PRs for the generic fixes (stage rates, `local_files_only`, others still needed). Record the PR links in this phase file. Any PR not merged when phase 08 starts uses the fallback in the table, so phase 08 never waits on upstream review.
6. Write down the remaining diff against upstream: `git diff upstream/main -- src/lerobot ':!src/lerobot/**/nexarm*'`. It should only contain the registration imports and any patches waiting on PRs.

## Verification

- Full test suite and NexArm tests on Linux are at least as good as the phase-01 baseline.
- `make test-nexarm-ete` passes.
- Sim dataset generation smoke test: `examples/nexarm/generate_stack_bowls_dataset.py` with 2 episodes.
- Real-arm smoke test before merging to `main`: teleoperate, then record 2 episodes (with merge into a collection), then replay 1 episode.
- `uv run pre-commit run --all-files` is clean.

## Risk / Rollback

High conflict count. The work stays on a branch until the real-arm smoke test passes. Roll back with `git reset --hard pre-reorg` on the branch.

## Execution notes (2026-10-24)

Branch `chore/upstream-sync`, made from `main` `7f12ba1d59cc1d5511c37a0ab5140fe17b1fe781`. Merged `upstream/main` `d3793b7e3` with `git merge -X no-renames`, because the fork's renames hid the pruned files from rename detection. Not merged into `main` and not pushed. No upstream PRs were opened (not authorized).

### Disposition applied per row

| Edit                                                                     | What was done                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Registration imports                                                     | Each script was rebuilt from upstream's file plus the `nexarm_*` names in its `from lerobot.robots/teleoperators import (...)` block: `lerobot_record/rollout/teleoperate/calibrate/replay/find_joint_limits`. NexArm factory branches were added to `robots/utils.py`, `teleoperators/utils.py` and `rl/*`, and `NexArmMotorsBus` to `motors/__init__.py`. The fork's gamepad/keyboard registration additions were dropped.                                                                                                                                                                               |
| Pruned robots, motors, `lerobot_setup_can.py`, `lerobot_setup_motors.py` | Upstream's files were restored (84 files, plus 5 that the fork had renamed away). Their scripts and the robot extras were put back in `pyproject.toml`. The fork's `rl/robot_kinematic_processor.py` copy was dropped; upstream keeps it in `robots/so_follower/`.                                                                                                                                                                                                                                                                                                                                         |
| `gradient_accumulation_steps`                                            | Took upstream's. Upstream names the flag `--accelerator.gradient_accumulation.steps`, so `stack_bowls_act.yaml`, `AGENT_GUIDE.md`, `docs/source/nexarm/sim_training.md`, `examples/nexarm/stack_bowls_act.md` and the gym integration test were updated.                                                                                                                                                                                                                                                                                                                                                   |
| Per-stage success rates                                                  | Kept in `lerobot_eval.py` (43 lines). Needs an upstream PR.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `local_files_only`                                                       | Upstream does not have it, so it was re-applied to upstream's `dataset_metadata.py` and `lerobot_dataset.py` (16 lines each). Needs an upstream PR.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| PyAV fallback                                                            | Upstream already has it. Took upstream's `streaming_dataset.py` and `import_utils.py`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Lazy `groot` imports                                                     | Still needed: `tests/policies/test_optional_policy_imports.py` fails with upstream's eager imports. Kept (25 lines). Needs an upstream PR.                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `--merge_root`, `--rerun_save_path`                                      | `lerobot_record.py` is now upstream's file plus the registration imports and `robot_type=robot.robot_type` (see below). The collection flags moved into `examples/nexarm/record_collection.py`, which keeps the same flag names. `examples/nexarm/record.py` was already an argparse launcher, so the wrapper got its own name. `record.py` and `prepare_collection.py` now launch the wrapper. `manual_control`, `recording_log` and `root_repo_id` moved with them. `datasets/collection_merge.py` and `utils/recording_session.py` are new fork files with no upstream conflict; they move in phase 08. |
| New fork files                                                           | No conflict. Three imports were fixed for upstream's rename of `lerobot.types` to `lerobot.lerobot_types`, and `lerobot_nexarm_sim_benchmark.py` now imports `SyncInferenceConfig` from `lerobot.inference`.                                                                                                                                                                                                                                                                                                                                                                                               |

### Patches kept that would need upstream PRs (none opened)

- `scripts/lerobot_eval.py`: per-stage success rates (`info["stage/<name>"]`, `pc_<stage>`).
- `datasets/dataset_metadata.py`, `datasets/lerobot_dataset.py`: `local_files_only`.
- `policies/groot/__init__.py`: lazy imports.
- `scripts/lerobot_record.py`, `rollout/context.py`: `robot_type=robot.robot_type` instead of `robot.name`. The upstream base class sets `robot_type = name`, so upstream robots are unaffected, and sim datasets get `nexarm_follower`.
- `rollout/configs.py`, `scripts/lerobot_rollout.py`, `utils/visualization_utils.py`, `utils/rerun_visualization.py`: `--rerun_save_path` (a `.rrd` file via `rr.save`).
- `common/train_utils.py`: a directory-junction fallback for the `last` checkpoint link on Windows without symlink rights.
- `scripts/lerobot_dataset_viz.py`: `--video-backend`.
- `utils/keyboard_input.py`: `init_keyboard_listener(manual_control=True)` (quiet events, debounce, Enter = next), used by the collection wrapper.

The fork's other edits are no longer applied: the `async_inference` changes (upstream deleted the module), the `find_port --help` change, and the OpenCV fps-rounding and typing tweaks.

### Remaining diff against upstream

`git diff upstream/main --stat -- src/lerobot ':!src/lerobot/**/*nexarm*'` shows 61 files, +5701/-25. Most of that is fork-owned new code that moves in phase 08:

- `nexarm_service/**`
- `datasets/collection_merge.py` (+225)
- `utils/recording_session.py` (+73)
- the NexArm env config in `envs/configs.py` (+106) and `envs/__init__.py`

Without those files, 25 upstream files differ by +233/-25. That diff is only the registration imports and the patches listed above.

### Other changes the merge required

- CI: upstream's `benchmark_tests.yml` already guards every job with `github.repository == 'huggingface/lerobot'`, so the fork's added `if: ${{ false }}` was a duplicate key and was removed. In the other workflows the fork's jobs were kept with upstream's pinned action SHAs.
- Lint and types: upstream's mypy now covers all of `src/lerobot`, and pydocstyle is enabled with a per-directory opt-out. The type errors in NexArm modules were fixed. `src/lerobot/nexarm_service/**`, `sim/**` and `plans/**` were added to the pydocstyle opt-out list, in the same way upstream lists its own unconverted modules.
- Tests adapted to upstream API changes (no assertions changed):
  - the coaching fake calls `DAggerStrategy._apply_transition`, which upstream made an instance method that also takes the teleop;
  - the eval stage-rate test gives the stand-in processors `reset()`, because `eval_policy` now resets them.
- `uv lock`: rerun-sdk 0.33.1 -> 0.36.3; `xacro` added to the test extra.

### Verification (Windows Server 2022, CPU)

| Check                                                                                                                                                                   | Baseline                          | After                                                                                                                                                                                                                                                                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pytest tests -k nexarm --ignore=tests/utils/test_process.py`                                                                                                           | 338 passed, 10 failed, 28 skipped | 337 passed, 11 failed, 5 skipped. The extra failure (the eval stage-rate test) was fixed afterwards and passes on its own, which leaves the same 10 environment failures (MuJoCo has no OpenGL context here). The collected test IDs are identical to `main`'s (349). Fewer skips because more extras are installed. |
| `pytest tests/nexarm_service`                                                                                                                                           | 128 passed                        | 128 passed (118 plus 10 coaching tests after the fake was adapted)                                                                                                                                                                                                                                                   |
| Wrapper and re-applied patches (`tests/scripts`, dataset metadata/dataset, optional policy imports, train utils, visualization, keyboard input, rollout, control robot) | n/a                               | 331 passed, 1 skipped                                                                                                                                                                                                                                                                                                |
| Broader suite                                                                                                                                                           | not recorded                      | BROAD_RESULT                                                                                                                                                                                                                                                                                                         |
| `pre-commit run --all-files`                                                                                                                                            | n/a                               | Hooks fail only on things the merge did not cause: Prettier on Markdown files this branch does not touch, and ruff/ruff-format on `examples/nexarm/export_episode_video.py`; both hook revisions and both file sets are unchanged from `main`. mypy also reports 8 `termios` attr-defined errors in upstream's `utils/keyboard_input.py`, which happen only when mypy runs on Windows. yaml, zizmor, typos, bandit and the rest pass.                                                                                                                                                                                                                                                                                                     |
| `uv sync --locked --extra all --extra test --extra dev`                                                                                                                 | fails on `main` too               | fails: `kinematics` and `hilserl` pull `placo` -> `pin` -> `cmeel-*` sdists, which do not build on Windows (CMake 4 policy error; `cmeel-tinyxml2` fails even with `CMAKE_POLICY_VERSION_MINIMUM=3.5`). Synced with every other `all` extra plus `test`, `dev` and `service`.                                        |
| Sim smoke test, `generate_stack_bowls_dataset.py --episodes 2`                                                                                                          | n/a                               | fails only on OpenGL: `mujoco.FatalError: gladLoadGL error` at the first camera render. Needs a box with EGL/OSMesa.                                                                                                                                                                                                 |

### Open checks

- Run on the Linux GPU server: `uv run pytest tests -k nexarm -q`, `uv run pytest tests/nexarm_service -q`, the full suite, and `make test-nexarm-ete`.
- Sim dataset smoke test (2 episodes of `examples/nexarm/generate_stack_bowls_dataset.py`) on a machine with OpenGL.
- `uv sync --locked --extra all` on Linux, where the `cmeel-*` wheels exist.
- Real-arm smoke test before merging to `main`: teleoperate, record 2 episodes with `examples/nexarm/record_collection.py --merge_root ...`, then replay 1 episode.
- Decide whether to open upstream PRs for the patches above. Apply the table's fallbacks if they are not merged when phase 08 starts.
