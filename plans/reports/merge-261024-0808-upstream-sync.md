# Upstream sync merge report

Date: 2026-10-24. Branch `chore/upstream-sync`, made from `main` at `7f12ba1d59cc1d5511c37a0ab5140fe17b1fe781`. It merges `upstream/main` at `d3793b7e36522321c1c455f0dc4826db3793dceb`. The branch is not merged into `main` and not pushed, and no upstream PRs were opened.

Details for each disposition, the patches that need upstream PRs, the remaining diff and the open checks are in the phase file's "Execution notes (2026-10-24)": [`phase-02-upstream-sync-and-retire-core-edits.md`](../261024-0016-research-repo-reorganization/phase-02-upstream-sync-and-retire-core-edits.md).

## Commits

1. `chore: merge upstream lerobot main into the NexArm fork` (merge commit, `-X no-renames`)
2. `fix(nexarm): import robot types from lerobot.lerobot_types`
3. `feat(nexarm): move collection recording flags into a record wrapper`
4. `fix(nexarm): use upstream's accelerator.gradient_accumulation.steps`
5. `docs: describe the fork as tracking upstream LeRobot hardware layers`
6. `ci: rely on upstream's repository guard for benchmark jobs`
7. `fix(nexarm): satisfy upstream's lint and type checks`
8. `test(nexarm): drive the coaching fake through upstream's DAgger transition API`
9. `test(nexarm): give the eval stage-rate test resettable processors`
10. `docs(plans): record the upstream sync execution notes`

## Judgment calls the disposition table did not settle

- **Wrapper name.** `examples/nexarm/record.py` was already an argparse launcher that runs a subprocess, so the wrapper is a new file, `examples/nexarm/record_collection.py`. `record.py` and `prepare_collection.py` now launch it.
- **Extra flags moved.** `--manual_control`, `--recording_log` and `--root_repo_id` were also fork additions to `lerobot_record.py`, so they moved into the wrapper with `--merge_root` and `--rerun_save_path`.
- **Library modules stay.** `datasets/collection_merge.py` and `utils/recording_session.py` stayed as new fork files in `src/` (no upstream conflict), and the wrapper imports them. They move in phase 08.
- **`robot_type`.** `robot_type=robot.robot_type` was kept in `lerobot_record.py` and `rollout/context.py`, so sim datasets keep `robot_type="nexarm_follower"`. It does not change upstream robots. It is a PR candidate.
- **`--rerun_save_path` in rollout and the viewer.** The flag was also kept in `lerobot-rollout` and the visualization helpers, because the wrapper reuses the visualization part. It is a PR candidate.
- **Small generic patches kept.** These are PR candidates:
  - the Windows junction fallback in `train_utils`
  - `--video-backend` in `lerobot_dataset_viz`
  - `manual_control` in `keyboard_input`
- **Fork edits dropped:**
  - the `async_inference` changes (upstream deleted the module)
  - the fork's `rl/robot_kinematic_processor.py` copy
  - the gamepad/keyboard registration imports
  - the `find_port --help` change
  - the OpenCV fps-rounding and typing tweaks
- **Gradient accumulation flag.** Upstream names the flag `--accelerator.gradient_accumulation.steps`. The fork's YAML, docs and test were moved to it.
- **Workflows.** The fork's jobs were kept with upstream's pinned action SHAs (release, docker publish and the rest). In `benchmark_tests.yml`, the fork's `if: ${{ false }}` was removed because upstream's repository guard already disables those jobs in the fork.
- **Lint and type config.** Upstream turned on mypy over all of `src/lerobot` and pydocstyle with per-directory opt-outs. The type errors in NexArm code were fixed rather than silenced. `nexarm_service/**`, `sim/**` and `plans/**` were added to the pydocstyle opt-out list, the same way upstream lists its own unconverted modules. One real bug turned up: `lerobot_nexarm_sim_benchmark.py` imported `SyncInferenceConfig` from `lerobot.rollout`, but upstream moved it to `lerobot.inference`.
- **Tests adapted to upstream API changes, with no assertion changes:**
  - the coaching fake: `_apply_transition` is now an instance method that also takes the teleop;
  - the eval stage-rate test: `eval_policy` now resets the processors.
- **Extras that do not install on Windows.** `--extra all` fails on Windows for `kinematics` and `hilserl`: the `cmeel-*` sdists under `placo` -> `pin` do not build here. This already happens on `main`. The environment was synced with every other extra.
- **Plan notes in the merge commit.** The merge commit also contains "Scouted facts" additions to the phase-04 and phase-07 plan files. They were uncommitted changes in the working tree when the merge was committed. They were left in place, not reverted.

## Test counts (Windows Server 2022, CPU)

| Suite | Before (baseline) | After |
|---|---|---|
| `pytest tests -k nexarm --ignore=tests/utils/test_process.py` | 338 passed, 10 failed, 28 skipped | 337 passed, 11 failed, 5 skipped. The eleventh failure was then fixed and passes on its own. The 10 that remain are the baseline's OpenGL/CRLF failures. Test IDs are identical to `main`'s. |
| `pytest tests/nexarm_service` | 128 passed | 128 passed |
| Wrapper and re-applied patches | n/a | 331 passed, 1 skipped |
| Broader suite | n/a | BROAD_RESULT |
