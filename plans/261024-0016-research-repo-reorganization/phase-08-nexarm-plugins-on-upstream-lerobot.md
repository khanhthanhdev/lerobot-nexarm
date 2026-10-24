---
phase: 8
title: "NexArm plugins on upstream LeRobot"
status: in-progress
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

## Scope decision (user, 2026-10-24)

The upstream merge kept more core patches than this phase planned fallbacks for: robot_type stamping in record/rollout, `--rerun_save_path` in rollout and the viz helpers (including the `.rrd` sink fix), the Windows `last`-checkpoint junction fallback, `--video-backend` in the dataset viewer, `manual_control` in `keyboard_input`, lazy `groot` imports, per-stage eval success rates and `local_files_only`. No upstream PRs are open.

Decision: do steps 1-3 now. Move the NexArm code into the `packages/` plugins and remove the registration imports from the core scripts. `src/lerobot/` stays, carrying only those patches. Removing it (step 4, and the acceptance criterion "`src/lerobot/` no longer exists") waits until each patch has a merged upstream PR or a NexArm-side fallback.

## Execution notes (2026-10-24)

Steps 1-3 done on `chore/upstream-sync`, one commit per package, not pushed: `3cc39d267` (teleoperator), `eb5db6c8d` (robot), `902cec292` (env), `4749ca05a` (service), `3d3c2b8af` (mypy, CI path filters, Docker, AGENTS.md and README layout). Report: [nexarm-plugins-261024-1408-phase-08.md](../reports/nexarm-plugins-261024-1408-phase-08.md).

### Module map

| Old                                                                | New                                                                                  |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `lerobot.teleoperators.nexarm_leader`                              | `lerobot_teleoperator_nexarm`                                                        |
| `lerobot.motors.nexarm`                                            | `lerobot_robot_nexarm.motors`                                                        |
| `lerobot.robots.{nexarm_follower,nexarm_sim,mobile_bi_nexarm_sim}` | `lerobot_robot_nexarm.{nexarm_follower,nexarm_sim,mobile_bi_nexarm_sim}`             |
| `lerobot.datasets.{collection_merge,mobile_nexarm_artifact}`       | `lerobot_robot_nexarm.{collection_merge,mobile_nexarm_artifact}`                     |
| `lerobot.utils.recording_session`                                  | `lerobot_robot_nexarm.recording_session`                                             |
| `lerobot.envs.nexarm`, `lerobot.envs.configs.NexArmEnv`            | `lerobot_env_nexarm.nexarm`, `lerobot_env_nexarm.configs.NexArmEnv`                  |
| `lerobot.scripts.lerobot_nexarm_sim_benchmark`                     | `lerobot_env_nexarm.sim_benchmark` (entry point `lerobot-nexarm-sim-benchmark` kept) |
| `lerobot.nexarm_service`                                           | `nexarm_service` (entry point `lerobot-nexarm-service` kept)                         |

Each package lives in `packages/<name>/src/<name>/`. The teleoperator, env and service packages depend on the robot package, and the service also depends on the teleoperator. The TurboVLA plugin now depends on `lerobot_robot_nexarm` for the gripper constants. Plain `uv sync` installs the robot, teleoperator and env plugins through a new default dependency group, `nexarm`. The `nexarm` extra points at the same three packages, and `service` points at `nexarm_service`.

### Decisions

- Type names are unchanged. A dataset's `info.json` stores `robot_type: "nexarm_follower"` and a checkpoint's `train_config.json` stores `env.type: "nexarm"`. Neither stores a Python path, so no shim was left in `src/lerobot`.
- The only core changes are deletions:
  - the `nexarm_*` names in the import blocks of the record, rollout, teleoperate, calibrate, replay and find-joint-limits scripts and of `rl/{actor,learner,gym_manipulator,eval_policy}.py`;
  - the NexArm branches in `robots/utils.py` and `teleoperators/utils.py`;
  - `NexArmMotorsBus` in `motors/__init__.py`;
  - `NexArmEnv` in `envs/configs.py` and `envs/__init__.py`, which both match upstream again.

  The factories now find the NexArm classes through upstream's `make_device_from_device_class` fallback. This was checked for all four device types.

- Tests stay in `tests/` at their old paths, as one rule for all of them. Most of them import `tests.mocks`, the `tests.nexarm_service` helpers or `examples.nexarm.*`, or reach repo assets via `parents[2]`. `uv run pytest tests -k nexarm`, `make test-nexarm-ete` and `mobile_nexarm_sim.yml` keep working as they are, so the Makefile needed no edit.
- The `nexarm_service` CLI now calls `register_third_party_plugins()` before it starts the server. Rollout sessions read a checkpoint's policy type with `PreTrainedConfig.from_pretrained`, which only works for plugin policies (`turbovla`, `bspline`) once they are registered. `hardware_factory.py` imports the robot and teleoperator classes directly. Of the examples that resolve types from the command line, `record_collection.py` already calls it; the other examples import the classes or run the `lerobot-*` CLIs.
- Three files found the checkout root with `Path(__file__).parents[4]` and now use `parents[5]`: `nexarm_sim/mujoco_backend.py`, `mobile_bi_nexarm_sim/contract.py` and `motors/kinematics_dynamics.py`.
- No package data had to move: the NexArm JSON/XML assets are in `sim/`, not in `src/lerobot/envs`.
- Tooling:
  - `packages/**` is in the pydocstyle opt-out, and the four package names are isort first-party.
  - mypy `files` and the hook's `files:` pattern include `packages/*/src`.
  - interrogate's `paths` includes `packages` (53.1%, gate 52).
  - `fast_tests.yml` and `full_tests.yml` also run when `packages/**` changes.
  - Both Dockerfiles copy `policies/` and `packages/` before the dependency layer and use `--no-install-workspace`, because `uv sync --locked` needs the members' `pyproject.toml`. This was already broken for `policies/`.

### Verification (Windows Server 2022, CPU)

| Check                                                              | Before                                                                   | After                                                                                                                                                                     |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pytest tests -k nexarm -q --ignore=tests/utils/test_process.py`   | 338 passed, 10 failed, 5 skipped                                         | 338 passed, 10 failed, 5 skipped. The junit files of the two runs list the same 353 test IDs with the same outcomes                                                       |
| `pytest tests/nexarm_service -q`                                   | 128 passed                                                               | 128 passed                                                                                                                                                                |
| `pytest policies -q`                                               | 28 passed, 2 skipped                                                     | 28 passed, 2 skipped                                                                                                                                                      |
| `lerobot-record --help`                                            | lists the NexArm types                                                   | lists `nexarm_follower`, `nexarm_sim`, `mobile_bi_nexarm_sim` and `nexarm_leader`                                                                                         |
| `lerobot-eval --env.type=nexarm --help`                            | resolves                                                                 | resolves (`NexArmEnv` from `lerobot_env_nexarm.configs`)                                                                                                                  |
| `lerobot-nexarm-service --help`, `--dump-openapi`, `--dump-events` | n/a                                                                      | works. Both dumps match `docs/service/*.json` apart from line endings                                                                                                     |
| Old dataset and checkpoint                                         | a 2-step ACT checkpoint trained before the move with `--env.type=nexarm` | the `thanhkt/nexarm_stack_bowls` metadata and a frame load. The checkpoint's `train_config.json` parses, and `select_action` returns the same 6 values as before the move |
| `pre-commit run --files <changed>`                                 | n/a                                                                      | clean, except the 8 known Windows `termios` mypy errors in `utils/keyboard_input.py`                                                                                      |

The 10 failures are the baseline's OpenGL and CRLF failures, with the same causes as before (9 `gladLoadGL`, 1 `build_isaac_assets.py` check), and no import error. Rewritten test files that `-k nexarm` does not select (`test_collection_merge`, `test_record_collection`, `test_recording_session`, `test_generate_stack_bowls_dataset`, `test_rollout_turbovla`, `test_train_stack_bowls_act`, `test_sim_exports`) give 66 passed and 4 failed. The 4 are the Windows-only failures phase 02 recorded (`test_recording_session` manual keys, `test_sim_exports` rendering). `pytest tests packages policies --collect-only` collects 5637 tests with no errors. In one intermediate full run, `tests/nexarm_service/test_coaching.py::test_reset_holds_the_follower_at_rest_until_the_next_handover` failed. It passed 3 times out of 3 on its own and passed in the final full run.

### What remains for removing `src/lerobot/`

`git diff upstream/main -- src/lerobot` is now 13 files, +190/-23. They hold no NexArm code, only these generic patches:

- `scripts/lerobot_record.py`, `rollout/context.py`: `robot_type=robot.robot_type`.
- `rollout/configs.py`, `scripts/lerobot_rollout.py`, `utils/visualization_utils.py`, `utils/rerun_visualization.py`: `--rerun_save_path` and the `.rrd` sink.
- `common/train_utils.py`: the Windows junction fallback for `last`.
- `scripts/lerobot_dataset_viz.py`: `--video-backend`.
- `utils/keyboard_input.py`: `manual_control`.
- `policies/groot/__init__.py`: lazy imports.
- `scripts/lerobot_eval.py`: per-stage success rates.
- `datasets/dataset_metadata.py`, `datasets/lerobot_dataset.py`: `local_files_only`.

Each patch needs a merged upstream PR or a NexArm-side fallback first.

Route (user left the choice to the maintainer, 2026-10-24): NexArm-side fallbacks by default, so the removal never waits on upstream review. That matches the phase-02 decision to open no upstream PRs. A patch goes upstream only when it is a plain upstream bug fix that no NexArm code can work around cleanly, and its fallback still lands first.

| Patch                                               | Fallback                                                                                                                                                                                                                                                                                                                                                |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `robot_type=robot.robot_type` in record and rollout | Upstream's `Robot.__init__` sets `robot_type = name` and then stamps `robot.name`, so `NexArmSim` (which sets `robot_type = "nexarm_follower"`) records `nexarm_sim`. The NexArm wrappers (`examples/nexarm/record_collection.py`, `rollout.py`) correct `meta.info["robot_type"]` after the dataset is created. Also an upstream bug-fix PR candidate. |
| `--rerun_save_path` and the `.rrd` sink             | A NexArm helper in `lerobot_robot_nexarm` that tees the Rerun stream to the viewer and the file. The wrappers call it, as `record_collection.py` already does by wrapping `lerobot_record.init_visualization`; `examples/nexarm/rollout.py` does the same for `lerobot_rollout`.                                                                        |
| Windows junction for `checkpoints/last`             | Drop. Training runs on Linux servers; on Windows, Developer Mode allows symlinks.                                                                                                                                                                                                                                                                       |
| `--video-backend` in `lerobot-dataset-viz`          | Drop. Document `LeRobotDataset(..., video_backend="pyav")` for the Windows TorchCodec case.                                                                                                                                                                                                                                                             |
| `manual_control` in `keyboard_input`                | Move into `lerobot_robot_nexarm.recording_session`: its own listener built from upstream's `create_key_listener`, `TerminalKeyListener` and `apply_recording_control`, with the debounce, the Enter key, the re-record gate and quiet output. `record_collection.py` calls it.                                                                          |
| Lazy `groot` imports                                | Drop once the pinned upstream imports `lerobot.policies.factory` without GR00T's extra dependencies. Check that before the pin.                                                                                                                                                                                                                         |
| Per-stage success rates in `lerobot-eval`           | `lerobot-nexarm-sim-benchmark` (in `lerobot_env_nexarm`) reports the `stage/*` flags that `NexArmEnv` already puts in `info`. Plain `lerobot-eval` then reports `pc_success` only.                                                                                                                                                                      |
| `local_files_only` in `LeRobotDataset` / metadata   | A NexArm helper that checks `meta/` and the episode files under `root` before building `LeRobotDataset` with `HF_HUB_OFFLINE=1`. `train_stack_bowls_act.py` and the TurboVLA plugin tests use it.                                                                                                                                                       |

### Fallbacks done (2026-10-24)

`git diff upstream/main -- src/lerobot` is empty. One commit per patch, each restoring its upstream files:

| Patch                 | What replaced it                                                                                                                                                                                                                                                                                                      |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| junction, viz backend | Dropped. `sim/README.md` shows a one-line PyAV `visualize_dataset` call for the Windows TorchCodec case.                                                                                                                                                                                                              |
| lazy `groot` imports  | Dropped with `tests/policies/test_optional_policy_imports.py`. Upstream's `groot/__init__.py` imports `cv2`, which the test blocked, but `opencv-python-headless` is a base dependency, so a normal install always has it.                                                                                            |
| `local_files_only`    | `lerobot_robot_nexarm.local_dataset` checks `meta/` and the selected episodes' data and video files under `root`, then builds the upstream classes. No `HF_HUB_OFFLINE`: with a `root`, LeRobot only contacts the Hub when one of those files is missing. Tests in `tests/datasets/test_local_dataset.py`.            |
| `--rerun_save_path`   | `lerobot_robot_nexarm.rerun_recording` calls upstream `init_rerun`, then sets a gRPC and a file sink. `record_collection.py` and `teleoperate.py` use it; `lerobot-nexarm-rollout` (also behind `examples/nexarm/rollout.py`) is `lerobot-rollout` plus the flag. Plain `lerobot-rollout` no longer takes it.         |
| `manual_control`      | `recording_session.init_manual_keyboard_listener` sets the session events directly (no `quiet` flag needed). Its key test now also passes on Windows.                                                                                                                                                                 |
| `robot_type`          | Not a wrapper post-fix: `NexArmSim.name` is now `nexarm_follower`, so upstream's `robot_type=robot.name` stamps the physical type in `lerobot-record`, `lerobot-rollout` and the service. `--robot.type=nexarm_sim` still selects the sim (config registry); no code dispatches on the robot's name.                  |
| per-stage rates       | Not `lerobot-nexarm-sim-benchmark` (it only runs pick-and-place): `lerobot-nexarm-eval` is `lerobot-eval` that asks each env (via `VectorEnv.call`) to log its seeded episodes' final `stage/*` flags and adds `pc_<stage>` to `overall`, `per_group` and `per_task`. Plain `lerobot-eval` reports `pc_success` only. |

`all` no longer includes `lerobot[nexarm]`; `uv sync` installs the plugins through the default `nexarm` group.

### Pin (done 2026-10-24)

1. Done: `lerobot = { git = "https://github.com/huggingface/lerobot", rev = "d3793b7e36522321c1c455f0dc4826db3793dceb" }` (the upstream merge base `src/lerobot/` matched) in the root `[tool.uv.sources]`. The root project is the non-published `lerobot-nexarm` workspace root (`[tool.uv] package = false`); its extras forward to `lerobot[...]` under the same names. `src/`, `setup.py`, `MANIFEST.in`, the root `conftest.py`, `utils/` and `scripts/ci/` are gone.
2. Done: the members' `lerobot = { workspace = true }` sources are removed, so they resolve the root pin. `mujoco` moved into `lerobot_robot_nexarm`.
3. Changed: the NexArm tests stay in the root `tests/`. Only upstream's tests were deleted; `tests/mocks/` (three mocks) and `tests/nexarm_service/` are NexArm-owned and stay, so nothing needed to move. Policy tests stay in `policies/*/tests/`.
4. Done: "Upgrading LeRobot" in `CONTRIBUTING.md`.
5. Upstream docs, non-NexArm examples, upstream workflows and GPU/benchmark Dockerfiles are removed; CI is `quality.yml`, `tests.yml`, `security.yml` and the manual `mobile_nexarm_sim.yml`.

Consequences: `rerun-sdk` resolves to 0.33.1 (upstream's `<0.34` cap; the fork had 0.36), and the `.rrd` recording tests pass on it. The installed `lerobot` has no `py.typed`, so mypy no longer type-checks calls into LeRobot. Windows results are unchanged: 446 passed, 1 skipped, the 11 known OpenGL/asset failures.

### Open

- `lerobot-find-joint-limits` and the HIL-SERL entry points (`rl/actor.py`, `rl/learner.py`, `rl/gym_manipulator.py`, `rl/eval_policy.py`) do not call `register_third_party_plugins()`, and upstream's don't either. `--robot.type=nexarm_*` therefore no longer parses there. No NexArm doc or example uses these commands, and adding the call would be a new core patch.
- On Linux: `pytest tests policies` and `make test-nexarm-ete` on the pinned layout (the `tests.yml` workflow runs both).
- Publishing: done, `all` leaves the NexArm packages out. `--extra nexarm` and `--extra turbovla` still name packages that are not on PyPI, so they only work from this repo.
- `Dockerfile.user` builds on Linux (2026-10-24, `marinelab`, from `86d7f4b5`, image 20.2 GB). Inside it, `register_third_party_plugins()` registers `nexarm_follower`, `nexarm_sim`, `mobile_bi_nexarm_sim`, `nexarm_leader`, env `nexarm` and policy `bspline`. `turbovla` is absent as expected: the `all` extra does not include the `turbovla` or `service` extras. `Dockerfile.internal` was removed with the pin. Rebuild `Dockerfile.user` from the pinned layout on Linux.
- The desktop app against the real arm: health, sessions, teleop and recording.
