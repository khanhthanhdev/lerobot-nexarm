# NexArm Consistency Fixes

Status: implemented, uncommitted on the branch · Branch: `fix/nexarm-consistency` · Source: `plans/reports/review-261011-0754-cross-project-consistency.md`

## Outcome

Fix every inconsistency in the review so sim, real recording, training, rollout and docs share one set of conventions.

## User decisions (fixed)

- **D1 Ports:** leader `/dev/ttyUSB0`, follower `/dev/ttyUSB1` everywhere.
- **D2 Sim tick scale:** arm joints use the servo constant `rad = (raw - 2048) * 2π / 4096` (`raw_to_radians` in `motors/nexarm/nexarm.py`); `ctrlrange` only clamps. Existing sim datasets and calibrations become stale and must be regenerated.
- **D3 Gripper:** raw 2833 = closed, 1195 = open (`GRIPPER_CLOSED_POS` / `GRIPPER_OPEN_POS`).
- **D5 Collection root:** default `--root-repo-id` becomes `thanhkt/nexarm_stack_bowls_top` (3 cameras, created by the first session). The old 2-camera root stays usable with `--no-top-cam --root-repo-id thanhkt/nexarm_stack_bowls`.
- **D4 Top camera:** on by default everywhere (real collection, record, rollout, inference yaml, sim). Disable with `--no-top-cam`. Missing RealSense or `pyrealsense2` must give a clear error naming `--no-top-cam`.

## Lead decisions

- Shared constants live in `src/lerobot/motors/nexarm/nexarm.py` (done): `POSITION_CENTER`, `TICKS_PER_REVOLUTION`, `GRIPPER_*_POS`, `GRIPPER_MID_POS`, `raw_to_radians`, `radians_to_raw`. No new literal 1195/2833/2048 elsewhere.
- One raw↔MuJoCo converter implementation, used by both `mujoco_backend.py` and `kinematics_dynamics.py`.
- `NexArmSim.robot_type = "nexarm_follower"` (same embodiment; registry `name` stays `nexarm_sim`) so sim+real datasets merge and `sanity_check_dataset_robot_compatibility` / policy `robot_type` agree; generators write `robot.robot_type`.
- Public names/signatures of `NexArmKinematicsDynamics`, `NexArmMujocoBackend`, `NexArmSim` stay unchanged (W2 consumes them).
- Threshold _values_ (stack-bowls 2400/1600 hysteresis, rollout 1800, cartesian labels) stay; only name them and make open/closed direction agree.
- TurboVLA checkpoints without the new `config.json` fields still roll out (fallback to current heuristic and 224).
- Rollout scripts compare opened cameras with the policy's expected image inputs at startup and fail clearly.
- `NexArmSim.send_action` returns the commanded (clamped) action like `NexArmFollower`; stack-bowls labels the commanded action.
- Physics: backend sets `model.opt.timestep = control_period / steps_per_action` so physics time equals frame time exactly.
- Calibration: refuse a calibration whose fps differs from the run fps. Precedence everywhere: explicit config/CLI value > calibration > default.
- Calibration + DR options available in pick-place generator, stack-bowls generator, `NexArmEnv` config, and benchmark. DR on by default in both generators (`--no-dr` disables). Object DR covers every task object (cube, bowls), not a hard-coded `cube`. Calibration only overrides DR ranges of joints it fitted.
- TurboVLA `config.json` records `task_type`, `camera_keys` (ordered), `image_size`, `fps`, `task`; rollout and diagnose load them through one helper.
- One shared camera-argument helper (`examples/nexarm/camera_config.py`) for int index or `/dev/v4l/by-id` path, fourcc, and RealSense auto-detect; used by collection, record, teleoperate, rollout, rollout_turbovla.
- Collection checks merge compatibility with the root dataset **before** recording; first session for a new root creates it instead of failing.

## Non-goals

- Speed-cap modelling in sim (firmware behaviour unknown; see unresolved).

## Workstreams (disjoint file ownership)

| ID  | Scope                                                                                                                                                                                                                                                                                                                                                                               | Runs             |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| W1  | Sim core: `src/lerobot/robots/nexarm_sim/*`, `motors/nexarm/kinematics_dynamics.py`, `envs/nexarm.py`, `envs/configs.py` (nexarm), `scripts/lerobot_nexarm_sim_benchmark.py`, `sim/README.md`, matching tests                                                                                                                                                                       | parallel with W2 |
| W2  | Real/TurboVLA/collection: train/rollout/diagnose TurboVLA, rollout, record, teleoperate(\_cartesian), vision_grasp, calibrate_arms, calibrate_camera_alignment, prepare/merge collection, merge_hub_datasets, upload_dataset, train_stack_bowls_act, yamls, `scripts/nexarm/*`, `datasets/collection_merge.py`, `scripts/lerobot_record.py`, new `camera_config.py`, matching tests | parallel with W1 |
| W3  | Sim generators: generate_sim_dataset, generate_stack_bowls_dataset, sim_dataset_utils, calibrate_sim, stack_bowls_sim, pick_place_sim, matching tests                                                                                                                                                                                                                               | after W1         |
| W4  | Docs: run.md, README.md, AGENT_GUIDE.md, DATA_COLLECTION_GUIDE.md, docs/source/nexarm/\*, stack_bowls_act.md                                                                                                                                                                                                                                                                        | after W1–W3      |

## Acceptance

Baseline OpenGL-limited failures on this host (headless Windows, `gladLoadGL error`), expected to keep failing here: `tests/envs/test_nexarm_env.py::{test_nexarm_env_gym_checker,test_nexarm_env_obs_types[pixels_agent_pos],test_nexarm_env_obs_types[pixels],test_nexarm_env_render_rgb}`, `tests/envs/test_nexarm_sim_gym_integration.py::{test_nexarm_sim_gym_raw_position_contract,test_nexarm_train_eval_integration}`, `tests/robots/test_nexarm_sim.py::{test_backend_steps_and_renders,test_sim_robot_matches_physical_feature_contract}`, `tests/scripts/test_generate_nexarm_sim_dataset.py::test_generator_writes_dataset_report_and_episode_log`, `tests/test_sim_exports.py::test_description_scene_camera_rendering`.

- `uv run --no-sync pre-commit run --files <touched>` clean (ruff, mypy strict for envs/motors/configs, typos).

- Every review item fixed or listed as unresolved with reason.
- `uv run --no-sync pytest tests -k "nexarm or turbovla or stack_bowls or collection or recording_session"` passes except tests that need an OpenGL context (environment limit on this host).
- `ruff check` / `ruff format --check` clean on touched files.
- Every documented command's flags exist in code.

## Unresolved

- Does CMD 56 speed cap apply to CMD 97 writes? (`nexarm.py:48` vs `:275-278`)
- Does sim shoulder_lift sign/zero and top-camera pose match the real rig?

## Result (2026-10-11)

- W1–W4 implemented; independent diff review found one high and several medium/low issues, all fixed: stale-data guards (`joint_mapping` marker in calibrations and generation reports, `robot_type` check on resume), private-root detection when logged out (creating the first session root now needs `hf auth login`), interrupted sharded-resume recovery, `ImportError` handling for `pyrealsense2`, generic merge-preflight hint, env backend cleanup on calibration failure, `POSITION_CENTER` in homing code, bowl-colour doc correction.
- Tests: NexArm-related suites pass except 10 tests that need an OpenGL context. Full suite on this Windows host also fails about 58 tests in cameras, datasets, processors and policies (missing Hugging Face cache files, OpenCV camera threads, Windows temp-file permissions); these were not compared against a baseline run at HEAD, but the failing test files do not reference any changed module. Causes seen: missing Hugging Face cache files, OpenCV camera threads, Windows temp-file permissions; the setup errors in `tests/policies/test_relative_actions.py` and the `test_policies.py` failure were not inspected. `tests/utils/test_process.py` cannot be collected on Windows (`signal.SIGHUP`).
- Pre-commit clean on all changed files.
- Left as is: `prepare_collection.py` run directly defaults to camera indices 0/1 (rig by-id paths and MJPG/YUYV live in `collect.sh`); firmware speed-cap modelling (non-goal).
