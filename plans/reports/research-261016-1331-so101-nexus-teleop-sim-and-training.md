# so101-nexus: real-leader → sim data collection, and the training environment

Source: `C:\Users\Administrator\code\so101-nexus` at `7bdb671`. Read from source (no binaries involved).

## 1. Data collection: real leader arm drives a simulated follower

The core idea: the simulator is wrapped as a **LeRobot `Robot`** (`SimSOFollower`), so the real
`SO101Leader` teleoperator talks to it exactly as it would to a physical follower.

### Pieces

| Piece | File | Role |
|---|---|---|
| Leader factory | `src/so101_nexus/teleop/leader.py:113` | Stock LeRobot `SO101Leader(use_degrees=True)` over serial. Real leader calibration converts ticks → degrees (body) and 0–100 (gripper). |
| Sim follower | `src/so101_nexus/lerobot_adapter/sim_follower.py:108` | `Robot` subclass registered as `sim_so_follower` (`sim_follower_config.py:12`). Owns a `gym.make(env_id)` env. |
| Sim camera | `lerobot_adapter/sim_camera_config.py:10` | `CameraConfig` type `sim`; `source` = obs key (`wrist_camera`, `overhead_camera`) or `"render"` (side view). |
| Unit conversion | `lerobot_adapter/normalization.py` | LeRobot units ↔ "motor ticks" ↔ sim radians, reusing LeRobot's own `FeetechMotorsBus._normalize/_unnormalize`. |
| Synthetic calibration | `lerobot_adapter/synthetic_calibration.py:27` | Sim-only `MotorCalibration` JSON: mid = 2048, drive_mode 0, range sized from MJCF `ctrlrange`; gripper 0..4095. |
| Recording loop | `teleop/recorder.py:385` | Countdown → per-step loop at fixed FPS. |
| Dataset schema | `teleop/dataset.py:701` (`build_features`), `:750` (`build_frame`) | LeRobot v3 features. |
| UI | `teleop/app.py` (Gradio walkthrough: setup → init → record → review → approve/discard → push) | Entry: `so101-nexus teleop` (`cli.py:32`). |

### Unit chain (joint-space, the default)

```
leader ticks --(real leader calibration)--> degrees / 0-100
   --(+wrist_roll offset, default -90 deg)-->            leader.py:50, recorder.py:444
   --unnormalize with SYNTHETIC follower calib--> ticks   normalization.py:64
   --(ticks - mid)/TICKS_PER_RADIAN--> sim rad            normalization.py:113
   gripper: 0-100 fraction --> [ctrl_low, ctrl_high] of sim gripper actuator
   --clip to actuator ctrlrange--> env.step(qpos_target)  normalization.py:279
```

Because the synthetic calibration centres every body joint on tick 2048 with no homing offset,
the chain collapses to roughly **leader degrees → sim radians (deg·π/180)**. The only
hand-tuned correction is the wrist-roll offset (a UI slider).

### Per-step loop (`recorder.py:439-465`)

1. `leader.get_action()` → apply wrist-roll offset.
2. `obs = follower.get_observation()` **before** stepping (so the frame is `(s_t, a_t, r_t)`).
   `observation.state` is **sim readback** converted back to LeRobot units, not the leader echo.
3. `sent = follower.send_action(action)` → `env.step`; the *clipped* action actually sent is what
   gets recorded as `action`.
4. Store reward, `reward_components`, `success`, `done`, privileged `environment_state`, and
   wrist / overhead / side images.
5. Stop on `max_steps`, user stop, truncation, or `success_hold_seconds` (0.5 s) after
   `terminated`.

Other details worth copying:

- **Episode start pose = leader pose.** Before each reset the current leader pose is read and
  passed as `env.reset(options={"init_qpos": ...})` (`recorder.py:232`, `sim_follower.py:249`,
  `mujoco/base_env.py:500`), so the sim arm doesn't jump when recording starts.
- **Control mode pinned** to `pd_joint_pos` while recording (`session.py:198`); delta labels are
  derived offline (`compute_delta_actions`, `app.py:1013`).
- **Per-episode seed** = `seed + episodes_completed`; reproducibility JSON (env id, seed, full
  initial `qpos/qvel/ctrl`, camera poses, rendering metadata) goes to
  `meta/reproducibility/episode_XXXXXX.json`, plus `meta/placement_contracts/` and
  `meta/so101_nexus_env.json` (`teleop/dataset.py:46`, `app.py:1152`).
- **Review before save**: episodes are buffered in memory; approve writes them with `add_frame` and
  `save_episode`, and discard clears the buffer (`app.py:1004`, `:1112`).
- A `LeRobotDataset` subclass squeezes the `(1,)` scalar buffers before `save_episode`
  (`teleop/dataset.py:817`).
- The single env thread (`RecordingWorker`, `recorder.py:321`) owns MuJoCo rendering and resets.

### Timing caveat

The UI FPS defaults to 30 (`app.py:1313`), and the loop runs one `env.step` per frame. Each
`env.step` advances 0.02 s of sim time (`PhysicsConfig.control_period_s`), and recording leaves
`physics` unchanged. At 30 FPS, sim time therefore runs at about 0.6× wall time while the operator
moves, and the dataset `fps`/timestamps (1/30 s) disagree with sim dt (1/50 s).
`bc_ppo_warp.py` differences with `control_dt`, so it stays consistent with sim time.

### Alternative: stock `lerobot-record` (reduced fidelity)

Because of the registry decorators, stock LeRobot can drive the sim follower
(`docs/content/docs/concepts/lerobot.mdx:63`). It is **not** equivalent to the Gradio app:

- **No wrist-roll offset.** The -90° offset exists only in `recorder.py:444` and
  `_seed_follower_from_leader`, and `SimSOFollower.send_action` doesn't apply it.
- **No leader-pose reset seeding.** `set_initial_leader_action` is called only by the Gradio
  recorder. The env resets to its rest pose, and the arm snaps to the leader pose on the first
  step (`max_relative_target` defaults to `None`).
- **Joints and cameras only.** `observation_features` (`sim_follower.py:158`) is motors plus
  cameras. `environment_state`, reward, success, done and `reward_components` never enter the
  schema, so the dataset **cannot feed `bc_ppo_warp.py`**, which needs
  `observation.environment_state`. The record-loop test only asserts joint names
  (`tests/core/test_lerobot_adapter_record_loop.py:218`).

Usage:

```bash
lerobot-record --robot.discover_packages_path=so101_nexus.lerobot_adapter \
  --robot.type=sim_so_follower --robot.env_id=MuJoCoTouch-v1 --robot.id=my_robot \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=my_leader \
  --dataset.repo_id=user/my_sim_reach --dataset.num_episodes=10 --dataset.single_task="..."
```

You can point the follower at a **real follower's calibration** so that the sim and real datasets
share identical normalized units.

### Recorded schema

`action`, `observation.state` (6, body deg + gripper 0–100),
`observation.images.{wrist,overhead[,side]}` (h264 video), `observation.environment_state`
(named privileged vector: TCP pose, object pose, grasp, …), `reward`, `success`, `done`,
`reward_components.*`, `task`. `lerobot_dataset.py:52` has `dataset_row_to_sim_qpos` /
`sim_qpos_to_dataset_row` to convert back and forth.

## 2. The training environment

### Env construction (MuJoCo backend)

- Registered in `mujoco/__init__.py:12+`: `MuJoCo{PickLift,PickReturn,PickAndPlace(-v2),StackCube}-v1`
  (1024 steps), `Touch` (512 steps), `LookAt` and `Move` (256 steps). Backend comes from the id
  prefix (`env_ids.py:9`): `MuJoCo*` CPU, `Warp*` GPU (mujoco-warp, one model with `nworld=N`),
  `MJBatch*` threaded CPU batch.
- Base class `SO101NexusMuJoCoBaseEnv` (`mujoco/base_env.py:164`). Each task env sets a
  `default_config_cls` (e.g. `PickConfig`, `config.py:1163`). An `EnvironmentConfig` dataclass
  (`config.py:911`) holds physics, render, reward, robot, spawn, `obs_mode`, observations, and
  `terminate_on_success`.
- Scene: SO101 MJCF (`assets/SO101/so101_new_calib.xml`, position actuators kp≈998, kv≈2.7)
  plus a programmatically generated object scene (`build_object_scene_xml`): cubes and primitives,
  YCB, GSO, and scanned meshes, placed with `spawn_utils.py`.
- Physics: `timestep 0.005 s`, `control_period 0.02 s` → 4 substeps, i.e. a **50 Hz** control
  rate (`config.py:829`). Truncation comes only from gym `max_episode_steps`.
- Control modes: `pd_joint_pos`, `pd_joint_delta_pos`, `pd_joint_target_delta_pos`,
  `pd_ee_pose`, `pd_ee_delta_pose` (EE modes go through IK, `_action_to_ctrl` `base_env.py:1066`).
- Observations are **composable components** (`observations.py`): state components plus
  `WristCamera` and `OverheadCamera`. The flat privileged state comes from
  `_compute_obs_components` (`base_env.py:912`).
- Domain randomization: robot colours, wrist-camera pose and FOV, side-camera
  azimuth/elevation, initial-pose noise, and object placement.
- Reward: four potential/tanh-shaped terms (reach, grasp, task objective, completion bonus) at
  0.25 weight each (`RewardConfig`, `config.py:532`). The breakdown goes in
  `info["reward_components"]` and success in `info["success"]`. Grasp detection uses bilateral
  contact with opposing normals (`base_env.py:708`).

### How the data is trained on

The documented pipeline (`docs/content/docs/workflow/overview.mdx`) is
**record → behaviour cloning → PPO with a persistent BC loss**, using the repo's own scripts
rather than `lerobot-train`:

- `examples/bc_ppo_warp.py` (recommended): `WarpPickLift-v1`, 1024 GPU worlds,
  `control_mode=pd_joint_delta_pos`.
  - `load_demo_transitions` (`:675`) reads the dataset parquet directly. Observation is the
    recorded **privileged `environment_state`**, re-laid by name to the current layout
    (`relabel_environment_state`). The action label is the **realised joint delta**
    `qpos[t+1]-qpos[t]` (from `observation.state` converted to radians), divided by the
    delta-action scale and clipped to [-1, 1] (`:753-766`).
  - The actor mean is BC-pretrained with MSE (2000 updates), then PPO runs with an added
    `bc_coef * bc_loss`. The critic is never BC-trained.
  - Evaluation runs on the CPU MuJoCo twin (`Warp`→`MuJoCo` id) with deterministic actions.
    Checkpoints store the model plus observation normalization stats.
- `examples/ppo_warp.py` and `examples/ppo.py`: pure PPO on privileged state, with no demos.
- So the policies are **state-based** (privileged-state actor). The camera streams are recorded,
  but these scripts don't consume them.

### LeRobot-facing surfaces (for VLA / IL policies)

- `envhub.py` `make_env` (+ `envhub/envs/*.py` shims) exposes each env to LeRobot EnvHub /
  `lerobot-eval`-style rollouts: keys `agent_pos` / `environment_state` / `pixels`,
  `info["final_info"]["is_success"]`, actions in **radians**.
- `processors/lerobot_env_wrapper.py` renames `state`→`observation.state` and
  `*_camera`→`observation.images.*` and converts HWC→CHW.
- `policy_adapters/recorder.py` (`RolloutRecorder`) runs a LeRobot-style policy that speaks
  degrees, converts to radians, and records the rollout back into a LeRobot dataset.
- The repo has no `lerobot-train` invocation. Training an IL policy on these datasets is left to
  stock LeRobot.

## 3. Relevance to lerobot-nexarm

The transferable pattern is to **make the sim a LeRobot `Robot` plus a `sim` camera type**, keep
recorded units equal to the real follower's units, seed the reset from the leader pose, pin
absolute joint control while recording, and record privileged state, reward and success alongside
the images. For NexArm the unit bridge is simpler: `rad = (raw - 2048)·2π/4096` already matches in
hardware and sim. The open item is the known sim joint-mapping mismatch (real Home is upright),
which has to be fixed before leader→sim teleop gives correct poses.

## Unresolved

- Bodies of the pick-and-place and stack-cube reward functions, and the Warp substep loop, were not
  read line by line.
- The repo shows no evidence of `lerobot-train` / `lerobot-eval` being run against EnvHub; that
  path is documented but untested here.
