# Calibration: real to sim to real

Every calibration of the NexArm is on this page: the real arms, the MuJoCo twin fitted to them, and running a policy trained in simulation back on the real arm. Other pages link here instead of repeating it. Follow the steps in order and run every command from the repository root with `uv run`.

The loop has three parts:

1. **Real arm** (Step 0): zero the servos, check that leader and follower agree, and fix the follower settings. Every later number depends on these.
2. **Real to sim** (Steps 1–8): make the simulation move, look and measure like your arm: joint mapping, latency and servo dynamics, cameras, scene.
3. **Sim to real** (Steps 9–10): generate data with the calibration, train on real and sim data, run the policy on the real arm, and go back to the step its failures point to.

**What gets calibrated**

| What                                                                                    | Tool                                                   | Result                          |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------- |
| Servo zero positions; leader/follower agreement at Home                                 | Vendor desktop app, `calibrate_arms.py`                | Zeros stored in the servos      |
| Per-joint sign and zero offset between the real servos and the sim joints               | `probe_joint_mapping.py`, `overlay_recorded_frames.py` | Joint mapping JSON              |
| Action latency and per-joint servo time constant (and `kp` with `--fit-kp`), and spread | `calibrate_sim.py`                                     | `SimCalibration` JSON           |
| Camera pose and field of view                                                           | `calibrate_camera_alignment.py`, `realsense_fovy.py`   | Edits you make to the scene XML |
| Arm mount height, cube and bowl sizes and masses, table and target zone                 | Measure by hand                                        | Edits you make to the scene XML |

**What is not calibrated:** contact physics such as cube friction and slip. Keep real contact-rich demonstrations in your training mix, since simulation widens coverage but does not replace them.

## All calibration commands

The short version, in order. Each line links to the step that explains it and how to read its output.

```bash
# Step 0: real arms at the vendor Home pose (needs --extra service)
uv run python tools/nexarm/calibrate_arms.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1

# Step 0: joint mapping (follower power-cycled, desktop app closed), then check it on a recorded episode
uv run python tools/nexarm/probe_joint_mapping.py --port /dev/ttyUSB1
MUJOCO_GL=egl uv run python tools/nexarm/overlay_recorded_frames.py --repo-id <real_repo_id> --episode 0 \
    --mapping outputs/nexarm/joint_mapping_probe.json

# Step 1: record 20 free-motion episodes
uv run python examples/nexarm/record.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 \
    --repo-id <user>/nexarm_sysid --num-episodes 20 --episode-time 20

# Step 2: audit the recording; optionally time the servos directly
uv run python tools/nexarm/audit_real_dataset.py --repo-id <sysid_repo_id> --episode-time-s 20
uv run python tools/nexarm/servo_step_test.py --port /dev/ttyUSB1 --joint elbow_flex

# Step 3: fit latency and dynamics
MUJOCO_GL=egl uv run python tools/nexarm/calibrate_sim.py --repo-id <sysid_repo_id> \
    --joint-mapping outputs/nexarm/joint_mapping_probe.json \
    --out outputs/calibration/nexarm_sim_calibration.json

# Step 5: generate calibrated sim data
MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --repo-id local/nexarm_stack_bowls_cal --root outputs/datasets/nexarm_stack_bowls_cal --episodes 200 \
    --calibration outputs/calibration/nexarm_sim_calibration.json \
    --joint-mapping outputs/nexarm/joint_mapping_probe.json

# Step 7: cameras (field of view first, then pose in the live overlay)
uv run python tools/nexarm/realsense_fovy.py --width 640 --height 480
uv run python tools/nexarm/calibrate_camera_alignment.py --camera-name front --cam-index 0
uv run python tools/nexarm/calibrate_camera_alignment.py --camera-name wrist --cam-index 1
uv run python tools/nexarm/calibrate_camera_alignment.py --camera-name top --cam-index /dev/video4

# Step 10: run the sim-trained policy on the real arm
uv run python examples/nexarm/rollout.py --follower-port /dev/ttyUSB1 --front-cam 0 --wrist-cam 1 \
    --policy-path <checkpoint> --task "<training prompt>"
```

Leave out `--joint-mapping` when the probe reports the identity mapping (every sign `+1`, every offset within a few degrees).

## Calibration files and where they are used

| File                                              | Made by                  | Pass it to                                                                                                                                                                                           | Refused when                                                 |
| ------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `outputs/nexarm/joint_mapping_probe.json`         | `probe_joint_mapping.py` | `--joint-mapping` (`calibrate_sim.py`, both generators); `--env.joint_mapping_path` (gym env, `lerobot-eval`); `--robot.joint_mapping_path` / `joint_mapping_path` (`nexarm_sim`, `NexArmSimConfig`) | Data or calibrations were made under another mapping version |
| `outputs/calibration/nexarm_sim_calibration.json` | `calibrate_sim.py`       | `--calibration` (both generators, `lerobot-nexarm-sim-benchmark`); `--env.calibration_path` (gym env); `calibration_path` (`NexArmSimConfig`)                                                        | Another fps, joint mapping or servo model (`dynamics_model`) |
| Scene XML in `sim/description/mjcf/`              | You, from Steps 7–8      | Used by every sim tool; regenerate sim data after a change                                                                                                                                           | —                                                            |

Copy a mapping and a calibration you keep to `sim/calibration/` and commit them together, since the calibration is fitted on top of the mapping. The repository ships neither: the default mapping is the identity, and the servo model `hx_servo_v2` already carries time constants fitted to `thanhkt/nexarm_stack_bowls` (Step 5). Explicit flags beat the calibration file, which beats the defaults: `--action-delay-steps` overrides the calibrated delay.

## Order and time

Do the steps in this order. Each step depends on the ones before it: a wrong joint mapping makes every dynamics fit wrong, and wrong dynamics make camera checks on a moving arm misleading.

| #   | Step                                                                         | Where          | Time      | Done when                                                |
| --- | ---------------------------------------------------------------------------- | -------------- | --------- | -------------------------------------------------------- |
| 0   | [Real arm: zero, settings, joint mapping](#step-0-before-you-start)          | Robot PC       | 30 min    | Wrist overlays line up on a recorded episode             |
| 1   | [Record calibration episodes](#step-1-record-system-identification-episodes) | Robot PC       | 30–45 min | 20 free-motion episodes recorded                         |
| 2   | [Check the recording](#step-2-check-the-recording)                           | Any PC         | 5 min     | Audit shows no lag outliers, or you noted them           |
| 3–4 | [Fit and read the result](#step-3-fit-the-simulation)                        | PC with OpenGL | 15–30 min | Holdout RMSE drops, few scales at a limit                |
| 5–6 | [Use and verify the calibration](#step-5-use-the-calibration)                | PC with OpenGL | 30 min    | Scripted tasks still pass, videos look like the real arm |
| 7   | [Align the cameras](#step-7-align-the-cameras)                               | Robot PC       | 30–60 min | Overlays line up for front, top and wrist                |
| 8   | [Match the scene](#step-8-match-the-scene)                                   | Robot PC       | 20 min    | Measured sizes and heights are in the scene XML          |
| 9   | [Train on real and sim data](#step-9-train-on-real-and-sim-data)             | GPU server     | hours     | Checkpoint passes in simulation                          |
| 10  | [Run on the real arm and iterate](#step-10-run-on-the-real-arm-and-iterate)  | Robot PC       | days      | Real success rate measured on a fixed protocol           |

The sim's world frame, used for every position below: the origin is on the floor under the base's pan axis, `-y` points toward the work area (the robot's front), `+x` is the robot's left, and `+z` is up. Distances are in metres.

## Part 1. Real arm

### Step 0. Before you start

**Zero the arms.** Zero positions live in the servos, not in software. Set them with the vendor desktop app (follower: Servo Advanced page; leader: Synchronizer page). Then place both arms by hand at the vendor Home pose (every joint at raw 2048: upper arm vertical, forearm and closed gripper horizontal) and check them:

```bash
uv run python tools/nexarm/calibrate_arms.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1
```

It prints both arms' raw readings and the leader-to-follower delta per joint, flags joints beyond `--tolerance` ticks, and writes identity calibration files, backing up any older software offsets. `lerobot-calibrate` writes the same identity files. Fix any offset it reports in the desktop app, not in software, and check that [teleoperation](../getting-started/hardware.md#leader-follower-teleoperation) tracks all six joints. The script needs `--extra service` in your `uv sync` line.

**Fix the follower settings.** `motion_speed` and `motion_acc` in `examples/nexarm/record.yaml` change how the arm responds, so a calibration only holds for the values used when you recorded. Decide them now, use them for every recording and rollout, and re-run the calibration if you change them.

**Check the simulation runs.** You can generate simulation data ([pipeline, section 1a](pipeline.md#1a-simulation-scripted-demonstrations)) and `MUJOCO_GL=egl` works on your machine.

**Measure the joint mapping.** Check that the sim joint mapping matches your arm. The sim maps raw 2048 onto the exported CAD pose, which is the follower's Home pose (an L: upper-arm tube vertical, forearm and closed gripper horizontal). Servo signs and zero errors still vary per arm, and until they are measured the sim can pose real joint values slightly differently from the real arm.

1. Measure the per-joint sign and zero offset by posing the un-energised follower by hand. The probe uses a straight-up pose as its reference and reads where it is, so it works whatever shape Home has. Power-cycle the follower and close any app that holds its serial port (such as the Hiwonder desktop app) first:

   ```bash
   uv run python tools/nexarm/probe_joint_mapping.py --port /dev/ttyUSB1
   ```

2. Confirm the result on a recorded episode. This needs OpenGL; use `MUJOCO_GL=egl` on headless Linux.

   ```bash
   uv run python tools/nexarm/overlay_recorded_frames.py --repo-id <real_repo_id> --episode 0 --mapping outputs/nexarm/joint_mapping_probe.json
   ```

   The wrist-camera overlays should line up. Fitting dynamics with a wrong mapping gives meaningless numbers.

In more detail:

- **Before the probe.** Power-cycle the follower, connect it by USB, and quit the Hiwonder desktop app. Clear the space around the arm. The power-cycle matters: any LeRobot program (`lerobot-record`, a rollout, an earlier probe) leaves the follower in bridge mode, which only a restart ends, and in bridge mode the probe cannot turn the motors off.
- **Turning the motors off.** The probe first sends the desktop app's "All off" command and asks whether the arm moves freely. If you answer `n`, it sends per-servo torque-off writes and asks again. Hold the arm before you press Enter: it drops as soon as the motors turn off.
- **During the probe.** It asks you to pose the arm straight up (every link vertical, gripper pointing at the ceiling), then to turn each joint by about a quarter turn. After each turn it asks which way the gripper or link moved: `f` (front, toward the work area), `b` (back), `l` (the robot's left) or `r` (its right).
- **Reading the output.** It prints one line per joint, such as `shoulder_pan sign +1 offset +0.40 deg`, and writes `outputs/nexarm/joint_mapping_probe.json`.
  - Every sign `+1` and every offset within a few degrees means the identity mapping is right, and you can skip the rest of this item.
  - A sign of `-1` means the sim turns that joint the wrong way.
  - An offset above about 5° means the servo's zero differs from the CAD pose.
  - A `WARNING` about the base pan being a quarter turn off, or about an offset outside the joint range, means you should repeat the probe more carefully before trusting it.
- **Checking the result.** Run `overlay_recorded_frames.py` twice on the same episode: once without `--mapping` and once with it. Open the PNG images in `outputs/nexarm/overlay/`. Each shows real, sim and a 50% blend side by side. In the blend, the gripper jaws and the table edge should coincide. Use episodes that move a lot. On `thanhkt/nexarm_stack_bowls`, episode 19 is a good test: without a correct mapping its replay puts the gripper about 39 mm into the panel behind the arm.
- **Applying a non-identity mapping.** Pass the file everywhere the sim reads real joint values:
  - `--joint-mapping outputs/nexarm/joint_mapping_probe.json` to `calibrate_sim.py`, `generate_sim_dataset.py` and `generate_stack_bowls_dataset.py`;
  - `--env.joint_mapping_path` to the gym environment;
  - `joint_mapping_path` in `NexArmSimConfig`.

  Calibrations and datasets record the mapping version and refuse to mix with another, so refit (Step 3) and regenerate after you apply it. Copy the file to `sim/calibration/` so it is versioned with the calibration fitted on top of it.

- **If a joint still moves the wrong way in the sim.** Correct the mapping as above; do not change the CAD joint axis to hide a mismatch between the controller and the sim. The mapping formula and the CAD zero are described in [`sim/README.md`](../../sim/README.md).

## Part 2. Real to sim

### Step 1. Record system-identification episodes

The fit needs episodes where the arm moves freely and moves a lot. A recording of a pick-and-place task works poorly, because contact with the cube biases the fit and several joints barely move.

Record 15–25 episodes with the same follower, leader, fps (30) and settings you use for real data collection. No object is needed. `record.py` still opens and records its cameras (front, wrist and, by default, the RealSense top camera; add `--no-top-cam` if you have none), but calibration reads only joint data.

```bash
uv run python examples/nexarm/record.py \
    --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 \
    --repo-id <user>/nexarm_sysid --num-episodes 20 --episode-time 20
```

The recorder appends a timestamp to the repo id and prints the final `Dataset repo_id:` (e.g. `<user>/nexarm_sysid_20261011_093000`) and `Dataset root:`. Use that printed id as `<sysid_repo_id>` below.

Per episode, move the leader arm like this:

1. Spend about 10 s moving **one joint at a time** through most of its range, once slowly and once quickly. Do this across the episodes so every arm joint (shoulder pan, shoulder lift, elbow, wrist flex, wrist roll) gets several slow and fast sweeps.
2. Add a few episodes that move all joints together, as in a normal task.
3. Keep the gripper open and still. The gripper is not fitted by default.
4. Do not touch objects or the table.

A 20-episode plan that covers every joint (about 20 s each, so add `--episode-time 20`):

| Episodes | Move                                                                                         |
| -------- | -------------------------------------------------------------------------------------------- |
| 1–2      | Shoulder pan only: slow sweep left to right, then fast back and forth                        |
| 3–4      | Shoulder lift only, with the elbow bent so the arm stays clear of the table                  |
| 5–6      | Elbow only, through most of its range                                                        |
| 7–8      | Wrist flex only                                                                              |
| 9–10     | Wrist roll only                                                                              |
| 11–14    | One joint per episode again, now with short fast reversals (small moves back and forth)      |
| 15–20    | All joints together, as when reaching to the bowls and lifting, at your normal and fast pace |

Before each episode, start from the same rest pose. Between episodes, check that the cable has not wrapped around the wrist.

Tips:

- Use at least 300 frames (10 s) per episode. The tool uses up to 600 frames by default.
- Include some small, fast reversals. They are the movements that reveal latency.
- Record a different set of episodes if you later want an independent check.

### Step 2. Check the recording

```bash
uv run python - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("<sysid_repo_id>")
print(ds.meta.total_episodes, "episodes", ds.meta.total_frames, "frames", ds.fps, "fps")
print(ds.meta.features["action"]["names"])
EOF
```

Then audit the joint data:

```bash
uv run python tools/nexarm/audit_real_dataset.py --repo-id <sysid_repo_id> --episode-time-s <episode_time_s>
```

The audit reports:

- **which episodes have trustworthy timing.** Frames are stamped at a nominal 1/fps, so "Record loop is running slower" warnings inflate speeds. Only episodes that ran their full time limit are used for speeds.
- **per-joint speeds.** A joint that never saturated was only moved that fast, so its top speed is a lower bound, not a servo limit.
- **episodes whose command-to-state lag differs from the rest.** These usually come from a session that logged differently; skip them with `--exclude-episodes` in `calibrate_sim.py`.
- **gripper levels.** If releases stay above the stack-bowls grasp-assist release threshold, raise `assist_release_pos` on `NexArmStackBowlsTask` before replaying real demos in the sim.

To measure servo speed and lag directly with real timestamps, run `tools/nexarm/servo_step_test.py --port /dev/ttyUSB1 --joint elbow_flex`. It compares `motion_speed` 2000 with 0.

Confirm that:

- the fps is 30 (or the fps you will generate with),
- the action names are the six `*.pos` joints,
- there are at least 10 episodes with 20 or more frames each.

### Step 3. Fit the simulation

```bash
MUJOCO_GL=egl uv run python tools/nexarm/calibrate_sim.py \
    --repo-id <sysid_repo_id> \
    --episodes 20 \
    --out outputs/calibration/nexarm_sim_calibration.json
```

The tool replays each episode's recorded `action` stream through the simulator, starting from the real first state, and compares the simulated joint positions with the real `observation.state`. It then fits:

1. the action delay (searched over `--delay-range`, default 0–6 control steps),
2. each joint's time-constant scale (limited to 0.25×–4×): how fast the servo follows a small step, `(kv + damping) / kp`. This is the one dynamics parameter that task demos identify,
3. a spread for it, which becomes the domain-randomization range.

`kp`, damping and friction stay at the servo model's values. With the time constant held, `kp` only sets the holding stiffness and damping the saturated top speed, and on the stack-bowls demos halving or doubling `kp` changes the replay error by 0.1 raw units. `--fit-kp` and `--fit-friction` fit them anyway; use those only with free-air sweeps that saturate the joints, and check the scripted tasks afterwards (Step 6).

By default 20% of the episodes are held out (`--holdout-frac`) and never used for fitting.

Useful options:

| Option                              | Use                                                   |
| ----------------------------------- | ----------------------------------------------------- |
| `--root <path>`                     | The dataset is in a local folder instead of the cache |
| `--joints shoulder_lift,elbow_flex` | Fit only some joints                                  |
| `--exclude-episodes 19,69-92`       | Skip episodes that do not replay                      |
| `--delay-range 0 10`                | The latency may be longer than 6 steps                |
| `--max-frames 900`                  | Use longer episodes                                   |
| `--holdout-frac 0`                  | No holdout (use only for very few episodes)           |

### Step 4. Read the result

The tool prints a table like this:

```
RMSE (raw servo units)
joint          fit_before  holdout_before  fit_after  holdout_after
shoulder_lift       32.58           31.52       5.33           2.56
```

The units are raw servo units, where one degree is about 11.4. Check each of these:

| Check                                                             | Good                            | If not                                                                                                              |
| ----------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `holdout_after` is lower than `holdout_before` for the arm joints | Yes, clearly                    | The fit does not generalize. Record more varied episodes (Step 1)                                                   |
| `holdout_after` is close to `fit_after`                           | Within about 2×                 | The fit overfits. Use more episodes or `--holdout-frac 0.3`                                                         |
| `action_delay_steps`                                              | Strictly inside `--delay-range` | A value at either edge of the range means the search range may be wrong. Widen it                                   |
| Time-constant scales                                              | Mostly between 0.5× and 2×      | Many scales at 0.25× or 4× mean the data is too weak to identify them. Treat the result as unreliable and re-record |
| Spread                                                            | 0.10–0.50                       | Values pinned at 0.10 or 0.50 mean there is little information. The clamp is protecting you                         |

The gripper RMSE is large because it is not fitted. This is expected.

If the fit is poor, the cause is almost always the data, not the tool: too few episodes, too little motion, or a different `motion_speed` than the one you use normally.

#### Optional: choose `kp`

The default fit keeps the servo model's `kp`. A `kp` scale no longer changes the response time (the tool recomputes `kv` to hold the time constant), so it only matters for how stiffly the arm holds against loads. To let the real data set it, you need recordings where the arm is pushed or carries a known load:

1. Fit with `--fit-kp` into a separate file.
2. Compare the holdout RMSE (the `holdout_after` mean over the five arm joints) with the default fit. Keep the `kp` scales only if they lower it clearly.
3. Check the scripted success over 20 seeds of each task (Step 6, check 1). It must stay at 20/20 on both.

### Step 5. Use the calibration

The repository ships no calibration file. The servo model in the MJCF (`hx_servo_v2`) already carries the time constants fitted to `thanhkt/nexarm_stack_bowls`, so the sim is usable without one:

- Without a calibration file, the model replays held-out `thanhkt/nexarm_stack_bowls` episodes with about 11 raw units of error (mean over the arm joints), and both scripted tasks score 20/20 with and without domain randomization.
- A refit on the same dataset moved the time constants by 0.84–1.30× and changed the holdout error by less than 1 raw unit, which is inside the noise of these demos. That is why no fitted file is shipped.

Fit your own file (Step 3) when you have free-air recordings of your follower, and pass it to the generator:

```bash
MUJOCO_GL=egl uv run python examples/nexarm/generate_sim_dataset.py \
    --repo-id local/nexarm_sim_pick_place \
    --root outputs/datasets/nexarm_sim_pick_place_cal \
    --calibration sim/calibration/nexarm_sim_calibration_<name>.json \
    --episodes 200 --workers 4 --gpus 0,1,2,3
```

This sets the simulated action delay to the fitted value (sampled per episode within ±1), scales the joint dynamics, and centers domain randomization (on by default in both generators) on the fitted values with the fitted spreads, for the joints it fitted only. The calibration file's hash is recorded in `generation_report.json`. `generate_stack_bowls_dataset.py` accepts the same `--calibration`, `--action-delay-steps` and `--action-delay-range` options.

The file records the `fps` it was fitted at; every loader refuses a run at a different fps (the generators reject it at argument parsing). Precedence everywhere is explicit value > calibration > default: `--action-delay-steps` overrides the calibrated delay.

The same calibration is available in:

- the gym environment: `--env.calibration_path`, `--env.action_delay_steps`, `--env.enable_domain_randomization`,
- the benchmark: `lerobot-nexarm-sim-benchmark --calibration PATH --action-delay-steps N --dr`,
- the robot class: `calibration_path` in `NexArmSimConfig`.

Calibrations fitted before the servo-constant joint mapping (`rad = (raw - 2048) * 2π / 4096`) are stale and are refused on load (they carry no matching `joint_mapping`): re-run Step 3, then regenerate any sim data made with them.

The fitted scales correct one sim servo model, so the file also records its `dynamics_model` (now `hx_servo_v2`, see "Servo model" in `sim/README.md`). A calibration without it, or fitted against another model (including `hx_servo_v1`), is refused on load: re-run Step 3, then regenerate any sim data made with it. The setpoint speed and acceleration caps and the armature are fixed priors of the model; the calibration does not fit them.

Use `--split eval` to generate a held-out evaluation set. Its seeds never overlap with the training seeds.

### Step 6. Verify the calibrated data

Do these checks before training:

1. **Scripted success check.** A calibration that makes the sim arm track the real one better can still break the scripted grasps that generate the data. Run 20 seeds of each task with no retries, first with domain randomization off and then with it on (drop `--no-dr`):

   ```bash
   CAL=sim/calibration/nexarm_sim_calibration_<name>.json
   for task in generate_sim_dataset generate_stack_bowls_dataset; do
     MUJOCO_GL=egl uv run python examples/nexarm/$task.py --calibration $CAL \
         --repo-id local/check_$task --root outputs/datasets/check_$task \
         --episodes 20 --max-attempts 20 --no-video --no-dr
     uv run python -c "import json; r=json.load(open('outputs/datasets/check_$task/generation_report.json')); print('$task', len(r['accepted_seeds']), '/', r['attempts'])"
   done
   ```

   Expect 20/20 for both without domain randomization, and at least 19/20 with it. The servo model without a calibration scores 20/20 on both tasks, with and without domain randomization. Delete the check folders before rerunning.

   To watch the episodes, run again without `--no-video` and with `--episodes 3`, then export them with `tools/nexarm/export_episode_video.py`. Look at whether the arm moves faster or more sharply than the real one, and whether objects slip in the gripper.

2. **Replay check.** Re-run Step 3 on a _different_ real dataset with `--holdout-frac 0` and the same model. The RMSE should stay close to the holdout value from Step 4. Then compare it to `SimCalibration()` (no calibration) to see the gain.
3. **Distribution check.** Compare the real and the simulated datasets for episode length, the range of each joint, and per-step action changes. `generation_report.json` gives the simulated numbers. Large gaps mean the scripted motion differs from how your operator moves. Adjust `--action-noise` or the trajectory variation in `generate_sim_dataset.py`.
4. **Look at frames.** Export a real and a simulated episode with `export_episode_video.py` and compare the views side by side.

### Step 7. Align the cameras

Policies see images, so wrong camera geometry causes a bigger sim-to-real gap than the dynamics.

```bash
uv run python tools/nexarm/calibrate_camera_alignment.py --cam-index 0 --camera-name front
```

The tool overlays the simulated view on the live camera. Keys: `m` switches mode, `+` and `-` change the blend, `s` saves a snapshot, `q` quits. Compare the table edge, the base of the arm and the target zone. If they do not line up, edit the camera `pos`, `quat`/`xyaxes` or `fovy` in the scene XML (`sim/description/mjcf/`) until they do, then repeat with `--camera-name wrist` and `--camera-name top`. For the RealSense `top` camera, `uv run python tools/nexarm/realsense_fovy.py` reads the colour intrinsics and prints the matching `fovy`. For `top`, pass the RealSense RGB V4L2 node as `--cam-index` (e.g. `/dev/video4`); the sim top camera looks straight down from 0.75 m. Check that the resolution and fps match your real cameras.

Known open point: the `front` camera sits at z = 0.817 m in `scene.xml` and 0.717 m in `bowl_stack_scene.xml`. Only one can match your rig, so measure the real height and correct the other file.

Step by step for each camera. Fix the field of view first; a pose error cannot be tuned away while the field of view is wrong.

1. **Field of view.**
   - **`top` (D435i):** run `uv run python tools/nexarm/realsense_fovy.py --width 640 --height 480` and set the printed `fovy` on the `top` camera.
   - **`front` (webcam):** calibrate it with a printed checkerboard (OpenCV `calibrateCamera`, or any camera-calibration app) to get `fy` at 640×480, then set `fovy = 2 * atan(240 / fy)` in degrees.
   - **`wrist`:** its `fovy` is in `sim/description/mjcf/nexarm.xml`.
2. **Measure the real camera position** with a tape, in the world frame above, to the lens centre:
   - `x`, left or right of the base's pan axis;
   - `y`, forward of it (negative toward the work area);
   - `z`, above the table.

   Current sim values:

   | Camera  | Position (x, y, z) in m                        | Points                                 |
   | ------- | ---------------------------------------------- | -------------------------------------- |
   | `front` | -0.03, -0.87, 0.82 (pick-place) / 0.72 (bowls) | Back at the arm, tilted down about 45° |
   | `top`   | -0.05, -0.35, 0.77                             | Straight down                          |
   | `wrist` | On the gripper; edit it in `nexarm.xml`        | Along the jaws                         |

3. **Set the measured `pos`** in both `scene.xml` and `bowl_stack_scene.xml`, unless the camera really is in a different place for the two tasks. Today the `front` camera height differs between them by 0.1 m, so check which one is right.
4. **Fine-tune the orientation in the overlay.** Run the alignment tool, look at where the table edge and the arm base land, edit `xyaxes`, and restart the tool. `xyaxes` holds the camera's right direction and its up direction, both in world coordinates.
   - Tilt (the image is too high or too low): change the second vector, for example `0 0.71 0.70` (about 45° down) toward `0 0.77 0.64` (about 50° down, looking further down).
   - Pan (the image is shifted left or right): rotate both vectors about `z`.
   - Size (the scene looks too big or too small): this is `fovy` or the distance, not the orientation. Go back to steps 1–2.
5. **Save a snapshot** with `s` once the table edge, the base and the target zone line up within a few pixels. Keep it with the commit that changes the XML.

After any change to the scene cameras, rerun the scripted success check (Step 6). Regenerate sim data before training, because the existing data was rendered from the old view.

### Step 8. Match the scene

Measure these on your real setup and update the scene XML if they differ:

- cube size and mass (the simulated cube is 20 mm and 20 g),
- table height and the target zone size,
- background and table color.

What to measure and where it lives in the sim:

| Measure on the real setup                              | Sim value                                                           | Where to change it                                                                |
| ------------------------------------------------------ | ------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Height of the shoulder-lift axis above the table       | 0.139 m                                                             | `nexarm_mount` `pos` z in `sim/description/mjcf/nexarm.xml` (moves the whole arm) |
| Anything behind or beside the arm (wall, panel, clamp) | Check in the viewer: `uv run python examples/nexarm/simulate.py`    | Add or move a box geom in the scene XML                                           |
| Cube: edge length and mass                             | 20 mm, 20 g                                                         | `cube_collision` in `scene.xml`                                                   |
| Target zone: centre and radius                         | (-0.09, -0.24), radius 0.05 m                                       | `target_zone` in `scene.xml`                                                      |
| Bowls: mass, rim diameter, height, start positions     | 60 g each; red (-0.11, -0.27), blue (0, -0.18), black (0.11, -0.27) | `bowl_stack_scene.xml`; keep the spawn slots in `stack_bowls_task.py` in step     |
| Table and background colour, lighting                  | Grey floor; domain randomization varies colour                      | Floor `rgba` in the scene XML; `DomainRandomizationRanges`                        |

Measure heights from the table surface, which is the sim floor (`z = 0`). If the real arm stands on a plate or clamp, measure from that, then add its height to the mount `z`. A mount error of 1–2 cm is enough to make replays touch the table, as episodes 1, 3 and 12 of `thanhkt/nexarm_stack_bowls` do (12–25 mm below it).

Then adjust the visual randomization range in `DomainRandomizationRanges` (`packages/lerobot_robot_nexarm/src/lerobot_robot_nexarm/nexarm_sim/mujoco_backend.py`) so it covers the lighting and color you actually see.

## Part 3. Sim to real

### Step 9. Train on real and sim data

1. Generate the training data with the calibration, the joint mapping (if not the identity) and domain randomization on (Step 5). Start with 200–300 episodes per task; more randomized episodes help sim-to-real more than more training steps. Generating on several GPUs: [Sim training](sim-training.md).
2. Train with real and simulated data mixed. `lerobot-train` takes one dataset, so merge the real and sim datasets first (they share `robot_type="nexarm_follower"`; they also need the same cameras and fps), and set the mix through the episode counts; start with real data at about 25–50% of the episodes:

   ```bash
   uv run lerobot-edit-dataset \
       --new_repo_id local/nexarm_real_sim_mix \
       --new_root outputs/datasets/nexarm_real_sim_mix \
       --operation.type merge \
       --operation.repo_ids "['thanhkt/nexarm_stack_bowls_top', 'local/nexarm_stack_bowls']" \
       --operation.roots "['$HOME/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top', 'outputs/datasets/nexarm_stack_bowls']"
   ```

   Then train on it as in [the pipeline guide, section 2](pipeline.md#2-training).

3. Check the checkpoint in the calibrated simulation before it touches the arm. Pass the same calibration and mapping to the env:

   ```bash
   MUJOCO_GL=egl uv run lerobot-eval --policy.path=<checkpoint> \
       --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=20 \
       --env.calibration_path=outputs/calibration/nexarm_sim_calibration.json \
       --env.joint_mapping_path=outputs/nexarm/joint_mapping_probe.json \
       --output_dir=outputs/eval/sim_check
   ```

   A checkpoint that fails here will not succeed on the real arm. Simulated success rate does not predict the real one, only rules out a broken checkpoint.

### Step 10. Run on the real arm and iterate

Before the first run:

- The follower uses the same `motion_speed` and `motion_acc` as when you recorded the calibration episodes (Step 0).
- The real cameras are where the sim cameras are: run `calibrate_camera_alignment.py` for each camera (Step 7) and check that the overlay still lines up. A bumped camera is the most common sim-to-real failure.
- The real scene matches the sim (Step 8): bowl or cube positions inside the trained spawn area, the same lighting as far as possible.
- The policy gets exactly the cameras it was trained on; `rollout.py` refuses a mismatch. Use `--no-top-cam` for a 2-camera checkpoint.
- Keep a hand near the power switch the first time a new checkpoint moves the arm.

Run it with the training prompt:

```bash
uv run python examples/nexarm/rollout.py --follower-port /dev/ttyUSB1 --front-cam 0 --wrist-cam 1 \
    --policy-path <checkpoint> \
    --task "Stack the bowls with red on bottom, blue in middle, and black on top."
```

To keep each trial as a dataset for later analysis, add `--strategy sentry --repo-id <user>/eval_nexarm`; every rollout option is in [Rolling out a trained policy](rollout.md).

Evaluate on **real held-out trials** with a fixed protocol: the same number of trials (at least 10–20), the same set of start layouts written down in advance, and the same success criterion for every checkpoint. Then read how the trials fail and go back to the step that owns it:

| How the real rollouts fail                                   | Likely cause                     | Go back to                                                            |
| ------------------------------------------------------------ | -------------------------------- | --------------------------------------------------------------------- |
| A joint moves the wrong way or to a consistently wrong angle | Joint mapping                    | [Step 0](#step-0-before-you-start), then regenerate                   |
| Motion starts late, overshoots or oscillates                 | Latency or servo dynamics        | [Steps 1–5](#step-1-record-system-identification-episodes)            |
| The gripper reaches a consistent offset from the object      | Camera pose or field of view     | [Step 7](#step-7-align-the-cameras)                                   |
| Hits the table, misses heights, or collides with the rig     | Mount height or scene geometry   | [Step 8](#step-8-match-the-scene)                                     |
| Reaches well, but grasps slip or fail                        | Contact physics (not calibrated) | More real demonstrations in the mix (Step 9)                          |
| Works in some lighting or backgrounds and not others         | Visual gap                       | Widen `DomainRandomizationRanges` ([Step 8](#step-8-match-the-scene)) |

Fix one cause at a time, regenerate the sim data, retrain, and rerun the same protocol so the success rates compare.

## Troubleshooting

| Symptom                                                 | Likely cause and fix                                                                                   |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `No usable episodes found`                              | Episodes have fewer than 20 frames, or the repo id or `--root` is wrong                                |
| Delay fits to 0 and RMSE stays high                     | The real data has little fast motion. Add fast reversals (Step 1)                                      |
| Delay fits to the maximum                               | Raise `--delay-range`, and check that the dataset fps equals the control rate                          |
| Scales hit 0.25× or 4×                                  | Too little excitation for that joint. Record more sweeps of it                                         |
| Good fit, but real rollouts still fail                  | The gap is probably visual or contact related. Do Steps 7 and 8 and keep real contact data in training |
| Results change after changing `motion_speed`            | Expected. Record new episodes with the new setting and re-fit                                          |
| A sim joint moves opposite to the real one              | Stop sim-to-real testing and measure the joint mapping (Step 0)                                        |
| A calibration or dataset is refused on load             | It was made under another fps, joint mapping or servo model. Re-run Step 3 and regenerate the data     |
| `calibrate_arms.py`: `No module named 'nexarm_service'` | Add `--extra service` to your `uv sync` line                                                           |
| `probe_joint_mapping.py` cannot turn the motors off     | The follower is still in bridge mode from a LeRobot program. Power-cycle it and close the desktop app  |

## Limits of this calibration

- It fits latency and joint tracking from free-air motion. It does not measure grasp friction, cube mass or slip.
- Per-joint scales are not independent. With weak data a higher damping can trade off against a lower gain, so judge the fit by the holdout RMSE, not by the individual numbers.
- One calibration describes one arm with one set of follower settings.
