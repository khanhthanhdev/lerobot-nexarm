# Calibrating the NexArm simulation to your real arm

This guide makes the MuJoCo twin behave like your physical arm, then generates simulation data from it. Follow the steps in order and run every command from the repository root with `uv run`.

**What gets calibrated**

| What                                                                                    | Tool                            | Result                          |
| --------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------- |
| Action latency, per-joint gain (`kp`), damping and friction, and how much each can vary | `calibrate_sim.py`              | `SimCalibration` JSON           |
| Camera pose and field of view                                                           | `calibrate_camera_alignment.py` | Edits you make to the scene XML |
| Cube size, mass, table height                                                           | Measure by hand                 | Edits you make to the scene XML |

**What is not calibrated:** contact physics such as cube friction and slip. Keep real contact-rich demonstrations in your training mix, since simulation widens coverage but does not replace them.

## Step 0. Before you start

- The arm is calibrated (`lerobot-calibrate`) and teleoperation works.
- You can generate simulation data (see `pipeline.md`, section 1a) and `MUJOCO_GL=egl` works on your machine.
- Decide the follower settings now and keep them. `motion_speed` and `motion_acc` in `examples/nexarm/record.yaml` change how the arm responds, so a calibration only holds for the values used when you recorded. Re-run the calibration if you change them.

## Step 1. Record system-identification episodes

The fit needs episodes where the arm moves freely and moves a lot. A recording of a pick-and-place task works poorly, because contact with the cube biases the fit and several joints barely move.

Record 15–25 episodes with the same follower, leader, fps (30) and settings you use for real data collection. No object is needed. `record.py` still opens and records its cameras (front, wrist and, by default, the RealSense top camera; add `--no-top-cam` if you have none), but calibration reads only joint data.

```bash
uv run python examples/nexarm/record.py \
    --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 \
    --repo-id <user>/nexarm_sysid --num-episodes 20
```

The recorder appends a timestamp to the repo id and prints the final `Dataset repo_id:` (e.g. `<user>/nexarm_sysid_20261011_093000`) and `Dataset root:`. Use that printed id as `<sysid_repo_id>` below.

Per episode, move the leader arm like this:

1. Spend about 10 s moving **one joint at a time** through most of its range, once slowly and once quickly. Do this across the episodes so every arm joint (shoulder pan, shoulder lift, elbow, wrist flex, wrist roll) gets several slow and fast sweeps.
2. Add a few episodes that move all joints together, as in a normal task.
3. Keep the gripper open and still. The gripper is not fitted by default.
4. Do not touch objects or the table.

Tips:

- Use at least 300 frames (10 s) per episode. The tool uses up to 600 frames by default.
- Include some small, fast reversals. They are the movements that reveal latency.
- Record a different set of episodes if you later want an independent check.

## Step 2. Check the recording

```bash
uv run python - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("<sysid_repo_id>")
print(ds.meta.total_episodes, "episodes", ds.meta.total_frames, "frames", ds.fps, "fps")
print(ds.meta.features["action"]["names"])
EOF
```

Confirm that:

- the fps is 30 (or the fps you will generate with),
- the action names are the six `*.pos` joints,
- there are at least 10 episodes with 20 or more frames each.

## Step 3. Fit the simulation

```bash
MUJOCO_GL=egl uv run python examples/nexarm/calibrate_sim.py \
    --repo-id <sysid_repo_id> \
    --episodes 20 \
    --out outputs/calibration/nexarm_sim_calibration.json
```

The tool replays each episode's recorded `action` stream through the simulator, starting from the real first state, and compares the simulated joint positions with the real `observation.state`. It then fits:

1. the action delay (searched over `--delay-range`, default 0–6 control steps),
2. each joint's `kp`, damping and friction scales (limited to 0.25×–4×),
3. a spread for damping and friction, which becomes the domain-randomization range.

By default 20% of the episodes are held out (`--holdout-frac`) and never used for fitting.

Useful options:

| Option                              | Use                                                   |
| ----------------------------------- | ----------------------------------------------------- |
| `--root <path>`                     | The dataset is in a local folder instead of the cache |
| `--joints shoulder_lift,elbow_flex` | Fit only some joints                                  |
| `--delay-range 0 10`                | The latency may be longer than 6 steps                |
| `--max-frames 900`                  | Use longer episodes                                   |
| `--holdout-frac 0`                  | No holdout (use only for very few episodes)           |

## Step 4. Read the result

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
| Scales                                                            | Mostly between 0.5× and 2×      | Many scales at 0.25× or 4× mean the data is too weak to identify them. Treat the result as unreliable and re-record |
| Spread                                                            | 0.10–0.50                       | Values pinned at 0.10 or 0.50 mean there is little information. The clamp is protecting you                         |

The gripper RMSE is large because it is not fitted. This is expected.

If the fit is poor, the cause is almost always the data, not the tool: too few episodes, too little motion, or a different `motion_speed` than the one you use normally.

## Step 5. Use the calibration

Pass it to the generator:

```bash
MUJOCO_GL=egl uv run python examples/nexarm/generate_sim_dataset.py \
    --repo-id local/nexarm_sim_pick_place \
    --root outputs/datasets/nexarm_sim_pick_place_cal \
    --calibration outputs/calibration/nexarm_sim_calibration.json \
    --episodes 200 --workers 4 --gpus 0,1,2,3
```

This sets the simulated action delay to the fitted value (sampled per episode within ±1), scales the joint dynamics, and centers domain randomization (on by default in both generators) on the fitted values with the fitted spreads, for the joints it fitted only. The calibration file's hash is recorded in `generation_report.json`. `generate_stack_bowls_dataset.py` accepts the same `--calibration`, `--action-delay-steps` and `--action-delay-range` options.

The file records the `fps` it was fitted at; every loader refuses a run at a different fps (the generators reject it at argument parsing). Precedence everywhere is explicit value > calibration > default: `--action-delay-steps` overrides the calibrated delay.

The same calibration is available in:

- the gym environment: `--env.calibration_path`, `--env.action_delay_steps`, `--env.enable_domain_randomization`,
- the benchmark: `lerobot-nexarm-sim-benchmark --calibration PATH --action-delay-steps N --dr`,
- the robot class: `calibration_path` in `NexArmSimConfig`.

Calibrations fitted before the servo-constant joint mapping (`rad = (raw - 2048) * 2π / 4096`) are stale and are refused on load (they carry no matching `joint_mapping`): re-run Step 3, then regenerate any sim data made with them.

Use `--split eval` to generate a held-out evaluation set. Its seeds never overlap with the training seeds.

## Step 6. Verify the calibrated data

Do these checks before training:

1. **Replay check.** Re-run Step 3 on a _different_ real dataset with `--holdout-frac 0` and the same model. The RMSE should stay close to the holdout value from Step 4. Then compare it to `SimCalibration()` (no calibration) to see the gain.
2. **Distribution check.** Compare the real and the simulated datasets for episode length, the range of each joint, and per-step action changes. `generation_report.json` gives the simulated numbers. Large gaps mean the scripted motion differs from how your operator moves. Adjust `--action-noise` or the trajectory variation in `generate_sim_dataset.py`.
3. **Look at frames.** Export a real and a simulated episode with `export_episode_video.py` and compare the views side by side.

## Step 7. Align the cameras

Policies see images, so wrong camera geometry causes a bigger sim-to-real gap than the dynamics.

```bash
uv run python examples/nexarm/calibrate_camera_alignment.py --cam-index 0 --camera-name front
```

The tool overlays the simulated view on the live camera. Keys: `m` switches mode, `+` and `-` change the blend, `s` saves a snapshot, `q` quits. Compare the table edge, the base of the arm and the target zone. If they do not line up, edit the camera `pos`, `quat`/`xyaxes` or `fovy` in the scene XML (`sim/fusion_export/`) until they do, then repeat with `--camera-name wrist` and `--camera-name top`. For `top`, pass the RealSense RGB V4L2 node as `--cam-index` (e.g. `/dev/video4`); the sim top camera looks straight down from 0.75 m. Check that the resolution and fps match your real cameras.

## Step 8. Match the scene

Measure these on your real setup and update the scene XML if they differ:

- cube size and mass (the simulated cube is 20 mm and 20 g),
- table height and the target zone size,
- background and table color.

Then adjust the visual randomization range in `DomainRandomizationRanges` (`src/lerobot/robots/nexarm_sim/mujoco_backend.py`) so it covers the lighting and color you actually see.

## Step 9. Train and iterate

1. Train with real and simulated data mixed. Start with real data about 25–50% of the batches, as in `pipeline.md`, section 2.
2. Evaluate on **real held-out trials**, not on simulated ones.
3. Look at how the real rollouts fail: late or early motion points to delay or gains (repeat Steps 1–5), missed grasps point to the scene or contact (Step 8).
4. Regenerate and retrain.

## Troubleshooting

| Symptom                                      | Likely cause and fix                                                                                   |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `No usable episodes found`                   | Episodes have fewer than 20 frames, or the repo id or `--root` is wrong                                |
| Delay fits to 0 and RMSE stays high          | The real data has little fast motion. Add fast reversals (Step 1)                                      |
| Delay fits to the maximum                    | Raise `--delay-range`, and check that the dataset fps equals the control rate                          |
| Scales hit 0.25× or 4×                       | Too little excitation for that joint. Record more sweeps of it                                         |
| Good fit, but real rollouts still fail       | The gap is probably visual or contact related. Do Steps 7 and 8 and keep real contact data in training |
| Results change after changing `motion_speed` | Expected. Record new episodes with the new setting and re-fit                                          |

## Limits of this calibration

- It fits latency and joint tracking from free-air motion. It does not measure grasp friction, cube mass or slip.
- Per-joint scales are not independent. With weak data a higher damping can trade off against a lower gain, so judge the fit by the holdout RMSE, not by the individual numbers.
- One calibration describes one arm with one set of follower settings.
