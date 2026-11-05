# Servo baseline (before the servo model)

Measured 2026-11-04 on branch `feat/sim-fidelity-servo-model` at `772b9b201`, MuJoCo 3.8.1, 30 fps, no action
delay, no domain randomization, no rendering. The harness was a scratch script; its method is recorded here so
the "after" run repeats it.

## Method

- **Scripted success:** `generate_episode` from `examples/nexarm/generate_sim_dataset.py` (pick-place,
  `scene.xml`) and `examples/nexarm/generate_stack_bowls_dataset.py` (stack-bowls, `bowl_stack_scene.xml`,
  generator defaults, so `grasp_mode="physical"`), seeds 0–19, `record_frame=None`, `camera_names=()`.
  Frames counts `NexArmSim.send_action` calls.
- **Step test:** from `HOME_POSITIONS`, hold the start pose for 40 frames, then command one step for 2 s and
  take the largest 200 ms (6-frame) displacement. Each step was chosen so no arm geom touches the workcell,
  the floor or another link (checked by listing contacts):

  | Joint         | Pre-pose         | Step (ticks) |
  | ------------- | ---------------- | -----------: |
  | shoulder_pan  | home             |        +1200 |
  | shoulder_lift | elbow_flex −1200 |         −900 |
  | elbow_flex    | home             |        −1200 |
  | wrist_flex    | home             |        +1200 |
  | wrist_roll    | home             |        +1200 |
  | gripper       | open (1195)      |       → 2833 |

  From home, a shoulder step of ±900 hits the workcell or the floor, and an elbow step of +1200 folds the
  gripper into the base.

- **Gripper stroke:** time to 95% of a full open→closed and closed→open stroke.
- **Droop:** for each pick-place reset (seeds 0–19), IK (`solve_ik`, keys without `.pos`) to 10 cm above the
  cube, hold that command 1.5 s, and take the worst arm-joint error.

## Results

| Measure                                    | Before            | Target (plan criterion 3)     |
| ------------------------------------------ | ----------------- | ----------------------------- |
| shoulder_pan sustained speed               | 5650 ticks/s      | within ±10% of the cap (3413) |
| shoulder_lift sustained speed              | 4531 ticks/s      | within ±10% of the cap (3413) |
| elbow_flex sustained speed                 | 5379 ticks/s      | within ±10% of the cap (3413) |
| wrist_flex sustained speed                 | 5571 ticks/s      | within ±10% of the cap (3413) |
| wrist_roll sustained speed                 | 5921 ticks/s      | within ±10% of the cap (3413) |
| gripper sustained speed                    | 8190 ticks/s      | about 3700 (real, saturated)  |
| Full gripper stroke (open→closed, 95%)     | 0.033 s           | 0.35–0.5 s                    |
| wrist_flex error 2 s after its step        | 62 ticks          | —                             |
| Droop at the IK grasp poses (median / max) | 13.4 / 13.9 mrad  | under 2 mrad                  |
| Scripted pick-place                        | 20/20, 380 frames | at least the baseline         |
| Scripted stack-bowls (physical grasp)      | 20/20, 428 frames | at least the baseline         |

## Real-arm reference

From the servo model fitted to the 7 full-length episodes of `thanhkt/nexarm_stack_bowls`
(`../state-feedback-ensembling/results/nexarm_servo_model.json`, a first-order lag per frame, `alpha`, with a
per-frame speed clip):

| Joint         | alpha | Time constant | Steady error, empty (median, IQR) | Holding (median, IQR) |
| ------------- | ----: | ------------: | --------------------------------- | --------------------- |
| shoulder_pan  | 0.455 |       0.055 s | 0 (−2…2) ticks                    | −2 (−2…1)             |
| shoulder_lift | 0.258 |       0.112 s | 1 (−3…5)                          | −7 (−7…7)             |
| elbow_flex    | 0.269 |       0.107 s | 2 (−4…8)                          | 2 (−8…16)             |
| wrist_flex    | 0.298 |       0.094 s | 0 (−1…3)                          | 2 (−3…6)              |
| wrist_roll    | 0.784 |       0.022 s | 0 (−1…1)                          | 0 (−1…1)              |
| gripper       | 0.149 |       0.207 s | 48 (0…49)                         | 70 (66…75)            |

Time constant = −1 / (30 · ln(1 − alpha)). Wrist_roll uses the per-episode median alpha; its pooled fit (0.038)
is poorly identified because the joint barely moves in the demos.

The real arm holds a median of 0–2 ticks from its command with the gripper empty, so the under-2 mrad droop
target (1.3 ticks) matches it; the spread looks like servo deadband, which the sim does not model. Small moves
close only 15–45% of the gap per frame, so a stiff `kp` alone (`kp=998`, `kv=2.731`, time constant about
3 ms) would track small commands far faster than the real servos.

## In progress: the stiff arm and the physical bowl grasp (2026-11-04)

Scripted success with the servo MJCF (stiff `kp=998.22`, kv from the real τ, the jaw models above) and the speed
clip, without an acceleration cap: pick-place 17/20 (seeds 7, 12, 19), stack-bowls 0/20 (all at
`nested_middle`). The stack-bowls default grasp is `physical`: the rim lifts to about 59 mm, then slides out of
the jaws.

Bisect on stack-bowls seeds 0–5, generator edits kept in every row:

| Variant                                                    | Stack-bowls |
| ---------------------------------------------------------- | ----------: |
| Old MJCF (HEAD), limiter on or off                         |         6/6 |
| New MJCF, limiter off                                      |         0/6 |
| New MJCF, the whole old gripper (kp, kv, 50 N, pads, jaws) |         0/6 |
| New MJCF, old pad solref / 60 N / 40 N gripper force       |         0/6 |
| New MJCF, old soft arm gains (kp 20–80)                    |         6/6 |
| New MJCF, acceleration cap 10k / 20k / 40k ticks/s²        |    6/6 each |
| New MJCF, acceleration cap 80k / 160k ticks/s²             |    3/6, 0/6 |

The stiff arm tracks the setpoint's instant speed change, and that jerk breaks the friction grasp; the old soft
arm low-passed it. At 30 fps the real demos show |Δ²state|·fps² p99 of 12.6k–28.8k ticks/s² on the arm joints
(p99.9 18.9k–56.7k), so a 20k cap is in the real range. The scripted sim motions stay below that at frame
resolution with or without a cap: the difference lies inside a frame.

With a 20k ticks/s² cap (all joints, 20 seeds): pick-place 20/20 (387 frames), stack-bowls 19/20 (seed 9,
456 frames); step speeds 3211/2518/2720/2781/3126/3683 ticks/s; gripper stroke 0.6 s; droop max 1.08 mrad.

**Decision (user, 2026-11-04):** keep the stiff arm and cap the setpoint acceleration. The default
`SERVO_MAX_ACCEL_TICKS_S2` caps the five arm joints at 20k ticks/s² and leaves the gripper uncapped, which keeps
its stroke at 0.5 s.

## After (servo MJCF, speed clip, arm acceleration cap; 20 seeds)

| Measure                         | Before                        | After                         |
| ------------------------------- | ----------------------------- | ----------------------------- |
| Step speed, ticks/s (200 ms)    | 5650/4531/5379/5571/5921/8190 | 3211/2518/2720/2781/3126/3699 |
| Gripper full stroke             | 0.033 s                       | 0.5 s (final error 4 ticks)   |
| Droop at the grasp poses        | 13.4 mrad max                 | 1.08 mrad max, 0.96 median    |
| Pick-place scripted             | 20/20, 380 frames             | 20/20, 386 frames             |
| Stack-bowls scripted (physical) | 20/20, 428 frames             | 20/20, 452 frames             |

The gravity-loaded joints run 14–26% under the 3413 ticks/s cap: kv damps them, as in the real demos, where no
arm joint saturates (criterion 3 was amended on 2026-11-05 to ask ±10% only of the unloaded joints). The gripper stroke meets
0.35–0.5 s with no margin, accepted.

Fidelity gap: the sim gripper settles 4 ticks from its command, while the real empty gripper stops 48 ticks
short and a holding one about 70. A policy sees that offset in `observation.state`, and the sim does not
reproduce it.
