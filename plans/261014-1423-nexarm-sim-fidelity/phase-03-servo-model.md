---
phase: 3
title: "Servo model"
status: pending
priority: P1
effort: "5h"
dependencies: [1, 2]
---

# Phase 3: Servo model

## Goal

Joint stiffness, torque, speed and gripper motion match the HX servos and the speeds measured on the real follower.

## Files to Create / Modify

- Modify: `C:/Users/Administrator/code/lerobot-nexarm/sim/description/mjcf/nexarm.xml`
  - Add default classes `hx65hm`, `hx30hm` and `hx12h`:
    - joint damping = stall ÷ no-load speed (1.156 / 0.533 / 0.225);
    - armature 0.028, frictionloss 0.052;
    - `<position kp="998.22" kv="2.731" forcerange="±stall">`.
  - Assign joint 2 to `hx65hm`, joints 1/3/4 to `hx30hm`, and joint 5 to `hx12h`.
  - Gripper: convert HX-12H through r = 10.15 mm.
    - Actuator: kp = 998.22/r², kv = 2.731/r², forcerange ±116 N. This is spec-derived; the user decided to
      lower it only if the resting-cube pick slips or penetrates. <!-- Updated: Validation Session 1 - grip force policy -->
    - Each jaw gets half the reflected armature, damping and frictionloss.
  - Add a comment with the source (Menagerie SO-101 STS3215 prior, vendor stall/speed).
- Modify: `src/lerobot/robots/nexarm_sim/mujoco_backend.py`
  <!-- Updated: user chose per-joint caps fitted from real data; motion_speed=2000 does not cap real moves -->
  - Add a servo setpoint in raw ticks. Each physics substep moves it toward the delayed commanded target by at most `max_speed_ticks_s[j] × dt`.
  - Add an optional acceleration limit.
  - Seed the setpoint in `reset()` and `set_joint_positions()`.
  - The constructor gains `max_speed_ticks_s: tuple[float, ...]` (6 values) and `max_accel_ticks_s2: float | None`.
- Done already: `calibrate_sim.py --exclude-episodes`, and `examples/nexarm/audit_real_dataset.py` for speeds,
  saturation, lag and gripper levels. Record the chosen caps and the dataset revision in `reports/servo-baseline.md`.
- Modify: `src/lerobot/robots/nexarm_sim/config_nexarm_sim.py`
  - Add `max_speed_ticks_s: tuple[float, ...]`, defaulting to the servo spec of 3413 for every joint.
  - Teleop speeds are only lower bounds on the cap, so fit a joint's cap only where the data shows saturation,
    i.e. measured speed falls below commanded speed after lag alignment. `audit_real_dataset.py` finds only the
    gripper saturates (p99.9 about 3700 ticks/s). All arm joints keep the spec of 3413. Record the choice in a comment.
  - Gripper: the audit measures about 3700 ticks/s (saturated), so derive the HX-12H gripper damping from that
    observed speed instead of stall ÷ 3413. The spec-derived 0.225 cannot exceed 3413 ticks/s.
  - Add `max_accel_ticks_s2: float | None = None`, and validate 6 positive values.
- Modify: `src/lerobot/robots/nexarm_sim/nexarm_sim.py` and `src/lerobot/envs/nexarm.py`, plus the `NexArmEnv` config in `src/lerobot/envs/configs.py`: pass the new options through.
- Modify: `src/lerobot/motors/nexarm/nexarm.py`: fix the `CMD_SET_MOTION_PARAMS` comment at line 49 and the
`write_motion_params` docstring. Real data recorded at `motion_speed=2000` reaches 3050 ticks/s on wrist flex,
so neither may claim that CMD 56 caps CMD 97 moves at `motion_speed` ticks/s.
<!-- Updated: Validation Session 1 - record dynamics_model in sim dataset reports -->
- Modify: `src/lerobot/robots/nexarm_sim/mujoco_backend.py`: export `SIM_DYNAMICS_MODEL = "hx_servo_v1"`. It is the single source for the calibration and the dataset reports.
- Modify: `examples/nexarm/sim_dataset_utils.py`
  - Add `"dynamics_model": SIM_DYNAMICS_MODEL` to the report next to `joint_mapping` (line 294).
  - When appending to a dataset whose report has a missing or different `dynamics_model`, log a warning and do not refuse (the user decision is that old datasets stay usable).
- Modify: `src/lerobot/robots/nexarm_sim/calibration.py`
  - Add `dynamics_model: str = SIM_DYNAMICS_MODEL` to `SimCalibration`.
  - `load` refuses a missing or different value with a message to re-run `calibrate_sim.py`.
- Modify: `examples/nexarm/calibrate_sim.py`: its docstring states that armature and the speed cap are fixed priors, not fitted.
- Modify: task timeouts in `pick_place_task.py`, `stack_bowls_task.py` and the env `max_episode_steps`, only if the baseline comparison shows a need.
- Modify: `sim/README.md`: a servo-model section, plus a note that earlier sim datasets and policies used faster dynamics.
- Modify: `docs/source/nexarm/sim_calibration.md`: the new calibration field.
- Modify/Create tests:
  - `tests/robots/test_nexarm_sim.py`: step-speed, gripper-stroke and droop tests.
  - `tests/robots/test_nexarm_sim_calibration.py`: old-file refusal.
  - New scripted-success test with rendering stubbed.

## Tasks & Steps

- [ ] Baseline before any change, recorded in `plans/261014-1423-nexarm-sim-fidelity/reports/servo-baseline.md`:
  - [ ] scripted pick-place and stack-bowls success over seeds 0–19, with render stubbed;
  - [ ] episode durations;
  - [ ] step-test speeds;
  - [ ] droop at the IK grasp poses.
- [ ] Add the servo classes and gripper conversion to the MJCF.
- [ ] **Precondition for the speed fit and the `nexarm.py` comment change:** confirm the real time base.
      Dataset timestamps are frame index ÷ fps, so speeds assume the record loop held 30 Hz. Either:
  - the user confirms that `lerobot-record` never printed "Record loop is running slower";
  - or the user runs one timed step test on the robot (one joint, `perf_counter`) at `motion_speed=2000` and at 0.
    If the loop was slower, rescale the speeds, and revisit the speed-cap decision with the user if the result is about 2000.
  - **Status: met.** The slow-loop warning was printed often, but the 7 episodes that ran the full 15 s
    (440–449 frames; 2, 19, 23, 25, 26, 29 and 37) averaged 29.3–29.9 Hz. They still show wrist flex at
    2700–3050 ticks/s.
    - Fit the caps on those episodes.
    - The timed step test (`examples/nexarm/servo_step_test.py`) is optional confirmation.
- [ ] Run `examples/nexarm/audit_real_dataset.py --repo-id thanhkt/nexarm_stack_bowls --episode-time-s 15` and set the `max_speed_ticks_s` defaults from it.
- [ ] Implement the backend rate limiter and its config plumbing.
- [ ] Gripper check: pick the resting cube with the scripted policy.
  - [ ] If it slips or penetrates more than 1 mm, tune `finger_collision` solref/solimp first and the force limit second.
  - [ ] Document any deviation from the derived values in the XML.
  - [ ] Real target: the closed, empty real gripper settles a median 72 ticks short of the command (interquartile
        range 49–83), per `plans/reports/real-data-261014-1805-nexarm-stack-bowls-analysis.md`. Compare the sim's
        steady-state error against it.
- [ ] Add `dynamics_model` to the calibration, update the calibration tests, and grep for committed calibration JSONs.
- [ ] Record `dynamics_model` in the sim dataset reports. Warn, and do not refuse, when appending across dynamics.
      Extend `tests/scripts/test_generate_nexarm_sim_dataset.py` to check the report field; that test needs the GL
      skip from phase 4, or a stubbed renderer.
- [ ] Re-run the baseline measurements. Adjust timeouts if needed, and record the before/after numbers in the report.
- [ ] Add the tests:
  - [ ] Step test: each joint's sustained 200 ms speed is within ±10% of its `max_speed_ticks_s` cap.
  - [ ] A full gripper stroke takes 0.35–0.5 s.
- [ ] After the servo model lands, run `calibrate_sim.py --repo-id thanhkt/nexarm_stack_bowls`, excluding
      episodes 69–89, with 67–100 ms (2–3 frames) as the starting latency. Record the replay RMS per joint before and
      after in `reports/servo-baseline.md` (acceptance criterion 7).
  - [ ] Droop under 2 mrad at the IK grasp poses.
  - [ ] An old calibration is refused.
  - [ ] Scripted success is at least the baseline.

## Verification

- `uv run pytest tests/robots/test_nexarm_sim.py tests/robots/test_nexarm_sim_calibration.py tests/robots/test_nexarm_stack_bowls.py tests/envs -k nexarm -q`
- `uv run mypy src/lerobot/robots/nexarm_sim src/lerobot/envs/nexarm.py`
- The before/after table in `reports/servo-baseline.md` meets acceptance criteria 3 and 4.

## Risk / rollback

Contact chatter from the stiff gains is the main risk; watch for penetration over 1 mm in the generator runs.
The XML classes and the backend limiter are independent, so either can be reverted alone.
