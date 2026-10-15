---
phase: 0
title: "Real-to-sim joint mapping"
status: pending
priority: P0
effort: "3h + robot session"
dependencies: []
---

# Phase 0: Real-to-sim joint mapping

## Goal
The sim poses the arm the way the real follower is posed for the same raw ticks. Every real-data step (replay
RMS, `calibrate_sim.py`, bowl-layout inference, real-to-sim replays, co-training) depends on this.

## Context
- The real follower stands straight up at raw 2048 (vendor Home; user-confirmed).
- The sim maps `rad = (raw - 2048) * 2π / 4096` with no sign or offset, so 2048 lands on the CAD export pose.
  In that pose the upper arm leans back about 35°, the forearm angles forward and down, and the gripper points
  forward.
- The episode 150 grasp frame confirms the mismatch: see Finding 4 in
  `plans/reports/real-data-261014-1805-nexarm-stack-bowls-analysis.md`.
- Standing the sim arm straight up needs qpos ≈ (0, −35.7°, −150.7°, 24.8°, 0). The elbow value lies outside
  the current MJCF range of ±135°, so the joint ranges must be re-centred together with the mapping.

## Steps
- [ ] User: run `examples/nexarm/probe_joint_mapping.py --port /dev/ttyUSB1` and share
  `outputs/nexarm/joint_mapping_probe.json`.
- [ ] User (machine with OpenGL): run `examples/nexarm/overlay_recorded_frames.py` with `--mapping` on episodes
  0, 150 and one from 69–92, and confirm that the wrist overlays line up.
- [ ] Apply the mapping in `src/lerobot/motors/nexarm/mujoco_mapping.py`:
  - add the per-joint sign and offset, and bump `JOINT_MAPPING_VERSION` to `servo_4096_v2`;
  - re-centre the MJCF joint ranges, and change the `HOME_POSITIONS` meaning if needed;
  - update the inverse kinematics in the scripted generators and `kinematics_dynamics.py` (which uses the same
    mapping);
  - update the URDF/Isaac joint origins.
- [ ] Re-run the bowl and pick-place generators and confirm the success rates. Existing sim datasets record
  `joint_mapping`, so the version check refuses mixing them with v2 data.
- [ ] Re-run `audit_real_dataset.py`, then check that forward kinematics puts real grasps at table height
  (jaw z within 0.02–0.08 m).

## Risk / rollback
This changes what every sim dataset and policy means in joint space, which the version bump is meant to
surface. Roll back by reverting the commit; v1 datasets stay valid under v1.
