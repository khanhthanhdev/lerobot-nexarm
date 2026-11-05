---
title: "NexArm single-arm sim fidelity"
description: "Make the single-arm MuJoCo twin match the real follower's servos, object masses and asset layout."
status: in-progress
priority: P1
effort: 14h
branch: main
tags: [bugfix, infra, tech-debt]
blockedBy: [261019-1440-nexarm-cad-home-pose]
blocks: []
created: 2026-10-14
---

# NexArm single-arm sim fidelity

Source review: `plans/reports/sim-review-261014-1356-nexarm-sim-assets.md`.

## Outcome

The single-arm MuJoCo twin behaves like the real follower within known hardware specs:

- servo speed, stiffness and torque match the Hiwonder HX servos;
- objects have their real masses;
- the model loads from one canonical asset tree on any OS.

Sim datasets and policies then see the same kinematic timing as hardware.

## Decisions already made (user, 2026-10-14)

- ~~The speed limit mirrors the follower config.~~ Superseded on 2026-10-14 by real data
  (`plans/reports/real-data-261014-1805-nexarm-stack-bowls-analysis.md`). All 163 real episodes were recorded
  with `motion_speed=2000` and `motion_acc=100` (user-confirmed). Even so, the elbow sustains 2300–3200 ticks/s
  and the gripper about 4000, so CMD 56 does not cap CMD 97 moves at `motion_speed` ticks/s. This assumes the
  Slow-loop warnings were frequent, but 7 episodes that ran the full 15 s (about 30 Hz average) confirm it.
  - **New decision:** fit per-joint speed caps from the real demos and fit latency and gains with
    `calibrate_sim.py`. A cap is fitted only for joints that saturate. `audit_real_dataset.py` (lag-aligned) finds only
    the gripper saturates (about 3700 ticks/s); every arm joint tracks its command, so arm joints keep the servo spec of 3413 ticks/s.
  - `motion_speed` is not used as the sim cap. The acceleration ramp stays optional and off by default.
- Scope is the single arm first. Rebuilding the mobile bimanual sim from the fixed arm is a follow-up plan.
- `sim/description/` becomes canonical. `sim/fusion_export/` stays raw Fusion exporter output that nothing loads
  at runtime.
- Bowl mass is 60 g (user, 2026-11-04). This supersedes the 55 g first noted; the sim has used 60 g since
  e8a582ad6, and `test_bowls_weigh_60_g_and_start_apart` checks it.

## Hardware facts used

- Follower servos (vendor BOM):
  - HX-65HM: joint 2 (6.37 N·m).
  - HX-30HM: joints 1, 3 and 4 (2.94 N·m).
  - HX-12H: joint 5 and the gripper (1.18 N·m).
  - All are rated about 0.19–0.2 s/60° at 11.1 V. This matches the current `forcerange` values.
- Servo dynamics prior: the STS3215 class identified in MuJoCo Menagerie SO-101 (same 30 kg·cm class). Values
  were verified against the raw `so101.xml`: kp 998.22, kv 2.731, armature 0.028, frictionloss 0.052.
- Damping per servo is stall torque ÷ no-load speed, a linear DC-motor torque–speed line: 1.156 for HX-65HM,
  0.533 for HX-30HM and 0.225 for HX-12H. Copying SO-101's 0.6 leaves the HX-12H gripper unable to move at
  3 rad/s, which was measured.
- Gripper pinion radius = 0.0255 m ÷ (1638 ticks × 2π/4096) = 10.15 mm. Rotary servo values convert through it:
  force = τ/r ≈ 116 N, kp = kp_rot/r², armature = arm_rot/r².

## Feasibility already measured (scratch experiments, 1.961 ms backend timestep)

- The stiff servos are stable, and final tracking error is under 1 mrad.
- Gravity droop at IK grasp poses falls from 9–12 mrad to under 1 mrad.
- The raw-tick rate limiter holds the 95th-percentile joint speed at its cap (tested at 2000 ticks/s, 3.07 rad/s).
  Without the accel ramp, move starts spike to about 4.5 rad/s for roughly 0.1 s.
- Open: a free-floating cube is squeezed out by the stiff 116 N gripper. A cube resting on the table is untested;
  phase 3 handles it.

## Phases

| #   | Phase                                                                         | Status                                      | Effort | Depends on |
| --- | ----------------------------------------------------------------------------- | ------------------------------------------- | ------ | ---------- |
| 0   | [Real-to-sim joint mapping](phase-00-joint-mapping.md)                        | pending (needs a robot session)             | 3h     | —          |
| 1   | [Canonical asset tree](phase-01-canonical-asset-tree.md)                      | done (on `main` before session 3)           | 3h     | —          |
| 2   | [Scene and object fixes](phase-02-scene-object-fixes.md)                      | done (`top` fy read on the device pending)  | 2h     | 1          |
| 3   | [Servo model](phase-03-servo-model.md)                                        | done, except the replay RMS (needs phase 0) | 5h     | 1, 2       |
| 4   | [Collision refit, URDF/Isaac sync, GL skip](phase-04-collision-refit-sync.md) | done, pending the user's viewer review      | 4h     | 3          |

## Acceptance criteria

1. Exactly one robot MJCF exists (`sim/description/mjcf/nexarm.xml`). It is loaded by `NexArmSimConfig`, the
   gym env, `kinematics_dynamics.py`, the examples and the tests. No symlinks remain under `sim/`, and every scene
   compiles with any working directory.
2. Each bowl body mass is 0.060 kg. Scene keyframes have no contact deeper than 1 mm, and the corrected `ready`
   pose has no self-contact.
3. Step test with the default config:
   - the unloaded joints' (pan, roll) sustained (200 ms) speed is within ±10% of their `max_speed_ticks_s` cap,
     and no joint exceeds its cap by more than 2% (the gravity-loaded joints run under it, as on the real arm);
   - a full gripper stroke takes 0.35–0.5 s (real: about 0.4 s at 30 Hz; rescale if the time-base check finds a
     slower loop);
   - droop at the IK grasp poses is under 2 mrad.
4. Scripted pick-place and stack-bowls success rates on fixed seeds are at least the baseline recorded before
   phase 3.
5. Loading a `SimCalibration` without the new `dynamics_model` value fails with a re-run instruction.
6. `uv run pytest tests -k "nexarm or sim_exports" -q` passes on this host. Render tests skip, not fail, without
   OpenGL. `generate_assets.py --check` fails only on the already-known axis drift.
7. After phase 0, report the replay RMS per joint before and after phase 3, using the real task demos in
   `thanhkt/nexarm_stack_bowls`. Exclude episodes 69–89 until their zero lag is explained. Contact biases this
   metric, so it is a check, not a fit. Free-air sweep recordings, if made later, replace it.
   <!-- Updated: real dataset analysed in plans/reports/real-data-261014-1805-nexarm-stack-bowls-analysis.md -->

## Non-goals

- Mobile bimanual sim (follow-up plan; its `spec.json` source path is repointed only).
- A physical bowl grasp, replacing the teleport assist.
- Gear backlash modelling.
- Changing the Fusion exporter's own output.

## Risks

- A slower arm can lengthen scripted episodes past their timeouts. Mitigation: measure the baseline first and
  raise timeouts only as needed.
- Stiff gains can chatter in contact. Mitigation: check penetration in the generator runs, and tune
  `finger_collision` solref before touching gains.
- Old sim datasets and policies were produced with the faster dynamics. This will be documented and not migrated.

## Validation Log

### Session 1 (2026-10-14)

Questions asked: 4, plus 3 decisions made before planning: speed cap, scope, canonical tree.

| Topic                | Decision                                                                                                       | Propagated to |
| -------------------- | -------------------------------------------------------------------------------------------------------------- | ------------- |
| Gripper force        | Start at the spec-derived 116 N. Lower it, or soften pads, only if the resting-cube pick slips or penetrates.  | phase 3       |
| Old sim datasets     | Record `dynamics_model` in new dataset reports. Old datasets stay usable, and an append across dynamics warns. | phase 3       |
| `sim/fusion_export/` | Untrack and gitignore it. The exporter still writes there for manual review.                                   | phase 1       |
| Collision refit      | Keep it in the plan. The user reviews it in the viewer before merge.                                           | phase 4       |

### Verification Results

- Claims checked: 35 (file paths, symbols, test names, line references).
- Verified: 35 | Failed: 0 | Unverified: 0. Tier: Standard (4 phases).
- Gaps found and added before validation: `src/lerobot/motors/nexarm/kinematics_dynamics.py:92`,
  `tests/test_sim_exports.py`, `.gitignore:186-190`, `sim/mobile_nexarm/spec.json:5-6` (read by the mobile CI
  `--check`) and `docs/source/nexarm/sim_calibration.md:158`.
- `validate_source` compares only axes and ranges, so the servo-gain changes do not affect the mobile check.

### Whole-Plan Consistency Sweep

- `dynamics_model` has one source (`SIM_DYNAMICS_MODEL` in `mujoco_backend.py`). Both the calibration and the
  dataset reports reference it.
- The fusion_export decision is consistent across plan.md (decisions, non-goals) and phase 1 (files, `.gitignore`).
- The gripper force figure (116 N) matches between plan.md hardware facts and phase 3.
- No unresolved contradictions.

### Session 3 (2026-11-04): stiffness from real data, start phases 2–3

Phase 1 was already done on `main` (no symlinks, one `bowl.stl`, every default under `sim/description/`,
`tests/robots/test_nexarm_sim_assets.py`). The phase files still name the pre-plugin `src/lerobot/...` paths;
the code now lives under `packages/` and `tools/nexarm/`. Baseline: `reports/servo-baseline.md`.

| Topic         | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Propagated to |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Joint gains   | Keep `kp=998.22` (stiff hold: the real arm's empty-gripper steady error is a median 0–2 ticks), but set `kv` per joint so the small-step time constant `(kv + damping) / kp` matches the real first-order fit (0.055–0.112 s; gripper 0.207 s). This replaces the SO-101 `kv=2.731`.                                                                                                                                                                                                                                                                                                            | phase 3       |
| Goal          | Sim-to-real (train in sim, deploy on the real follower). The `front` camera stays at its current pose (user, 2026-11-04).                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | plan          |
| Accel cap     | On by default, replacing "the acceleration ramp stays optional and off": 20k ticks/s² on the five arm joints, none on the gripper. Without it the stiff arm's jerk shakes the bowl out of the physical grasp (stack-bowls 0/20). The value was tuned on that grasp (10k–40k pass, 80k fails) and is consistent with the real demos' arm p99 of 12.6k–28.8k ticks/s², which cannot tell 10k from 40k at 30 fps. Evidence: `reports/servo-baseline.md`.                                                                                                                                           | phase 3       |
| Bowl mass     | 60 g, not 55 g (user). The sim already used 60 g.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | plan, phase 2 |
| Phase 2 scope | Done: the keyframes moved into `scene.xml` and `bowl_stack_scene.xml` with full qpos; the bowls already spawn at the task slots; `cam_mount` already has collision boxes. Not done: the `top` fovy, which needs a `pyrealsense2` read. Dropped: the floor `material="groundplane"`. It is a dark checker (rgb 0.1–0.4), and domain randomization sets the floor `geom_rgba` to 0.65–0.95, which multiplies the texture, so the floor would render near black and its colour randomization would mostly vanish. Match the real floor (colour or photo texture) instead, once a reference exists. | phase 2       |
| Criterion 3   | **Amended (2026-11-05).** The user replied "Continue to done" to this proposal; taken as acceptance. The gravity-loaded joints (lift, elbow, wrist flex) reach 74–81% of the cap, because kv damps them, as in the real demos, where no arm joint saturates. Criterion 3 now asks ±10% only of the unloaded joints and "never above the cap ×1.02" of the rest, as the tests check. The gripper stroke (0.5 s) is accepted at the edge of 0.35–0.5 s.                                                                                                                                           | phase 3       |
| Gripper force | Stays at the spec-derived 116 N; jaw pads use a direct contact stiffness (`solref="-230000 -150"`). With the accel cap both scripted tasks are 20/20.                                                                                                                                                                                                                                                                                                                                                                                                                                           | phase 3       |

### Session 4 (2026-11-05): phase 4, `top` fovy, calibration

| Topic            | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Propagated to   |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `top` camera     | User: it is a D435i. `fovy` 42.6° in both scenes from the nominal fy 615 px at 640×480; the device fy read and pose check are still to do on the robot PC.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | phase 2, README |
| Phase 4          | The refit and URDF/Isaac sync had already landed; measured coverage 91–100%, no self-contact in 40 scripted episodes. Added the OpenGL skip helper (CI fails instead) and the IK-pose self-contact test. Awaiting the user's viewer review.                                                                                                                                                                                                                                                                                                                                                                                  | phase 4         |
| Calibration      | `calibrate_sim.py` now keeps `kp` fixed (`--fit-kp` opts in): a fitted kp hit the 4× bound and dropped scripted stack-bowls to 6/20 (4/20 with kp alone). Refit with `--exclude-episodes 1,3,12,19,69-92` (the replays that hit the floor or the back panel): delay 0, every arm-joint RMSE within 1 tick of the nominal model, so damping and friction are poorly identified by these demos (several scales sit at a bound). Both scripted tasks pass with it and domain randomization on (10/10 each). Output `outputs/calibration/nexarm_sim_calibration_hx_servo_v1.json`, copied to vishc-server-2 beside the old file. | phase 3         |
| Phase 0 evidence | Posing the sim at real recorded states: episode 19 puts the gripper 39 mm into the back panel (it replays 280–480 ticks off), and episodes 1, 3 and 12 go 12–25 mm below the floor. Flipping the shoulder-pan sign removes the episode-19 hit only. Check the pan sign and the mount height on the robot before trusting replay fits.                                                                                                                                                                                                                                                                                        | phase 0         |

#### kp choice (user asked for the best-fitting kp, 2026-11-05)

`--fit-kp` (with `--exclude-episodes 1,3,12,19,69-92`) gives kp scales 1.41/2.83/3.67/4.0/2.83 (pan, lift, elbow, wrist flex, roll).
Scaling them as `fitted ** alpha` (damping and friction from the same fit; the first 20 kept episodes are
0–23 minus 1, 3, 12 and 19, and the holdout is 20–23; 20 seeds per task):

| kp                              | Holdout RMSE mean (pan/lift/elbow/wflex/roll) | Pick-place | Stack-bowls |
| ------------------------------- | --------------------------------------------- | ---------: | ----------: |
| servo model (alpha 0)           | 17.4 (6.4/22.7/26.9/27.5/3.4)                 |      20/20 |       20/20 |
| alpha 0.25                      | 13.2                                          |      20/20 |       18/20 |
| alpha 0.5                       | 11.6 (6.9/15.2/15.5/15.0/5.5)                 |      20/20 |       12/20 |
| alpha 0.75 / 1.0                | 12.2 / 13.6                                   |      20/20 |  4/20, 0/20 |
| **alpha 0.5, elbow alpha 0.25** | **12.4 (6.9/14.3/20.1/15.2/5.5)**             |  **20/20** |   **20/20** |

Only the elbow breaks the grasp on its own (16/20 with just its kp raised; the other four joints alone stay
20/20), and lowering the acceleration cap does not recover it (13/20 at 15k, 15/20 at 10k). Chosen: the last
row, kp scales 1.19/1.68/1.38/2.0/1.68. With domain randomization on (generator default) it gives pick-place
20/20 and stack-bowls 19/20. Check videos (3 episodes per task, front | top | wrist):
`outputs/videos/sim_{stack_bowls,pick_place}_kp_check.mp4`. The user reviewed both videos and found them fine
(2026-11-05).

Where the kp lives (the user asked for the best option): the calibration is tracked as
`sim/calibration/nexarm_sim_calibration_hx_servo_v1.json`, and the MJCF keeps `kp=998.22`. What was approved is
the whole calibration (kp with the fitted damping, friction and spreads), so shipping the file ships exactly
what was tested. Baking kp into the MJCF would invalidate the kv derivation in the servo comment, would need a
`dynamics_model` bump so that existing calibrations do not apply the scales twice, and would ship an untested
combination of kp with the nominal damping.

### Session 2 (2026-10-14): real dataset

Source: `plans/reports/real-data-261014-1805-nexarm-stack-bowls-analysis.md`.

| Topic              | Decision                                                                                                                                                                                         | Propagated to                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| Speed cap          | Fit per-joint caps from the real demos, only for saturating joints. `motion_speed` is not a real cap; confirmed on the 7 full-length (about 30 Hz) episodes despite frequent slow-loop warnings. | plan decisions, criterion 3, phase 3 |
| Recording settings | All 163 episodes used `motion_speed=2000`, `motion_acc=100` (user).                                                                                                                              | plan decisions                       |
| Replay metric      | Use the task demos for criterion 7, excluding episodes 69–89.                                                                                                                                    | criterion 7, phase 3                 |
| Camera             | The dataset has no `top` stream; the D435i fovy still needs a `pyrealsense2` read.                                                                                                               | non-goals removed, phase 2 unchanged |
