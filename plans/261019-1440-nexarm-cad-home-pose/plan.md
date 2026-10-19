---
title: "NexArm CAD zero pose = real Home pose"
description: "Re-pose the Fusion CAD so raw 2048 means the same arm pose in sim and on the real follower, then rebuild everything that depends on it."
status: in-progress
priority: P0
effort: "CAD session + 4h + robot session"
branch: main
tags: [bugfix, sim, cad, handoff]
blockedBy: []
blocks: [261014-1423-nexarm-sim-fidelity]
created: 2026-10-19
---

# NexArm CAD zero pose = real Home pose

Handoff written 2026-10-19 so work can continue on another laptop. It lists every open problem found while
reviewing `sim/`, what was already fixed, and the workflow to finish.

Files in this folder:

| File                              | What it is                                                                                    |
| --------------------------------- | --------------------------------------------------------------------------------------------- |
| `real-follower-home.jpg`          | Photo of the real follower after the desktop app's "All home" (every servo at raw 2048).      |
| `sim-cad-vs-real-home.png`        | Sim side view, front to the right. Left: current CAD at raw 2048. Right: target pose (photo). |
| `follower-calibration-guide.html` | Follower-only probe guide; open it in a browser.                                              |

## Decision (user, 2026-10-19)

Fix the CAD instead of compensating in software: with every Fusion joint at value 0, the arm must be in the
real Home pose. Then raw 2048 maps to the same pose in sim and on hardware, the joint mapping offsets become
about 0, and the MJCF joint ranges are centred on Home without extra work.

This supersedes the context line in `261014-1423-nexarm-sim-fidelity/phase-00-joint-mapping.md` that says the
real follower stands straight up at raw 2048. The photo shows it does not.

## The real Home pose (target)

From `real-follower-home.jpg`, at raw 2048 on every servo:

- base pan facing the work area;
- upper arm vertical;
- elbow bent 90°, forearm horizontal, pointing toward the work area;
- wrist straight, gripper horizontal and in line with the forearm;
- gripper half open at raw 2048 (between open 1195 and closed 2833); the CAD zero is closed (raw 2833,
  jaws touching), which is the jaw slide's zero;
- wrist camera on top of the gripper.

The pose as built, in the old CAD zero's MuJoCo joint frames (sim front is -y). The earlier fit (−35.7° /
−60.7° / +24.8°) made the joint-to-joint pivot line vertical; the upper-arm tube is offset from that line, so
making the tube itself vertical adds about 9.3° to the shoulder and takes it back at the elbow:

| Joint           | MuJoCo joint               | Old-zero qpos of Home | Result                         |
| --------------- | -------------------------- | --------------------: | ------------------------------ |
| `shoulder_pan`  | `joint_1_base_to_link_1`   |                  0.0° | unchanged                      |
| `shoulder_lift` | `joint_2_link_1_to_link_2` |                −45.0° | upper-arm tube vertical        |
| `elbow_flex`    | `joint_3_link_2_to_link_3` |                −69.9° | forearm horizontal             |
| `wrist_flex`    | `joint_4_link_3_to_link_4` |                +24.9° | gripper in line                |
| `wrist_roll`    | `joint_5_link_4_to_link_5` |                  0.0° | camera on top                  |

`repose_models.py` (this folder) fits these from the new Fusion extract and prints them.

## Outcome (2026-10-19)

- Fusion `NexArm` v25: every joint at 0 is Home, gripper zero is closed (pinion 0 to −145.5°, slides 0 to
  25.39 mm). Arm limits are symmetric about Home: ±135° / ±120° / ±135° / ±100° / ±180°, matching the MJCF
  ranges, so problem 2 needed no range change. `NexArm-Sim-Single-Arm` v3 references v25.
- `repose_models.py` moved every hand-kept arm-frame value with its link (MJCF joints, finger and gripper
  collisions, sites, wrist camera, `ready` keyframe; URDF/xacro joint and visual origins) and dropped the
  ±25.4 mm jaw visual offsets. `apply_fusion_extract.py` then passed its axis-line check unchanged (no
  override flag needed, problem 3) and merged meshes, inertials, collision boxes and the Isaac scene JSON.
- `ready` is now `0 -0.0154 0.3197 1.0655` (same physical pose). `JOINT_MAPPING_VERSION` is `servo_4096_v2`.
- Both generators had a jump-to-target bug when the task ended mid-stage (settling sent the stage's final
  waypoint in one frame); it surfaced as `quality_jerk_spike` with the new IK branches and is fixed.
- Not taken from the extract: the Fusion C270 mount is 100 mm higher than the committed `front` camera.
  Decision (user): keep the sim's `front` pose; the merge no longer moves `front` (`CAMERAS` in
  `apply_fusion_extract.py`).
- Still open in problem 5: the Isaac USDs need `build_usd.py` on a machine with Isaac Sim, and
  `sim/mobile_nexarm/generate_assets.py` refuses to run because `spec.json` already disagreed with the MJCF
  at HEAD (`joint_3` axis −x vs +x, gripper slide range). The mobile arm is unused, so its CI workflow is
  manual-only until the spec is updated. The Rerun demo `sim/nexarm_demo.rrd` was rebuilt.

## Open problems

Status key: **open** needs work, **blocked** waits on another item, **known** accepted for now.

| #   | Problem                                                                                                                                                                                                                                                                                                              | Status             | Where                                                                                         |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | --------------------------------------------------------------------------------------------- |
| 1   | CAD zero pose is not the real Home pose. At raw 2048 the sim upper arm leans back ~36°, the elbow is folded and the forearm slopes down. Every sim dataset, IK target and policy is posed differently from hardware.                                                                                                 | fixed              | Fusion design → `sim/description/`                                                            |
| 2   | MJCF joint ranges are centred on the CAD pose, not on raw 2048. With the current CAD the sim elbow can rise only ~74° from Home and cannot reach straight up; shoulder lift and wrist flex lose ~36° and ~25° on one side. `probe_joint_mapping.py` does not warn because each offset is inside its range.           | fixed              | `sim/description/mjcf/nexarm.xml` joint `range`                                               |
| 3   | `sim/apply_fusion_extract.py` refuses any extract whose joint axis lines move more than 1 mm from the committed MJCF. A re-posed CAD moves them on purpose, and there is no override flag.                                                                                                                           | fixed (no flag)    | `sim/apply_fusion_extract.py` (check near line 254)                                           |
| 4   | Values tuned against the old zero pose: the `ready` keyframe (`0 -0.8 -0.9 1.5 0 0`), scripted generator waypoints and IK seeds, the wrist camera placement, and the hard-coded range table in `sim/test_sim_models.py` (lines ~254–280).                                                                            | fixed              | `sim/description/mjcf/nexarm.xml:186`, `examples/nexarm/generate_*`, `sim/test_sim_models.py` |
| 5   | Derived assets must be rebuilt after the CAD change: URDF/xacro, Isaac scene JSON and USDs, the mobile bimanual assets, and the Rerun demo.                                                                                                                                                                          | fixed (Isaac USD and mobile assets open, see Outcome) | `sim/isaac_lab/`, `sim/mobile_nexarm/generate_assets.py`, `sim/nexarm_demo.rrd`               |
| 6   | Joint directions (signs) are still unmeasured on the real arm. The CAD fix sets the zero, not the direction each servo turns.                                                                                                                                                                                        | open (needs robot) | `examples/nexarm/probe_joint_mapping.py` → `outputs/nexarm/joint_mapping_probe.json`          |
| 7   | Fitting the sim's motion (`calibrate_sim.py`) needs recordings with commands and measured positions. Only the leader can produce them today; desktop-app recordings with torque off have positions only. A follower-only recorder (scripted sweeps saved as a LeRobot dataset) would remove the need for the leader. | open (not written) | `examples/nexarm/calibrate_sim.py`                                                            |
| 8   | Existing sim datasets and sim calibrations encode the old CAD pose. They must be regenerated after the CAD change.                                                                                                                                                                                                   | open (regenerate)  | `outputs/datasets/`, `outputs/calibration/`, any Hub sim datasets                             |
| 9   | The Isaac Lab model and stack-bowls task have never been run in Isaac. Only the asset sync check passes.                                                                                                                                                                                                             | known              | `sim/isaac_lab/`                                                                              |
| 10  | The pyupgrade pre-commit hook corrupts the "Reach Undulation" f-string in `examples/nexarm/visualize_rerun_sim.py` (adds a backtick each run). Present on `main` too. Run other hooks with `SKIP=pyupgrade` on that file until fixed.                                                                                | known              | `examples/nexarm/visualize_rerun_sim.py:577`                                                  |
| 11  | GPU (EGL) rendering differs by a few pixels between identical resets, so camera observations are not bit-deterministic. Physics state is. The gym checker test now checks determinism on the `state` observation only.                                                                                               | known (handled)    | `tests/envs/test_nexarm_env.py`                                                               |

## Fixed in this session (uncommitted on the old laptop)

Commit and push these before switching laptops, or they stay behind:

| File                                                     | Change                                                                                                                                |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/envs/test_nexarm_env.py`                          | Gym checker runs on `state` (full determinism) and on `pixels_agent_pos` (API only, marked nondeterministic).                         |
| `src/lerobot/envs/nexarm.py`                             | `grasp_mode` typed as the task's `Literal`; mypy passes.                                                                              |
| `examples/nexarm/probe_joint_mapping.py`                 | Uses a hand-posed straight-up reference and its measured raw values instead of assuming raw 2048 is upright; turns torque off itself. |
| `tests/scripts/test_nexarm_real2sim_tools.py`            | New test for an arm whose Home is bent (upright elbow reads 3072); torque-off asserted.                                               |
| `examples/nexarm/visualize_rerun_sim.py`                 | The pick-and-place demo now drives the arm with the scripted IK episode (it used to sit still) and replays the first successful seed. |
| `sim/README.md`, `docs/source/nexarm/sim_calibration.md` | Joint-mapping section rewritten; no longer claims the real arm is upright at 2048.                                                    |

Not in git: `sim/nexarm_demo.rrd` (git-ignored, regenerate on the new laptop) and anything under `outputs/`.

## Workflow

### 0. Set up the new laptop

```bash
git clone https://github.com/khanhthanhdev/lerobot-nexarm.git && cd lerobot-nexarm
git lfs install && git lfs pull                       # STL meshes are in LFS
git submodule update --init --recursive               # TurboVLA, bspline-policy
uv sync --locked --extra all
```

Check the baseline before changing anything:

```bash
MUJOCO_GL=egl uv run python sim/test_sim_models.py
MUJOCO_GL=egl uv run pytest tests -k "nexarm and sim or nexarm_env or mujoco" -q
uv run python sim/isaac_lab/build_isaac_assets.py --check
```

Expected: all model checks pass, 123+ passed / 0 failed, Isaac assets up to date. On Windows or macOS drop
`MUJOCO_GL=egl`.

### 1. Re-pose the CAD (problem 1)

In Fusion, with every joint at value 0, pose the arm as in `real-follower-home.jpg` (table above). Keep the
occurrence and joint names that section 11 of `sim/README.md` requires. Joint axes must stay where they are on
each link; only the zero pose changes.

### 2. Export and merge (problem 3)

```bash
uv run python sim/run_export.py --skip-apply          # extract only
uv run python sim/apply_fusion_extract.py --check     # expect: axis line ... mm from the MJCF axis line
```

The check is expected to fail with axis-line errors. Add an explicit opt-in flag to `apply_fusion_extract.py`
(for example `--accept-new-kinematics`) that skips only the axis-line comparison and prints what moved, then:

```bash
uv run python sim/apply_fusion_extract.py --accept-new-kinematics
uv run python sim/isaac_lab/build_usd.py              # Isaac USDs (needs Isaac Sim)
```

The merge already refits collision boxes, updates inertials in MJCF/URDF/xacro, rewrites `workcell.xml`,
places cameras and runs `build_isaac_assets.py`.

### 3. Re-centre ranges and fix pose-dependent values (problems 2 and 4)

- Joint ranges in `nexarm.xml` stay symmetric (servo travel around raw 2048): ±135° pan, ±120° lift,
  ±135° elbow, ±100° wrist flex, ±180° roll. With the new zero they are now centred on Home. Confirm the arm
  does not self-collide at the range ends.
- Re-derive the `ready` keyframe for the new zero.
- Check the wrist camera pose against the real wrist camera.
- Update the range table in `sim/test_sim_models.py` if it changed.
- Bump `JOINT_MAPPING_VERSION` in `src/lerobot/motors/nexarm/mujoco_mapping.py` so datasets made with the old
  CAD are refused.

### 4. Verify the sim

```bash
MUJOCO_GL=egl uv run python sim/test_sim_models.py
MUJOCO_GL=egl uv run pytest tests -k "nexarm and sim or nexarm_env or mujoco or real2sim" -q
uv run python sim/isaac_lab/build_isaac_assets.py --check
uv run python examples/nexarm/simulate.py             # eyeball: sliders at 2048 look like the photo
```

Smoke-test both scripted generators; they must still produce accepted episodes:

```bash
MUJOCO_GL=egl uv run python examples/nexarm/generate_sim_dataset.py --repo-id=local/smoke \
  --root=outputs/datasets/smoke --episodes=1 --max-attempts=3 --camera-width=96 --camera-height=96 --no-video
MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py --repo-id=local/smoke2 \
  --root=outputs/datasets/smoke2 --episodes=1 --max-attempts=5 --camera-width=96 --camera-height=96 --no-video
```

If IK fails at a stage, adjust the generator waypoints or IK seeds for the new zero.

### 5. Rebuild derived assets (problem 5)

```bash
uv run python sim/mobile_nexarm/generate_assets.py
MUJOCO_GL=egl uv run python sim/visualize_rerun.py --mode all --view save   # writes sim/nexarm_demo.rrd
uv run rerun sim/nexarm_demo.rrd                                            # check the pick-and-place succeeds
```

### 6. Measure joint directions on the real follower (problem 6)

Follow `follower-calibration-guide.html`:

1. Desktop app: run All home, then close the app (it holds the serial port).
2. `uv run python examples/nexarm/probe_joint_mapping.py --port /dev/ttyUSB1` (or the `COM` port on Windows).
3. Support the arm; pose it straight up (raise the forearm 90° from Home); do the quarter turns.

With the fixed CAD every offset should be close to 0° (wrist roll 0° or ±90° depending on the camera side).
An offset near ±90° means a wrong answer in the probe; redo it. Any sign of −1 means that servo turns opposite
to the sim joint: keep the JSON and pass it with `--joint-mapping` / `joint_mapping_path`.

### 7. Regenerate data (problem 8)

Regenerate sim datasets and, once motion recordings exist, the sim calibration with the new CAD and mapping:

```bash
MUJOCO_GL=egl uv run python examples/nexarm/generate_sim_dataset.py \
  --repo-id=local/nexarm_sim_pick_place --root=outputs/datasets/nexarm_sim_pick_place \
  --episodes=20 --joint-mapping outputs/nexarm/joint_mapping_probe.json
```

### 8. Optional: motion fit without the leader (problem 7)

Write a follower-only recorder that commands per-joint sweeps (slow, fast, quick reversals) through
`NexArmFollower` and saves `action` + `observation.state` as a LeRobot dataset, then run
`calibrate_sim.py --repo-id <that dataset> --joint-mapping outputs/nexarm/joint_mapping_probe.json`.

## Reproduce the pose fit

Prints the target qpos for the photo pose and the upright pose against the current MJCF:

```bash
uv run python - <<'EOF'
import numpy as np
from examples.nexarm import probe_joint_mapping as probe
sim = probe.SimArm("sim/description/mjcf/nexarm.xml")
targets = {1: np.array([0, 0, 1.0]), 2: np.array([0, -1.0, 0]), 3: np.array([0, -1.0, 0])}
q = np.zeros(5)
for _ in range(5):
    for i in (1, 2, 3):
        sim.forward(q)
        tip = sim.anchor(i + 1) if i < 3 else sim.jaw()
        a = sim.axis(i); link = tip - sim.anchor(i)
        cur = link - a * (link @ a); want = targets[i] - a * (targets[i] @ a)
        q[i] += np.arctan2(a @ np.cross(cur, want), cur @ want)
for name, value in zip(probe.ARM_JOINTS, np.degrees(q)):
    print(f"{name:14s} {value:+7.1f} deg")
EOF
```

After the CAD fix this should print about 0° for every joint.

## Done when

- At raw 2048 the sim arm matches `real-follower-home.jpg` (all fitted angles within ±3°).
- Sim test suite, model checks and Isaac asset check pass; both generators accept episodes.
- `probe_joint_mapping.py` on the real follower reports offsets within ±5° of 0 (wrist roll may be ±90°).
- `sim/nexarm_demo.rrd` regenerated and the pick-and-place reaches SUCCESS.
