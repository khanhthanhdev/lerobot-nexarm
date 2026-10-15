# NexArm simulation review: model, settings, meshes

Date: 2026-10-14. Scope: `sim/` assets (MJCF, URDF, meshes, scenes, Isaac config), `src/lerobot/robots/nexarm_sim/`,
`src/lerobot/envs/nexarm.py`, `src/lerobot/motors/nexarm/mujoco_mapping.py`, and the bimanual mobile sim.
Method: read the sources, then load the models in MuJoCo 3.8.1 and measure them. This host has no OpenGL, so
no images were rendered. Camera checks are pinhole projections.

## What is sound

- URDF and MJCF kinematics agree exactly. Over 200 random joint poses, all 11 link meshes land in the same world
  pose in both models (0.0 mm error).
- The raw-to-joint mapping is centralized in `mujoco_mapping.py`. It is versioned (`servo_4096_v1`), and
  calibrations recorded with a different mapping are refused.
- Gripper geometry matches the mapping. At slide 0 (raw 2833, closed), the jaw pads touch (0.0 mm gap). At slide
  0.0255 (raw 1195, open), the gap is 50.9 mm. The visual fingertips agree with the collision pads (0.0 mm closed,
  51.0 mm open), so grasp images match the physics.
- Declared link inertias agree with uniform-density mesh centroids within 1–8 mm, except `cam_mount` (23 mm,
  which is plausible if the camera module's mass is included).
- Across a sweep of arm poses, gravity torque stays well within actuator `forcerange` (worst case: joint 2 at 2.04 Nm against
  6.37 Nm; joint 3 at 0.73 Nm against 2.94 Nm).
- All three cameras see the workspace. Front and top both frame the cube and the target. The wrist camera sees
  the pinch site 58 mm ahead and the cube near the image centre.
- Bowl-task reset enforces a spacing of at least 140 mm and rejects overlapping layouts.
- The sim feature contract (joint keys plus `front`, `wrist` and `top` at 640×480) matches the camera set in
  `examples/nexarm/record.py`. This was checked statically, because the runtime contract test needs OpenGL.

## Findings (most severe first)

### 1. Bowls weigh 705 g each (high, verified)
`sim/fusion_export/bowl_stack_scene.xml:42` (and the same line pattern for the blue and black bowls). Each bowl
has 33 collision proxies at 1.818 g, which totals 60 g as intended. However, the visual mesh geom has no `mass`
or `density`, so MuJoCo fills the mesh at the default density of 1000 kg/m³ and adds about 645 g. Measured body
mass: 704.7 g. The assisted grasp teleports a held bowl, so the arm never carries that load. Pushes and contacts
against a 0.7 kg bowl are still unrealistic, and domain randomization scales the inflated mass.
Fix: set `mass="0"` (or `density="0"`) on the three `*_visual` geoms, or give the bowl bodies an explicit
`<inertial>`.

### 2. The `ready` keyframe is in deep self-collision (medium, verified)
`sim/description/mjcf/nexarm.xml:189` (and the identical `fusion_export/NexArm-sim.xml`). The pose
`(0, -0.5236, 1.0472, -0.5236, 0)` folds the forearm back into the base. The jaws sit 31.5 mm inside `link_1`,
and stepping from it leaves the elbow 111 mrad off target. The mirrored pose `(0, +0.5236, -1.0472, +0.5236, 0)`
is collision-free, so the signs look copied from a different arm convention. No code loads this keyframe, but the
MuJoCo viewer's "load key" does.
Fix: flip the signs of joints 2–4 in that keyframe.

### 3. Keyframes break any scene with free objects (medium, verified)
The keyframes declare 8 qpos values, but `scene.xml` has nq=15. MuJoCo zero-pads the rest, which places the cube
at the world origin with an all-zero quaternion, 10 mm inside the base and the floor. The backend uses
`mj_resetData`, so training is unaffected; only manual viewer use is.
Fix: move the keyframes into each scene file with the full qpos, or drop them from the robot file.

### 4. Joint and gripper speeds are unlimited in the MJCF (medium, verified)
`nexarm.xml` actuators set no velocity limit. Measured step responses:
- Joint 1: peak 16 rad/s.
- Joint 2: peak 12 rad/s.
- Joint 5: peak 25 rad/s.
- Gripper: full close in 78 ms.

The URDF and `sim/isaac_lab/nexarm_cfg.py` cap the joints at 5.24–5.51 rad/s and the jaws at 0.042 m/s (a full
stroke of about 0.6 s). Sim calibration (`calibration.py`) fits only kp, damping, frictionloss, and latency, so
it cannot recover a speed cap. Teleop at 30 fps mostly sends small steps, which hides this. Policy chunks with
large deltas will move 2–5× faster in sim than on hardware.
Fix: rate-limit the commanded target per physics step in the backend (servo speed), or move to
`intvelocity` / velocity-clamped actuators.

### 5. Position actuators droop under gravity (low/medium, verified)
The actuators are PD (kp plus kv) with no integral term, so gravity leaves a steady-state offset:
- Joint 2: up to about 25 mrad (2.04 Nm / kp 80), roughly 16 servo ticks. A clean step from home to q2=1.0
  settled 9.8 mrad short.
- Joint 3: up to about 18 mrad.

Real bus servos hold position more stiffly, so this is a pose-dependent offset between sim and real observations.
`kp_scale` in the calibration can absorb part of it, but only for fitted joints.

### 6. Duplicated asset trees (medium, maintainability, verified)
`sim/description/mjcf/nexarm.xml` is byte-identical to `sim/fusion_export/NexArm-sim.xml` (after whitespace).
`scene.xml` matches as well. `sim/description/urdf/nexarm.urdf` is identical to `sim/fusion_export/nexarm.urdf`.
All 15 STLs in `sim/assets/meshes/visual/` are identical to those in `sim/fusion_export/meshes/` (about 18 MB
checked in twice).

Defaults disagree on which tree to use:
- `NexArmSimConfig` and every example script default to `fusion_export/`.
- The gym env defaults to `description/` (`envs/nexarm.py:37`).
- `bowl_stack_scene.xml` exists only in `fusion_export/`.

Any edit to one copy silently splits the robot used for data generation from the one used for env evaluation.

The likely reason for the copy is portability. `sim/description/mjcf/meshes` and `sim/description/urdf/meshes`
are git symlinks (mode 120000). Git for Windows sets `core.symlinks=false` at system level, which this host only
avoids because of a global override. A default Windows clone checks the symlinks out as text files, so the
`description` scenes cannot find their meshes. The gym env's fallback to `fusion_export` does not help either: it
triggers only on `FileNotFoundError` for `scene.xml`, and the mesh error is raised later inside
`NexArmMujocoBackend`.

Fix: keep one real `meshes/` directory and one robot MJCF. An `<include>` from another directory is not a drop-in
replacement. I tested it: MuJoCo resolved the meshes relative to the wrong directory, so the include needs an
explicit `meshdir` (or `assetdir`) in the including scene. Add a test that the env default and the robot default
compile to the same model.

### 7. The four bowl meshes are the same file (low, verified)
`bowl.stl`, `bowl_red.stl`, `bowl_blue.stl`, and `bowl_black.stl` share an md5 hash (`dd41ff8e…`, 1.6 MB each,
both trees). Colour comes from materials. Fix: use one `bowl.stl` and drop the copies.

### 8. The default bowl scene spawns interpenetrating bowls (low, verified)
In `bowl_stack_scene.xml`, the bowls sit 100 mm apart but are 133 mm wide. Loading the raw file gives 5.8 mm
penetration, and the blue bowl ends up perched on the other two. Task reset repositions them, so only the viewer
and `simulate.py` are affected. Fix: use the reset slot positions (±0.11, 0.14 m apart) in the XML.

### 9. URDF collision uses full CAD meshes, contradicting the docs (low, verified)
`sim/assets/meshes/collision/README.md` says collisions use primitives "inside the URDF and MJCF". The MJCF does,
but every URDF `<collision>` references the visual STL (up to 78 k faces for `link_1`). Isaac conversion uses
`convex_decomp=False`, so each link becomes a single convex hull.
Fix: port the MJCF primitives into the URDF/xacro, or correct the README.

### 10. Small parameter drift between the three model descriptions (low, verified)
- Jaw damping: URDF 0.05 vs MJCF 0.01.
- Jaw friction: URDF 0.02 vs MJCF 0.
- Pinion damping: URDF 0.01 vs MJCF 0.08 (inherited from `childclass="nexarm"`).
- Wrist flex (joint 4): Isaac stiffness 40 / damping 2 vs MJCF kp 35 / kv 1.5.
- Isaac disables self-collisions; MJCF enables them.

### 11. Self-collision is common across the commanded joint range (informational, verified)
Of 3000 uniformly sampled poses within joint limits, 41% have more than 2 mm self-penetration. The most frequent
pairs are `link_2`–`base_link` and `link_2`–`link_4`/`link_5`. The `link_2` capsule is undersized rather than generous: 5% of the mesh vertices lie 66 mm or more from its axis, against a 32 mm radius, so real self-contacts are, if anything, under-reported. Hardware also self-collides at the extremes, so this is expected, but the backend never
rejects or warns on self-colliding targets, and position actuators will push through contacts with up to their
full `forcerange`.

### 12. Camera intrinsics are not tied to real hardware (verify)
`top` and `front` use fovy 55°, and `wrist` uses 92.2°. If `top` is a RealSense D435 colour stream, its vertical
FOV is about 42°. `examples/nexarm/calibrate_camera_alignment.py` exists; confirm whether its output was folded
back into the scene files.

### 13. The bowl grasp is not physical (informational)
Both `grasp_mode`s in `stack_bowls_task.py:177` teleport the held bowl to the gripper every physics step
(position only; orientation stays frozen). The docstring states this. Datasets generated this way do not teach
grasp-force or slip behaviour.

### 14. Rendering tests fail instead of skipping when OpenGL is missing (low, verified)
The single-arm sim suites ran with 83 passed and 9 failed. All 9 failures are `mujoco.FatalError: gladLoadGL
error`, because this Windows Server host has no OpenGL driver. No failure points to a model defect. Tests that
render (`test_backend_steps_and_renders`, `test_nexarm_env_*` pixel tests,
`test_sim_robot_matches_physical_feature_contract`, the gym integration tests, and the dataset generator test)
should skip when no GL context can be created, the same way hardware tests use skip decorators from
`tests/utils.py`.

### 15. Smaller asset gaps (low, verified)
- `cam_mount` has no collision geom, so the camera bracket passes through objects and other links.
- `floor` never uses the `groundplane` material, so the checker texture defined in both scenes goes unused.
- Bowl collision walls are 11 mm-thick capsules, while the mesh wall is about 4 mm thick. This changes how deep
  bowls nest, which the stacking thresholds (`min/max_nesting_delta_m`) depend on.

## Bimanual mobile sim (`sim/mobile_nexarm/`, `robots/mobile_bi_nexarm_sim/`)

The mobile model is a hand-written capsule chain built by `generate_assets.py`. It does not reuse the single-arm
MJCF, so none of the single-arm geometry, gains or gripper carries over. Its 10 tests pass. I re-checked
findings 1 and 5 myself; the rest come from a delegated review that measured them in MuJoCo.

1. **The CI asset check fails (high, verified).** `.github/workflows/mobile_nexarm_sim.yml:33` runs `generate_assets.py --check`, which raises
   `axis drift for joint_3_link_2_to_link_3: [1,0,0] != [-1,0,0]`. A second drift is hidden behind the first: the
   gripper axis and range in `spec.json:81-82` predate the jaw flip in 26a85819. The generated xml/urdf/usda
   match the generator's output for the current spec, so the stale part is the spec.
2. **The kinematics do not match the real arm (high).**
   - Link lengths are [.08, .19, .15, .07, .06] m vs [.067, .235, .145, .051, .075] m measured on the source.
   - At raw 2048 the mobile arm is straight and horizontal (TCP 0.605 m out) instead of folded (0.121 m
     forward, 0.181 m up).
   - shoulder_lift and wrist_flex move the TCP in the opposite direction to hardware.
3. **The gripper cannot hold the 50 g cube (high).** The jaws stop 18 mm apart when closed (vs 0 mm on the
   source arm), so the 20 mm cube is barely squeezed and falls 123 mm. Moving the jaw bodies to ±8 mm fixes the
   hold.
4. **The scripted data engine yields 0/10 successes (high).** Over seeds 0–9:
   - 8 seeds end with `forbidden_collision`: the jaws hit the table because the position-only IK leaves the
     gripper pitched 27–37°.
   - 2 seeds fail IK. `writer.finalize` then raises on an empty episode, which aborts the whole run
     (`generate_mujoco.py:89-94, 132`).
5. **The base moves twice the commanded distance (medium, verified in code).** `mujoco_backend.py:256-270` adds
   v·dt to qpos and also sets qvel=v, and `mj_step` then integrates qvel again. The measured displacement is
   exactly 2× at x.vel=0.2. The existing test checks direction only.
6. **Isaac replay integrates base velocity in the world frame (medium).** `isaac/runner.py:66` treats the
   velocities as world-frame, while MuJoCo treats them as body-frame, so replays diverge whenever yaw ≠ 0.
7. **Actuators are unrealistic (medium).** Every joint uses kp=100, kv=0.3, no forcerange, and `gravcomp=1`.
   The jaws lack gravcomp, which plausibly causes the 15 mm lift sag and the steady arm errors.
8. **Collision filtering is too loose (medium).**
   - Arm capsules never collide with each other (contype 1, conaffinity 6).
   - The palm and lift-carriage boxes are unnamed, so `_has_forbidden_collision` cannot flag them.
   - The carriage overlaps the table slab and hits it within 3 steps of driving forward.
9. **The right arm faces backward (medium).** `spec.json:31` mounts it at yaw +π/2, which puts its TCP at
   x=−0.6 m, away from the table. This is hidden because first-task mode freezes that arm.
10. **The URDF and USD exports are skeletons (medium).** The URDF has no inertials, collisions or visuals, and
    the USD loses joint axis signs and link offsets. ManiSkill therefore loads a massless, non-colliding arm.

Recommended direction: build each mobile arm from `sim/description/mjcf/nexarm.xml` with `MjSpec` attach and
`left_`/`right_` prefixes. That removes findings 1–3 and 7 at the source and keeps one arm definition.

## Unresolved questions

- Which RealSense model and resolution is the real `top` camera, and was `calibrate_camera_alignment.py` output
  ever applied to the scenes?
- Is the right arm's +π/2 mount intended as a mirror, or is it a bug?
- Which asset tree should be canonical: `sim/description/` or `sim/fusion_export/`?
