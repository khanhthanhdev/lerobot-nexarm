---
phase: 2
title: "Scene and object fixes"
status: pending
priority: P1
effort: "2h"
dependencies: [1]
---

# Phase 2: Scene and object fixes

## Goal
Scene objects have real masses, the scenes load collision-free, and keyframes work in every scene.

## Files to Create / Modify
- Modify: `C:/Users/Administrator/code/lerobot-nexarm/sim/description/mjcf/bowl_stack_scene.xml`
  - Set `mass="0"` on each `bowl_*_visual` geom.
  - Set the `bowl_col` class mass to `0.0016667` (55 g ÷ 33 geoms).
  - Spawn the bowls at the task reset slots: (−0.11, −0.27), (0, −0.18) and (0.11, −0.27).
  - Add the floor material and full-qpos keyframes.
- Modify: `sim/description/mjcf/scene.xml`: add the floor `material="groundplane"` and full-qpos `home` and `ready` keyframes, including the cube pose.
- Modify: `sim/description/mjcf/nexarm.xml`
  - Remove `<keyframe>`; it is zero-padded in scenes.
  - Add a `cam_mount` collision box sized to its mesh extent.
  - Add an exclude between `cam_mount` and `link_6_gripper_base`.
- Modify: `src/lerobot/robots/nexarm_sim/stack_bowls_task.py`: no logic change. Confirm `base_positions` matches the scene spawn.
- Modify: the `top` camera in `scene.xml` and `bowl_stack_scene.xml`. <!-- Updated: user confirmed top camera is a RealSense D435i -->
  - The D435i colour stream at 640×480 has fy ≈ 615 px, so fovy = 2·atan(240/fy) ≈ 42.5°. The current 55° matches the horizontal FOV.
  - Read the exact fy from the device with `pyrealsense2` (`profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()`) and set `fovy` from it.
  - Re-check the camera pose with `examples/nexarm/calibrate_camera_alignment.py --camera-name top`.
- Create tests in `tests/robots/test_nexarm_sim_assets.py`.

## Tasks & Steps
- [ ] Fix the bowl masses. Assert that `body_mass` is 0.055 ± 1e-6 for each bowl.
- [ ] Move the keyframes into the scenes, with `ready` = `(0, 0.5236, -1.0472, 0.5236, 0)` plus the gripper and object poses.
  - Check that the pose has no self-contact. If it does, search for a nearby collision-free pose and record it.
- [ ] Add the `cam_mount` collision box and exclude.
  - Verify there is no contact at home.
  - Verify there is no new self-contact along the scripted pick and bowl IK trajectories, which phase 3 also uses.
- [ ] Apply the floor material.
- [ ] Add tests:
  - [ ] Bowl mass.
  - [ ] Every keyframe in every scene has no contact deeper than 1 mm.
  - [ ] The raw bowl scene settles for 1 s with no bowl-bowl contact.

## Verification
- `uv run pytest tests/robots/test_nexarm_sim_assets.py tests/robots/test_nexarm_stack_bowls.py -q`

## Risk / rollback
Domain randomization scales the bowl mass around 55 g, which is intended. The bowl assist is kinematic, so
existing bowl datasets stay valid. Roll back by reverting the commit.
