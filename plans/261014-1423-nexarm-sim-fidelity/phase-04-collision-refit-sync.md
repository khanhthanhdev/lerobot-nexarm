---
phase: 4
title: "Collision refit, URDF/Isaac sync, GL skip"
status: done, pending the user's viewer review
priority: P2
effort: "4h"
dependencies: [3]
---

# Phase 4: Collision refit, URDF/Isaac sync, GL skip

## Goal

Collision primitives cover the link meshes, and the URDF and Isaac configs carry the same dynamics as the MJCF.
Tests that need OpenGL skip cleanly when it is missing.

## Status (2026-11-05)

The refit and the sync already landed with the CAD Home pose and Isaac twin commits (`47f4c5721`, `e8a582ad6`):
`apply_fusion_extract.py` fits oriented boxes that enclose every mesh vertex, and `build_isaac_assets.py` writes
the URDF boxes and joint damping/friction and the Isaac drive gains (stiffness `kp`, damping `kv` + joint damping,
effort, armature) from the MJCF. Measured on the current MJCF:

| Check                                                      | Result                                        |
| ---------------------------------------------------------- | --------------------------------------------- |
| Mesh vertices inside the link's collision boxes (0 mm)     | 91.4% `base_link`, 95.4–100% every other link |
| Random joint poses in self-contact (2000 samples)          | 25% (41% measured at planning time)           |
| Self-contact along 20+20 scripted episodes (16 751 frames) | none; pick-place and stack-bowls 20/20        |

Added here: `tests/opengl.py` (`require_opengl`, a cached `mujoco.Renderer` probe; CI sets
`NEXARM_REQUIRE_OPENGL=1` so it fails instead of skipping), applied to the 10 render tests, and a test that the
IK grasp-approach poses have no self-contact. The URDF/Isaac velocity limit stays at 5.51 rad/s, and the Isaac
twin still has no setpoint speed or acceleration limiter.

## Context (at planning time)

- The `link_2` capsule (r 32 mm) is undersized: 5% of its mesh vertices lie 66 mm or more from its axis.
- Self-contacts appear in 41% of random poses, but none at the IK grasp poses.
- The URDF uses raw CAD meshes for collision.
- Isaac imports with `convex_decomp=False`.

## Files to Create / Modify

- Modify: `C:/Users/Administrator/code/lerobot-nexarm/sim/description/mjcf/nexarm.xml`: refit the arm-link collision primitives. Split `link_2` into 2–3 primitives if needed.
- Modify: `sim/description/urdf/nexarm.urdf` and `nexarm.urdf.xacro`:
  - use the same primitive collisions;
  - copy joint `dynamics` damping/friction and the limit effort/velocity from the MJCF classes.
- Modify: `sim/isaac_lab/nexarm_cfg.py`: per-servo stiffness 998.22, damping, effort and armature as in the MJCF. The velocity limits stay at the servo spec.
- Modify: `sim/assets/meshes/collision/README.md`: describe the actual collision approach.
- Modify: `tests/utils.py`: add one `require_opengl` skip helper that tries to create a `mujoco.Renderer` once and caches the result.
- Apply the helper to the 9 render-dependent tests:
  - `tests/robots/test_nexarm_sim.py::test_backend_steps_and_renders` and `::test_sim_robot_matches_physical_feature_contract`.
  - `tests/envs/test_nexarm_env.py`: the four pixel/render tests.
  - `tests/envs/test_nexarm_sim_gym_integration.py`: the two tests.
  - `tests/scripts/test_generate_nexarm_sim_dataset.py::test_generator_writes_dataset_report_and_episode_log`.
- Create tests in `tests/robots/test_nexarm_sim_assets.py`: collision coverage, self-contact and URDF/MJCF agreement.

## Tasks & Steps

- [ ] Measure the share of each link's mesh vertices that fall inside its collision geoms.
- [ ] Refit the primitives until each arm link reaches at least 90% coverage, while keeping:
  - [ ] no self-contact at the keyframes and along the scripted IK trajectories;
  - [ ] scripted success no lower than the phase-3 result.
- [ ] Report the share of random poses in self-collision before and after; this is informational.
- [ ] Sync the URDF collisions and dynamics, and the Isaac config.
- [ ] Add the GL skip helper and apply it.
- [ ] Add tests:
  - [ ] Coverage of at least 90% per link.
  - [ ] No self-contact at the keyframes or IK poses.
  - [ ] URDF and MJCF mesh placement agree within 0.1 mm over random poses.
  - [ ] URDF joint damping equals MJCF joint damping.

## Verification

- `uv run pytest tests -k "nexarm or sim_exports" -q` passes. On a host without OpenGL, the render tests are reported as skipped.
- `uv run pre-commit run --all-files --show-diff-on-failure`

## Merge gate

<!-- Updated: Validation Session 1 - user reviews collision fit in viewer -->

The user reviews the refit collisions in the MuJoCo viewer (with group 3 visible) before merge. The implementer
stops after this phase and asks for that review.

## Risk / rollback

Primitives that are too fat block real poses. Validate on the scripted trajectories, not only on keyframes. The
coverage fit cannot be checked visually here, so the user should look at it in the viewer
(`uv run python examples/nexarm/simulate.py`) before merging. Roll back by reverting the commit.
