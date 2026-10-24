---
phase: 1
title: "Canonical asset tree"
status: pending
priority: P1
effort: "3h"
dependencies: []
---

# Phase 1: Canonical asset tree

## Goal

Load every single-arm consumer from `sim/description/`, with no symlinks and no duplicate models or meshes.

## Files to Create / Modify

- Modify: `C:/Users/Administrator/code/lerobot-nexarm/sim/description/mjcf/nexarm.xml`: set `meshdir="../../assets/meshes/visual"`.
- Modify: `C:/Users/Administrator/code/lerobot-nexarm/sim/description/urdf/nexarm.urdf` and `nexarm.urdf.xacro`: mesh paths become `../../assets/meshes/visual/<name>.stl`.
- Delete: `sim/description/mjcf/meshes` and `sim/description/urdf/meshes` (both are symlinks).
- Create (move): `sim/description/mjcf/bowl_stack_scene.xml`, which includes `nexarm.xml`. All three bowl meshes use `bowl.stl`.
- Delete: `sim/assets/meshes/visual/bowl_{red,blue,black}.stl`; these are identical to `bowl.stl`.
- Delete: `sim/fusion_export/{NexArm-sim.xml,scene.xml,bowl_stack_scene.xml,nexarm.urdf}` and `sim/fusion_export/meshes/`.
- Modify: `.gitignore` lines 186–190: remove the `!sim/fusion_export/...` re-includes and ignore `sim/fusion_export/`. It becomes untracked exporter output (user decision). <!-- Updated: Validation Session 1 - untrack fusion_export -->
- Modify defaults to `sim/description/mjcf/scene.xml` or `bowl_stack_scene.xml`:
  - `src/lerobot/robots/nexarm_sim/config_nexarm_sim.py:26`.
  - `src/lerobot/envs/nexarm.py:37-38`. Also remove `FALLBACK_SCENE_PATH`, because it never catches mesh errors.
  - `src/lerobot/motors/nexarm/kinematics_dynamics.py:92`.
  - `src/lerobot/scripts/lerobot_nexarm_sim_benchmark.py:552`.
  - The `--model` default in each of `examples/nexarm/{calibrate_camera_alignment,calibrate_sim,generate_sim_dataset,generate_stack_bowls_dataset,pick_place_sim,rollout_turbovla,simulate,stack_bowls_sim}.py`.
  - `sim/test_sim_models.py`.
  - `sim/isaac_lab/convert_urdf_to_usd.py` (line 4 and line 53).
- Modify: `sim/mobile_nexarm/spec.json:5-6` to `../description/mjcf/nexarm.xml` and `../assets/meshes/visual`.
- Modify tests:
  - `tests/test_sim_exports.py`: drop the fusion checks and keep the description ones.
  - `tests/robots/test_nexarm_sim.py`, `test_nexarm_sim_calibration.py` and `test_nexarm_stack_bowls.py`.
  - `tests/scripts/test_generate_nexarm_sim_dataset.py` and `test_generate_stack_bowls_dataset.py`.
- Modify docs:
  - `sim/README.md`: layout section; state that `fusion_export/` is exporter output to review and copy into `description/` by hand.
  - `run.md`, `sim/isaac_lab/README.md` and `docs/source/nexarm/sim_calibration.md:158`.
  - Exporter sources keep writing to `fusion_export/` (non-goal).
- Create: `tests/robots/test_nexarm_sim_assets.py`.

## Tasks & Steps

- [ ] Run `git grep -n fusion_export -- ':!plans'` and confirm that the list matches the files above before editing.
- [ ] Repoint `meshdir` and the URDF mesh paths, then delete the two symlinks.
- [ ] Move the bowl scene and collapse the bowl meshes to `bowl.stl`.
- [ ] Repoint all defaults, tests and docs. Remove the env fallback.
- [ ] Repoint `spec.json` and confirm `generate_assets.py --check` still fails only with `axis drift for joint_3_link_2_to_link_3`.
- [ ] Delete the checked-in `fusion_export` copies and update `.gitignore`.
- [ ] Add tests:
  - [ ] Every scene compiles from a temporary working directory.
  - [ ] The env default and the `NexArmSimConfig` default compile to the same nq, nu, nmesh and body masses.
  - [ ] No git symlink exists under `sim/` (`git ls-files -s sim | awk '$1==120000'` is empty).

## Verification

- `uv run pytest tests/robots/test_nexarm_sim_assets.py tests/test_sim_exports.py -q`
- `git grep -n "fusion_export" -- ':!plans' ':!sim/export_fusion_to_mujoco.py' ':!sim/exporter' ':!sim/run_export.py' ':!sim/README.md' ':!.gitignore'` returns nothing.
- `uv run python sim/mobile_nexarm/generate_assets.py --check` reports only the known axis drift.

## Risk / rollback

User scripts that pass `--model sim/fusion_export/...` will break; call this out in the commit message. Roll back
by reverting the commit.
