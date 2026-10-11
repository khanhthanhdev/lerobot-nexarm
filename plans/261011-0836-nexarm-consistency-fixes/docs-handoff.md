# Docs Handoff — code changes the docs must follow

## Conventions (now enforced in code)

- Ports: leader `/dev/ttyUSB0`, follower `/dev/ttyUSB1` (defaults in record, teleoperate, rollout, rollout_turbovla, prepare_collection, calibrate_arms, vision_grasp, teleoperate_cartesian, yamls, collect.sh).
- Gripper raw: 1195 = open, 2833 = closed.
- Sim arm mapping: `rad = (raw - 2048) * 2π / 4096`, clamped to joint range. Reachable raw: pan/elbow 512–3584, lift 683–3413, wrist_flex 910–3186, wrist_roll 0–4095. Gripper raw maps inverted onto the 0..0.0255 m jaw slide. Existing sim datasets and sim calibrations are stale and must be regenerated.
- Sim physics timestep is adjusted so an integer number of steps equals exactly 1/fps.
- `NexArmSim.robot_type == "nexarm_follower"` (registry type stays `nexarm_sim`), so sim and real datasets share `robot_type`.
- Top camera ON by default everywhere (real and sim). `--no-top-cam` disables. Missing RealSense or `pyrealsense2` → error naming `--no-top-cam` and `uv sync --extra intelrealsense`.
- Rollout (`rollout.py`, `rollout_turbovla.py`) exits at startup if opened cameras differ from the policy's image inputs. Legacy 2-camera ACT models (trained on `thanhkt/nexarm_stack_bowls`) and legacy 2-view TurboVLA checkpoints need `--no-top-cam`.
- Collection: default root `thanhkt/nexarm_stack_bowls_top` (3 cameras, created by the first session). Old 2-camera root: `--no-top-cam --root-repo-id thanhkt/nexarm_stack_bowls`. Merge compatibility (fps, robot_type, cameras, shapes) is checked before recording; Hub/network errors suggest `--merge-root` or `--no-merge`.
- `lerobot-record` stamps `_YYYYMMDD_HHMMSS` onto `repo_id` and prints the final stamped `repo_id` and root after recording; follow-up commands must use that id (or `--root`).

## Flag changes (W2: real / TurboVLA / collection)

| Script                                         | Flag / key                            | Old                                                      | New                                                                                                       |
| ---------------------------------------------- | ------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| prepare_collection                             | `--leader-port` / `--follower-port`   | USB1 / USB0                                              | USB0 / USB1                                                                                               |
| prepare_collection, merge_collection           | `--root-repo-id`                      | `thanhkt/nexarm_stack_bowls`                             | `thanhkt/nexarm_stack_bowls_top`                                                                          |
| prepare_collection                             | `--front-cam` / `--wrist-cam`         | by-id paths                                              | `0` / `1` (index or path accepted; `collect.sh` passes rig by-id paths)                                   |
| prepare_collection                             | `--front-fourcc` / `--wrist-fourcc`   | MJPG / YUYV                                              | auto (`collect.sh` passes MJPG / YUYV)                                                                    |
| record, teleoperate                            | `--leader-port` / `--follower-port`   | required                                                 | default USB0 / USB1                                                                                       |
| rollout                                        | `--follower-port`                     | required                                                 | default USB1                                                                                              |
| record, teleoperate, rollout, rollout_turbovla | `--front-cam` / `--wrist-cam`         | int                                                      | index or device path, default 0 / 1                                                                       |
| same                                           | `--front-fourcc`, `--wrist-fourcc`    | —                                                        | new, default auto                                                                                         |
| same                                           | `--top-cam`                           | optional serial, off                                     | default `auto` (exactly one RealSense), on                                                                |
| same                                           | `--no-top-cam`                        | —                                                        | new                                                                                                       |
| record                                         | `--fps`                               | cameras only                                             | also sets dataset fps                                                                                     |
| rollout_turbovla                               | `--task`                              | pick-cube prompt                                         | default: checkpoint's saved task, else task-type training prompt; stack-bowls sim uses sampled bowl order |
| rollout_turbovla                               | `--fps`                               | 30                                                       | default: checkpoint fps, else 30                                                                          |
| rollout_turbovla                               | `--task-type auto`                    | name guess                                               | saved `task_type` first                                                                                   |
| train_turbovla                                 | `--task-type`                         | —                                                        | `auto` / `pick_place` / `stack_bowls`                                                                     |
| train_turbovla                                 | `--front-cam-key` / `--wrist-cam-key` | only given keys                                          | always 2 views [front, wrist]; unset one matched by name; combining with `--cameras` is an error          |
| train_turbovla                                 | `config.json`                         | —                                                        | adds `task_type`, `camera_keys`, `fps`, `task`                                                            |
| diagnose_turbovla                              | `--cameras`                           | always used                                              | only when checkpoint lacks `camera_keys`                                                                  |
| merge_hub_datasets                             | `--delete-sources`                    | sources deleted by default                               | opt-in; default keeps sources (`--keep-sources` still accepted)                                           |
| merge_hub_datasets                             | `--output-dir`                        | help said cache                                          | temp dir deleted after upload                                                                             |
| upload_dataset                                 | `--large-folder/--no-large-folder`    | broken toggle                                            | BooleanOptionalAction, default on                                                                         |
| calibrate_camera_alignment                     | `--camera-name`                       | front, wrist                                             | adds `top`; `--cam-index` takes index or path                                                             |
| inference/record/teleoperate yaml              | `robot.cameras.top`                   | commented                                                | enabled (intelrealsense, 640x480@30)                                                                      |
| scripts/nexarm/train_turbovla\*                | output dir / dataset flag             | `nexarm_turbovla_{cluster,ddp,single}`, `--dataset-root` | `outputs/train/nexarm_turbovla`, `--sim-dataset-root`                                                     |
| collect.sh                                     | args                                  | none                                                     | passes ports, by-id cameras, MJPG/YUYV; user args override                                                |
| train_stack_bowls_act                          | fps / resolution                      | fixed 30 / 480x640                                       | read from dataset metadata                                                                                |

## Flag changes (W1: sim core)

| Where                          | Flag                                                                                                                   | New     |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | ------- |
| `lerobot-nexarm-sim-benchmark` | `--calibration PATH`, `--action-delay-steps N`, `--dr/--no-dr` (default off)                                           | new     |
| env `nexarm`                   | `--env.calibration_path`, `--env.enable_domain_randomization`, `--env.action_delay_steps`                              | new     |
| calibration precedence         | explicit config/CLI > calibration > 0; calibration fps must equal run fps                                              | new     |
| DR                             | `object_friction`, `object_mass_scale`, `object_bodies`, `object_rgb_jitter` (was `cube_*`); applies to cube and bowls | renamed |

## W3 (generators)

| Script                       | Old                                              | New                                                                                                                                     |
| ---------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| generate_sim_dataset         | `--dr` no-op                                     | DR on by default; `--no-dr` disables; `--dr/--domain-randomization` explicit on                                                         |
| generate_sim_dataset         | `--workers` 1; `--gpus` ignored unless workers>1 | `--workers` defaults to number of `--gpus` ids else 1; `--gpus` = comma-separated GPU ids (e.g. `0,1`), round-robin                     |
| generate_sim_dataset         | bad calibration fps failed at connect            | rejected at argument parsing                                                                                                            |
| generate_stack_bowls_dataset | DR off by default                                | DR on by default; `--no-dr` disables; `--dr/--domain-randomization` explicit on                                                         |
| generate_stack_bowls_dataset | —                                                | `--calibration PATH`, `--action-delay-steps N`, `--action-delay-range MIN MAX` (CLI > calibration > 0; calibrated delay ±1 per episode) |
| generate_stack_bowls_dataset | —                                                | `--split {train,eval}` (eval seeds start at 1000000), `--workers N`                                                                     |
| generate_stack_bowls_dataset | `--seed-start` 0 / episode count on resume       | split start, or seed after last logged on `--resume`                                                                                    |
| generate_stack_bowls_dataset | —                                                | `--resume` on an old-layout dataset (no `generation_episodes.jsonl`) errors: regenerate                                                 |
| both generators              | —                                                | `--resume` works with `--workers`/`--gpus`                                                                                              |
| stack_bowls_sim              | `--model .../scene_stack_bowls.xml` (missing)    | `sim/fusion_export/bowl_stack_scene.xml`                                                                                                |
| calibrate_sim                | fps/fitted_joints in `source` only               | also top-level `fps`, `fitted_joints`                                                                                                   |

Provenance (both generators): `<root>/generation_report.json` and `<root>/generation_episodes.jsonl`; `meta/generation.jsonl` no longer written. Both write `robot_type="nexarm_follower"`.
