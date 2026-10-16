# NexArm in Isaac Lab (Isaac Sim)

`Isaac-NexArm-StackBowls-Direct-v0` is the Isaac Lab twin of the MuJoCo `NexArmStackBowls-v0` task: the same
robot, workcell, cameras, bowls, action units and success rule, so either simulator can train a policy for the real
stack-bowls task. Targets Isaac Lab 2.x (Isaac Sim 4.5 or 5.x).

## 1. Where the design comes from

The Fusion export updates `sim/description/` (see `sim/README.md`, section 11). The MJCF there is the reference,
and `build_isaac_assets.py` mirrors it for Isaac. `sim/apply_fusion_extract.py` runs it after every export, and a
test fails if its outputs are stale.

| Output                                            | What it carries                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sim/description/urdf/nexarm.urdf` (and `.xacro`) | MJCF collision boxes (not the raw CAD meshes), joint damping and friction, and a fixed `gripper_frame` link at the MJCF site. Tests check the boxes and that link against MuJoCo within 10 µm over random poses.                                                                                  |
| `sim/isaac_lab/nexarm_scene.json`                 | Robot base pose (on the 33 mm plate), drive gains, raw-tick action map, gripper frame, MJCF contact excludes, table, the 38 workcell fixtures, the `front`/`top`/`wrist` cameras with their MJCF poses and fields of view, the 60 g bowls with their 33 contact proxies, and the task thresholds. |

To refresh them by hand, or to check them:

```bash
uv run python sim/isaac_lab/build_isaac_assets.py [--check]
```

## 2. Build the USD assets (once per export)

```bash
./isaaclab.sh -p sim/isaac_lab/build_usd.py --headless      # Windows: isaaclab.bat -p ...
```

This writes `sim/isaac_lab/usd/` (git-ignored):

- `nexarm.usd`: the URDF through Isaac Lab's converter. Fixed joints are kept, so `cam_mount` and
  `link_6_gripper_base` stay separate prims. The mimic jaw becomes a normal joint, and the environment drives
  both jaws with the same target.
- `bowl_red.usd`, `bowl_blue.usd`, `bowl_black.usd`: rigid bowls that collide through the same capsule ring and
  base disc as the MJCF bowl. A convex hull would fill a bowl, and the bowls could not nest. Only `pxr` is
  needed for these (`--bowls-only` with `pip install usd-core`).

## 3. Check the environment

```bash
./isaaclab.sh -p sim/isaac_lab/check_env.py --headless
./isaaclab.sh -p sim/isaac_lab/check_env.py --headless --enable_cameras --save_images outputs/isaac_frames
```

It checks that the bowls settle on the table, that the arm holds home and follows a raw-tick command, and that a
released, nested stack counts as success. With cameras, it saves one frame per camera to compare with the MuJoCo
renders.

## 4. Train

```bash
./isaaclab.sh -p sim/isaac_lab/train.py --headless --num_envs 1024
```

PPO (rsl_rl) runs on state observations, and checkpoints go to `logs/rsl_rl/nexarm_stack_bowls/`. The PPO
settings are a starting point: three-bowl stacking from dense rewards is a long-horizon task, and no reference
run exists yet.

## 5. Environment contract

|                    | Isaac (`NexArmStackBowlsEnv`)                                                                                                   | MuJoCo (`NexArmStackBowls-v0`)                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Action             | 6 follower features in raw servo ticks; `action_mode="normalized"` maps [-1, 1] over each raw range                             | same (`control_mode="raw"` or `"normalized"`)                      |
| Raw tick → joint   | affine map from `nexarm_scene.json`, tested equal to the MuJoCo backend                                                         | `raw_to_joint_position`                                            |
| Control / physics  | 30 Hz, 17 steps of 1.96 ms                                                                                                      | same                                                               |
| Drives             | MJCF `kp`; damping = MJCF `kv` + joint damping; MJCF force range and armature                                                   | position actuators                                                 |
| Grasp              | contact assist: gripper commanded past 2400 ticks with both jaws on one bowl carries it rigidly; below 1600 releases            | same                                                               |
| Success            | concentric (< 35 mm), nested (20–38 mm apart), released, still, upright, for 0.5 s                                              | same                                                               |
| Policy observation | normalized joint positions, joint velocities, gripper frame, bottom/middle/top bowl positions, held flags, previous action (33) | `agent_pos`, `environment_state` (bowls bottom/middle/top), pixels |
| Cameras            | `enable_cameras=True`: TiledCameras at the MJCF poses, 640×480, `obs["images"]`                                                 | `front`, `wrist`, `top`                                            |
| Instruction        | `env.task_descriptions()`                                                                                                       | `task_description`                                                 |

Known differences:

- PhysX joints have no friction loss (MuJoCo `frictionloss`, 0.02–0.08 N·m). The JSON records it.
- Contacts are rigid in PhysX and soft in MuJoCo. The robot links keep PhysX's default friction; the bowls and
  table use the MJCF values with `max` combining, as MuJoCo does.
- Calibrated action latency and domain randomization exist only in the MuJoCo env.

## 6. Files

- `build_isaac_assets.py`: MJCF → URDF collisions and `nexarm_scene.json` (MuJoCo only, no Isaac).
- `build_usd.py`: URDF → `usd/nexarm.usd`, and the bowl USDs.
- `nexarm_isaac/`: the Isaac Lab package. `scene_spec.py` reads the JSON, `robot_cfg.py` holds the articulation
  config, `stack_bowls_env.py` is the task, and `agents/` holds the PPO config. Importing it registers the gym id.
- `train.py`, `check_env.py`: entry points run with Isaac Lab's Python.
