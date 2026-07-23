# NexArm MuJoCo simulation

The NexArm simulation is ready for interactive control, physical-leader
teleoperation, LeRobot data collection, dataset replay, and policy rollout. It
is not a firmware-in-the-loop simulator: the simulated follower uses MuJoCo
position actuators instead of executing the Hiwonder ESP32/AT32 firmware.

The shortest complete workflow is:

1. Run `examples/nexarm/pick_place_sim.py` to verify the scene and controls.
2. Record demonstrations with `lerobot-record` and the physical NexArm leader.
3. Inspect at least one episode with `lerobot-dataset-viz`.
4. Train a policy with `lerobot-train`.
5. Run it with `lerobot-rollout`, then record good and failed rollouts as a
   separate evaluation dataset.
6. Compare checkpoints with `lerobot-nexarm-sim-benchmark` on identical seeds.

## What is ready

| Capability                                        | Status          | Notes                                                                             |
| ------------------------------------------------- | --------------- | --------------------------------------------------------------------------------- |
| MuJoCo viewer and actuator sliders                | Ready           | Six controls: five arm joints and the gripper.                                    |
| Physical NexArm leader to simulated follower      | Ready           | The leader uses the Hiwonder firmware and USB protocol.                           |
| LeRobot `Robot` interface                         | Ready           | Registered as `--robot.type=nexarm_sim`.                                          |
| Front and wrist RGB cameras                       | Ready           | Both render at 640 x 480 by default.                                              |
| Object contact and grasping                       | Ready           | The scene contains a free red cube and jaw collision bodies.                      |
| LeRobot record, replay, and rollout               | Ready           | Uses the same six action/state feature names as the physical follower.            |
| Seeded pick/place reset and success detection     | Ready           | `NexArmPickPlaceTask` checks grasp, release, stable placement, drop, and timeout. |
| Gymnasium reward environment                      | Not implemented | The current task wrapper is intentionally smaller than a benchmark environment.   |
| Hiwonder follower firmware in the simulation loop | Not implemented | MuJoCo replaces the follower firmware and servo controller.                       |
| Calibrated HX-30HM motor dynamics                 | Approximate     | Position gains and friction are stable defaults, not an identified motor model.   |
| Isaac Sim model                                   | Not implemented | The current implementation is MuJoCo-first.                                       |

## Controller and firmware boundary

The physical NexArm follower follows this path:

```text
LeRobot action (raw servo positions)
  -> NexArmFollower
  -> USB CommProtocol
  -> CMD 68/96/97/98
  -> follower ESP32 and AT32
  -> HX-30HM servos
```

The simulated follower follows a different path:

```text
LeRobot action (same raw servo positions)
  -> NexArmSim
  -> raw-position to radian/metre conversion
  -> MuJoCo position actuators
  -> physics step and rendered cameras
```

Therefore, MuJoCo does not execute `CMD 68`, `CMD 97`, servo acceleration
registers, or the Hiwonder follower control loop. It preserves the external
LeRobot action contract so datasets and policies can address the real and
simulated followers with the same feature names.

When a physical leader is used, its side of the pipeline does use the Hiwonder
firmware:

```text
physical leader arm
  -> leader ESP32 firmware
  -> CMD 96 position read over USB
  -> NexArmLeader.map_leader_to_follower()
  -> NexArmSim
  -> MuJoCo
```

The existing leader mapping mirrors `shoulder_lift` and maps the useful gripper
range before the command reaches the simulator. Do not run two programs against
the same leader serial port at the same time.

## Model files

- `fusion_export/NexArm-sim.xml` contains the robot, inertias, joints, limits,
  six position actuators, the jaw equality constraint, wrist camera, and primitive
  contact geometry.
- `fusion_export/scene.xml` includes the robot and adds the front camera, floor,
  lighting, red cube, and green target zone.
- `fusion_export/meshes/` contains detailed STL visual meshes. These are
  visual-only because using the complete meshes for contact produced unstable
  overlapping convex hulls. New Fusion exports use low-refinement meshes to
  reduce simulator startup and rendering cost.
- `export_fusion_to_mujoco.py` is executed inside Fusion 360 through Fusion MCP
  and regenerates the model, scene, and STL files.
- `../src/lerobot/robots/nexarm_sim/` contains the LeRobot simulation backend
  and `NexArmSim` robot implementation. `pick_place_task.py` adds the optional
  seeded single-arm task without changing the six-feature robot contract.
- `../examples/nexarm/simulate.py` launches the viewer and optional leader
  control loop.
- `../examples/nexarm/pick_place_sim.py` launches the seeded pick/place test.

The model uses a 2 ms physics timestep, or 500 physics steps per second. At the
default 30 Hz LeRobot control rate, each action advances approximately 17
MuJoCo steps.

`nexarm_mount` recenters the Fusion assembly around the base axis and provides
one transform for placing the complete robot in a larger scene. Arm joints use
damped, actuated defaults; the lightweight jaw joints use separate defaults so
pinion-scale inertia cannot lock the gripper. Non-adjacent primitive
self-collision is enabled, while explicit exclusions cover connected link pairs.

## Joint and action mapping

The simulated robot exposes the same LeRobot keys as the physical follower:

| LeRobot feature     | MuJoCo joint               | Raw input |      MuJoCo control range |
| ------------------- | -------------------------- | --------: | ------------------------: |
| `shoulder_pan.pos`  | `joint_1_base_to_link_1`   |    0-4095 | -2.356194 to 2.356194 rad |
| `shoulder_lift.pos` | `joint_2_link_1_to_link_2` |    0-4095 | -2.094395 to 2.094395 rad |
| `elbow_flex.pos`    | `joint_3_link_2_to_link_3` |    0-4095 | -2.356194 to 2.356194 rad |
| `wrist_flex.pos`    | `joint_4_link_3_to_link_4` |    0-4095 | -1.745329 to 1.745329 rad |
| `wrist_roll.pos`    | `joint_5_link_4_to_link_5` |    0-4095 | -3.141593 to 3.141593 rad |
| `gripper.pos`       | `right_jaw_slide_joint`    | 1195-2833 |            -0.0255 to 0 m |

The conversion is linear:

```text
ratio = (raw - raw_min) / (raw_max - raw_min)
control = control_min + ratio * (control_max - control_min)
```

The five arm joints reset to raw position `2048`. The gripper resets to `2833`,
which maps to the closed position. Raw gripper position `1195` maps to the open
position. An equality constraint moves the left jaw in the opposite direction,
giving 51 mm of total additional jaw travel.

The pinion remains in the visual model but is fixed to the gripper body. A
246:1 pinion-to-slide equality made the constraint poorly conditioned without
improving manipulation physics, so jaw motion is represented by the two linear
slides only.

This mapping is interface-compatible, but it has not yet been measured against
the physical arm at multiple poses. Before sim-to-real deployment, compare real
and simulated zero positions, endpoints, and direction for every joint.

## 1. Install

From the repository root:

```bash
uv sync --locked
```

For a physical leader, recording, and visualization tools:

```bash
uv sync --locked --extra core_scripts
```

Confirm the model loads:

```bash
uv run python -c 'import mujoco; m=mujoco.MjModel.from_xml_path("sim/fusion_export/scene.xml"); print(f"actuators={m.nu}, cameras={m.ncam}, geoms={m.ngeom}")'
```

Expected output includes six actuators and two cameras.

## 2. Play using MuJoCo sliders

```bash
uv run python examples/nexarm/simulate.py
```

Use the MuJoCo viewer's Control panel to move the six position actuators. The
red cube is a free dynamic object and the invisible simplified robot collision
bodies can contact it.

You can load a different scene explicitly:

```bash
uv run python examples/nexarm/simulate.py --model sim/fusion_export/scene.xml --fps 30
```

For the repeatable single-arm test, use:

```bash
uv run python examples/nexarm/pick_place_sim.py --seed 0
```

Move the red cube into the green zone, open the gripper, and leave the cube
stable for 0.5 seconds. Add `--auto-reset` to start the next deterministic seed
after success, drop, or timeout.

## 3. Control MuJoCo with the physical leader

Find the leader serial port:

```bash
uv run lerobot-find-port
```

Linux example:

```bash
uv run python examples/nexarm/simulate.py --leader-port /dev/ttyUSB0 --fps 30
```

Windows example:

```powershell
uv run python examples/nexarm/simulate.py --leader-port COM18 --fps 30
```

This command opens the MuJoCo viewer, reads the physical leader through the
Hiwonder protocol, applies the existing leader-to-follower mapping, and sends
the result to `NexArmSim`. It does not connect to or power the physical follower.

## 4. Use standard LeRobot teleoperation

The standard CLI is useful when camera observations should be sent to Rerun:

```bash
uv run lerobot-teleoperate \
  --robot.type=nexarm_sim \
  --robot.id=nexarm_sim \
  --robot.model_path=sim/fusion_export/scene.xml \
  --robot.fps=30 \
  --teleop.type=nexarm_leader \
  --teleop.id=nexarm_leader \
  --teleop.port=/dev/ttyUSB0 \
  --display_data=true
```

This path renders the `front` and `wrist` camera observations but does not open
the interactive MuJoCo viewer. Use `examples/nexarm/simulate.py` when the main
goal is visually playing with the scene.

## 5. Record a LeRobot simulation dataset

Set a Hub-compatible repository ID even when keeping the dataset local:

```bash
uv run lerobot-record \
  --robot.type=nexarm_sim \
  --robot.id=nexarm_sim \
  --robot.model_path=sim/fusion_export/scene.xml \
  --robot.fps=30 \
  --teleop.type=nexarm_leader \
  --teleop.id=nexarm_leader \
  --teleop.port=/dev/ttyUSB0 \
  --dataset.repo_id=YOUR_HF_USERNAME/nexarm_sim_pick_cube \
  --dataset.root=outputs/datasets/nexarm_sim_pick_cube \
  --dataset.single_task="Pick up the red cube" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true
```

The dataset contains six raw joint state values, six raw action values, and the
front and wrist RGB streams. Keeping `--dataset.root` explicit makes local
recording, visualization, replay, and backup use the same directory.
`NexArmPickPlaceTask.reset(seed=...)` provides
repeatable cube and target randomization for custom recording loops. The
standard `lerobot-record` command still calls `NexArmSim.reset()` directly, so
it does not opt into task randomization automatically.

## 6. Inspect the recorded camera video, state, and actions

Install the local dataset viewer:

```bash
uv sync --locked --extra dataset_viz
```

Open episode 0 in Rerun:

```bash
uv run lerobot-dataset-viz \
  --repo-id=YOUR_HF_USERNAME/nexarm_sim_pick_cube \
  --root=outputs/datasets/nexarm_sim_pick_cube \
  --episode-index=0
```

Rerun shows both camera streams next to the six state and six action curves, so
use it to find blurred frames, control lag, bad demonstrations, and inconsistent
end states before training. Save a portable Rerun recording instead of opening
the viewer immediately with:

```bash
uv run lerobot-dataset-viz \
  --repo-id=YOUR_HF_USERNAME/nexarm_sim_pick_cube \
  --root=outputs/datasets/nexarm_sim_pick_cube \
  --episode-index=0 \
  --save=1 \
  --output-dir=outputs/dataset_visualizations
```

The encoded source camera videos are under
`outputs/datasets/nexarm_sim_pick_cube/videos/`. LeRobot v3 may concatenate
multiple episodes into an MP4 shard, so `lerobot-dataset-viz` is the reliable
way to seek one episode using its metadata. If the dataset was pushed to the
Hub, paste its repository ID into the
[LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).

## 7. Replay a recorded episode

```bash
uv run lerobot-replay \
  --robot.type=nexarm_sim \
  --robot.id=nexarm_sim \
  --robot.model_path=sim/fusion_export/scene.xml \
  --dataset.repo_id=YOUR_HF_USERNAME/nexarm_sim_pick_cube \
  --dataset.root=outputs/datasets/nexarm_sim_pick_cube \
  --dataset.episode=0
```

Replay is a useful action-contract test: the same dataset action keys can be
sent to `nexarm_sim`, and a simulation dataset can later be tested cautiously
against `nexarm_follower` after joint calibration is verified.

## 8. Train an ACT policy

After checking the recorded camera frames and actions:

```bash
uv run lerobot-train \
  --dataset.repo_id=YOUR_HF_USERNAME/nexarm_sim_pick_cube \
  --dataset.root=outputs/datasets/nexarm_sim_pick_cube \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/act_nexarm_sim \
  --job_name=act_nexarm_sim \
  --batch_size=8 \
  --wandb.enable=false \
  --policy.repo_id=YOUR_HF_USERNAME/act_nexarm_sim
```

Use `--policy.device=mps` on Apple silicon or an appropriate CPU/device setting
when CUDA is unavailable.

## 9. Run a policy in MuJoCo

For a quick autonomous test without recording a dataset:

```bash
uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/act_nexarm_sim/checkpoints/last/pretrained_model \
  --robot.type=nexarm_sim \
  --robot.id=nexarm_sim \
  --robot.model_path=sim/fusion_export/scene.xml \
  --robot.fps=30 \
  --task="Pick up the red cube" \
  --duration=60 \
  --display_data=true \
  --rerun_save_path=outputs/rollouts/act_nexarm_sim.rrd
```

`base` runs the policy autonomously and exits after `--duration` seconds. It
does not save a dataset. `--display_data=true` streams the simulated front and
wrist observations to Rerun; it does not open the interactive MuJoCo viewer.
`--rerun_save_path` preserves the visual trace so it can be opened again with
`uv run rerun outputs/rollouts/act_nexarm_sim.rrd`.

To record the policy rollout as an evaluation dataset, switch to `sentry` and
provide a dataset destination:

```bash
uv run lerobot-rollout \
  --strategy.type=sentry \
  --policy.path=outputs/train/act_nexarm_sim/checkpoints/last/pretrained_model \
  --robot.type=nexarm_sim \
  --robot.id=nexarm_sim \
  --robot.model_path=sim/fusion_export/scene.xml \
  --robot.fps=30 \
  --dataset.repo_id=YOUR_HF_USERNAME/eval_nexarm_sim \
  --dataset.root=outputs/datasets/eval_nexarm_sim \
  --dataset.single_task="Pick up the red cube" \
  --duration=60 \
  --display_data=true
```

This is policy rollout through the LeRobot `Robot` interface. It does not run
the seeded task success gate or produce comparable policy metrics; use the
benchmark command below for that. Inspect this evaluation dataset with the same
`lerobot-dataset-viz` command from step 6, changing `--repo-id` and `--root`.

## 10. Benchmark ACT, pi0, and SmolVLA

Install the policy-specific server dependencies once:

```bash
uv sync --locked --extra dataset --extra pi --extra smolvla
```

Pass each trained checkpoint with a unique label. Policies are loaded one at a
time so large VLA checkpoints do not remain together in GPU memory:

```bash
MUJOCO_GL=egl uv run lerobot-nexarm-sim-benchmark \
  --policy act=outputs/train/act_nexarm_sim/checkpoints/last/pretrained_model \
  --policy pi0=outputs/train/pi0_nexarm_sim/checkpoints/last/pretrained_model \
  --policy smolvla=outputs/train/smolvla_nexarm_sim/checkpoints/last/pretrained_model \
  --episodes=20 \
  --device=cuda \
  --output-dir=outputs/nexarm_sim_benchmark
```

The command runs the same seed range for every policy and writes
`benchmark.json` plus a flat `benchmark.csv`. Results include pick/place
success rate, termination reasons, inference mean/p50/p95 latency, achieved
control rate, load time, parameter count, and peak CUDA memory. Add
`--realtime` when wall-clock control timing matters; without it, MuJoCo runs as
fast as inference and rendering allow.

Treat success rate as the primary task metric, then use termination reasons to
separate drops from timeouts and p95 latency to check whether a policy can
sustain the target control rate. The benchmark intentionally does not encode
videos because rendering and encoding would distort timing; use a recorded
`sentry` rollout for qualitative video comparison.

Each checkpoint must have been trained or fine-tuned on the six NexArm action
features and the configured `front`/`wrist` cameras. A generic base VLA
checkpoint does not have the NexArm normalization statistics or action
contract, so it is not directly comparable.

The same entry point is available as:

```bash
MUJOCO_GL=egl uv run python examples/nexarm/benchmark_sim.py --policy act=PATH
```

## 11. Re-export from Fusion 360

The Fusion design is the source of truth. Moving through Onshape is unnecessary
for the current pipeline.

The exporter expects these occurrence names:

- `base_link:1`
- `link_1:1` through `link_5:1`
- `link_6:1`, `gripper_base:1`, and `pinion_gear:1`
- `left_jaw:1` and `right_jaw:1`
- `cam_mount:1`

It expects these movable joint names:

- `joint_1_base_to_link_1` through `joint_5_link_4_to_link_5`
- `left_jaw_slide_joint`
- `right_jaw_slide_joint`

The Fusion root component needs a `mujoco/config` JSON attribute containing the
model and default actuator configuration. Each actuated native joint needs a
`mujoco/config` attribute such as:

```json
{ "actuated": true, "actuator": "position" }
```

Run `sim/export_fusion_to_mujoco.py` through the Fusion MCP script runner. It
writes through the configured WSL UNC path into `sim/fusion_export/`.

The primitive collision coordinates currently match this exact CAD export.
After changing link dimensions, joint origins, jaw geometry, or the camera
mount, update `COLLISION_SPECS` and camera poses in the exporter, re-export, and
run the validation tests.

The detailed STL files are rendering assets, not collision or inertia sources.
Repair non-manifold CAD bodies in Fusion before re-exporting if those meshes
will be converted for another simulator.

## 12. Validate after model or controller changes

Run the focused simulation tests:

```bash
uv run --locked --extra test pytest tests/robots/test_nexarm_sim.py -q
```

The tests verify:

- raw servo value to MuJoCo control conversion and reverse conversion;
- simulation stepping and both camera outputs;
- the same six LeRobot features as the physical follower;
- active damping, friction loss, armature, and collision masks;
- increasing jaw separation for the documented open command;
- stable two-jaw contact with the 20 mm task cube.

Run the style check for the simulation implementation:

```bash
uv run --locked --extra dev ruff check src/lerobot/robots/nexarm_sim examples/nexarm/simulate.py tests/robots/test_nexarm_sim.py sim/export_fusion_to_mujoco.py
```

## 13. Troubleshooting

**The model file is missing:** Re-export from Fusion or pass an explicit
`--robot.model_path`. The default is `sim/fusion_export/scene.xml`.

**The Controls panel is empty:** Load `scene.xml` or `NexArm-sim.xml` from
`fusion_export/`. A model without the six `<position>` actuators cannot expose
control sliders.

**The robot shakes or explodes:** Keep detailed STL geometry visual-only. Check
for overlapping primitive collision bodies, invalid inertias, excessive gains,
or a collision shape intersecting the floor at the reset pose.

**A camera is reported missing:** The default `camera_names` are `front` and
`wrist`. Both must exist in the loaded MJCF, or pass a different configured
camera list.

**The leader times out:** Confirm the correct serial port, 1 Mbps support,
power, USB permissions, and that no second process has opened the port.

**A physical joint moves in the opposite direction:** Stop sim-to-real testing
and measure the real follower mapping. Correct the per-joint direction/zero
mapping in the simulation backend instead of changing the CAD axis only to
hide a controller mismatch.

**Running without a desktop:** The interactive viewer needs a display. For
offscreen LeRobot camera rendering on a Linux server, set `MUJOCO_GL=egl` and
use record or rollout without launching `examples/nexarm/simulate.py`.

## Remaining work for a high-fidelity simulator

1. Measure the physical follower's raw zero, direction, and reachable raw range
   for all six joints and store a simulation calibration profile.
2. Identify HX-30HM torque, velocity, damping, deadband, backlash, and command
   latency instead of relying on generic position gains.
3. Add sensor noise, control delay, camera calibration, textures, lighting
   randomization, and physical friction measurements for sim-to-real training.
4. Implement a Gymnasium environment with deterministic reset, cube
   randomization, reward, termination, and success metrics.
5. Add automatic episode resets so large simulation datasets do not require
   manual object placement.
