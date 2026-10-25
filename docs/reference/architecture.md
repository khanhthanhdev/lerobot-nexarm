# Architecture and conventions

How this repository sits on top of LeRobot, where the code lives, and the conventions every script and guide shares.

## LeRobot in 60 seconds

LeRobot = **datasets + policies + envs + robot control**, unified by a small set of strong abstractions.

- **`LeRobotDataset`** — episode-aware dataset (video or images + actions + state), loadable from the Hub or disk.
- **Policies** (`ACT`, `Diffusion`, `SmolVLA`, `π0`, `π0.5`, `Wall-X`, `X-VLA`, `VQ-BeT`, `TD-MPC`, …) — all inherit `PreTrainedPolicy` and can be pushed/pulled from the Hub.
- **Processors** — small composable transforms between dataset → policy → robot.
- **Envs** (sim: `nexarm`, etc.) and **Robots** (real: `nexarm_follower`; simulated: `nexarm_sim`) — same action/observation contract so code swaps cleanly.
- **CLI** — `lerobot-record`, `lerobot-train`, `lerobot-eval`, `lerobot-teleoperate`, `lerobot-calibrate`, `lerobot-find-port`, `lerobot-replay`, `lerobot-rollout`.

LeRobot's own documentation is at <https://huggingface.co/docs/lerobot>.

## How this repository uses LeRobot

This repository contains no LeRobot sources. `lerobot` is installed from upstream, pinned by commit in `[tool.uv.sources]` of `pyproject.toml`. The NexArm robot, teleoperator, env and desktop-engine code are LeRobot plugins in `packages/`, and the research policies are plugins in `policies/`; `uv sync` installs them all. LeRobot's CLIs find them through `register_third_party_plugins()`, so `--robot.type=nexarm_follower`, `--teleop.type=nexarm_leader`, `--env.type=nexarm` and `--policy.type=turbovla` / `bspline` work in the stock commands. Upstream's robots, policies and envs stay available next to them.

`lerobot-find-joint-limits` and the HIL-SERL commands do not load plugins, so they do not see the NexArm types. NexArm behaviour that LeRobot lacks lives in the plugins and wrappers: `examples/nexarm/record_collection.py` (collection flags), `lerobot-nexarm-eval` (per-stage success rates) and `lerobot-nexarm-rollout` (`.rrd` recording). Moving to a newer LeRobot is described in [`CONTRIBUTING.md`, Upgrading LeRobot](../../CONTRIBUTING.md#upgrading-lerobot).

## Code layout

```
packages/                            # LeRobot plugins, each under <package>/src/<package>/
├── lerobot_robot_nexarm/
│   ├── motors/                      # NexArmMotorsBus (UART framing, torque, bridge mode), kinematics_dynamics.py
│   ├── nexarm_follower/             # Follower robot driver (connect, observe, send_action)
│   ├── nexarm_sim/                  # MuJoCo single-arm simulation, pick-and-place and stack-bowls tasks
│   └── mobile_bi_nexarm_sim/        # Bimanual mobile robot simulation
├── lerobot_teleoperator_nexarm/     # Leader arm driver (torque disable, joint mirroring)
├── lerobot_env_nexarm/              # Gymnasium environments (NexArmPickPlace-v0, NexArmStackBowls-v0), sim benchmark
└── nexarm_service/                  # Desktop-app engine (lerobot-nexarm-service)
policies/                            # lerobot_policy_turbovla, lerobot_policy_bspline
examples/nexarm/                     # Ready-to-run tutorials and launchers
tools/nexarm/                        # Calibration, probes, dataset audits, merges, uploads, videos
sim/                                 # MuJoCo MJCF, URDF, Isaac Lab configs, meshes, exporters
experiments/                         # One YYMMDD-<slug> folder per experiment
firmware/, hardware/                 # Vendor controller firmware and protocol docs; mechanical files
```

[`AGENTS.md`](../../AGENTS.md) has the per-module table.

## Hardware

| Component        | Description                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| **Leader arm**   | ESP32 board driving 6 × HX-30HM servos. Operator freely moves this arm during teleoperation.            |
| **Follower arm** | ESP32 + AT32F421 co-processor driving 6 × HX-30HM servos. Mirrors the leader or executes policy output. |
| **Servos**       | HX-30HM serial bus servos — 12-bit resolution (0–4095), 1 Mbps, high torque.                            |
| **Cameras**      | `front` and `wrist` USB cameras plus an Intel RealSense D435i `top` camera (RGB), all 640×480 @ 30 FPS. |

The follower enters LeRobot bridge mode (CMD 68) when connected. The leader runs with torque disabled (CMD 98) so the operator moves it freely.

### Communication protocol

NexArm uses a custom CommProtocol over USB serial:

```
Frame: [0xFF][0xFF][ID][LEN][CMD][ARGS...][CHECKSUM]
```

| CMD | Function                                                         | Direction            |
| --- | ---------------------------------------------------------------- | -------------------- |
| 56  | Set motion speed and acceleration (`motion_speed`, `motion_acc`) | Host → Follower      |
| 68  | Enter/exit LeRobot bridge mode (follower only)                   | Host → Follower      |
| 96  | Read 6 servo positions (12-byte reply)                           | Host → Device → Host |
| 97  | Write 6 servo positions (12 bytes, no reply)                     | Host → Device        |
| 98  | Enable/disable torque                                            | Host → Device        |

The vendor protocol documents are in [`firmware/`](../../firmware/README.md).

## Conventions

Shared constants live in `packages/lerobot_robot_nexarm/src/lerobot_robot_nexarm/motors/nexarm.py`; keep code and docs consistent with them.

### Joints

| Joint | Name            | Notes                                             |
| ----- | --------------- | ------------------------------------------------- |
| 1     | `shoulder_pan`  | Base rotation                                     |
| 2     | `shoulder_lift` | Mirrored between leader and follower (4096 − pos) |
| 3     | `elbow_flex`    | Elbow                                             |
| 4     | `wrist_flex`    | Wrist pitch                                       |
| 5     | `wrist_roll`    | Wrist rotation                                    |
| 6     | `gripper`       | Raw range [1195, 2833]: 1195 open, 2833 closed    |

Arm joints use the servo constant `rad = (raw - 2048) * 2π / 4096` (4096 ticks per revolution), in hardware and in the MuJoCo simulation. Datasets store raw servo units for state and action.

### Ports, cameras and datasets

- Leader `/dev/ttyUSB0`, follower `/dev/ttyUSB1` on Linux.
- Cameras are named `front`, `wrist` and `top`; dataset keys are `observation.images.<name>`. The RealSense `top` camera is on by default; `--no-top-cam` disables it. A policy must run with exactly the cameras it was trained on.
- `lerobot-record`, `record.py` and `record_collection.py` append `_YYYYMMDD_HHMMSS` to `repo_id`.
- Simulation datasets use `robot_type="nexarm_follower"`, the same as real datasets, and record the joint mapping in `generation_report.json`.
