# Hiwonder NexArm LeRobot VLA Open-Source 6-Axis Robotic Arm

English | [中文](./README_cn.md) | **[Run Guide (`run.md`)](./run.md)** | **[Agent Guide](./AGENT_GUIDE.md)**

> 📖 **Looking for ready-to-run instructions? See [`run.md`](./run.md)** for complete, copy-pasteable guides across simulation (MuJoCo, Gymnasium, Cartesian teleop), physical hardware (leader-follower, vision grasping), dataset recording, training, and Sim-to-Real calibration.

[NexArm](https://www.hiwonder.com/products/nexarm6-axis) is an open-source, [🤗 LeRobot](https://github.com/huggingface/lerobot)-native robotic arm designed for embodied AI research and rapid validation of imitation and reinforcement learning policies. Its dual‑chip architecture (ESP32 + AT32) enables synchronous leader‑follower teleoperation with millisecond‑level tracking latency, generating clean demonstration data that feeds directly into LeRobot training pipelines.

The robotic arm features an all‑metal chassis driven by 65 kg·cm magnetic encoder servos, delivering ±2 mm repeatability and smooth, jitter‑free motion. Combined with advanced inverse kinematics and curve smoothing, it natively computes complex trajectories while minimizing start‑stop vibrations for fluid movement.

For on‑device perception, NexArm integrates a 6 TOPS K230 vision module, allowing you to run multimodal large models and computer vision pipelines—such as YOLO tracking and hand‑eye coordinated grasping—without a PC. All schematics, firmware, and code are open source, making NexArm a transparent and modifiable platform for academic research and real‑world robotic prototyping.

<p align="center">
  <img src="./media/readme/VLA_architecture.jpg" alt="nexarm" width="600"/>
</p>

## 🎯 Compared to SO‑ARM101:

- Payload: 500 g (150% improvement over SO‑ARM101)
- Repeatability: ±2 mm (33% improvement over SO‑ARM101)
- Architecture: ESP32 + AT32 dual‑chip design
- Build: All‑metal body with rotating base and parallel‑rail gripper
- Versatility: Supports multiple control methods out of the box

## End-to-End Robotic Arm Comparison

| Model               | NexArm                                                              | SO-ARM101                                                           |
| ------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Payload             | 500 g                                                               | 200 g                                                               |
| Repeatability       | ±2 mm                                                               | ±3 mm                                                               |
| Workspace           | 0.5 m                                                               | 0.4 m                                                               |
| Joint Servo         | Dual-output-shaft magnetic encoder bus servo                        | Single-output-shaft magnetic encoder bus servo with fixed shaft     |
| Body Material       | Aerospace-grade metal structure                                     | PLA 3D-printed structural parts                                     |
| End Effector        | Parallel rail gripper                                               | 3D-printed vertical-opening gripper                                 |
| Product Positioning | Advanced end-to-end development / advanced embodied AI applications | Entry-level end-to-end development / basic embodied AI applications |

---

## Table of Contents

- [Hardware Overview](#hardware-overview)
- [Installation & Environment Setup](#installation--environment-setup)
- [Quickstart Path A: Simulation (No Hardware Needed)](#quickstart-path-a-simulation-no-hardware-needed)
  - [Interactive MuJoCo Physics Viewer](#interactive-mujoco-physics-viewer)
  - [6-DOF Cartesian Keyboard Teleoperation](#6-dof-cartesian-keyboard-teleoperation)
  - [Interactive Pick-and-Place Task](#interactive-pick-and-place-task)
  - [Gymnasium RL Environment](#gymnasium-rl-environment)
  - [Sim Dataset Generation with Domain Randomization](#sim-dataset-generation-with-domain-randomization)
  - [Rerun 3D Simulation Showcase](#rerun-3d-simulation-showcase)
- [Quickstart Path B: Physical Hardware](#quickstart-path-b-physical-hardware)
  - [Step 1: Find Serial Ports & Set Permissions](#step-1-find-serial-ports--set-permissions)
  - [Step 2: Find Cameras](#step-2-find-cameras)
  - [Step 3: Leader-Follower Teleoperation](#step-3-leader-follower-teleoperation)
  - [Step 4: 6-DOF Cartesian Hardware Teleoperation](#step-4-6-dof-cartesian-hardware-teleoperation)
  - [Step 5: Autonomous Vision-Guided Grasping](#step-5-autonomous-vision-guided-grasping)
  - [Step 6: Collect a Dataset](#step-6-collect-a-dataset)
  - [Step 7: Train a Policy](#step-7-train-a-policy)
  - [Step 8: Run Inference & Rollout](#step-8-run-inference--rollout)
- [Sim-to-Real Pre-Flight Alignment](#sim-to-real-pre-flight-alignment)
- [Kinematics & Dynamics Python API](#kinematics--dynamics-python-api)
- [Code Architecture](#code-architecture)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Hardware Overview

### Components

| Component        | Description                                                                                               |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| **Leader arm**   | ESP32 board driving 6 × HX-30HM servos. Operator freely moves this arm during teleoperation.              |
| **Follower arm** | ESP32 + AT32F421 co-processor driving 6 × HX-30HM servos. Mirrors the leader or executes policy output.   |
| **Servos**       | HX-30HM serial bus servos — 12-bit resolution (0–4095), 1 Mbps, high torque.                              |
| **Cameras**      | 2 × USB cameras: `front` (top-down workspace view) and `wrist` (end-effector close-up), 640×480 @ 30 FPS. |

### Joint Layout (6 DOF)

| Joint | Name            | Notes                                             |
| ----- | --------------- | ------------------------------------------------- |
| 1     | `shoulder_pan`  | Base rotation                                     |
| 2     | `shoulder_lift` | Mirrored between leader and follower (4096 − pos) |
| 3     | `elbow_flex`    | Elbow                                             |
| 4     | `wrist_flex`    | Wrist pitch                                       |
| 5     | `wrist_roll`    | Wrist rotation                                    |
| 6     | `gripper`       | Open/close, mapped range [1195, 2833]             |

### Communication Protocol

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

---

## Installation & Environment Setup

### Recommended — uv (locked and reproducible)

[`uv`](https://docs.astral.sh/uv/) creates a reproducible virtual environment using Python 3.12+ and installs the exact versions in `uv.lock`.

```bash
git clone https://github.com/Hiwonder-official/lerobot-nexarm.git
cd lerobot-nexarm
```

Choose the dependency profile matching your workflow:

```bash
# 1. Simulation Only (MuJoCo, Gymnasium, PyTorch, OpenCV, PySerial)
uv sync --locked

# 2. Physical Hardware & Teleoperation (adds serial protocols + Rerun viz)
uv sync --locked --extra nexarm --extra viz

# 3. Hardware + Policy Training (adds datasets, wandb, accelerate)
uv sync --locked --extra nexarm --extra training --extra viz

# 4. Development & Testing (adds pytest, ruff, mypy, pre-commit)
uv sync --locked --extra nexarm --extra dev --extra test

# 5. Full Stack (all policies, simulation tools, benchmarks)
uv sync --locked --extra all
```

> **Important**: Always run commands through `uv run <command>` (e.g. `uv run python ...` or `uv run lerobot-train`). Do not manually activate the `.venv` or install packages with `pip`.

### Git LFS & Hugging Face Hub

```bash
# Install Git LFS test artifacts and 3D meshes
git lfs install && git lfs pull

# Log into Hugging Face (required to push/pull private datasets and policies)
uv run hf auth login
```

### Linux Serial Permissions

If operating physical hardware or USB cameras on Linux (Ubuntu/Debian), grant your user read/write access to serial ports:

```bash
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyUSB*
```

_(Log out and log back in for the group membership change to take effect)._

### Hardware Connections (For Physical Robot)

1. Plug the **follower arm** ESP32 into the PC via USB and connect its 12V / 5A power supply.
2. Plug the **leader arm** ESP32 into the PC via USB (powered via USB).
3. Plug in both USB cameras (`front` top-down workspace view and `wrist` end-effector view).

### Platform Support

| Platform      | Status   | Notes                                                   |
| ------------- | -------- | ------------------------------------------------------- |
| Ubuntu 20.04+ | Verified | Port format `/dev/ttyUSB0`; add user to `dialout` group |
| Windows 10/11 | Verified | Install CH340 driver; port format `COM19`               |
| macOS         | Verified | Port format `/dev/tty.usbserial-xxx`                    |

### Verify Installation

```bash
uv run python -c "from lerobot.robots.nexarm_sim import NexArmSim; from lerobot.robots.nexarm_follower import NexArmFollower; print('NexArm installation OK!')"
```

---

## Quickstart Path A: Simulation (No Hardware Needed)

You can explore, teleoperate, and train policies on the NexArm platform immediately in simulation without any physical hardware.

### Interactive MuJoCo Physics Viewer

Launch the interactive MuJoCo viewer to inspect the 6-DOF NexArm model and drag joints with actuator sliders:

```bash
uv run python examples/nexarm/simulate.py
```

### 6-DOF Cartesian Keyboard Teleoperation

Control the simulated robot's end-effector in 3D Cartesian task space ($X, Y, Z$, Roll, Pitch, Yaw) with real-time Damped Least-Squares (DLS) Inverse Kinematics:

```bash
uv run python examples/nexarm/teleoperate_cartesian.py --robot sim
```

**Keybindings:**

- **`W` / `S`**: Translate $\pm X$ (Forward / Backward)
- **`A` / `D`**: Translate $\pm Y$ (Left / Right)
- **`R` / `F`**: Translate $\pm Z$ (Up / Down)
- **`U` / `J`**: Rotate $\pm \text{Roll}$
- **`I` / `K`**: Rotate $\pm \text{Pitch}$
- **`O` / `L`**: Rotate $\pm \text{Yaw}$
- **`Space`**: Toggle Gripper (Open $\leftrightarrow$ Close)
- **`1` / `2` / `3`**: Step size: Fine (2 mm), Medium (5 mm), Coarse (15 mm)
- **`H`**: Return to Home pose | **`Q`**: Exit

> _Headless / CI mode_: Add `--dry-run` to run 50 headless steps without a GUI:
>
> ```bash
> uv run python examples/nexarm/teleoperate_cartesian.py --robot sim --dry-run
> ```

### Interactive Pick-and-Place Task

Run the benchmark pick-and-place simulation where the robot grasps a red cube and deposits it into the green target zone:

```bash
uv run python examples/nexarm/pick_place_sim.py --seed 0
```

### Gymnasium RL Environment

Run the registered Gymnasium environment `NexArmPickPlace-v0` with camera rendering and step rewards:

```bash
uv run python examples/nexarm/demo_gym_env.py
```

### Sim Dataset Generation with Domain Randomization

Generate synthetic demonstration datasets with visual and dynamics domain randomization, plus simulated USB transport latency:

```bash
uv run python examples/nexarm/generate_sim_dataset.py \
  --repo-id local/nexarm_sim2real \
  --num-episodes 50 \
  --domain-randomization \
  --action-delay-steps 2
```

### Rerun 3D Simulation Showcase

Visualize kinematics, joint sweeps, and multi-axis trajectories in the [Rerun](https://rerun.io) viewer:

```bash
uv run python examples/nexarm/visualize_rerun_sim.py --mode joint_sweep
```

---

## Quickstart Path B: Physical Hardware

Follow these steps to operate the physical NexArm leader-follower system.

### Step 1: Find Serial Ports & Set Permissions

Identify which serial port corresponds to the leader arm and which to the follower arm:

```bash
uv run lerobot-find-port
```

- **Linux**: Typically `/dev/ttyUSB0` (Leader) and `/dev/ttyUSB1` (Follower).
- **Windows**: Typically `COM18` (Leader) and `COM19` (Follower).

> _Tip_: Plug in one arm at a time if you are unsure which port belongs to which arm.

### Step 2: Find Cameras

Scan and identify the OpenCV indices for the `front` and `wrist` cameras:

```bash
uv run lerobot-find-cameras opencv
```

Save test snapshots to confirm views:

```bash
uv run python -c "
import cv2
for idx in [0, 1]:
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'cam_{idx}.png', frame)
            print(f'Captured cam_{idx}.png')
        cap.release()
"
```

- **`front`**: Top-down view of the entire workspace.
- **`wrist`**: Close-up view of the gripper and end-effector.

### Step 3: Leader-Follower Teleoperation

Verify the leader-follower link. The leader arm runs torque-free so the operator moves it freely; the follower mirrors every joint in real time.

```bash
# Linux
uv run python examples/nexarm/teleoperate.py \
  --leader-port /dev/ttyUSB0 \
  --follower-port /dev/ttyUSB1 \
  --front-cam 0 --wrist-cam 1 --fps 30

# Windows
uv run python examples/nexarm/teleoperate.py \
  --leader-port COM18 \
  --follower-port COM19 \
  --front-cam 0 --wrist-cam 1 --fps 30
```

**Save a replayable Rerun recording:**

```bash
uv run python examples/nexarm/teleoperate.py \
  --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 \
  --rerun-save-path outputs/rerun/nexarm_teleop.rrd
```

**Or use the LeRobot CLI directly:**

```bash
uv run lerobot-teleoperate \
  --robot.type=nexarm_follower --robot.port=/dev/ttyUSB1 \
  --teleop.type=nexarm_leader  --teleop.port=/dev/ttyUSB0 \
  --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30}}' \
  --display_data=true
```

### Step 4: 6-DOF Cartesian Hardware Teleoperation

Control the physical arm directly using keyboard Cartesian inputs without needing a physical leader arm:

```bash
uv run python examples/nexarm/teleoperate_cartesian.py \
  --robot real \
  --port /dev/ttyUSB1
```

### Step 5: Autonomous Vision-Guided Grasping

Execute autonomous visual object detection (HSV color tracking), 2D-to-3D workspace projection, and smooth pick-and-place trajectories on hardware:

```bash
uv run python examples/nexarm/vision_grasp.py \
  --robot real \
  --port /dev/ttyUSB1 \
  --camera-index 0
```

### Step 6: Collect a Dataset

Record demonstration episodes via teleoperation for imitation learning:

```bash
uv run python examples/nexarm/record.py \
  --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 \
  --repo-id local/nexarm_pick \
  --task "Pick up the red block and place it into the tray" \
  --num-episodes 50 --episode-time 10 --reset-time 10
```

**Episode Controls:**

- **`Enter`**: Start / confirm next episode.
- **`←` (Left Arrow)**: Redo current episode (discards bad demonstration).
- **`ESC`**: Finish recording session early and save.

**Verify recorded episodes:**

```bash
# Visualize dataset
uv run lerobot-dataset-viz --repo-id local/nexarm_pick --episode-index 0

# Replay episode on follower arm
uv run lerobot-replay --robot.type=nexarm_follower --robot.port=/dev/ttyUSB1 \
  --dataset.repo_id=local/nexarm_pick --dataset.episode=0
```

### Step 7: Train a Policy

Train an ACT (Action Chunking with Transformers) policy on the collected dataset:

```bash
uv run lerobot-train \
  --dataset.repo_id=local/nexarm_pick \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/nexarm_act \
  --job_name=nexarm_act \
  --batch_size=32 \
  --steps=100000 \
  --save_freq=25000 \
  --wandb.enable=false
```

Checkpoints are saved to:

```
outputs/train/nexarm_act/checkpoints/last/pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_postprocessor.json
└── train_config.json
```

### Step 8: Run Inference & Rollout

Deploy the trained policy on the physical robot. The follower arm executes actions predicted by the model:

```bash
uv run python examples/nexarm/rollout.py \
  --follower-port /dev/ttyUSB1 \
  --policy-path outputs/train/nexarm_act/checkpoints/last/pretrained_model \
  --front-cam 0 --wrist-cam 1 --strategy sentry
```

---

## Sim-to-Real Pre-Flight Alignment

Before deploying a policy trained in simulation onto the real arm, verify that the physical camera perspective and workspace match the simulation:

```bash
uv run python examples/nexarm/calibrate_camera_alignment.py \
  --camera top \
  --real-camera-index 0 \
  --blend-alpha 0.5
```

- Press **`M`**: Cycle view mode (Alpha Blend $\to$ Side-by-Side $\to$ Canny Edge Overlay).
- Press **`[` / `]`**: Adjust transparency.
- Press **`S`**: Save alignment snapshot.

---

## Kinematics & Dynamics Python API

Module: [`src/lerobot/motors/nexarm/kinematics_dynamics.py`](file:///home/thanh/code/vinuni/lerobot-nexarm/src/lerobot/motors/nexarm/kinematics_dynamics.py)

```python
import numpy as np
from lerobot.motors.nexarm import NexArmKinematicsDynamics

kd = NexArmKinematicsDynamics()

# 5 arm joint positions in radians
q = np.array([0.0, 0.4, 0.8, -0.4, 0.0])

# 1. Forward Kinematics (FK)
fk = kd.forward_kinematics(q)
print("EE Position (x, y, z):", fk["position"])
print("EE Orientation (RPY deg):", np.rad2deg(fk["rpy"]))

# 2. 6x5 Spatial Geometric Jacobian
J = kd.jacobian(q)

# 3. Damped Least-Squares Inverse Kinematics (IK)
target_pos = np.array([0.22, 0.0, 0.15])
success, q_sol = kd.inverse_kinematics(target_pos, q_init=q)

# 4. Generalized Mass Matrix M(q)
M = kd.mass_matrix(q)

# 5. Gravity Compensation Torques g(q)
tau_grav = kd.gravity_torques(q)
```

### Hardware Protection (Idle Torque Timeout)

To protect the bus servos against thermal degradation and mechanical fatigue, `NexArmFollowerConfig` has an automatic `idle_timeout_s` (defaults to `20.0` seconds). If no action is received for $> 20$ seconds, motor torque automatically turns off; the next `send_action()` smoothly re-engages torque.

---

## Code Architecture

```
src/lerobot/
├── motors/nexarm/
│   ├── __init__.py                  # Exports NexArmMotorsBus, NexArmKinematicsDynamics
│   ├── nexarm.py                    # CommProtocol UART framing, read/write, torque, bridge mode
│   └── kinematics_dynamics.py       # C-accelerated FK, IK, Jacobian, M(q), g(q)
├── robots/
│   ├── nexarm_follower/             # Follower robot driver (connect, observe, send_action)
│   ├── nexarm_sim/                  # MuJoCo single-arm simulation & pick-and-place task
│   └── mobile_bi_nexarm_sim/        # Bimanual mobile robot simulation
├── teleoperators/nexarm_leader/     # Leader arm driver (torque disable, joint mirroring)
└── envs/nexarm.py                   # Gymnasium environment (NexArmPickPlace-v0)
```

---

## Testing & Quality Assurance

```bash
# 1. Run full unit test suite
uv run pytest tests/ -v

# 2. Run NexArm robot and motor tests
uv run pytest tests/motors/ tests/robots/ -v

# 3. Code formatting and lint checks
uv run ruff check .
uv run ruff format . --check

# 4. Full pre-commit suite (lint, format, types, typos, security)
uv run pre-commit run --all-files --show-diff-on-failure
```

---

## Troubleshooting & FAQs

**Serial port permission denied (Linux)**

```bash
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyUSB*
# Log out and back in
```

**Camera not found or busy**

- Run `uv run lerobot-find-cameras opencv` to scan available indices.
- Close other applications using the camera (OBS, browser, Zoom).
- Re-scan after unplugging and replugging USB devices.

**Follower arm does not move during teleoperation**

1. Confirm follower port and leader port are not swapped.
2. Confirm the follower ESP32 firmware supports CMD 68 (LeRobot bridge mode).
3. Try power-cycling the follower arm (ensure 12V power supply is plugged in).

**TimeoutError during data collection**

```
TimeoutError: No position reply from NexArm
```

The leader firmware prints debug lines over Serial that corrupt protocol frames. The driver automatically retries 3 times. To eliminate this permanently, comment out `Serial.printf` calls in `Nex_Arm.ino` inside the `lerobotMode == true` branch and reflash the firmware.

**Training loss does not decrease / arm moves hesitantly**

- Ensure you have at least 50 demonstration episodes with consistent initial states.
- If the arm collapses to the mean, lower KL weight: `--policy.kl_weight=5.0` or `1.0`.
- Increase batch size: `--batch_size=32` or `--batch_size=64`.
- Train longer: `--steps=200000`.

**Low inference FPS on CPU**
ACT uses action chunking (`chunk_size=100`) so CPU inference is normally 20–30 Hz. If slower:

- Verify no background processes saturate CPU cores.
- Dual-camera capture adds ~45 ms per frame — this is normal.

**Checksum errors**
The leader firmware has a known checksum bug. The driver automatically accepts both correct and vendor checksums for maximum compatibility.
