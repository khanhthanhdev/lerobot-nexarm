# NexArm Run Guide: Simulation, Hardware, Teleoperation, Training & Sim2Real

This guide provides copy-pasteable instructions for setting up, running, and developing with the **Hiwonder NexArm** platform on LeRobot. Whether you are running pure simulation without physical hardware or operating a real dual-arm leader-follower rig, this document covers every workflow.

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
   - [Install `uv` & Clone](#install-uv--clone)
   - [Dependency Profiles (Choose Your Extra)](#dependency-profiles-choose-your-extra)
   - [Git LFS & Hugging Face Authentication](#git-lfs--hugging-face-authentication)
   - [Linux Serial Permissions](#linux-serial-permissions)
2. [Quickstart Path A: Simulation (Zero Hardware Needed)](#2-quickstart-path-a-simulation-zero-hardware-needed)
   - [2.1 Interactive MuJoCo Physics Viewer](#21-interactive-mujoco-physics-viewer)
   - [2.2 6-DOF Cartesian Keyboard Teleoperation](#22-6-dof-cartesian-keyboard-teleoperation)
   - [2.3 Interactive Pick-and-Place Task](#23-interactive-pick-and-place-task)
   - [2.4 Gymnasium RL Environment Demo](#24-gymnasium-rl-environment-demo)
   - [2.5 Sim Dataset Generation with Domain Randomization](#25-sim-dataset-generation-with-domain-randomization)
   - [2.6 Rerun 3D Simulation Showcase](#26-rerun-3d-simulation-showcase)
3. [Quickstart Path B: Physical Hardware Setup & Teleoperation](#3-quickstart-path-b-physical-hardware-setup--teleoperation)
   - [3.1 Hardware Wiring & Power Checklist](#31-hardware-wiring--power-checklist)
   - [3.2 Find Serial Ports](#32-find-serial-ports)
   - [3.3 Find & Verify USB Cameras](#33-find--verify-usb-cameras)
   - [3.4 Real-Time Leader-Follower Teleoperation](#34-real-time-leader-follower-teleoperation)
   - [3.5 6-DOF Cartesian Hardware Teleoperation](#35-6-dof-cartesian-hardware-teleoperation)
   - [3.6 Autonomous Vision-Guided Grasping](#36-autonomous-vision-guided-grasping)
   - [3.7 Sim-to-Real Pre-Flight Camera Alignment](#37-sim-to-real-pre-flight-camera-alignment)
4. [Imitation Learning Pipeline: Record -> Train -> Rollout](#4-imitation-learning-pipeline-record---train---rollout)
   - [4.1 Record Demonstrations](#41-record-demonstrations)
   - [4.2 Inspect & Visualize Dataset](#42-inspect--visualize-dataset)
   - [4.3 Replay Demonstrations on Hardware](#43-replay-demonstrations-on-hardware)
   - [4.4 Train a Policy (ACT / Diffusion)](#44-train-a-policy-act--diffusion)
   - [4.5 Deploy & Rollout Trained Policy](#45-deploy--rollout-trained-policy)
5. [Python API: Kinematics, Dynamics & Hardware Control](#5-python-api-kinematics-dynamics--hardware-control)
   - [5.1 C-Accelerated Kinematics & Dynamics](#51-c-accelerated-kinematics--dynamics)
   - [5.2 Direct Motor Bus Access](#52-direct-motor-bus-access)
   - [5.3 Hardware Protections (Idle Torque Timeout)](#53-hardware-protections-idle-torque-timeout)
6. [Testing & Quality Assurance](#6-testing--quality-assurance)
7. [Troubleshooting & FAQs](#7-troubleshooting--faqs)

---

## 1. Prerequisites & Environment Setup

### Install `uv` & Clone

[`uv`](https://docs.astral.sh/uv/) is the required package manager. It creates reproducible, locked virtual environments with Python 3.12+.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/Hiwonder-official/lerobot-nexarm.git
cd lerobot-nexarm
```

### Dependency Profiles (Choose Your Extra)

Sync the exact dependencies needed for your workflow:

```bash
# Profile 1: Simulation only (MuJoCo, Gymnasium, OpenCV, PyTorch, PySerial)
uv sync --locked

# Profile 2: Physical Hardware & Teleoperation (adds serial protocols + Rerun viz)
uv sync --locked --extra nexarm --extra viz

# Profile 3: Hardware + Policy Training (adds datasets, wandb, accelerate)
uv sync --locked --extra nexarm --extra training --extra viz

# Profile 4: Development & Testing (adds pytest, ruff, mypy, pre-commit)
uv sync --locked --extra nexarm --extra dev --extra test

# Profile 5: Everything (all policies, simulation tools, benchmarks)
uv sync --locked --extra all
```

> **Note**: Always invoke commands through `uv run <command>` (e.g. `uv run python ...` or `uv run lerobot-teleoperate`). Do not manually activate the `.venv` or install packages with `pip`.

### Git LFS & Hugging Face Authentication

```bash
# Pull test fixtures and assets
git lfs install && git lfs pull

# Log into Hugging Face Hub (required for uploading datasets or downloading private models)
uv run hf auth login
```

### Linux Serial Permissions

On Linux (Ubuntu/Debian), grant your user read/write access to serial USB ports:

```bash
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyUSB*
```

_(Log out and log back in for group membership to take effect)._

---

## 2. Quickstart Path A: Simulation (Zero Hardware Needed)

You do **not** need physical robot hardware to start development. The repository includes native MuJoCo physics models, 6-DOF Cartesian solvers, and Gymnasium environments.

### 2.1 Interactive MuJoCo Physics Viewer

Launch the passive MuJoCo viewer to inspect the NexArm model and drag joints with interactive sliders:

```bash
uv run python examples/nexarm/simulate.py
```

Optional flags:

- `--model sim/fusion_export/scene.xml`: Specify a custom MJCF scene model.
- `--fps 30`: Target simulation frame rate.

### 2.2 6-DOF Cartesian Keyboard Teleoperation

Control the simulated robot's end-effector in 3D Cartesian task space ($X, Y, Z$, Roll, Pitch, Yaw) using real-time Damped Least-Squares (DLS) Inverse Kinematics:

```bash
uv run python examples/nexarm/teleoperate_cartesian.py --robot sim
```

**Keybindings & Controls:**

| Key                 | Action                                                                                   |
| :------------------ | :--------------------------------------------------------------------------------------- |
| **`W` / `S`**       | Translate End-Effector **$\pm X$** (Forward / Backward)                                  |
| **`A` / `D`**       | Translate End-Effector **$\pm Y$** (Left / Right)                                        |
| **`R` / `F`**       | Translate End-Effector **$\pm Z$** (Up / Down)                                           |
| **`U` / `J`**       | Rotate End-Effector **$\pm \text{Roll}$**                                                |
| **`I` / `K`**       | Rotate End-Effector **$\pm \text{Pitch}$**                                               |
| **`O` / `L`**       | Rotate End-Effector **$\pm \text{Yaw}$**                                                 |
| **`Space`**         | **Toggle Gripper** (Open $\leftrightarrow$ Close)                                        |
| **`1` / `2` / `3`** | Step resolution: **Fine** (2 mm / 1°), **Medium** (5 mm / 2.5°), **Coarse** (15 mm / 5°) |
| **`H`**             | Return to default **Home Pose**                                                          |
| **`Q`**             | **Exit** gracefully                                                                      |

> **Headless / CI Mode**: Add `--dry-run` to run 50 headless steps without opening a window:
>
> ```bash
> uv run python examples/nexarm/teleoperate_cartesian.py --robot sim --dry-run
> ```

### 2.3 Interactive Pick-and-Place Task

Run the benchmark pick-and-place simulation where the robot must grasp a red cube and place it into the green target zone:

```bash
uv run python examples/nexarm/pick_place_sim.py --seed 0
```

Add `--auto-reset` to continuously loop through randomized episodes.

### 2.4 Gymnasium RL Environment Demo

Run the registered Gymnasium environment `NexArmPickPlace-v0` with camera rendering and step rewards:

```bash
uv run python examples/nexarm/demo_gym_env.py
```

### 2.5 Sim Dataset Generation with Domain Randomization

Generate synthetic demonstration datasets with visual and dynamics domain randomization, plus simulated USB transport latency:

```bash
# Standard dataset
uv run python examples/nexarm/generate_sim_dataset.py \
    --repo-id local/nexarm_sim_standard \
    --num-episodes 20

# Sim2Real-hardened dataset (Visual DR + Dynamics DR + 2-step latency)
uv run python examples/nexarm/generate_sim_dataset.py \
    --repo-id local/nexarm_sim2real \
    --num-episodes 50 \
    --domain-randomization \
    --action-delay-steps 2
```

### 2.6 Rerun 3D Simulation Showcase

Visualize kinematics, joint sweeps, and multi-axis trajectories in the [Rerun](https://rerun.io) viewer:

```bash
# Single joint limits sweep
uv run python examples/nexarm/visualize_rerun_sim.py --mode joint_sweep

# Full workspace reach visualization
uv run python examples/nexarm/visualize_rerun_sim.py --mode workspace

# Complete multi-mode showcase saved to .rrd
uv run python examples/nexarm/visualize_rerun_sim.py --mode all --view save
```

---

## 3. Quickstart Path B: Physical Hardware Setup & Teleoperation

### 3.1 Hardware Wiring & Power Checklist

1. **Follower Arm**:
   - Connect 12V / 5A DC power supply.
   - Connect USB cable from the ESP32 board to the host PC.
2. **Leader Arm**:
   - Connect USB cable from the master ESP32 board to the host PC (powered via USB).
3. **Cameras**:
   - `front` camera: Mounted overlooking the workspace table (top-down view).
   - `wrist` camera: Mounted on the arm near the gripper facing the end-effector.

### 3.2 Find Serial Ports

Run the port finder utility to identify leader and follower ports:

```bash
uv run lerobot-find-port
```

- Unplug and replug each arm when prompted.
- **Linux**: Typically `/dev/ttyUSB0` (Leader) and `/dev/ttyUSB1` (Follower).
- **Windows**: Typically `COM18` (Leader) and `COM19` (Follower).
- **macOS**: Typically `/dev/tty.usbserial-*`.

### 3.3 Find & Verify USB Cameras

Identify the OpenCV camera indices for the `front` and `wrist` cameras:

```bash
uv run lerobot-find-cameras opencv
```

To visually check the camera feeds, save snapshots with:

```bash
uv run python -c "
import cv2
for idx in [0, 1, 2]:
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f'cam_{idx}.png', frame)
            print(f'Captured cam_{idx}.png')
        cap.release()
"
```

### 3.4 Real-Time Leader-Follower Teleoperation

Move the leader arm by hand; the follower arm mirrors every joint in real time:

```bash
# Linux
uv run python examples/nexarm/teleoperate.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --front-cam 0 \
    --wrist-cam 1 \
    --fps 30

# Windows
uv run python examples/nexarm/teleoperate.py \
    --leader-port COM18 \
    --follower-port COM19 \
    --front-cam 0 \
    --wrist-cam 1
```

**Save a replayable Rerun recording:**

```bash
uv run python examples/nexarm/teleoperate.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --rerun-save-path outputs/rerun/teleop_session.rrd
```

**Run via LeRobot CLI:**

```bash
uv run lerobot-teleoperate \
    --robot.type=nexarm_follower --robot.port=/dev/ttyUSB1 \
    --teleop.type=nexarm_leader  --teleop.port=/dev/ttyUSB0 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30}}' \
    --display_data=true
```

### 3.5 6-DOF Cartesian Hardware Teleoperation

Control the physical arm directly using keyboard Cartesian inputs without needing a physical leader arm:

```bash
uv run python examples/nexarm/teleoperate_cartesian.py \
    --robot real \
    --port /dev/ttyUSB1
```

### 3.6 Autonomous Vision-Guided Grasping

Execute autonomous visual object detection (HSV color tracking), 2D-to-3D workspace projection, and smooth pick-and-place trajectories:

```bash
# Run on physical hardware
uv run python examples/nexarm/vision_grasp.py \
    --robot real \
    --port /dev/ttyUSB1 \
    --camera-index 0

# Test in simulation first
uv run python examples/nexarm/vision_grasp.py --robot sim
```

### 3.7 Sim-to-Real Pre-Flight Camera Alignment

Overlay live physical camera frames onto the MuJoCo simulated camera view to align field of view, focal angle, and table placement:

```bash
uv run python examples/nexarm/calibrate_camera_alignment.py \
    --camera top \
    --real-camera-index 0 \
    --blend-alpha 0.5
```

- Press **`M`** to cycle view modes: Alpha Blend $\to$ Side-by-Side $\to$ Canny Edge Overlay.
- Press **`[` / `]`** to adjust transparency.
- Press **`S`** to save a snapshot to disk.

---

## 4. Imitation Learning Pipeline: Record -> Train -> Rollout

### 4.1 Record Demonstrations

Record demonstration episodes using leader-follower teleoperation:

```bash
# Using example script wrapper
uv run python examples/nexarm/record.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --front-cam 0 \
    --wrist-cam 1 \
    --repo-id local/nexarm_pick_cube \
    --task "Pick up the red cube and place it into the green zone" \
    --num-episodes 50 \
    --episode-time 12 \
    --reset-time 8

# Using lerobot-record CLI directly
uv run lerobot-record \
    --robot.type=nexarm_follower --robot.port=/dev/ttyUSB1 \
    --teleop.type=nexarm_leader  --teleop.port=/dev/ttyUSB0 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30}}' \
    --dataset.repo_id=local/nexarm_pick_cube \
    --dataset.single_task="Pick up the red cube and place it into the green zone" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=12 \
    --dataset.reset_time_s=8 \
    --display_data=true
```

**Episode Controls During Recording:**

- **`Enter`**: Start / confirm next episode.
- **`←` (Left Arrow)**: Redo current episode (discards bad demonstration).
- **`ESC`**: Finish recording session early and save dataset.

### 4.2 Inspect & Visualize Dataset

Visualize recorded episodes, camera video streams, and joint state trajectories:

```bash
uv run lerobot-dataset-viz --repo-id local/nexarm_pick_cube --episode-index 0
```

### 4.3 Replay Demonstrations on Hardware

Verify recorded actions by playing back an episode directly on the follower arm:

```bash
uv run lerobot-replay \
    --robot.type=nexarm_follower \
    --robot.port=/dev/ttyUSB1 \
    --dataset.repo_id=local/nexarm_pick_cube \
    --dataset.episode=0
```

### 4.4 Train a Policy (ACT / Diffusion)

Train an **Action Chunking with Transformers (ACT)** policy on your dataset:

```bash
uv run lerobot-train \
    --dataset.repo_id=local/nexarm_pick_cube \
    --policy.type=act \
    --policy.device=cuda \
    --output_dir=outputs/train/nexarm_act \
    --job_name=nexarm_act \
    --batch_size=32 \
    --steps=100000 \
    --save_freq=25000 \
    --wandb.enable=false
```

_Note_: If VRAM is limited (< 8 GB), reduce `--batch_size=16` or `--batch_size=8`.

### 4.5 Deploy & Rollout Trained Policy

Run real-time inference on the physical follower arm (no leader arm required):

```bash
# Using rollout script
uv run python examples/nexarm/rollout.py \
    --follower-port /dev/ttyUSB1 \
    --policy-path outputs/train/nexarm_act/checkpoints/last/pretrained_model \
    --front-cam 0 \
    --wrist-cam 1 \
    --strategy sentry

# Using lerobot-rollout CLI
uv run lerobot-rollout \
    --robot.type=nexarm_follower \
    --robot.port=/dev/ttyUSB1 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30}}' \
    --policy.path=outputs/train/nexarm_act/checkpoints/last/pretrained_model \
    --strategy.type=sentry \
    --display_data=true
```

---

## 5. Python API: Kinematics, Dynamics & Hardware Control

### 5.1 C-Accelerated Kinematics & Dynamics

Import the kinematics/dynamics solver backed by MuJoCo:

```python
import numpy as np
from lerobot.motors.nexarm import NexArmKinematicsDynamics

kd = NexArmKinematicsDynamics()

# 5 arm joint positions in radians
q = np.array([0.0, 0.4, 0.8, -0.4, 0.0])

# 1. Forward Kinematics
fk = kd.forward_kinematics(q)
print("EE Position [x, y, z]:", fk["position"])
print("EE RPY (degrees):", np.rad2deg(fk["rpy"]))

# 2. Geometric Jacobian (6 x 5)
J = kd.jacobian(q)
print("Jacobian shape:", J.shape)

# 3. Damped Least-Squares Inverse Kinematics
target_pos = np.array([0.22, 0.0, 0.15])
success, q_sol = kd.inverse_kinematics(target_pos, q_init=q)
if success:
    print("IK Solution (rad):", q_sol)

# 4. Mass / Inertia Matrix M(q)
M = kd.mass_matrix(q)

# 5. Gravity Compensation Torques
tau_g = kd.gravity_torques(q)
print("Gravity torques (Nm):", tau_g)
```

### 5.2 Direct Motor Bus Access

Directly query and command the 6 HX-30HM servos via `NexArmMotorsBus`:

```python
from lerobot.motors.nexarm import NexArmMotorsBus

bus = NexArmMotorsBus(port="/dev/ttyUSB1")
bus.connect()

# Set motion profile: speed (0-3400 raw units/s), acceleration ramp (0-254)
bus.set_motion_profile(speed=2000, acc=100)

# Read 6 servo positions (0-4095 range)
positions = bus.read_positions()
print("Current positions:", positions)

# Enable / disable torque
bus.set_torque(False)  # Free movement
bus.set_torque(True)   # Active hold

bus.disconnect()
```

### 5.3 Hardware Protections (Idle Torque Timeout)

To prevent servo overheating during pauses in teleoperation or data collection, `NexArmFollowerConfig` has an automatic idle torque timeout:

- `idle_timeout_s`: Defaults to `20.0` seconds.
- When no new command is received for $> 20$ seconds, motor torque automatically turns off.
- The next `send_action()` seamlessly re-engages torque.

---

## 6. Testing & Quality Assurance

Run the quality suite before submitting changes or running experiments:

```bash
# 1. Run all unit tests
uv run pytest tests/ -v

# 2. Run NexArm-specific robot and motor tests
uv run pytest tests/motors/ tests/robots/ -v

# 3. Fast lint and formatting checks
uv run ruff check .
uv run ruff format . --check

# 4. Pre-commit suite (full lint, format, typecheck, typos)
uv run pre-commit run --all-files --show-diff-on-failure
```

---

## 7. Troubleshooting & FAQs

### Q: `Permission denied: '/dev/ttyUSB*'`

**Fix**: Add your user to the `dialout` group and grant RW permissions:

```bash
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyUSB*
```

_(Log out and back in if this is the first time adding to `dialout`)_.

### Q: `TimeoutError: No position reply from NexArm`

**Fix**: The leader firmware may emit debug lines over Serial that interrupt protocol frames. The driver automatically retries up to 3 times. For a permanent fix, comment out `Serial.printf` lines in `Nex_Arm.ino` when `lerobotMode == true` and reflash.

### Q: Camera display errors (`cv2.error` or headless SSH)

**Fix**: Use `--dry-run` or `--save-snapshot`:

- In `teleoperate_cartesian.py`: add `--dry-run`
- In `vision_grasp.py`: add `--dry-run`
- In `calibrate_camera_alignment.py`: add `--save-snapshot outputs/alignment.png`

### Q: Policy training loss doesn't decrease / arm moves hesitantly

**Fix**:

- Ensure you have at least 50 high-quality demonstration episodes.
- If the arm collapses to the mean position, reduce the KL loss weight: `--policy.kl_weight=5.0` or `1.0`.
- Increase batch size: `--batch_size=32` or `--batch_size=64`.
- Train for more steps: `--steps=200000`.

### Q: Which port is Leader and which is Follower?

**Fix**: Run `uv run lerobot-find-port`. Plug in only one arm at a time if you are unsure.
