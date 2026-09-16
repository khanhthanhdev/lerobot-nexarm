# NexArm Comprehensive Run Guide: Teleoperation, Dynamics, Vision & Sim2Real

This guide provides complete, copy-pasteable instructions for running all simulation, hardware, teleoperation, dynamics, vision, and Sim-to-Real features on the **Hiwonder NexArm** platform with LeRobot.

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Quick Verification (Tests & Lints)](#2-quick-verification-tests--lints)
3. [6-DOF Cartesian Teleoperation](#3-6-dof-cartesian-teleoperation)
4. [Autonomous Vision-Guided Grasping](#4-autonomous-vision-guided-grasping)
5. [Sim2Real Pre-Flight Camera Alignment](#5-sim2real-pre-flight-camera-alignment)
6. [Dataset Generation with Domain Randomization & Latency](#6-dataset-generation-with-domain-randomization--latency)
7. [Kinematics & Dynamics Python API](#7-kinematics--dynamics-python-api)
8. [Hardware Protections (Idle Torque Timeout)](#8-hardware-protections-idle-torque-timeout)
9. [Troubleshooting & FAQs](#9-troubleshooting--faqs)

---

## 1. Prerequisites & Environment Setup

Ensure you have installed `uv` (modern Python package manager) and have Python 3.12+ available.

```bash
# Clone the repository (if not already in workspace)
cd /home/thanh/code/vinuni/lerobot-nexarm

# Sync all locked dependencies (including test, dev, and nexarm extras)
uv sync --locked --extra nexarm --extra dev

# Install Git LFS test artifacts
git lfs install && git lfs pull
```

> **Note on Serial Ports (Linux)**:
> If connecting to a physical robot or camera, ensure your user has access to serial ports:
>
> ```bash
> sudo usermod -a -G dialout $USER
> sudo chmod 666 /dev/ttyUSB*
> ```

---

## 2. Quick Verification (Tests & Lints)

Before running hardware or simulation loops, run the automated quality checks:

```bash
# 1. Run motor and robot unit tests (83 passing tests)
uv run pytest tests/motors/ tests/robots/ -v

# 2. Verify formatting and linting
uv run ruff check .
uv run ruff format . --check
```

---

## 3. 6-DOF Cartesian Teleoperation

Script: [`examples/nexarm/teleoperate_cartesian.py`](file:///home/thanh/code/vinuni/lerobot-nexarm/examples/nexarm/teleoperate_cartesian.py)

Controls the robot's end-effector in 3D Cartesian task space ($X, Y, Z$, Roll, Pitch, Yaw) with real-time Damped Least-Squares (DLS) Inverse Kinematics.

### A. Run in Simulation (MuJoCo)

```bash
uv run python examples/nexarm/teleoperate_cartesian.py --robot sim
```

### B. Run on Physical Hardware

```bash
uv run python examples/nexarm/teleoperate_cartesian.py \
    --robot real \
    --port /dev/ttyUSB0
```

### C. Headless / Dry-Run (CI or Remote SSH)

```bash
uv run python examples/nexarm/teleoperate_cartesian.py --robot sim --dry-run
```

### Keybindings & Controls

| Key           | Action                                                  |
| :------------ | :------------------------------------------------------ |
| **`W` / `S`** | Translate End-Effector **$\pm X$** (Forward / Backward) |
| **`A` / `D`** | Translate End-Effector **$\pm Y$** (Left / Right)       |
| **`R` / `F`** | Translate End-Effector **$\pm Z$** (Up / Down)          |
| **`U` / `J`** | Rotate End-Effector **$\pm \text{Roll}$**               |
| **`I` / `K`** | Rotate End-Effector **$\pm \text{Pitch}$**              |
| **`O` / `L`** | Rotate End-Effector **$\pm \text{Yaw}$**                |
| **`Space`**   | **Toggle Gripper** (Open $\leftrightarrow$ Close)       |
| **`1`**       | Set **Fine** Step (2 mm translation / 1.0° rotation)    |
| **`2`**       | Set **Medium** Step (5 mm translation / 2.5° rotation)  |
| **`3`**       | Set **Coarse** Step (15 mm translation / 5.0° rotation) |
| **`H`**       | Return to default **Home Pose**                         |
| **`Q`**       | **Exit** gracefully and restore terminal                |

---

## 4. Autonomous Vision-Guided Grasping

Script: [`examples/nexarm/vision_grasp.py`](file:///home/thanh/code/vinuni/lerobot-nexarm/examples/nexarm/vision_grasp.py)

Performs closed-loop visual object detection (HSV thresholding + contour centroid moments), projects 2D image coordinates into 3D Cartesian coordinates, generates smooth multi-waypoint trajectories, and executes a pick-and-place cycle.

### A. Run in Simulation (MuJoCo Top-Down Camera)

```bash
uv run python examples/nexarm/vision_grasp.py --robot sim
```

### B. Run on Physical Hardware

```bash
uv run python examples/nexarm/vision_grasp.py \
    --robot real \
    --port /dev/ttyUSB0 \
    --camera-index 0
```

### C. Dry-Run (Execute 1 complete cycle and exit)

```bash
uv run python examples/nexarm/vision_grasp.py --robot sim --dry-run
```

### Tuning Color Detection Flags

You can customize the target object's HSV range to track different colored cubes or items:

```bash
# Example: Track blue objects
uv run python examples/nexarm/vision_grasp.py \
    --robot sim \
    --hue-min 100 --hue-max 130 \
    --sat-min 120 --sat-max 255 \
    --val-min 70 --val-max 255
```

---

## 5. Sim2Real Pre-Flight Camera Alignment

Script: [`examples/nexarm/calibrate_camera_alignment.py`](file:///home/thanh/code/vinuni/lerobot-nexarm/examples/nexarm/calibrate_camera_alignment.py)

Before deploying a vision policy trained in simulation onto the real robot, verify that the physical camera perspective, focal distance, and field-of-view match the simulation.

### A. Live Interactive Comparison (Real Camera vs. Sim Camera)

```bash
uv run python examples/nexarm/calibrate_camera_alignment.py \
    --camera top \
    --real-camera-index 0 \
    --blend-alpha 0.5
```

### B. Headless Snapshot Generation (No display required)

```bash
uv run python examples/nexarm/calibrate_camera_alignment.py \
    --camera top \
    --save-snapshot outputs/calibration/alignment_check.png
```

### Alignment Modes (In Live Window)

- Press **`M`**: Cycle mode (**Alpha Blend** $\rightarrow$ **Side-by-Side** $\rightarrow$ **Canny Edge Overlay**).
- Press **`[` / `]`**: Decrease / increase alpha transparency.
- Press **`S`**: Save snapshot to disk.
- Press **`Q`**: Exit.

---

## 6. Dataset Generation with Domain Randomization & Latency

Script: [`examples/nexarm/generate_sim_dataset.py`](file:///home/thanh/code/vinuni/lerobot-nexarm/examples/nexarm/generate_sim_dataset.py)

To bridge the Sim-to-Real gap, policies should be trained on simulated data containing realistic serial transport delays and randomized dynamics/visuals.

### A. Generate Standard Dataset

```bash
uv run python examples/nexarm/generate_sim_dataset.py \
    --repo-id local/nexarm_standard \
    --num-episodes 20
```

### B. Generate Sim2Real-Hardened Dataset (DR + Latency)

```bash
uv run python examples/nexarm/generate_sim_dataset.py \
    --repo-id local/nexarm_sim2real \
    --num-episodes 50 \
    --domain-randomization \
    --action-delay-steps 2
```

**Parameters**:

- `--domain-randomization` (`--dr`): Enables:
  - **Visual DR**: Camera position jitter ($\pm 1.2$ cm, $\pm 2^\circ$), lighting direction/intensity, table/object color.
  - **Dynamics DR**: Contact friction ($\mu \in [0.8, 2.2]$), link mass jitter ($\pm 25\%$), joint damping and friction loss.
- `--action-delay-steps 2`: Simulates 2-step (66 ms at 30 Hz) physical USB transport and motor reaction latency.

---

## 7. Kinematics & Dynamics Python API

Module: [`src/lerobot/motors/nexarm/kinematics_dynamics.py`](file:///home/thanh/code/vinuni/lerobot-nexarm/src/lerobot/motors/nexarm/kinematics_dynamics.py)

You can import and use the C-accelerated kinematics and dynamics engine directly in your custom scripts:

```python
import numpy as np
from lerobot.motors.nexarm import NexArmKinematicsDynamics

# Initialize solver with NexArm MuJoCo model
kd = NexArmKinematicsDynamics()

# Current arm joint angles in radians (5 arm joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
q = np.array([0.0, 0.4, 0.8, -0.4, 0.0])

# 1. Forward Kinematics (FK)
fk = kd.forward_kinematics(q)
print("EE Position (x, y, z):", fk["position"])
print("EE Orientation (RPY deg):", np.rad2deg(fk["rpy"]))

# 2. 6x5 Spatial Geometric Jacobian
J = kd.jacobian(q)
print("Jacobian Shape:", J.shape)  # (6, 5)

# 3. Damped Least-Squares Inverse Kinematics (IK)
target_pos = np.array([0.22, 0.0, 0.15])
success, q_sol = kd.inverse_kinematics(target_pos, q_init=q)
if success:
    print("IK Solution (rad):", q_sol)

# 4. Generalized Mass / Inertia Matrix M(q)
M = kd.mass_matrix(q)
print("Mass Matrix (5x5):\n", M)

# 5. Static Gravity Compensation Torques g(q)
tau_grav = kd.gravity_torques(q)
print("Gravity Compensation Torques (Nm):", tau_grav)
```

---

## 8. Hardware Protections (Idle Torque Timeout)

To protect the bus servos against thermal degradation and mechanical fatigue during pauses in teleoperation or data collection:

- **Follower Config**: In [`src/lerobot/robots/nexarm_follower/config_nexarm_follower.py`](file:///home/thanh/code/vinuni/lerobot-nexarm/src/lerobot/robots/nexarm_follower/config_nexarm_follower.py), `idle_timeout_s` defaults to `20.0` seconds.
- **Behavior**:
  - If no `send_action()` command is received for $> 20$ seconds, the follower automatically disables motor torque to allow servos to cool down.
  - The moment a new `send_action()` command is issued, torque is automatically re-engaged and the goal is commanded smoothly.
- **Customizing Timeout**:
  ```bash
  # Example: Set idle timeout to 60 seconds
  lerobot-teleoperate \
      --robot.type=nexarm_follower \
      --robot.port=/dev/ttyUSB0 \
      --robot.idle_timeout_s=60.0 \
      ...
  ```

---

## 9. Troubleshooting & FAQs

### Q: `Permission denied: '/dev/ttyUSB0'`

**Fix**: Add your user to the `dialout` group and grant RW access:

```bash
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyUSB0
```

_(Log out and log back in if adding to `dialout` for the first time)_.

### Q: `cv2.error: The function is not implemented` / No GUI display on headless SSH

**Fix**: Use the `--dry-run` or `--save-snapshot` options provided in the scripts:

- In `teleoperate_cartesian.py`: Add `--dry-run`
- In `vision_grasp.py`: Add `--dry-run`
- In `calibrate_camera_alignment.py`: Use `--save-snapshot <path.png>`

### Q: Camera index mismatch (wrong camera selected)

**Fix**: List connected video devices using:

```bash
v4l2-ctl --list-devices
```

Then pass the appropriate index (e.g. `--camera-index 2`).

### Q: Arm moves too fast or jerks

**Fix**: Use the Cartesian step resolution keys in `teleoperate_cartesian.py`:

- Press `1` for fine 2 mm step adjustments.
- For scripted motion, decrease the step size in `vision_grasp.py` via `approach_steps=40`.
