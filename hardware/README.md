# 🛠️ NexArm Hardware & Sim-to-Real Reliability Specifications

This directory provides mechanical guidelines, 3D printing recommendations, and physical reliability protections for the **Hiwonder NexArm 6-DOF robot**, designed to ensure robust physical rollouts and repeatable Sim-to-Real policy transfer (modeled after the `reBot-DevArm` design standards).

---

## 📋 Table of Contents

- [1. 3D-Printable Brackets & Cable Management](#1-3d-printable-brackets--cable-management)
- [2. Standard Wrist Camera Mount Specifications](#2-standard-wrist-camera-mount-specifications)
- [3. Operational Envelope & Thermal Protection Guidelines](#3-operational-envelope--thermal-protection-guidelines)
- [4. Power Supply & Electrical Recommendations](#4-power-supply--electrical-recommendations)

---

## 1. 3D-Printable Brackets & Cable Management

NexArm uses daisy-chained 3-pin serial bus cables connecting six HX-30HM servos. During high-speed rollouts and continuous teleoperation, cable twisting and connector chafing represent a common failure mode.

### Recommended 3D-Printed Parts:

| Component                               | Function                                                                                       | Recommended Material | Print Settings          |
| :-------------------------------------- | :--------------------------------------------------------------------------------------------- | :------------------- | :---------------------- |
| **Joint 1 Base Harness Clip**           | Anchors the bus harness to the base plate to prevent wire strain on Joint 1 yaw rotations.     | ABS / PETG           | 0.2mm layer, 35% infill |
| **Link 2/3 Middle Cable Channel**       | Guides the serial bus cable along Link 2 without sagging into the elbow pinch point.           | PLA / PETG           | 0.2mm layer, 20% infill |
| **Wrist Cable Restraint (Joint 4/5)**   | Secures the wiring loop entering the wrist roll servo (Joint 5) and gripper.                   | ABS                  | 0.2mm layer, 40% infill |
| **Desktop G-Clamp Reinforcement Plate** | Flat mounting base with screw holes for 4-inch/6-inch desktop G-clamps to prevent base wobble. | PLA / PETG           | 0.2mm layer, 50% infill |

---

## 2. Standard Wrist Camera Mount Specifications

In imitation learning (ACT, Diffusion Policy, SmolVLA), the relative extrinsic pose between the wrist camera and the gripper center must be rigid and repeatable between the physical table and the simulation model (`sim/description/mjcf/nexarm.xml`).

### Supported Sensors & Coordinate Targets:

1. **Intel RealSense D405**:
   - **Recommended Sensor**: Sub-millimeter RGB-D precision at close range (7 cm – 50 cm).
   - **Mount Location**: Secured directly to Link 5 wrist roll collar.
   - **Optical Center Offset**: `pos="0.539 0.015 0.295"` (matches MuJoCo wrist camera frame).

2. **Standard 32×32 mm UVC Webcams**:
   - **Lens**: 90°–100° wide FOV, low distortion.
   - **Weight**: < 25 grams (preserves payload capacity of wrist joints 4 & 5).

---

## 3. Operational Envelope & Thermal Protection Guidelines

Empirical testing on high-torque serial bus servos (HX-30HM) demonstrates that holding static poses under load causes rapid coil temperature buildup.

### Recommended Operating Constraints:

1. **Maximum Workspace Reach**:
   - Total kinematic reach: ~450 mm.
   - **Recommended working radius**: $\le 70\%$ of reach (**$< 320\text{ mm}$**).
   - Operating at 100% reach places maximum cantilever load on Joint 2 (shoulder lift), dramatically increasing motor current.

2. **Maximum Safe Payload**:
   - Nominal dynamic payload: **$\le 300\text{ g}$**.
   - Maximum instantaneous payload: **$500\text{ g}$**.

3. **Software Idle Thermal Protection**:
   - `NexArmFollowerConfig` defaults to `idle_timeout_s = 20.0`.
   - When the arm is stationary between teleoperation episodes or evaluation rollouts for $> 20$ seconds, the driver automatically disengages holding torque to keep servo temperatures below 60°C.
   - Any new command (`send_action`) instantly re-engages torque without manual intervention.

4. **Continuous Duty Cycle**:
   - After **30 minutes** of continuous evaluation, allow a **5-minute resting interval** to extend servo gear and motor life.

---

## 4. Power Supply & Electrical Recommendations

- **Voltage**: **DC 11.1V – 12.0V** regulated.
- **Current Rating**: Minimum **5A**, recommended **8A – 10A** power supply (e.g. MeanWell LRS-100-12 or 3S LiPo battery).
- **Wiring**: Ensure power ground and USB serial ground are common to prevent logic packet corruption on the 1 Mbps bus.
