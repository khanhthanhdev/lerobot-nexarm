# Isaac Lab (Isaac Sim) NexArm Guide

This directory contains the necessary configurations and tools to use the exported NexArm robot model in **NVIDIA Isaac Lab** and **Isaac Sim**.

---

## 1. Quick Overview

The export pipeline from Fusion 360 produces:

- [sim/fusion_export/nexarm.urdf](file:///home/thanh/code/vinuni/lerobot-nexarm/sim/fusion_export/nexarm.urdf)
- [sim/fusion_export/meshes/](file:///home/thanh/code/vinuni/lerobot-nexarm/sim/fusion_export/meshes/)

These can be imported into Isaac Sim / Isaac Lab as an Articulation asset.

---

## 2. Converting URDF to USD

There are two ways to convert the URDF to Universal Scene Description (`.usd`):

### Option A: Using Isaac Sim's Graphical URDF Importer (Recommended for first time)

1. Launch Isaac Sim.
2. In the top menu, navigate to: **Isaac Utils** $\rightarrow$ **Workflows** $\rightarrow$ **URDF Importer**.
3. Under **Input File**, browse and select `sim/fusion_export/nexarm.urdf`.
4. Configure options:
   - **Fix Base Link**: Checked (true).
   - **Merge Fixed Joints**: Checked (true) to merge `cam_mount` and fixed flanges.
   - **Convex Decomposition**: Optional (uncheck if using visual meshes directly).
5. Click **Import**.
6. Inspect the robot in the viewport, then click **File $\rightarrow$ Save As** to save the stage as `sim/isaac_lab/nexarm.usd`.

### Option B: Using the CLI Script

Run using Isaac Sim's Python runtime:

```bash
cd /path/to/isaac-sim
./python.sh /home/thanh/code/vinuni/lerobot-nexarm/sim/isaac_lab/convert_urdf_to_usd.py \
    --input /home/thanh/code/vinuni/lerobot-nexarm/sim/fusion_export/nexarm.urdf \
    --output /home/thanh/code/vinuni/lerobot-nexarm/sim/isaac_lab/nexarm.usd
```

---

## 3. Using `NexArmCfg` in Isaac Lab Environments

In your Isaac Lab task or script, instantiate the robot using [sim/isaac_lab/nexarm_cfg.py](file:///home/thanh/code/vinuni/lerobot-nexarm/sim/isaac_lab/nexarm_cfg.py):

```python
from sim.isaac_lab.nexarm_cfg import NexArmCfg

# In your SceneCfg:
robot = NexArmCfg(usd_path="path/to/nexarm.usd")
```

### Key Joint Names in Isaac Lab:

- **Arm Joints**:
  - `joint_1_base_to_link_1` (revolute, $\pm 135^\circ$)
  - `joint_2_link_1_to_link_2` (revolute, $\pm 100^\circ$)
  - `joint_3_link_2_to_link_3` (revolute, $\pm 100^\circ$)
  - `joint_4_link_3_to_link_4` (revolute, $\pm 100^\circ$)
  - `joint_5_link_4_to_link_5` (revolute, $\pm 120^\circ$)
- **Gripper Joints**:
  - `right_jaw_slide_joint` (prismatic, 0 to -0.0255 m)
  - `left_jaw_slide_joint` (prismatic, 0 to -0.0255 m)
