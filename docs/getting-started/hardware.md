# Hardware quickstart

Connect the leader and follower arms and the cameras, then check them with teleoperation. [Install](install.md) first. The joint layout, servo units and port conventions are in [Architecture, Conventions](../reference/architecture.md#conventions); the vendor's own tutorials (firmware, ESP32, motion control) are in [`docs/hiwonder/`](../hiwonder/1_Getting_Started_NexArm.md).

## Wiring and power checklist

1. **Follower arm** (actuated):
   - Connect the 12V / 5A DC power supply (barrel jack). Servos have no torque without it.
   - Connect the USB-C cable from the follower ESP32 board to the host PC.
2. **Leader arm** (passive, moved by hand):
   - Connect the USB-C cable from the leader ESP32 board to the host PC (powered via USB).
3. **Cameras** (all 640x480 @ 30 FPS):
   - `front` camera: USB webcam (e.g. Logitech C270) at an angle over the workspace table; gives height, layer depth and transit clearance.
   - `wrist` camera: USB webcam on the arm near the gripper facing the end-effector; gives the fine alignment of the last few centimetres.
   - `top` camera: Intel RealSense D435i looking straight down at the table; gives occlusion-free $(X, Y)$ alignment. Only RGB is recorded. It is **on by default** in every recording and rollout script; pass `--no-top-cam` if you do not have one.

The camera set you record with is the set the policy needs at rollout.

## Find the serial ports

Run the port finder utility to identify leader and follower ports:

```bash
uv run lerobot-find-port
```

- Unplug and replug each arm when prompted. Plug in one arm at a time if you are unsure which is which.
- **Linux**: `/dev/ttyUSB0` (Leader) and `/dev/ttyUSB1` (Follower). Every script, YAML and guide in this repository uses these defaults; if your cables enumerate the other way, pass the ports explicitly.
- **Windows**: Typically `COM18` (Leader) and `COM19` (Follower).
- **macOS**: Typically `/dev/tty.usbserial-*`.

On Linux the ports need [serial permissions](install.md#linux-serial-permissions).

## Find and check the cameras

Identify the OpenCV camera indices for the `front` and `wrist` cameras and the RealSense serial for `top`:

```bash
uv run lerobot-find-cameras opencv
uv run lerobot-find-cameras realsense
```

The example scripts take `--front-cam` / `--wrist-cam` as an index or a device path such as `/dev/v4l/by-id/...` (defaults `0` / `1`), with optional `--front-fourcc` / `--wrist-fourcc` (e.g. `MJPG`). `--top-cam` defaults to auto-detecting exactly one RealSense; pass its serial when several are connected. A missing RealSense or `pyrealsense2` stops the script with an error naming `--no-top-cam` and `uv sync --extra intelrealsense`.

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

`uv run python tools/nexarm/prepare_collection.py --test-cameras --stress-test 30` runs all streams together, including RealSense RGB and depth, and reports fps, drops and stalls.

## Leader-follower teleoperation

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

Check that the follower tracks all 6 joints, including the gripper, with low latency. `Ctrl+C` stops.

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
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"top":{"type":"intelrealsense","serial_number_or_name":"<REALSENSE_SERIAL>","width":640,"height":480,"fps":30}}' \
    --display_data=true
```

The `lerobot-*` CLIs only open the cameras listed in `--robot.cameras`; remove the `top` entry to run with front and wrist only.

## 6-DOF Cartesian hardware teleoperation

Control the physical arm directly using keyboard Cartesian inputs without needing a physical leader arm (keys as in [the simulation](simulation.md#6-dof-cartesian-keyboard-teleoperation)):

```bash
uv run python examples/nexarm/teleoperate_cartesian.py \
    --robot real \
    --port /dev/ttyUSB1
```

## Vision-guided grasping

Execute autonomous visual object detection (HSV color tracking), 2D-to-3D workspace projection, and smooth pick-and-place trajectories:

```bash
# Run on physical hardware
uv run python examples/nexarm/vision_grasp.py \
    --robot real \
    --port /dev/ttyUSB1 \
    --cam-index 0

# Test in simulation first
uv run python examples/nexarm/vision_grasp.py --robot sim
```

## Align the cameras with the simulation

Before running a policy trained in simulation, overlay the live camera on the simulated view and match them:

```bash
uv run python tools/nexarm/calibrate_camera_alignment.py \
    --camera-name front \
    --cam-index 0
```

Keys, the top camera and how to edit the scene are in [Sim calibration, Step 7](../guides/sim-calibration.md#step-7-align-the-cameras).

Next: [record a dataset and train a policy](../guides/pipeline.md), or use the [stack-bowls collection rig](../guides/data-collection.md).
