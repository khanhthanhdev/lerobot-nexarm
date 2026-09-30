# NexArm Data Collection Guide: Step-by-Step

This document provides a concise, step-by-step guide to collecting teleoperation demonstration datasets using the Hiwonder NexArm (Leader-Follower dual arm rig) with LeRobot.

---

## 1. Physical Hardware Checklist

1. **Follower Arm (Slave / Actuated)**:
   - Connect the **12V / 5A DC power supply** (barrel jack). Servos will not have torque without external 12V power.
   - Connect the USB-C cable from the follower ESP32 base to the host PC.
2. **Leader Arm (Master / Passive)**:
   - Connect the USB-C cable from the leader ESP32 base to the host PC (powered directly via USB).
3. **Cameras**:
   - **Front camera** (top-down view overlooking workspace): e.g. `/dev/video0` (Logitech C270).
   - **Wrist camera** (mounted near gripper end-effector): e.g. `/dev/video2`.

---

## 2. Linux Permissions & Port Setup

Grant read/write permissions to serial ports:

```bash
# Option A: Grant access for current session
sudo chmod 666 /dev/ttyUSB*

# Option B: Permanent permission (requires logout/login)
sudo usermod -a -G dialout $USER
```

Identify Leader and Follower serial ports:

```bash
uv run lerobot-find-port
```

- Typical assignment on Linux:
  - Follower (`nexarm_follower`): `/dev/ttyUSB0`
  - Leader (`nexarm_leader`): `/dev/ttyUSB1`

Verify cameras:

```bash
uv run lerobot-find-cameras opencv
```

---

## 3. Pre-Flight Check: Teleoperation Test

Before recording full datasets, verify that the leader mirrors smoothly to the follower:

```bash
uv run python examples/nexarm/teleoperate.py \
    --leader-port /dev/ttyUSB1 \
    --follower-port /dev/ttyUSB0 \
    --front-cam 0 \
    --wrist-cam 2 \
    --fps 30
```

- Move the leader arm by hand.
- Confirm the follower tracks all 6 joints (including gripper) with low latency.
- Press `Ctrl+C` to stop.

---

## 4. Record Demonstrations

### Step 4.1: Set Hugging Face User

```bash
export HF_USER="$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')"
: "${HF_USER:?Run 'uv run hf auth login' first or export HF_USER manually}"
```

### Step 4.2: Run Recording Command

```bash
uv run lerobot-record \
    --robot.type=nexarm_follower \
    --robot.port=/dev/ttyUSB0 \
    --teleop.type=nexarm_leader \
    --teleop.port=/dev/ttyUSB1 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
    --dataset.repo_id="${HF_USER}/nexarm_stack_bowls" \
    --dataset.single_task="Stack the bowls with red on bottom, blue in middle, and black on top." \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=15 \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --display_data=false \
    --play_sounds=false
```

### Key Parameter Explanations:

| Flag | Why it is important |
| :--- | :--- |
| `"fourcc":"MJPG"` | **Critical for USB bandwidth**: Compresses front camera stream in hardware, preventing USB 2.0 bus contention that causes frame drops. |
| `--play_sounds=false` | Prevents audio daemon hangs (`espeak` / `say`) on headless or non-standard Linux audio setups when finishing episodes. |
| `--display_data=false` | Disables real-time Rerun visualizer logging to give 100% of CPU/thread resources to 30 Hz control and video encoding. |
| `--dataset.episode_time_s=15` | Recording duration for each demonstration episode. |
| `--dataset.reset_time_s=15` | Pause between episodes giving you time to reset props on the table. |

---

## 5. Keyboard Controls During Recording

| Key | Action |
| :--- | :--- |
| **`Enter`** or **`→` (Right Arrow)** | Start the next episode early (skips remaining reset time). |
| **`←` (Left Arrow)** | **Rerecord / Discard episode**: Discards the current demonstration if you made a mistake or dropped an object. |
| **`ESC`** | **Finish recording**: Stops the session early, saves all completed episodes, and finalizes video files. |

---

## 6. Inspect & Validate Recorded Dataset

### 6.1 Replay on Physical Robot

Test that the follower robot faithfully reproduces recorded demonstration 0:

```bash
uv run lerobot-replay \
    --robot.type=nexarm_follower \
    --robot.port=/dev/ttyUSB0 \
    --dataset.repo_id="${HF_USER}/nexarm_stack_bowls" \
    --dataset.episode=0
```

### 6.2 Visualize in Browser / Rerun

Inspect synchronized videos and joint angle trajectories:

```bash
uv run lerobot-dataset-viz \
    --repo-id "${HF_USER}/nexarm_stack_bowls" \
    --episode-index 0
```

---

## 7. Troubleshooting

- **Record loop runs at 5.0 Hz**:
  The follower arm was not in LeRobot bridge mode, so the recorder parsed the 5 Hz telemetry stream. Ensure the 12V power supply is on and restart the command.
- **FPS drops to 13–15 Hz**:
  Both cameras are sending uncompressed YUYV over a shared USB controller. Add `"fourcc":"MJPG"` to the front camera configuration.
- **`Permission denied: '/nexarm...'`**:
  `${HF_USER}` is empty. Run `export HF_USER="your_hf_username"`.
- **KeyboardInterrupt / Freeze on exit**:
  Add `--play_sounds=false` to bypass system text-to-speech calls.
