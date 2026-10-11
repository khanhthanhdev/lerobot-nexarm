# NexArm Data Collection Guide: Step-by-Step

This document provides a concise, step-by-step guide to collecting teleoperation demonstration datasets using the Hiwonder NexArm (Leader-Follower dual arm rig) with LeRobot.

Install the full NexArm profile from [`run.md`, Dependency Profiles](run.md#dependency-profiles) first. It includes `intelrealsense`, which the default `top` camera needs; `collect.sh` runs with `uv run --no-sync`, so it never installs missing extras for you.

## One-command manual collection

From this checkout, run:

```bash
./scripts/nexarm/collect.sh
```

The launcher checks access to the selected USB devices (automatically runs `sudo chmod 666` only if necessary), tests all cameras together at 640×480 / 30 fps, saves snapshots under `outputs/collection/<timestamp>/`, and connects the recorder. If sudo is needed, Linux may ask for your password. Defaults: leader `/dev/ttyUSB0`, follower `/dev/ttyUSB1`, Logitech front camera (MJPG) and icSpring wrist camera (YUYV) via stable `/dev/v4l/by-id/` paths, and the RealSense D435i `top` camera auto-detected (exactly one connected; pass `--top-cam <SERIAL>` otherwise). Any flag you pass overrides these, for example if your cables use different ports:

```bash
./scripts/nexarm/collect.sh --leader-port /dev/ttyUSB1 --follower-port /dev/ttyUSB0
```

Without a RealSense, add `--no-top-cam` and use the 2-camera root (see below). A missing RealSense or `pyrealsense2` stops the launcher with an error naming `--no-top-cam` and `uv sync --extra intelrealsense`.

Check the hardware without connecting or moving the arms:

```bash
./scripts/nexarm/collect.sh --prepare-only
# Cameras only:
./scripts/nexarm/collect.sh --test-cameras
```

The terminal shows a large **saved episode count** and the current phase:

- **READY:** press **Enter / →** to start the first episode.
- **COLLECTING:** press **Enter / →** to stop and save the episode.
- **RESET THE ENVIRONMENT:** reset the props and starting pose; follower teleoperation continues without recording. Press **Enter / →** to start the next episode.
- **← / r:** discard the current recording and reset before trying again.
- **Esc / q:** finish; a nonempty current recording is saved. Quitting at READY or during reset adds no episode.

Recording and reset have no timeout. Keys are ignored while saving (except quit), so wait for the reset screen before pressing again. Keep the terminal focused. Routine logs go to `outputs/collection/<timestamp>/recording.log`; setup/recording failures are printed.

Defaults collect 50 bowl-stacking episodes into a timestamped local session (`thanhkt/nexarm_stack_bowls_<YYYYMMDD_HHMMSS>`). When you press **Esc / q** or reach the episode limit, the session automatically merges into the **local** root dataset `thanhkt/nexarm_stack_bowls_top` (front, wrist and top cameras):

```text
~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top
```

**The collector never uploads or modifies the Hub dataset.** The screen shows **MERGING DATA...** until completion. Existing root episodes are preserved. The previous local root is kept at the sibling path `nexarm_stack_bowls_top.previous`; session recordings are also retained. If the local root does not exist yet, its Hub version is downloaded once as the starting point; if the Hub has no such dataset either, the first session creates the root. Checking the Hub needs a login (`hf auth login`): without one, a private root looks missing, so the collector stops instead of creating it. Subsequent merges use the local root and work offline.

Before recording starts, the session is checked against the root's fps, `robot_type`, camera set and image shapes, so an incompatible session fails immediately instead of after recording. If the root cannot be read from the Hub (network, login or gated repo), the error suggests `--merge-root` (a local copy of the root) or `--no-merge`.

To keep adding to the old 2-camera root instead, record without the top camera:

```bash
./scripts/nexarm/collect.sh --no-top-cam --root-repo-id thanhkt/nexarm_stack_bowls
```

The local root stores an import ledger in `meta/collection_sessions.json`. Retrying the same session adds no duplicates; resuming a previously merged session adds only its new episodes. A file lock prevents concurrent local merges, and the new dataset is built separately before replacing the local root.

```bash
./scripts/nexarm/collect.sh --num-episodes 100
# Resume an existing session by its exact timestamped ID:
./scripts/nexarm/collect.sh --resume --repo-id thanhkt/nexarm_stack_bowls_20261004_160000
```

Retry a failed local merge without recording again (the recorder prints this command with the exact session id and root):

```bash
uv run --no-sync python examples/nexarm/merge_collection.py \
    --repo-id thanhkt/nexarm_stack_bowls_20261004_160000
```

Use `--no-merge` to keep each recording session separate, or `--merge-root /path/to/local/root` to choose the local merged directory. `--root-repo-id` changes the root dataset ID used for metadata and the optional initial download. Camera/USB preflight checks never merge anything.

To train on the merged root, pass its local directory as shown in [`pipeline.md`, section 2](docs/source/nexarm/pipeline.md#2-training).

Called directly, `prepare_collection.py` defaults to `--front-cam 0 --wrist-cam 1` with automatic pixel formats; `collect.sh` passes the rig's by-id paths with MJPG (front) and YUYV (wrist). Override with `--front-cam` / `--wrist-cam` (index or device path) and `--front-fourcc` / `--wrist-fourcc`.

---

## 1. Physical Hardware Checklist

1. **Follower Arm (Slave / Actuated)**:
   - Connect the **12V / 5A DC power supply** (barrel jack). Servos will not have torque without external 12V power.
   - Connect the USB-C cable from the follower ESP32 base to the host PC.
2. **Leader Arm (Master / Passive)**:
   - Connect the USB-C cable from the leader ESP32 base to the host PC (powered directly via USB).
3. **Cameras**:
   - **Front camera** (overlooking the workspace): e.g. `/dev/video0` (Logitech C270).
   - **Wrist camera** (mounted near gripper end-effector): e.g. `/dev/video2`.
   - **Top camera** (Intel RealSense D435i looking down at the table; only RGB is recorded). Skip it with `--no-top-cam`.

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

- Default assignment on Linux (used by every script in this repository):
  - Leader (`nexarm_leader`): `/dev/ttyUSB0`
  - Follower (`nexarm_follower`): `/dev/ttyUSB1`

Verify cameras:

```bash
uv run lerobot-find-cameras opencv
uv run lerobot-find-cameras realsense
```

---

## 3. Pre-Flight Check: Teleoperation Test

Before recording full datasets, verify that the leader mirrors smoothly to the follower:

```bash
uv run python examples/nexarm/teleoperate.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --front-cam 0 \
    --wrist-cam 2 \
    --fps 30
```

- Move the leader arm by hand.
- Confirm the follower tracks all 6 joints (including gripper) with low latency.
- Press `Ctrl+C` to stop.

---

## 4. Record Demonstrations

`collect.sh` above is the recommended path. The equivalent `lerobot-record` command follows; its cameras must match the root (`thanhkt/nexarm_stack_bowls_top` has front, wrist and top).

### Step 4.1: Set Hugging Face User

```bash
export HF_USER="$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')"
: "${HF_USER:?Run 'uv run hf auth login' first or export HF_USER manually}"
```

### Step 4.2: Run Recording Command

```bash
uv run lerobot-record \
    --robot.type=nexarm_follower \
    --robot.port=/dev/ttyUSB1 \
    --teleop.type=nexarm_leader \
    --teleop.port=/dev/ttyUSB0 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30,"fourcc":"MJPG"},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30},"top":{"type":"intelrealsense","serial_number_or_name":"<REALSENSE_SERIAL>","width":640,"height":480,"fps":30}}' \
    --dataset.repo_id="${HF_USER}/nexarm_stack_bowls" \
    --dataset.single_task="Stack the bowls with red on bottom, blue in middle, and black on top." \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=inf \
    --dataset.reset_time_s=inf \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --display_data=false \
    --play_sounds=false \
    --manual_control=true \
    --dataset.push_to_hub=false \
    --root_repo_id=thanhkt/nexarm_stack_bowls_top
```

For the 2-camera root, remove the `top` entry and use `--root_repo_id=thanhkt/nexarm_stack_bowls`.

`lerobot-record` appends `_YYYYMMDD_HHMMSS` to `--dataset.repo_id` and prints the final `Dataset repo_id:` and `Dataset root:` when it finishes. Use that printed id in the commands below (shown as `${HF_USER}/nexarm_stack_bowls_<YYYYMMDD_HHMMSS>`), or the merged root `thanhkt/nexarm_stack_bowls_top`.

### Key Parameter Explanations:

| Flag                           | Why it is important                                                                                                                    |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| `"fourcc":"MJPG"`              | **Critical for USB bandwidth**: Compresses front camera stream in hardware, preventing USB 2.0 bus contention that causes frame drops. |
| `--play_sounds=false`          | Prevents audio daemon hangs (`espeak` / `say`) on headless or non-standard Linux audio setups when finishing episodes.                 |
| `--display_data=false`         | Disables real-time Rerun visualizer logging to give 100% of CPU/thread resources to 30 Hz control and video encoding.                  |
| `--dataset.episode_time_s=inf` | No recording timeout: record until you press **→**.                                                                                    |
| `--dataset.reset_time_s=inf`   | No reset timeout: reset the scene, then press **→** again to continue recording.                                                       |
| `--root_repo_id`               | Merge the finished session into this local root; compatibility is checked before recording starts.                                     |

---

## 5. Keyboard Controls During Recording

With `--manual_control=true`, wait for READY and press **Enter / →** to start. Complete your demonstration, then press **Enter / →** once to end the episode and enter reset mode. Reset the props and return the arm to its starting pose; these reset movements are not recorded, but the follower still follows the leader. Press **Enter / →** again when ready to record the next episode. Neither phase has a timeout.

Repeat until 50 episodes have been recorded, or press **ESC** to finish early. The last episode skips reset mode and finishes the session automatically.

| Key                                         | Action                                                                                                         |
| :------------------------------------------ | :------------------------------------------------------------------------------------------------------------- |
| **Enter**, **`→` (Right Arrow)** or **`n`** | While recording: end the episode and enter reset mode. While resetting: continue with the next episode.        |
| **`←` (Left Arrow)**                        | **Rerecord / Discard episode**: Discards the current demonstration if you made a mistake or dropped an object. |
| **`ESC`**                                   | **Finish recording**: Stops the session early, saves all completed episodes, and finalizes video files.        |

Run in an interactive terminal so keyboard controls are available. On Wayland or over SSH, keep the recording terminal focused. If arrow keys are intercepted, use **`n`** instead; Enter works when `--manual_control=true`. Without that flag, the original timed/arrow flow starts immediately.

---

## 6. Inspect & Validate Recorded Dataset

### 6.1 Replay on Physical Robot

Test that the follower robot faithfully reproduces recorded demonstration 0 of a session:

```bash
uv run lerobot-replay \
    --robot.type=nexarm_follower \
    --robot.port=/dev/ttyUSB1 \
    --dataset.repo_id="${HF_USER}/nexarm_stack_bowls_<YYYYMMDD_HHMMSS>" \
    --dataset.episode=0
```

### 6.2 Visualize in Browser / Rerun

Inspect synchronized videos and joint angle trajectories:

```bash
uv run lerobot-dataset-viz \
    --repo-id "${HF_USER}/nexarm_stack_bowls_<YYYYMMDD_HHMMSS>" \
    --episode-index 0
```

Use `--repo-id thanhkt/nexarm_stack_bowls_top` (or `--dataset.repo_id=` for replay) to inspect the merged root.

---

## 7. Troubleshooting

- **Record loop runs at 5.0 Hz**:
  The follower arm was not in LeRobot bridge mode, so the recorder parsed the 5 Hz telemetry stream. Ensure the 12V power supply is on and restart the command.
- **FPS drops to 13–15 Hz**:
  Cameras are sending uncompressed YUYV over a shared USB controller. Add `"fourcc":"MJPG"` to the front camera configuration (`--front-fourcc MJPG` in the scripts).
- **`This session cannot be merged into root dataset ...`**:
  The planned cameras, resolution, fps or `robot_type` differ from the root. Match the root (add or remove `--no-top-cam`), choose another `--root-repo-id`, or use `--no-merge`.
- **`Permission denied: '/nexarm...'`**:
  `${HF_USER}` is empty. Run `export HF_USER="your_hf_username"`.
- **KeyboardInterrupt / Freeze on exit**:
  Add `--play_sounds=false` to bypass system text-to-speech calls.
- **Upload failed / Internet disconnection during upload**:
  The recorded dataset is already finalized and safely stored locally in `~/.cache/huggingface/lerobot/<repo_id>` (the printed timestamped id). Re-upload it at any time without re-recording:
  ```bash
  uv run python examples/nexarm/upload_dataset.py --repo-id "${HF_USER}/nexarm_stack_bowls_<YYYYMMDD_HHMMSS>"
  ```
