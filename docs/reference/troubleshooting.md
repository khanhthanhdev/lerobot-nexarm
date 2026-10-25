# Troubleshooting

General problems with the arms, cameras and recording loop. Guide-specific problems are listed in each guide: [data collection](../guides/data-collection.md#5-troubleshooting), [sim training](../guides/sim-training.md#troubleshooting), [sim calibration](../guides/sim-calibration.md#troubleshooting) and [`sim/README.md`](../../sim/README.md#13-troubleshooting). Weak policies are covered in [Training tips](../guides/training-tips.md#18-troubleshooting-signal).

## `Permission denied: '/dev/ttyUSB*'`

Grant your user access to the serial ports, as in [Install, Linux serial permissions](../getting-started/install.md#linux-serial-permissions). Log out and back in the first time you add yourself to `dialout`.

## Which port is Leader and which is Follower?

Run `uv run lerobot-find-port`. Plug in only one arm at a time if you are unsure.

## Follower arm does not move during teleoperation

1. Confirm follower port and leader port are not swapped.
2. Confirm the follower ESP32 firmware supports CMD 68 (LeRobot bridge mode).
3. Try power-cycling the follower arm (ensure 12V power supply is plugged in).

## `Permission denied: '/nexarm_stack_bowls_YYYYMMDD_HHMMSS'`

**Cause**: `${HF_USER}` was empty, so the dataset repo ID expanded to `/nexarm_stack_bowls` and the recorder tried to create it at the filesystem root.

**Fix**: Log in and set the username before recording:

```bash
export HF_USER="$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')"
: "${HF_USER:?Run 'uv run hf auth login' first, then set HF_USER to your Hugging Face username}"
```

Then keep `--dataset.repo_id="${HF_USER}/nexarm_stack_bowls"` in your normal command and leave `--dataset.root` unset: the session is written to `~/.cache/huggingface/lerobot/${HF_USER}/nexarm_stack_bowls_<YYYYMMDD_HHMMSS>`, a new directory every run. `--dataset.root` is the dataset directory itself and is not timestamped, so a fixed value fails with `FileExistsError` on the second run; if you need a custom location, give each session its own path (e.g. `--dataset.root=outputs/recordings/nexarm_stack_bowls_$(date +%Y%m%d_%H%M%S)`). Do not use `outputs/datasets/nexarm_stack_bowls`, which is the stack-bowls sim generator's default output.

## `TimeoutError: No position reply from NexArm`

The leader firmware may emit debug lines over Serial that interrupt protocol frames. The driver automatically retries up to 3 times. For a permanent fix, comment out `Serial.printf` lines in `Nex_Arm.ino` when `lerobotMode == true` and reflash.

## Checksum errors

The leader firmware has a known checksum bug. The driver automatically accepts both correct and vendor checksums for maximum compatibility.

## Camera not found or busy

- Run `uv run lerobot-find-cameras opencv` to scan available indices.
- Close other applications using the camera (OBS, browser, Zoom).
- Re-scan after unplugging and replugging USB devices.

## Camera display errors (`cv2.error` or headless SSH)

Use `--dry-run` or `--save-snapshot`:

- In `teleoperate_cartesian.py`: add `--dry-run`
- In `vision_grasp.py`: add `--dry-run`
- In `calibrate_camera_alignment.py`: add `--save-snapshot outputs/alignment.png`

## Record loop running slower (5.0 Hz) than target FPS (30 Hz)

1. **Follower serial firmware desync (5.0 Hz telemetry)**:
   - When the physical follower arm is in default/standalone mode (not in LeRobot bridge mode), its AT32 coprocessor broadcasts telemetry packets (`CMD_GET_CUR_COORDS = 11`) at exactly **5.0 Hz (every 200 ms)**.
   - If `read_positions()` accepted these unrequested telemetry frames without verifying the reply command ID (`CMD_READ_POS = 96`), every read would block for ~200 ms, throttling the record loop to 5.0 Hz and confusing Cartesian coordinates with servo positions.
   - The driver in [`packages/lerobot_robot_nexarm/src/lerobot_robot_nexarm/motors/nexarm.py`](../../packages/lerobot_robot_nexarm/src/lerobot_robot_nexarm/motors/nexarm.py) filters incoming packets strictly for the expected command reply (`CMD_READ_POS`), and `rtscts=False` prevents Linux serial driver write blocking on CH340 USB converters. If the loop still runs at 5 Hz, check that the 12V supply is on and restart the command.

2. **USB 2.0 camera bandwidth contention (YUYV vs MJPG)**:
   - By default on Linux, OpenCV requests raw uncompressed `YUYV` (4:2:2). Two 640x480 @ 30 FPS cameras streaming uncompressed video consume ~37 MB/s, which exceeds single USB 2.0 controller bandwidth and causes the UVC driver to throttle or drop frames down to 5–15 FPS.
   - **Fix**: Specify `"fourcc":"MJPG"` for OpenCV cameras that support Motion-JPEG (like the Logitech C270 / `front` camera); the example scripts take `--front-fourcc MJPG`:

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
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=20 \
    --display_data=false \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2
```

3. **Rerun display overhead**:
   - Setting `--display_data=true` logs every uncompressed camera feed synchronously to Rerun every step. For high-FPS recording without frame drops, set `--display_data=false` or stream to a remote Rerun viewer with `--display_ip`.

## Policy training loss doesn't decrease / arm moves hesitantly

- Ensure you have at least 50 high-quality demonstration episodes with consistent initial states.
- If an ACT policy collapses to the mean position, reduce the KL loss weight: `--policy.kl_weight=5.0` or `1.0`.
- Increase batch size: `--batch_size=32` or `--batch_size=64`.
- Train for more steps: `--steps=200000`.

## Low inference FPS on CPU

ACT uses action chunking (`chunk_size=100`) so CPU inference is normally 20–30 Hz. If slower:

- Verify no background processes saturate CPU cores.
- Dual-camera capture adds ~45 ms per frame — this is normal.
