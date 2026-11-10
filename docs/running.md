# Running NexArm: every command on one page

The commands for running the real rig, in the order you use them: install, find the ports, find the cameras, set your settings, teleoperate, collect data, check and upload it, train, roll out, and calibrate for sim-to-real. Run every command from the repository root through `uv run`. Each section links to the page that owns the topic and explains it in full; update that page, not this one, when behaviour changes.

Defaults used below: leader `/dev/ttyUSB0`, follower `/dev/ttyUSB1`, `front` camera `0`, `wrist` camera `1`, RealSense `top` camera auto-detected, 640x480 at 30 FPS. On Windows replace the ports with your `COM` ports (for example `COM18` / `COM19`) and write each command on one line, or end lines with a backtick in PowerShell instead of `\`.

## 1. Install (once per machine)

```bash
uv sync --locked --extra test --extra dev --extra core_scripts --extra training --extra nexarm --extra intelrealsense
git lfs install && git lfs pull
uv run hf auth login
sudo usermod -a -G dialout $USER   # Linux only; log out and back in
```

Add `--extra turbovla` for TurboVLA and `--extra service` for the desktop engine and `calibrate_arms.py`; `uv sync` removes every extra you do not list, so repeat the whole line. Check the install:

```bash
uv run python -c "from lerobot_robot_nexarm.nexarm_follower import NexArmFollower; print('NexArm installation OK!')"
```

Details: [Install](getting-started/install.md).

## 2. Find the serial ports

Power the follower (12V / 5A) and plug both arms in over USB-C, then:

```bash
uv run lerobot-find-port
```

Unplug the arm it asks for and press Enter; it prints that arm's port. Run it once per arm, or plug in one arm at a time.

| OS      | Leader                 | Follower               |
| ------- | ---------------------- | ---------------------- |
| Linux   | `/dev/ttyUSB0`         | `/dev/ttyUSB1`         |
| Windows | `COM18` (typical)      | `COM19` (typical)      |
| macOS   | `/dev/tty.usbserial-*` | `/dev/tty.usbserial-*` |

Other ways to list ports:

```bash
ls /dev/ttyUSB* /dev/serial/by-id/                                  # Linux
uv run python -m serial.tools.list_ports -v                         # any OS
```

On Windows, Device Manager → Ports (COM & LPT) also lists them; install the CH340 driver if no port appears. Details: [Hardware quickstart, Find the serial ports](getting-started/hardware.md#find-the-serial-ports).

## 3. Find and test the cameras

```bash
uv run lerobot-find-cameras opencv       # USB webcams: index or path for --front-cam / --wrist-cam
uv run lerobot-find-cameras realsense    # RealSense D435i: serial for --top-cam
ls -l /dev/v4l/by-id/                    # Linux: stable device paths that survive replugging
```

Test all cameras together, without connecting the arms; snapshots go to `outputs/collection/<timestamp>/`:

```bash
uv run python tools/nexarm/prepare_collection.py --test-cameras --front-cam 0 --wrist-cam 1
uv run python tools/nexarm/prepare_collection.py --test-cameras --stress-test 30   # fps, drops and stalls over 30 s
```

No RealSense: add `--no-top-cam` to every command on this page. Details: [Hardware quickstart, Find and check the cameras](getting-started/hardware.md#find-and-check-the-cameras).

## 4. Settings

### Script flags

The `examples/nexarm/*.py` scripts and `tools/nexarm/prepare_collection.py` share these flags:

| Flag                               | Default                   | Set it to                                               |
| ---------------------------------- | ------------------------- | ------------------------------------------------------- |
| `--leader-port`                    | `/dev/ttyUSB0`            | Leader port from section 2                              |
| `--follower-port`                  | `/dev/ttyUSB1`            | Follower port from section 2                            |
| `--front-cam`, `--wrist-cam`       | `0`, `1`                  | Index or device path from section 3                     |
| `--front-fourcc`, `--wrist-fourcc` | auto                      | `MJPG` when frames drop on a shared USB bus             |
| `--top-cam`                        | auto-detect one RealSense | Its serial when several are connected                   |
| `--no-top-cam`                     | off                       | On when you have no RealSense or use a 2-camera dataset |
| `--fps`                            | `20`                      | Control loop and dataset rate (cameras capture at 30)   |

### LeRobot CLI flags and YAML configs

The `lerobot-*` commands take `--robot.port`, `--teleop.port` and a `--robot.cameras` JSON instead (examples in [Hardware quickstart](getting-started/hardware.md#leader-follower-teleoperation)), or a YAML file through `--config_path`. Edit the ports, camera indexes and `repo_id` in these files once, then run them:

| File                                                                      | Command                                                                                          |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [`examples/nexarm/teleoperate.yaml`](../examples/nexarm/teleoperate.yaml) | `uv run lerobot-teleoperate --config_path=examples/nexarm/teleoperate.yaml`                      |
| [`examples/nexarm/record.yaml`](../examples/nexarm/record.yaml)           | `uv run lerobot-record --config_path=examples/nexarm/record.yaml`                                |
| [`examples/nexarm/train.yaml`](../examples/nexarm/train.yaml)             | `uv run lerobot-train --config_path=examples/nexarm/train.yaml`                                  |
| [`examples/nexarm/inference.yaml`](../examples/nexarm/inference.yaml)     | `uv run lerobot-rollout --config_path=examples/nexarm/inference.yaml --policy.path=<checkpoint>` |

Any flag after `--config_path` overrides the file, for example `--robot.port=COM19`. Delete the `top` block of a YAML file to run without a RealSense.

### Hugging Face user and dataset location

```bash
export HF_USER="$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')"   # Linux / macOS
$env:HF_USER = "<your_hf_username>"                                                # Windows PowerShell
```

Datasets are written to `~/.cache/huggingface/lerobot/<repo_id>` (set `HF_LEROBOT_HOME` to move it). The recorders append `_YYYYMMDD_HHMMSS` to `repo_id` and print the final id; use that id in later commands. Headless Linux machines render the simulation with `MUJOCO_GL=egl`.

### Check the arm alignment

Place both arms at the vendor Home pose and check how far each joint is off (needs `--extra service`):

```bash
uv run python tools/nexarm/calibrate_arms.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1
```

Zero offsets are fixed in the servos with the vendor desktop app, not in software; see [Calibration, Step 0](guides/calibration.md#step-0-before-you-start).

## 5. Teleoperate

```bash
uv run python examples/nexarm/teleoperate.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 --front-cam 0 --wrist-cam 1
```

Move the leader by hand; the follower must track all six joints. `Ctrl+C` stops. Add `--rerun-save-path outputs/rerun/teleop.rrd` to save a replayable recording. Keyboard Cartesian control without a leader: `uv run python examples/nexarm/teleoperate_cartesian.py --robot real --port /dev/ttyUSB1`. Details: [Hardware quickstart](getting-started/hardware.md#leader-follower-teleoperation).

## 6. Collect data

### Stack-bowls rig (Linux)

```bash
./scripts/nexarm/collect.sh --prepare-only    # check USB access and cameras, arms not connected
./scripts/nexarm/collect.sh                   # record 50 episodes, then merge into the local root
./scripts/nexarm/collect.sh --num-episodes 100 --mouse
./scripts/nexarm/collect.sh --no-top-cam --root-repo-id thanhkt/nexarm_stack_bowls   # 2-camera root
```

`collect.sh` uses the rig's ports and `/dev/v4l/by-id/` camera paths; any flag you pass overrides them.

### Any machine, including Windows

`collect.sh` is a thin wrapper around this command, which runs on every OS:

```bash
uv run python tools/nexarm/prepare_collection.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 --front-cam 0 --wrist-cam 1 --num-episodes 50
```

Keys while recording:

| Key               | READY         | COLLECTING            | RESET              |
| ----------------- | ------------- | --------------------- | ------------------ |
| **Enter / → / n** | Start episode | Stop and save episode | Start next episode |
| **← / r**         | —             | Discard and reset     | —                  |
| **Esc / q**       | Finish        | Save and finish       | Finish             |

Sessions merge into the local root `thanhkt/nexarm_stack_bowls_top_20fps`; nothing is uploaded. Resume a session with `--resume --repo-id <exact timestamped id>`, keep it separate with `--no-merge`, and retry a failed merge with `uv run python tools/nexarm/merge_collection.py --repo-id <session id>`. Details: [Data collection](guides/data-collection.md).

### Your own task

```bash
uv run python examples/nexarm/record.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 \
    --front-cam 0 --wrist-cam 1 --repo-id local/nexarm_pick_cube \
    --task "Pick up the red cube and place it into the green zone" \
    --num-episodes 50 --episode-time 12 --reset-time 8
```

Details: [Pipeline, 1c](guides/pipeline.md#1c-any-task-recordpy-and-lerobot-record). Simulated datasets: [Pipeline, 1a](guides/pipeline.md#1a-simulation-scripted-demonstrations).

## 7. Inspect, replay and upload

```bash
uv run lerobot-dataset-viz --repo-id <repo_id> --episode-index 0
uv run lerobot-replay --robot.type=nexarm_follower --robot.port=/dev/ttyUSB1 --dataset.repo_id=<repo_id> --dataset.episode=0
uv run python tools/nexarm/export_episode_video.py --repo-id <repo_id> --root ~/.cache/huggingface/lerobot/<repo_id> --episodes 0
uv run python tools/nexarm/audit_real_dataset.py --repo-id <repo_id>
uv run python tools/nexarm/upload_dataset.py --repo-id "${HF_USER}/<dataset_name>"
```

## 8. Train

```bash
# ACT on your dataset
uv run lerobot-train --dataset.repo_id=<repo_id> --policy.type=act --policy.device=cuda \
    --policy.push_to_hub=false --output_dir=outputs/train/nexarm_act --batch_size=32 --steps=100000

# ACT on the merged stack-bowls root
uv run python examples/nexarm/train_stack_bowls_act.py --repo-id thanhkt/nexarm_stack_bowls_top \
    --dataset-root ~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top
```

TurboVLA, other policies, training length and multi-GPU: [Pipeline, 2](guides/pipeline.md#2-training), [Training tips](guides/training-tips.md), [TurboVLA](guides/turbovla.md), [Sim training](guides/sim-training.md), [Server training](guides/server-training.md).

## 9. Roll out and evaluate

```bash
# Real robot
uv run python examples/nexarm/rollout.py --follower-port /dev/ttyUSB1 --front-cam 0 --wrist-cam 1 \
    --policy-path outputs/train/nexarm_act/checkpoints/last/pretrained_model

# Simulation
MUJOCO_GL=egl uv run lerobot-eval --policy.path=<checkpoint> \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=20
```

The published TurboVLA, B-spline and Plan-Bind stack-bowls models run by name: `./scripts/nexarm/rollout_stack_bowls.sh planbind` on the rig ([Stack-bowls rollout](guides/stack-bowls-rollout.md)).

Give the policy exactly the cameras it was trained on (`--no-top-cam` for 2-camera checkpoints) and the same task prompt. Keep a hand near the power the first time you run a new checkpoint. Record rollouts as evaluation datasets with `--strategy sentry --repo-id "${HF_USER}/eval_<name>"`. Strategies, `lerobot-rollout` options, DAgger, measuring the success rate and troubleshooting: [Rolling out a trained policy](guides/rollout.md).

## 10. Calibrate for sim-to-real

Before a policy trained on simulated data runs on the real arm, fit the simulation to your arm. In order:

```bash
uv run python tools/nexarm/probe_joint_mapping.py --port /dev/ttyUSB1                       # joint signs and zeros
uv run python examples/nexarm/record.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 \
    --repo-id <user>/nexarm_sysid --num-episodes 20 --episode-time 20                       # free-motion episodes
MUJOCO_GL=egl uv run python tools/nexarm/calibrate_sim.py --repo-id <sysid_repo_id> \
    --out outputs/calibration/nexarm_sim_calibration.json                                   # latency and dynamics
uv run python tools/nexarm/calibrate_camera_alignment.py --camera-name front --cam-index 0  # repeat for wrist, top
```

Then generate sim data with `--calibration` (and `--joint-mapping` if the probe was not the identity). Every step, how to read each result, the files and the flags that take them, and the pre-flight checks for the first real run are in [Calibration: real to sim to real](guides/calibration.md).

## When something fails

| Symptom                             | First thing to try                                                 |
| ----------------------------------- | ------------------------------------------------------------------ |
| `Permission denied: '/dev/ttyUSB*'` | `sudo usermod -a -G dialout $USER`, then log out and back in       |
| Follower does not move              | Ports swapped? 12V supply on? Power-cycle the follower             |
| Loop runs at 5 Hz                   | Follower not in bridge mode: check the 12V supply and restart      |
| Loop drops to 13–15 Hz              | `--front-fourcc MJPG`                                              |
| Camera not found or busy            | Re-run `lerobot-find-cameras`; close OBS, browsers and video calls |
| RealSense or `pyrealsense2` missing | `--no-top-cam`, or add `--extra intelrealsense` to your sync line  |
| `Permission denied: '/nexarm...'`   | `HF_USER` is empty; set it as in section 4                         |

Everything else: [Troubleshooting](reference/troubleshooting.md) and [Data collection, Troubleshooting](guides/data-collection.md#5-troubleshooting).
