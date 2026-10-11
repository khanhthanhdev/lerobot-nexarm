# NexArm pipeline: data collection, training, rollout, benchmark

End-to-end workflow for the Hiwonder NexArm. Run every command from the repository root with `uv run`.

## 0. Cameras

All stages use up to three RGB cameras. Names must match between collection, training, and rollout.

| Name    | Real hardware                      | Simulation (MJCF camera) | Dataset key                |
| ------- | ---------------------------------- | ------------------------ | -------------------------- |
| `front` | USB webcam (OpenCV)                | `front`                  | `observation.images.front` |
| `wrist` | USB webcam on the gripper (OpenCV) | `wrist`                  | `observation.images.wrist` |
| `top`   | Intel RealSense D435i              | `top`                    | `observation.images.top`   |

All are 640x480 at 30 FPS. The `top` camera is on by default everywhere (real collection, record, teleoperate, rollout, `inference.yaml`, simulation); pass `--no-top-cam` to run with `front` and `wrist` only. A policy must be given exactly the cameras it was trained on: `rollout.py` and `rollout_turbovla.py` exit at startup when the opened cameras differ from the policy's image inputs. Legacy 2-camera models and datasets (for example `thanhkt/nexarm_stack_bowls`) need `--no-top-cam`.

Install: use the full NexArm profile from [`run.md`, Dependency Profiles](../../../run.md#dependency-profiles), which includes `intelrealsense`. Add `--extra smolvla --extra groot` to it for TurboVLA. `uv sync` removes extras you do not list, so do not sync a smaller set.

## 1. Data collection

### 1a. Simulation (scripted demonstrations)

```bash
MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --repo-id local/nexarm_stack_bowls \
    --root outputs/datasets/nexarm_stack_bowls \
    --episodes 50
```

- Cameras default to `front,wrist,top` (`--cameras front,wrist` for a 2-camera dataset). Only successful episodes are kept.
- Domain randomization (visual, dynamics, and every bowl's friction and mass; bowl colours stay fixed because the prompt names them) is on by default; `--no-dr` disables it.
- `--gpus 0,1` takes comma-separated GPU ids for EGL rendering; `--workers` defaults to one per id. `--split eval` writes a held-out set (seeds start at 1000000). `--calibration PATH` and `--action-delay-steps N` apply a sim calibration (see `sim_calibration.md`). Other flags: `--seed-start`, `--max-attempts`, `--no-video`.
- A fresh run on an existing `--root` is an error. `--resume` appends to it (also with `--workers`/`--gpus`), continuing after the last logged seed; it requires `generation_episodes.jsonl` and a `generation_report.json` whose `joint_mapping` matches the current mapping, and refuses to resume otherwise.
- Datasets generated before the servo-constant joint mapping (`rad = (raw - 2048) * 2π / 4096`) are stale and are refused by `--resume` (their report has no `joint_mapping`): regenerate them.
- Sim datasets record `robot_type="nexarm_follower"`, the same as real datasets. Provenance is in `<root>/generation_report.json` and `<root>/generation_episodes.jsonl`.

The pick-and-place generator `generate_sim_dataset.py` takes the same generation flags.

### 1b. Real robot (teleoperation with a leader arm)

Find the serial ports first (`uv run lerobot-find-port`; defaults are leader `/dev/ttyUSB0`, follower `/dev/ttyUSB1`), then check cameras and record in one step. On the collection rig, `./scripts/nexarm/collect.sh` runs the same script with the rig's ports, by-id camera paths and pixel formats:

```bash
uv run python examples/nexarm/prepare_collection.py \
    --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 \
    --front-cam 0 --wrist-cam 1 --num-episodes 50
```

- `--top-cam` auto-detects the RealSense when exactly one is connected; pass `--top-cam <SERIAL>` when several are. `--no-top-cam` records front and wrist only.
- `--test-cameras` checks the cameras only (no arm needed); `--stress-test [SECONDS]` runs all streams, including RealSense RGB and depth, together and reports fps, drops and stalls.
- Only RGB is recorded for `top`. Depth and a point cloud are checked during the test and saved to `outputs/collection/<timestamp>/`, but not stored in the dataset.
- Sessions merge into the local root `thanhkt/nexarm_stack_bowls_top` (3 cameras, created by the first session after `hf auth login`) at `~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top`; nothing is uploaded. For the old 2-camera root use `--no-top-cam --root-repo-id thanhkt/nexarm_stack_bowls`. Details, keys and merge recovery: [`DATA_COLLECTION_GUIDE.md`](../../../DATA_COLLECTION_GUIDE.md).

Simpler alternatives with the same camera flags (`--front-cam`, `--wrist-cam`, `--front-fourcc`, `--wrist-fourcc`, `--top-cam`, `--no-top-cam`): `examples/nexarm/record.py` and `examples/nexarm/teleoperate.py`, or `lerobot-record` with `--robot.cameras` including `top: {type: intelrealsense, serial_number_or_name: <serial>, width: 640, height: 480, fps: 30}`. `lerobot-record` (and `record.py`) appends `_YYYYMMDD_HHMMSS` to the repo id and prints the final `Dataset repo_id:` and `Dataset root:`; use those in later commands.

Check a recorded episode with `examples/nexarm/export_episode_video.py` (side-by-side front, wrist and top).

## 2. Training

Train from the dataset you collected. The merged real root is local only, so pass its directory (or your `--merge-root`):

```bash
REAL_ROOT=~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top
```

### TurboVLA

```bash
# Simulation data only
uv run python examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --output-dir outputs/train/nexarm_turbovla \
    --batch-size 16 --max-steps 50000 --save-steps 5000

# Co-train simulation + collected real data (real drawn 50% of the time)
uv run python examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --real-dataset-root "$REAL_ROOT" --real-ratio 0.5 \
    --output-dir outputs/train/nexarm_turbovla_cotrain
```

Use `--no-sim --real-dataset-root "$REAL_ROOT"` for real data only. `--cameras` defaults to the camera keys common to all datasets; `--front-cam-key` / `--wrist-cam-key` select exactly two views instead. `--task-type auto` saves `stack_bowls` or `pick_place` from the training prompts. The checkpoint's `config.json` records `task_type`, `camera_keys`, `image_size`, `fps` and `task`, which rollout reads. See [`turbovla_guide.md`](turbovla_guide.md) for fine-tuning from a pretrained checkpoint.

### ACT on the real stack-bowls dataset

```bash
# Collected 3-camera root (local)
uv run python examples/nexarm/train_stack_bowls_act.py \
    --repo-id thanhkt/nexarm_stack_bowls_top --dataset-root "$REAL_ROOT" --check-data
uv run python examples/nexarm/train_stack_bowls_act.py \
    --repo-id thanhkt/nexarm_stack_bowls_top --dataset-root "$REAL_ROOT"
```

Without `--repo-id`/`--dataset-root` the launcher downloads the legacy 2-camera Hub dataset `thanhkt/nexarm_stack_bowls`. It requires `front` and `wrist`, uses `top` when the dataset has it, and reads fps and resolution from the dataset metadata. See `examples/nexarm/stack_bowls_act.md` for multi-GPU and resume instructions.

### Other policies

Use `lerobot-train` with a dataset that contains the cameras you intend to use at rollout. Multi-GPU: see the ACT guide above or the LeRobot multi-GPU documentation.

## 3. Rollout

### TurboVLA

```bash
# Simulation (task, cameras, fps and image size come from the checkpoint; the stack-bowls
# prompt follows the sampled bowl order)
uv run python examples/nexarm/rollout_turbovla.py --robot sim \
    --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt \
    --save-video outputs/rollout.mp4

# Real robot (front, wrist and auto-detected top camera)
uv run python examples/nexarm/rollout_turbovla.py --robot real \
    --follower-port /dev/ttyUSB1 --front-cam 0 --wrist-cam 1 \
    --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt \
    --task "Stack the bowls with red on bottom, blue in middle, and black on top."
```

`--task` defaults to the prompt saved with the checkpoint; keep it identical to the training prompt. A 2-view checkpoint needs `--no-top-cam`; a camera mismatch exits with an error. Checkpoints without the new `config.json` fields fall back to front/wrist[/top] by view count, 224 px images, 30 fps and a task-type guess (override with `--task-type`). Ctrl+C stops the rollout.

### LeRobot policies (ACT, pi0, SmolVLA, ...)

```bash
uv run python examples/nexarm/rollout.py --follower-port /dev/ttyUSB1 \
    --policy-path <checkpoint dir or Hub id> --front-cam 0 --wrist-cam 1
```

Add `--no-top-cam` for 2-camera checkpoints. `--strategy` selects `base`, `sentry`, `highlight` or `dagger` (the last three need `--repo-id`). `examples/nexarm/inference.yaml` is a ready config with the `top` camera enabled; delete its `top` block for 2-camera checkpoints.

Keep the arm clear of obstacles and a hand near power the first time you run a new checkpoint.

## 4. Benchmark

`lerobot-nexarm-sim-benchmark` evaluates LeRobot-format checkpoints on the single-arm **pick-and-place** task (red cube into the green zone) on identical seeds and reports success rate and latency. It does not benchmark stack-bowls policies; evaluate those with `rollout_turbovla.py --robot sim` or a `sentry` rollout.

```bash
MUJOCO_GL=egl uv run lerobot-nexarm-sim-benchmark \
    --policy act=outputs/train/act_nexarm_sim/checkpoints/last/pretrained_model \
    --policy smolvla=outputs/train/smolvla_nexarm_sim/checkpoints/last/pretrained_model \
    --episodes 20 --device cuda \
    --output-dir outputs/nexarm_sim_benchmark
```

- Checkpoints must be trained on pick-and-place data (e.g. `generate_sim_dataset.py`), the six NexArm joints and the same cameras; a generic base VLA checkpoint is not comparable.
- `--cameras` defaults to `front wrist top`. For a policy trained on two cameras pass `--cameras front wrist`.
- `--calibration PATH` (fitted at `--fps`), `--action-delay-steps N` (explicit value > calibration > 0) and `--dr` (domain randomization, off by default for reproducible comparisons) match the generator options.
- Output: `benchmark.json` and `benchmark.csv` with pick/place success, termination reasons, inference mean/p50/p95 latency, achieved control rate, load time, parameter count and peak CUDA memory.
- Success rate is the primary metric; use termination reasons to separate drops from timeouts and p95 latency to check the policy can sustain 30 FPS. `--realtime` throttles to the control rate; `--seed-start` and `--episode-timeout` control the episode set.
- The benchmark does not encode video, to avoid distorting timing. For a qualitative comparison use a `sentry` rollout or `--save-video` in `rollout_turbovla.py`.

Real-robot throughput check: `uv run python examples/nexarm/prepare_collection.py --test-cameras --stress-test 30`.

See also `sim/README.md` (simulation details) and `AGENT_GUIDE.md` (setup and policy selection).
