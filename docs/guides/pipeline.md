# NexArm pipeline: data collection, training, rollout, benchmark

End-to-end workflow for the Hiwonder NexArm. Run every command from the repository root with `uv run`.

## 0. Cameras

All stages use up to three RGB cameras. Names must match between collection, training, and rollout.

| Name    | Real hardware                      | Simulation (MJCF camera) | Dataset key                |
| ------- | ---------------------------------- | ------------------------ | -------------------------- |
| `front` | USB webcam (OpenCV)                | `front`                  | `observation.images.front` |
| `wrist` | USB webcam on the gripper (OpenCV) | `wrist`                  | `observation.images.wrist` |
| `top`   | Intel RealSense D435i              | `top`                    | `observation.images.top`   |

All are 640x480 at 30 FPS. The `top` camera is on by default everywhere (real collection, record, teleoperate, rollout, `inference.yaml`, simulation); pass `--no-top-cam` to run with `front` and `wrist` only. A policy must be given exactly the cameras it was trained on: `rollout.py` exits at startup when the opened cameras differ from the policy's image inputs. Legacy 2-camera models and datasets (for example `thanhkt/nexarm_stack_bowls`) need `--no-top-cam`.

Install: use the full NexArm profile from [Install, Dependency profiles](../getting-started/install.md#dependency-profiles), which includes `intelrealsense`. Add `--extra turbovla` to it for TurboVLA. `uv sync` removes extras you do not list, so do not sync a smaller set.

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
- `--gpus 0,1` takes comma-separated GPU ids for EGL rendering; `--workers` defaults to one per id. `--split eval` writes a held-out set (seeds start at 1000000). `--calibration PATH` and `--action-delay-steps N` apply a sim calibration (see [Sim calibration](sim-calibration.md)). Other flags: `--seed-start`, `--max-attempts`, `--no-video`.
- A fresh run on an existing `--root` is an error. `--resume` appends to it (also with `--workers`/`--gpus`), continuing after the last logged seed; it requires `generation_episodes.jsonl` and a `generation_report.json` whose `joint_mapping` matches the current mapping, and refuses to resume otherwise.
- Datasets generated before the servo-constant joint mapping (`rad = (raw - 2048) * 2π / 4096`) are stale and are refused by `--resume` (their report has no `joint_mapping`): regenerate them.
- Sim datasets record `robot_type="nexarm_follower"`, the same as real datasets. Provenance is in `<root>/generation_report.json` and `<root>/generation_episodes.jsonl`.

```bash
# Append 100 new episodes without deleting existing data
MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --root outputs/datasets/nexarm_stack_bowls \
    --resume \
    --episodes 100
```

The pick-and-place generator `generate_sim_dataset.py` takes the same generation flags.

### 1b. Real robot: the stack-bowls collection rig

Set up the arms and cameras first ([Hardware quickstart](../getting-started/hardware.md)). Find the serial ports (`uv run lerobot-find-port`; defaults are leader `/dev/ttyUSB0`, follower `/dev/ttyUSB1`), then check cameras and record in one step. On the collection rig, `./scripts/nexarm/collect.sh` runs the same script with the rig's ports, by-id camera paths and pixel formats:

```bash
uv run python tools/nexarm/prepare_collection.py \
    --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1 \
    --front-cam 0 --wrist-cam 1 --num-episodes 50
```

- `--top-cam` auto-detects the RealSense when exactly one is connected; pass `--top-cam <SERIAL>` when several are. `--no-top-cam` records front and wrist only.
- `--test-cameras` checks the cameras only (no arm needed); `--stress-test [SECONDS]` runs all streams, including RealSense RGB and depth, together and reports fps, drops and stalls.
- Only RGB is recorded for `top`. Depth and a point cloud are checked during the test and saved to `outputs/collection/<timestamp>/`, but not stored in the dataset.
- Sessions merge into the local root `thanhkt/nexarm_stack_bowls_top` (3 cameras, created by the first session after `hf auth login`) at `~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top`; nothing is uploaded. For the old 2-camera root use `--no-top-cam --root-repo-id thanhkt/nexarm_stack_bowls`. Details, keys and merge recovery: [Data collection](data-collection.md).

### 1c. Any task: record.py and lerobot-record

Record demonstration episodes of your own task with leader-follower teleoperation. Both commands take the camera flags of the [Hardware quickstart](../getting-started/hardware.md#find-and-check-the-cameras):

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

# Using the lerobot-record flags directly (record_collection.py is lerobot-record plus the NexArm collection flags)
uv run python examples/nexarm/record_collection.py \
    --robot.type=nexarm_follower --robot.port=/dev/ttyUSB1 \
    --teleop.type=nexarm_leader  --teleop.port=/dev/ttyUSB0 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"top":{"type":"intelrealsense","serial_number_or_name":"<REALSENSE_SERIAL>","width":640,"height":480,"fps":30}}' \
    --dataset.repo_id=local/nexarm_pick_cube \
    --dataset.single_task="Pick up the red cube and place it into the green zone" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=12 \
    --dataset.reset_time_s=8 \
    --dataset.push_to_hub=false \
    --display_data=true
```

`lerobot-record` pushes to the Hub by default; keep `--dataset.push_to_hub=false` for a `local/` id (use `<hf_user>/...` to upload). `record.py` records front, wrist and the RealSense top camera by default (add `--no-top-cam` for front and wrist only); `--fps` sets both the camera and dataset frame rate.

**The recorder appends a timestamp to the repo id.** `local/nexarm_pick_cube` is saved as e.g. `local/nexarm_pick_cube_20261011_093000` under `~/.cache/huggingface/lerobot/`, and `record.py` / `record_collection.py` print the final `Dataset repo_id:` and `Dataset root:` when recording ends (upstream's plain `lerobot-record` does not print them; use the newest `<name>_YYYYMMDD_HHMMSS` folder in the cache instead). Use that printed id in every follow-up command below (shown as `local/nexarm_pick_cube_<YYYYMMDD_HHMMSS>`). Leave `--dataset.root` unset: a fixed root fails on the second run.

**Episode controls with timed episodes:**

- **`Enter`**: Start / confirm next episode.
- **`←` (Left Arrow)**: Redo current episode (discards bad demonstration).
- **`ESC`**: Finish recording session early and save dataset.

**Untimed episodes:** with `--dataset.episode_time_s=inf --dataset.reset_time_s=inf`, recording starts immediately and has no timeout. Press **→** (or **n**) to end the episode, reset the scene and arm, then press **→** again to record the next one. Reset movements are not recorded; the follower still follows the leader. Use an interactive terminal. `--manual_control=true` adds the READY/COLLECTING/RESET screens of the [collection rig](data-collection.md#3-keyboard-controls-during-recording).

### 1d. Inspect and replay

Visualize recorded episodes, camera video streams, and joint state trajectories:

```bash
uv run lerobot-dataset-viz --repo-id local/nexarm_pick_cube_<YYYYMMDD_HHMMSS> --episode-index 0
```

Verify recorded actions by playing back an episode directly on the follower arm:

```bash
uv run lerobot-replay \
    --robot.type=nexarm_follower \
    --robot.port=/dev/ttyUSB1 \
    --dataset.repo_id=local/nexarm_pick_cube_<YYYYMMDD_HHMMSS> \
    --dataset.episode=0
```

`tools/nexarm/export_episode_video.py` writes a side-by-side front, wrist and top video of one episode.

## 2. Training

For simulation datasets trained on several GPUs (generation, TurboVLA and ACT with Accelerate,
evaluation), follow [Sim training](sim-training.md).

Train from the dataset you collected. The merged real root is local only, so pass its directory (or your `--merge-root`):

```bash
REAL_ROOT=~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top
```

### TurboVLA

```bash
# Simulation data (same flags as scripts/nexarm/train_turbovla_single_gpu.sh)
uv run lerobot-train --policy.type=turbovla \
    --dataset.repo_id=local/nexarm_stack_bowls --dataset.root=outputs/datasets/nexarm_stack_bowls \
    --policy.device=cuda --policy.push_to_hub=false \
    --policy.vision_compute_precision=bf16_autocast --policy.interaction_compute_precision=bf16_autocast \
    --accelerator.mixed_precision=bf16 --ema.enable=true --ema.decay=0.999 \
    --batch_size=16 --steps=50000 --save_freq=5000 \
    --output_dir=outputs/train/nexarm_turbovla

# Collected real data: the same command with the real dataset
#   --dataset.repo_id=thanhkt/nexarm_stack_bowls_top --dataset.root="$REAL_ROOT" \
#   --output_dir=outputs/train/nexarm_turbovla_real
```

The policy uses every camera of the dataset unless `--policy.camera_keys` selects and orders them. The checkpoint folder records the cameras, image size and normalization statistics, which rollout reads. Mixing sim and real data with a ratio and fine-tuning from the official TurboVLA weights are not supported; see [the TurboVLA guide](turbovla.md#3-training-turbovla).

### ACT on the real stack-bowls dataset

```bash
# Collected 3-camera root (local)
uv run python examples/nexarm/train_stack_bowls_act.py \
    --repo-id thanhkt/nexarm_stack_bowls_top --dataset-root "$REAL_ROOT" --check-data
uv run python examples/nexarm/train_stack_bowls_act.py \
    --repo-id thanhkt/nexarm_stack_bowls_top --dataset-root "$REAL_ROOT"
```

Without `--repo-id`/`--dataset-root` the launcher downloads the legacy 2-camera Hub dataset `thanhkt/nexarm_stack_bowls`. It requires `front` and `wrist`, uses `top` when the dataset has it, and reads fps and resolution from the dataset metadata. See `examples/nexarm/stack_bowls_act.md` for multi-GPU and resume instructions.

### ACT on your own dataset

Train an **Action Chunking with Transformers (ACT)** policy on a dataset from section 1c:

```bash
uv run lerobot-train \
    --dataset.repo_id=local/nexarm_pick_cube_<YYYYMMDD_HHMMSS> \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/nexarm_act \
    --job_name=nexarm_act \
    --batch_size=32 \
    --steps=100000 \
    --save_freq=25000 \
    --wandb.enable=false
```

If VRAM is limited (< 8 GB), reduce `--batch_size=16` or `--batch_size=8`. How many steps and which batch to use: [Training tips](training-tips.md#3-how-long-should-i-train). Checkpoints are saved to:

```
outputs/train/nexarm_act/checkpoints/last/pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_postprocessor.json
└── train_config.json
```

### Other policies

Use `lerobot-train` with a dataset that contains the cameras you intend to use at rollout; [Training tips](training-tips.md#2-which-policy-should-i-train) compares the policies. Multi-GPU: see the ACT guide above or the LeRobot multi-GPU documentation.

## 3. Rollout

### TurboVLA

TurboVLA checkpoints are LeRobot policies, so they run like the policies below. Use the EMA weights in `pretrained_model_ema`:

```bash
# Simulation (the stack-bowls prompt follows the sampled bowl order; videos go to --output_dir)
MUJOCO_GL=egl uv run lerobot-eval \
    --policy.path=outputs/train/nexarm_turbovla/checkpoints/last/pretrained_model_ema \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=20 \
    --output_dir=outputs/eval/nexarm_turbovla

# Real robot (front, wrist and auto-detected top camera)
uv run python examples/nexarm/rollout.py --follower-port /dev/ttyUSB1 \
    --policy-path outputs/train/nexarm_turbovla/checkpoints/last/pretrained_model_ema \
    --front-cam 0 --wrist-cam 1 \
    --task "Stack the bowls with red on bottom, blue in middle, and black on top."
```

Keep the task identical to the training prompt. A 2-view checkpoint needs `--no-top-cam`; a camera mismatch exits with an error. `lerobot-rollout --strategy.type=base --policy.path=... --robot.type=nexarm_follower ...` works as well ([the TurboVLA guide, section 4](turbovla.md#4-rollout--evaluation)). Checkpoints from the former `train_turbovla.py` need a one-time conversion first, described in the same section. Ctrl+C stops the rollout.

### LeRobot policies (ACT, pi0, SmolVLA, ...)

```bash
# Using rollout script
uv run python examples/nexarm/rollout.py --follower-port /dev/ttyUSB1 \
    --policy-path <checkpoint dir or Hub id> --front-cam 0 --wrist-cam 1

# Using lerobot-rollout CLI
uv run lerobot-rollout \
    --robot.type=nexarm_follower \
    --robot.port=/dev/ttyUSB1 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"top":{"type":"intelrealsense","serial_number_or_name":"<REALSENSE_SERIAL>","width":640,"height":480,"fps":30}}' \
    --policy.path=outputs/train/nexarm_act/checkpoints/last/pretrained_model \
    --strategy.type=base \
    --display_data=true
```

Add `--no-top-cam` for 2-camera checkpoints (or remove the `top` entry from `--robot.cameras`). `--strategy` selects `base`, `sentry`, `highlight` or `dagger` (the last three need `--repo-id`). To save the rollout as an evaluation dataset, use `--strategy sentry --repo-id <user>/eval_nexarm` (`rollout.py`) or `--strategy.type=sentry --dataset.repo_id=<user>/eval_nexarm --dataset.single_task="<task>"` (`lerobot-rollout`). `examples/nexarm/inference.yaml` is a ready config with the `top` camera enabled; delete its `top` block for 2-camera checkpoints.

Keep the arm clear of obstacles and a hand near power the first time you run a new checkpoint.

## 4. Benchmark

`lerobot-nexarm-sim-benchmark` evaluates LeRobot-format checkpoints on the single-arm **pick-and-place** task (red cube into the green zone) on identical seeds and reports success rate and latency. It does not benchmark stack-bowls policies; evaluate those with `lerobot-eval --env.type=nexarm --env.task=NexArmStackBowls-v0` (section 3) or a `sentry` rollout. `lerobot-eval` reports `pc_success`. For how far the failed episodes got, run `lerobot-nexarm-eval` with the same flags: it adds `pc_<stage>` (`pc_picked_middle`, `pc_nested_middle`, ...) to the log and to `eval_info.json`, and each episode's stage flags to its task's metrics. It logs seeded episodes only, so keep `--seed` set (1000 by default).

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
- The benchmark does not encode video, to avoid distorting timing. For a qualitative comparison use a `sentry` rollout or the episode videos that `lerobot-eval` saves.

Real-robot throughput check: `uv run python tools/nexarm/prepare_collection.py --test-cameras --stress-test 30`.

See also [`sim/README.md`](../../sim/README.md) (simulation details) and [Training tips](training-tips.md) (policy choice, training length, evaluation targets).
