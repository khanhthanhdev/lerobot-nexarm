# NexArm pipeline: data collection, training, rollout, benchmark

End-to-end workflow for the Hiwonder NexArm. Run every command from the repository root with `uv run`.

## 0. Cameras

All stages use up to three RGB cameras. Names must match between collection, training, and rollout.

| Name    | Real hardware                           | Simulation (MJCF camera) | Dataset key                |
| ------- | --------------------------------------- | ------------------------ | -------------------------- |
| `front` | USB webcam (OpenCV)                     | `front`                  | `observation.images.front` |
| `wrist` | USB webcam on the gripper (OpenCV)      | `wrist`                  | `observation.images.wrist` |
| `top`   | Intel RealSense D435i (pass its serial) | `top`                    | `observation.images.top`   |

All are 640x480 at 30 FPS. `front` and `wrist` are required; `top` is optional on hardware. A policy trained with `top` must be given `top` at rollout and benchmark time, and a policy trained without it must not be.

Install:

```bash
uv sync --locked --extra test --extra dev --extra core_scripts --extra training --extra nexarm --extra intelrealsense
```

Do not run a bare `uv sync --locked --extra ...` with a smaller extra set than you need: `uv sync` removes packages that are not in the requested extras.

## 1. Data collection

### 1a. Simulation (scripted demonstrations)

```bash
MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --repo-id local/nexarm_stack_bowls \
    --root outputs/datasets/nexarm_stack_bowls \
    --episodes 50 \
    --cameras front,wrist,top
```

Useful flags: `--gpus N` (parallel generation), `--resume`, `--seed-start`, `--domain-randomization`, `--no-video`, `--max-attempts`. Only successful episodes are kept.

### 1b. Real robot (teleoperation with a leader arm)

Find the serial ports first (`uv run lerobot-find-port`), then check cameras and record in one step:

```bash
uv run python examples/nexarm/prepare_collection.py \
    --follower-port <FOLLOWER_PORT> --leader-port <LEADER_PORT> \
    --top-cam <REALSENSE_SERIAL> \
    --repo-id <user>/nexarm_stack_bowls --num-episodes 50
```

- `--test-cameras` checks the cameras only (no arm needed); `--stress-test [SECONDS]` runs all streams, including RealSense RGB and depth, together and reports fps, drops and stalls.
- `--top-cam` with no value auto-detects the RealSense when exactly one is connected.
- Only RGB is recorded for `top`. Depth and a point cloud are checked during the test and saved to `outputs/collection/<timestamp>/`, but not stored in the dataset.
- Recording has no time limit: press **Right Arrow** to end an episode, reset the scene, press **Right Arrow** again to start the next. **Left Arrow** re-records, **Esc** finishes.
- New sessions are merged into `--root-repo-id` unless `--no-merge` is given. Merge a session later with `examples/nexarm/merge_collection.py`.

Simpler alternatives that take the same camera flags: `examples/nexarm/record.py` and `examples/nexarm/teleoperate.py` (add `--top-cam <serial>`), or `lerobot-record` with `--robot.cameras` including `top: {type: intelrealsense, serial_number_or_name: <serial>, width: 640, height: 480, fps: 30}`.

Check a recorded episode with `examples/nexarm/export_episode_video.py` (side-by-side front, wrist and top).

## 2. Training

### TurboVLA

```bash
# Simulation data only
uv run python examples/nexarm/train_turbovla.py \
    --dataset-root outputs/datasets/nexarm_stack_bowls \
    --output-dir outputs/train/nexarm_turbovla \
    --batch-size 16 --max-steps 50000 --save-steps 5000

# Co-train simulation + real data (real drawn 50% of the time)
uv run python examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --real-repo-id <user>/nexarm_stack_bowls --real-ratio 0.5 \
    --output-dir outputs/train/nexarm_turbovla_cotrain
```

`--cameras` defaults to `front,wrist,top`. If your dataset has no `top` stream, pass `--cameras front,wrist`. See `turbovla_guide.md` for fine-tuning from a pretrained checkpoint (`--pretrained-checkpoint`, `--freeze-vision`).

### ACT on the real stack-bowls dataset

```bash
uv run python examples/nexarm/train_stack_bowls_act.py --check-data   # decode a real episode first
uv run python examples/nexarm/train_stack_bowls_act.py
```

Requires `front` and `wrist`; `top` is used automatically when the dataset contains it. See `examples/nexarm/stack_bowls_act.md` for multi-GPU and resume instructions.

### Other policies

Use `lerobot-train` with a dataset that contains the cameras you intend to use at rollout. Multi-GPU: see the ACT guide above or the LeRobot multi-GPU documentation.

## 3. Rollout

### TurboVLA

```bash
# Simulation
uv run python examples/nexarm/rollout_turbovla.py --robot sim \
    --checkpoint <path/to/checkpoint> --task "Stack the bowls with red on bottom, blue in middle, and black on top." \
    --save-video outputs/rollout.mp4

# Real robot
uv run python examples/nexarm/rollout_turbovla.py --robot real \
    --follower-port <FOLLOWER_PORT> --front-cam 0 --wrist-cam 1 --top-cam <REALSENSE_SERIAL> \
    --checkpoint <path/to/checkpoint> --task "<task>"
```

A 3-view checkpoint requires `--top-cam` on the real robot; the script exits with an error otherwise. Ctrl+C stops the rollout.

### LeRobot policies (ACT, pi0, SmolVLA, ...)

```bash
uv run python examples/nexarm/rollout.py --follower-port <FOLLOWER_PORT> \
    --policy-path <checkpoint dir or Hub id> --front-cam 0 --wrist-cam 1 --top-cam <REALSENSE_SERIAL>
```

Omit `--top-cam` for 2-camera checkpoints. `--strategy` selects `base`, `sentry`, `highlight` or `dagger` (the last three need `--repo-id`). `examples/nexarm/inference.yaml` is a ready config with a commented-out `top` camera.

Keep the arm clear of obstacles and a hand near power the first time you run a new checkpoint.

## 4. Benchmark

The simulation benchmark runs every policy on the same seeds and reports success rate and latency.

```bash
MUJOCO_GL=egl uv run lerobot-nexarm-sim-benchmark \
    --policy act=outputs/train/act_nexarm_sim/checkpoints/last/pretrained_model \
    --policy smolvla=outputs/train/smolvla_nexarm_sim/checkpoints/last/pretrained_model \
    --episodes 20 --device cuda \
    --output-dir outputs/nexarm_sim_benchmark
```

- `--cameras` defaults to `front wrist top`. For a policy trained on two cameras pass `--cameras front wrist`.
- Output: `benchmark.json` and `benchmark.csv` with pick/place success, termination reasons, inference mean/p50/p95 latency, achieved control rate, load time, parameter count and peak CUDA memory.
- Success rate is the primary metric; use termination reasons to separate drops from timeouts and p95 latency to check the policy can sustain 30 FPS. `--realtime` throttles to the control rate; `--seed-start` and `--episode-timeout` control the episode set.
- The benchmark does not encode video, to avoid distorting timing. For a qualitative comparison use a `sentry` rollout or `--save-video` in `rollout_turbovla.py`.
- `lerobot-nexarm-sim-benchmark` loads LeRobot-format checkpoints. TurboVLA checkpoints are evaluated with `rollout_turbovla.py --robot sim` instead (watch the success outcome and saved video).
- Checkpoints must be trained on the six NexArm joints and the same cameras; a generic base VLA checkpoint is not comparable.

Real-robot throughput check: `uv run python examples/nexarm/prepare_collection.py --test-cameras --stress-test 30`.

See also `sim/README.md` (simulation details) and `AGENT_GUIDE.md` (setup and policy selection).
