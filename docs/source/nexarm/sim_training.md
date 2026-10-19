# Train NexArm policies on simulation data with several GPUs

This guide goes from an empty machine to trained TurboVLA and ACT checkpoints, using datasets generated in the
MuJoCo simulation. Run every command from the repository root on a Linux machine with NVIDIA GPUs.

The steps:

1. Set up the machine.
2. Generate the simulation datasets (rendering spread across GPUs).
3. Train TurboVLA with `torchrun` (PyTorch DDP).
4. Train ACT with Accelerate.
5. Evaluate the checkpoints in simulation.

For single-GPU training, real-robot collection and rollout, see [`pipeline.md`](pipeline.md).

## Before you start

### What the simulation gives you

| Task           | Generator                                          | Prompt                                                                         | Typical length |
| -------------- | -------------------------------------------------- | ------------------------------------------------------------------------------ | -------------- |
| Stack bowls    | `examples/nexarm/generate_stack_bowls_dataset.py`  | One of six bowl orders, e.g. _"Stack the bowls with red on bottom, blue in middle, and black on top."_ | ~340 frames    |
| Pick and place | `examples/nexarm/generate_sim_dataset.py`          | _"Pick up the red cube, place it in the green target zone, and release it."_  | ~375 frames    |

Every episode records the `front`, `wrist` and `top` cameras (640 x 480, 30 FPS), the six joint positions as
state, and the six joint targets as action, in raw servo units. Sim datasets use `robot_type="nexarm_follower"`,
the same as real datasets, so the same trainers read both.

Raw 2048 on every arm servo is the real arm's Home pose in the simulation too: the upper arm vertical, the
forearm and closed gripper horizontal. Each dataset's `generation_report.json` records this as
`joint_mapping: servo_4096_v2`. Sim datasets or calibrations made before this mapping (`servo_4096_v1` or no
`joint_mapping` at all) describe a different pose for the same numbers. Regenerate them; do not mix them in.

### Machine

- Linux with NVIDIA GPUs and drivers that support EGL (headless OpenGL). Check with `nvidia-smi`.
- Disk: about 6 MB per 3-camera episode with video, so about 2 GB for 300 episodes.
- GPU memory: the ACT settings below start from a preset made for 8 GB GPUs. TurboVLA trains a DINOv3 ViT-B and
  needs considerably more; if batch 16 per GPU does not fit, see "CUDA out of memory" under Troubleshooting.

### Install

```bash
git submodule update --init TurboVLA
uv sync --locked --extra test --extra dev --extra core_scripts --extra training --extra nexarm \
    --extra intelrealsense --extra smolvla --extra groot
export MUJOCO_GL=egl
```

This is the full NexArm profile from [`run.md`](../../../run.md#dependency-profiles) plus the TurboVLA extras.
`uv sync` removes extras you do not list, so keep the whole line.

TurboVLA needs its DINOv3 and BERT backbones. DINOv3 is gated on Hugging Face: accept its licence on the
model page, log in, then download both:

```bash
hf auth login
uv run python examples/nexarm/setup_turbovla.py --download-backbones
uv run python examples/nexarm/setup_turbovla.py --download-pretrained   # optional: official TurboVLA weights
```

Check the setup before spending GPU hours:

```bash
uv run python sim/test_sim_models.py
uv run python sim/isaac_lab/build_isaac_assets.py --check
```

## 1. Generate the datasets

### Stack bowls

```bash
MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --repo-id local/nexarm_stack_bowls \
    --root outputs/datasets/nexarm_stack_bowls \
    --episodes 300 \
    --gpus 0,1,2,3
```

### Pick and place

```bash
MUJOCO_GL=egl uv run python examples/nexarm/generate_sim_dataset.py \
    --repo-id local/nexarm_sim_pick_place \
    --root outputs/datasets/nexarm_sim_pick_place \
    --episodes 300 \
    --gpus 0,1,2,3
```

How generation uses the GPUs: `--gpus` starts one worker process per listed GPU, and each worker renders its
cameras with EGL on its own GPU. Workers run separate seed blocks and their episodes are merged into one dataset
at `--root`. Use `--workers N` to run more processes than GPUs (they are assigned to the GPUs in turn). The
physics runs on the CPU, so more workers help only while CPU cores are free.

Useful flags (both generators):

| Flag                                     | Effect                                                                                                                 |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `--episodes N`                           | Number of **accepted** episodes. Failed or low-quality attempts are dropped and logged with their reason.               |
| `--max-attempts N`                       | Attempt budget, default 3 x `--episodes`. Acceptance is normally 90–100 %.                                              |
| `--cameras front,wrist`                  | Two cameras instead of three (stack bowls). The policy must then run with the same two cameras.                         |
| `--no-dr`                                | Turn off domain randomization (lighting, materials, camera jitter, dynamics, object friction and mass). Keep it on for training data. |
| `--split eval`                           | Write a held-out set. Eval seeds start at 1,000,000 and never overlap the training seeds.                               |
| `--resume`                               | Add episodes to an existing `--root`. Refused if the existing data used another joint mapping.                         |
| `--calibration PATH`                     | Apply a fitted sim calibration (`sim_calibration.md`), including its action delay.                                      |
| `--joint-mapping PATH`                   | Use a joint mapping measured on your follower with `probe_joint_mapping.py`.                                            |
| `--no-video`                             | Store PNG frames instead of video (larger on disk, faster to decode).                                                   |

A fresh run on an existing `--root` stops with an error; delete the folder or pass `--resume`.

How many episodes: start with 200–300 per task for ACT and TurboVLA. More episodes with domain randomization on
help sim-to-real more than more training steps do.

### Check what you generated

```bash
cat outputs/datasets/nexarm_stack_bowls/generation_report.json       # counts, rejects, joint_mapping
uv run python examples/nexarm/train_stack_bowls_act.py \
    --repo-id local/nexarm_stack_bowls --dataset-root outputs/datasets/nexarm_stack_bowls --check-data
uv run python examples/nexarm/export_episode_video.py --help           # side-by-side video of one episode
```

`--check-data` decodes episode 0 and checks the cameras, state, action chunks and episode-end padding. It works
for any NexArm dataset, not only stack bowls.

Optionally generate a held-out eval set the same way with `--split eval --episodes 30 --root
outputs/datasets/nexarm_stack_bowls_eval`.

## 2. Train TurboVLA on several GPUs

`examples/nexarm/train_turbovla.py` uses PyTorch DDP when started with `torchrun`: every GPU holds a full copy
of the model, reads a different part of each batch, and gradients are averaged. The first process (rank 0)
prints logs and writes checkpoints.

### Stack bowls, 4 GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 --master_port=29500 \
    examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --output-dir outputs/train/turbovla_stack_bowls_4gpu \
    --batch-size 16 \
    --max-steps 50000 \
    --save-steps 5000 \
    --lr 5e-5 \
    --horizon 16 \
    --num-workers 8
```

`scripts/nexarm/train_turbovla_multi_gpu.sh DATASET_ROOT OUTPUT_DIR BATCH MAX_STEPS SAVE_STEPS` runs the same
command on every visible GPU (`NUM_GPUS=4` to choose).

### Pick and place

Same command with `--sim-dataset-root outputs/datasets/nexarm_sim_pick_place` and its own `--output-dir`.
`--task-type auto` (the default) saves `pick_place` or `stack_bowls` in the checkpoint from the training
prompts. Train one checkpoint per task.

### Fine-tune from the official TurboVLA weights

Converges faster than training from DINOv3 + BERT alone:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 --master_port=29500 \
    examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --pretrained-checkpoint pretrained/TurboVLA/checkpoints/robotwin/steps_55000_ema_model.safetensors \
    --output-dir outputs/train/turbovla_stack_bowls_ft \
    --batch-size 16 --max-steps 30000 --save-steps 5000 --num-workers 8
```

Add `--freeze-vision` to train only the interaction layers and action head (less memory, faster per step).

### Co-train with real demonstrations

Add the real dataset and the share of samples drawn from it:

```bash
    --real-dataset-root ~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top --real-ratio 0.5
```

The mix holds on every GPU (each rank draws from both datasets in that ratio). The cameras used are the ones
every dataset has; the 3-camera sim data and the 3-camera real root train on three views.

### Batch size, learning rate and precision

- `--batch-size` is **per GPU**. Effective batch = `batch-size x grad-accum-steps x GPUs`; the 4-GPU example
  trains at 64.
- To match a single-GPU recipe exactly, divide its batch by the GPU count, or keep the per-GPU batch and expect
  fewer steps to be needed. Start with the default `--lr 5e-5`; if loss is unstable at a large effective batch,
  lower it or raise `--warmup-steps`.
- `--max-steps` and `--save-steps` count optimizer updates, not samples, so 4 GPUs see 4x the data per step.
- `--num-workers` is per GPU: 4 GPUs x 8 workers starts 32 video-decoding processes. Lower it if CPU or RAM runs
  out.
- Precision defaults to bf16 on GPUs that support it (Ampere and newer), otherwise fp16. Force it with
  `--precision`.
- `--wandb --wandb-project nexarm-turbovla` logs from rank 0 only.

### Outputs

In `--output-dir`:

- `steps_XXXX_ema_pytorch_model.pt` and `steps_XXXX_ema_model.safetensors`: averaged (EMA) weights, best for
  rollout. `final_ema_pytorch_model.pt` at the end.
- `steps_XXXX_model.pt`: raw weights.
- `stats_turbovla.json`: normalization statistics, read by rollout.
- `config.json`: task type, ordered camera keys, image size, fps and default prompt, read by rollout.

There is no resume with optimizer state. After an interruption, start a new run from the last
`steps_XXXX_model.pt` with `--pretrained-checkpoint` and the remaining number of steps; the optimizer and
learning-rate schedule start fresh.

## 3. Train ACT on several GPUs

`examples/nexarm/train_stack_bowls_act.py` checks the dataset, then starts the LeRobot trainer through
Accelerate (DDP). It accepts any NexArm dataset with the front and wrist cameras (and uses `top` when present),
so it trains on both sim tasks despite its name. Do not wrap it in `accelerate launch` yourself.

### Stack bowls, 4 GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python examples/nexarm/train_stack_bowls_act.py \
    --repo-id local/nexarm_stack_bowls \
    --dataset-root outputs/datasets/nexarm_stack_bowls \
    --num-gpus 4 --mixed-precision bf16 \
    --output-dir outputs/train/act_stack_bowls_4gpu \
    --batch_size=8 --gradient_accumulation_steps=1 --num_workers=4
```

Add `--dry-run` first to print the dataset summary and the exact `accelerate` command without starting
training.

### Pick and place

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python examples/nexarm/train_stack_bowls_act.py \
    --repo-id local/nexarm_sim_pick_place \
    --dataset-root outputs/datasets/nexarm_sim_pick_place \
    --num-gpus 4 --mixed-precision bf16 \
    --output-dir outputs/train/act_pick_place_4gpu \
    --batch_size=8 --gradient_accumulation_steps=1 --num_workers=4
```

### Settings

- The base settings are in `examples/nexarm/stack_bowls_act.yaml`: 100-action chunks (3.3 s at 30 FPS),
  100,000 steps, a checkpoint every 5,000 steps, and the last 10 % of episodes held out for loss evaluation.
  Any `--key=value` flag after the launcher's own flags overrides the YAML (for example `--steps=50000`).
- Effective batch = `batch_size x gradient_accumulation_steps x GPUs`. The YAML's single-GPU value is 2 x 8 = 16;
  the 4-GPU example uses 8 x 1 x 4 = 32. Raise `batch_size` and lower accumulation on large GPUs.
- `--mixed-precision bf16` needs Ampere or newer; use `fp16` on older GPUs.
- `--num_workers` is per GPU.
- ACT does not read the prompt. A stack-bowls ACT policy sees six bowl orders without being told which one, so it
  learns them only from the image. Use TurboVLA when the instruction matters.
- `--pretrained PATH_OR_HUB_ID` starts from an existing NexArm ACT checkpoint with the same cameras.

### Resume

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes=4 --multi_gpu --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    --config_path=outputs/train/act_stack_bowls_4gpu/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

This restores weights, optimizer state and step count.

Checkpoints are under `--output-dir/checkpoints/<step>/pretrained_model`, with `last` pointing to the newest.

## 4. Evaluate in simulation

### TurboVLA

```bash
MUJOCO_GL=egl uv run python examples/nexarm/rollout_turbovla.py --robot sim \
    --checkpoint outputs/train/turbovla_stack_bowls_4gpu/final_ema_pytorch_model.pt \
    --save-video outputs/rollout_turbovla.mp4
```

The task, cameras, image size and fps come from the checkpoint. In the stack-bowls scene the prompt follows the
sampled bowl order, so leave `--task` unset.

### ACT (pick and place)

```bash
MUJOCO_GL=egl uv run lerobot-nexarm-sim-benchmark \
    --policy act=outputs/train/act_pick_place_4gpu/checkpoints/last/pretrained_model \
    --episodes 50 --device cuda \
    --output-dir outputs/nexarm_sim_benchmark
```

It reports success rate, failure reasons and inference latency on fixed seeds. It covers pick and place only;
check stack-bowls ACT policies with a `sentry` rollout (`pipeline.md`, section 3).

Held-out loss tells you training is working, not that the arm succeeds. Judge checkpoints by simulated success
rate, then on the real arm.

## 5. Before running on the real arm

- Measure the follower's joint directions once with `examples/nexarm/probe_joint_mapping.py` (see
  [`sim_calibration.md`](sim_calibration.md)). If it reports a sign or offset that is not the identity,
  regenerate the sim data with `--joint-mapping outputs/nexarm/joint_mapping_probe.json`; otherwise sim-trained
  policies move those joints the wrong way.
- Run the real arm with exactly the cameras the policy was trained on; `rollout.py` and `rollout_turbovla.py`
  refuse a mismatch.
- Keep a hand near the power switch the first time a new checkpoint moves the arm.

## Troubleshooting

| Symptom                                                              | Fix                                                                                                                                  |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Generation fails to create an OpenGL context, or images are black    | `MUJOCO_GL=egl` is not set, or the driver has no EGL. Check `nvidia-smi` and that `libegl1` is installed.                            |
| Generator: `Dataset already exists: <root>`                          | Delete the folder or pass `--resume`.                                                                                                |
| Generator: `--resume` refused because of `joint_mapping`             | The folder was made with another mapping (e.g. before `servo_4096_v2`). Generate a new dataset in a new folder.                      |
| Many `quality_*` or `ik_failed_*` rejects                            | The reasons are listed in `generation_report.json`. A fresh checkout of the models usually accepts 90–100 %; a big drop means the sim assets changed. |
| `torchrun`: `Address already in use`                                 | Another run uses port 29500. Pass a different `--master_port`.                                                                       |
| NCCL hangs at start                                                  | Try `NCCL_P2P_DISABLE=1`; on cloud machines also check `NCCL_SOCKET_IFNAME`.                                                         |
| CUDA out of memory                                                   | Lower the per-GPU batch and raise gradient accumulation to keep the effective batch; for TurboVLA add `--freeze-vision`.             |
| GPUs mostly idle, CPU at 100 %                                       | Video decoding is the bottleneck. Lower `--num-workers` per GPU if RAM is short, or generate with `--no-video`.                       |
| TurboVLA: `Local sim dataset metadata not found`                     | `--sim-dataset-root` must be the folder that contains `meta/info.json`.                                                              |
| TurboVLA: `TurboVLA repository not found`                            | Run `git submodule update --init TurboVLA`.                                                                                          |
| ACT launcher: `Requested N GPUs, but only M are visible`             | `CUDA_VISIBLE_DEVICES` lists fewer GPUs than `--num-gpus`.                                                                           |
