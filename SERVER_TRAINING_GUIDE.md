# NexArm TurboVLA Server Training Guide

This guide provides complete, copy-pasteable instructions to set up, prepare datasets, train, and deploy **TurboVLA** for the **Hiwonder NexArm** (6-DOF manipulator) on a remote GPU server or cluster.

---

## Architecture Summary

| Component                  | NexArm TurboVLA Configuration                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Robot Platform**         | Hiwonder NexArm (6-DOF: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) |
| **Vision Backbone**        | DINOv3 ViT-B over the dataset's views (`front`, `wrist`, and `top` when present) at 224×224 by default        |
| **Language Encoder**       | Online BERT (`google-bert/bert-base-uncased`)                                                                 |
| **Interaction**            | 6-layer bidirectional cross-attention                                                                         |
| **Action Head**            | ACT-style transformer decoder (horizon: 16 steps, action dim: 6, state dim: 6)                                |
| **Control Rate / Latency** | 32 Hz closed-loop control with < 1 GB VRAM at inference time                                                  |

---

## 1. Fast Server Setup (One Command)

On your remote GPU server (Ubuntu 22.04 / 24.04 with NVIDIA drivers):

```bash
# Clone the repository with submodules (including TurboVLA)
git clone --recurse-submodules https://github.com/khanhthanhdev/lerobot-nexarm.git
cd lerobot-nexarm

# Run the automated server setup script
bash scripts/nexarm/setup_server.sh
```

The script automatically:

1. Installs or updates `uv`.
2. Syncs the environment with PyTorch, CUDA, Transformers, DINOv3 (timm), Accelerate, and LeRobot.
3. Caches foundation models (`facebook/dinov3-vitb16-pretrain-lvd1689m` and `google-bert/bert-base-uncased`).

The script syncs only `--extra smolvla --extra training --extra groot`, which is enough for a training server (including simulation data generation). On a machine that also records or rolls out on the real arm, use the full NexArm profile from [`run.md`, Dependency Profiles](run.md#dependency-profiles) plus `--extra smolvla --extra groot`.

_(Optional)_ If you want to fine-tune from official released TurboVLA weights:

```bash
uv run python examples/nexarm/setup_turbovla.py --download-pretrained
```

---

## 2. Dataset Preparation

TurboVLA trains on datasets in the **LeRobot 2.0/3.0 format** containing `front`, `wrist` and optionally `top` camera images, 6-DOF joint states, 6-DOF action targets, and task language descriptions. When several datasets are combined, only the cameras present in all of them are used.

### Path A: Generate Dataset Directly on the Server (Simulation)

You can generate hundreds of high-quality, physics-verified demonstration episodes on the server without any hardware:

```bash
# Generate 50 episodes of the 3-Bowl Stacking Task (with 6 language instruction permutations),
# front/wrist/top cameras, domain randomization on, rendered on GPUs 0 and 1
MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --repo-id local/nexarm_stack_bowls \
    --root outputs/datasets/nexarm_stack_bowls \
    --episodes 50 \
    --gpus 0,1

# Append 50 more episodes to the same dataset
MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --root outputs/datasets/nexarm_stack_bowls \
    --episodes 50 --gpus 0,1 --resume
```

Domain randomization (lighting, textures, colors, dynamics, bowl friction/mass) is on by default; `--no-dr` disables it. A fresh run on an existing `--root` is an error; `--resume` continues after the last logged seed and needs `generation_episodes.jsonl` at the root. Datasets generated before the servo-constant joint mapping change are stale: regenerate them. More flags: [`pipeline.md`, section 1a](docs/source/nexarm/pipeline.md#1a-simulation-scripted-demonstrations).

### Path B: Transfer Locally Recorded Teleoperation Dataset to Server

Real recordings live under `~/.cache/huggingface/lerobot/`: the collection launcher merges into `thanhkt/nexarm_stack_bowls_top`, and `record.py` / `lerobot-record` print the timestamped `Dataset root:` of each session.

```bash
# 1. On your local machine, compress the dataset root:
tar -czvf nexarm_dataset.tar.gz -C ~/.cache/huggingface/lerobot/thanhkt nexarm_stack_bowls_top

# 2. Transfer to server via rsync or scp:
rsync -avzP nexarm_dataset.tar.gz user@server-ip:~/lerobot-nexarm/

# 3. On the server, extract it and train with --real-dataset-root datasets/nexarm_stack_bowls_top:
mkdir -p datasets && tar -xzvf nexarm_dataset.tar.gz -C datasets
```

### Path C: Sync via Hugging Face Hub

```bash
# On local machine (upload the dataset root):
uv run python examples/nexarm/upload_dataset.py \
    --repo-id your_username/nexarm_stack_bowls_top \
    --root ~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top

# On server: pass --real-repo-id your_username/nexarm_stack_bowls_top to train_turbovla.py
```

---

## 3. Training on the Server

### Option 1: Single-GPU Training

Ideal for a workstation or single GPU (RTX 3090, 4090, A5000, A100):

```bash
# Using the preconfigured launcher script:
bash scripts/nexarm/train_turbovla_single_gpu.sh \
    outputs/datasets/nexarm_stack_bowls \
    outputs/train/nexarm_turbovla \
    16 50000 5000

# Or using the Python script directly:
CUDA_VISIBLE_DEVICES=0 uv run python examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --output-dir outputs/train/nexarm_turbovla \
    --batch-size 16 \
    --max-steps 50000 \
    --save-steps 5000 \
    --lr 5e-5 \
    --horizon 16 \
    --device cuda
```

### Option 2: Multi-GPU Training with PyTorch DDP (`torchrun`)

Automatically distributes batches across all available GPUs on the node:

```bash
# Uses all detected GPUs automatically (or set NUM_GPUS=4):
bash scripts/nexarm/train_turbovla_multi_gpu.sh \
    outputs/datasets/nexarm_stack_bowls \
    outputs/train/nexarm_turbovla \
    16 50000 5000

# Or explicitly:
uv run torchrun --nproc_per_node=4 \
    examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --output-dir outputs/train/nexarm_turbovla \
    --batch-size 16 \
    --max-steps 50000 \
    --save-steps 5000 \
    --lr 5e-5
```

### Option 3: HPC / SLURM Cluster Job

Submit a batch job to a SLURM cluster:

```bash
sbatch scripts/nexarm/train_turbovla.slurm
```

The job trains on `--sim-dataset-root outputs/datasets/nexarm_stack_bowls` and writes to `outputs/train/nexarm_turbovla`, like the launchers above; edit the script to change them. To co-train with real data, add to the `train_turbovla.py` commands `--real-dataset-root <root>` (or `--real-repo-id`) and `--real-ratio`, as in [`turbovla_guide.md`](docs/source/nexarm/turbovla_guide.md#co-training-with-simulation-and-real-data).

To monitor your SLURM job:

```bash
squeue -u $USER
tail -f logs/turbovla_<job_id>.out
```

### Option 4: Fine-Tuning from Pretrained TurboVLA (Fastest)

To achieve fast convergence with fewer steps on smaller demonstration datasets (< 50 episodes), initialize from the official release weights and freeze the visual backbone:

```bash
uv run python examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --pretrained-checkpoint pretrained/TurboVLA/checkpoints/robotwin/steps_55000_ema_model.safetensors \
    --output-dir outputs/train/nexarm_turbovla_ft \
    --freeze-vision \
    --batch-size 16 \
    --max-steps 30000 \
    --save-steps 5000
```

---

## 4. Monitoring & Metrics

### Terminal Progress

The training script displays step count, average L1 action loss, learning rate schedule, and steps per second (SPS):

```
TurboVLA Training:  42%|████▏     | 21000/50000 [35:12<48:38, 9.9sps, loss=0.0182, lr=3.41e-05]
```

### Weights & Biases (WandB)

Enable online logging by adding `--wandb`:

```bash
uv run python examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --output-dir outputs/train/nexarm_turbovla \
    --wandb --wandb-project nexarm-turbovla
```

---

## 5. Artifacts & Checkpoints Produced

In your `--output-dir` (e.g. `outputs/train/nexarm_turbovla/`), the training run generates:

| File                          | Purpose                                                                   |
| ----------------------------- | ------------------------------------------------------------------------- |
| `final_ema_pytorch_model.pt`  | **Recommended**: Exponential Moving Average weights for inference         |
| `final_ema_model.safetensors` | SafeTensors format of the EMA checkpoint                                  |
| `final_model.pt`              | Raw optimizer weights from the last step                                  |
| `stats_turbovla.json`         | Dataset min/max normalization stats (automatically loaded by rollout)     |
| `config.json`                 | Architecture plus `task_type`, `camera_keys`, `image_size`, `fps`, `task` |
| `steps_XXXX_*`                | Periodic checkpoints saved every `--save-steps`                           |

---

## 6. Deploying the Checkpoint to the Robot

Copy the checkpoint directory back to your robot controller or local machine:

```bash
# Transfer trained checkpoint folder from server to local robot machine:
rsync -avzP user@server-ip:~/lerobot-nexarm/outputs/train/nexarm_turbovla/ ./outputs/train/nexarm_turbovla/
```

### A. Test in MuJoCo Simulation:

```bash
uv run python examples/nexarm/rollout_turbovla.py \
    --robot sim \
    --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt
```

Rollout reads the task type, cameras, image size, fps and prompt from `config.json`; in the stack-bowls scene the prompt follows the sampled bowl order, so leave `--task` unset.

### B. Deploy on Physical NexArm Follower:

```bash
uv run python examples/nexarm/rollout_turbovla.py \
    --robot real \
    --follower-port /dev/ttyUSB1 \
    --front-cam 0 \
    --wrist-cam 1 \
    --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt \
    --task "Stack the bowls with red on bottom, blue in middle, and black on top."
```

The RealSense `top` camera is opened by default; add `--no-top-cam` for a 2-view checkpoint (the script exits if the cameras differ from the checkpoint's). See [`turbovla_guide.md`](docs/source/nexarm/turbovla_guide.md#4-rollout--evaluation) for legacy checkpoints.

---

## 7. Hyperparameter Recommendations

| Parameter        | Recommended Value | Description                                                               |
| ---------------- | ----------------- | ------------------------------------------------------------------------- |
| `--batch-size`   | `16` (per GPU)    | Total batch size across GPUs: 16–64                                       |
| `--horizon`      | `16`              | Action chunk size (matches `rollout_turbovla.py` 30 FPS execution)        |
| `--lr`           | `5e-5`            | Base AdamW learning rate for action head and interaction                  |
| `--dinov3-lr`    | `5e-5`            | Learning rate for DINOv3 backbone (if unfrozen)                           |
| `--warmup-steps` | `1000`            | Linear warmup steps before cosine decay                                   |
| `--ema-decay`    | `0.999`           | Exponential moving average decay rate                                     |
| `--precision`    | `bf16`            | BFloat16 mixed precision on Ampere/Ada/Hopper (RTX 30xx/40xx, A100, H100) |
| `--max-steps`    | `30k–50k`         | Convergence usually reached between 30k–50k steps on ~50 episodes         |
