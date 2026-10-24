# NexArm TurboVLA Server Training Guide

This guide provides complete, copy-pasteable instructions to set up, prepare datasets, train, and deploy **TurboVLA** for the **Hiwonder NexArm** (6-DOF manipulator) on a remote GPU server or cluster.

> **Using the desktop app?** The NexArm desktop app can do the server steps below for you: add the server in Lab: Train, pick it as the runner, and it uploads the dataset over an SSH tunnel, trains there and brings the checkpoints back. Prepare a fresh server once with `scripts/nexarm-gpu-bootstrap.sh` (see [`docs/service/README.md`](docs/service/README.md)). The manual steps in this guide remain valid for TurboVLA and anything the app does not cover.

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
# Clone the repository (the setup script installs TurboVLA as a pinned package)
git clone https://github.com/khanhthanhdev/lerobot-nexarm.git
cd lerobot-nexarm

# Run the automated server setup script
bash scripts/nexarm/setup_server.sh
```

The script automatically:

1. Installs or updates `uv`.
2. Syncs the environment with PyTorch, CUDA, Transformers, DINOv3 (timm), the pinned `turbovla` package, and LeRobot.
3. Caches foundation models (`facebook/dinov3-vitb16-pretrain-lvd1689m` and `google-bert/bert-base-uncased`).

The script syncs the full NexArm profile from [`run.md`, Dependency Profiles](run.md#dependency-profiles) plus `--extra turbovla`, which installs the pinned TurboVLA package. Use the same line on any other machine that trains or runs TurboVLA.

_(Optional)_ Download the official released TurboVLA weights. `lerobot-train` has no partial loading for weights whose heads differ from the NexArm model (the former `--pretrained-checkpoint`), so they cannot seed a NexArm run directly; a converted NexArm checkpoint can be fine-tuned with `--policy.path`. See the dropped features in [section 3](#3-training-on-the-server).

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

Real recordings live under `~/.cache/huggingface/lerobot/`: the collection launcher merges into `thanhkt/nexarm_stack_bowls_top`, and `record.py` / `examples/nexarm/record_collection.py` print the timestamped `Dataset root:` of each session.

```bash
# 1. On your local machine, compress the dataset root:
tar -czvf nexarm_dataset.tar.gz -C ~/.cache/huggingface/lerobot/thanhkt nexarm_stack_bowls_top

# 2. Transfer to server via rsync or scp:
rsync -avzP nexarm_dataset.tar.gz user@server-ip:~/lerobot-nexarm/

# 3. On the server, extract it and train with --dataset.root=datasets/nexarm_stack_bowls_top:
mkdir -p datasets && tar -xzvf nexarm_dataset.tar.gz -C datasets
```

### Path C: Sync via Hugging Face Hub

```bash
# On local machine (upload the dataset root):
uv run python examples/nexarm/upload_dataset.py \
    --repo-id your_username/nexarm_stack_bowls_top \
    --root ~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top

# On server: pass --dataset.repo_id=your_username/nexarm_stack_bowls_top (no --dataset.root) to lerobot-train
```

---

## 3. Training on the Server

TurboVLA trains through LeRobot's own trainer with the policy plugin (`--policy.type=turbovla`). The loss, parameter groups, frozen BERT, AdamW settings and warmup-cosine schedule are those of the former `train_turbovla.py`, and a 2k-step GPU run gave loss curves within 4% of it ([`experiments/261024-turbovla-nexarm-parity`](experiments/261024-turbovla-nexarm-parity/README.md)). The checkpoints load directly in `lerobot-eval`, `lerobot-rollout` and the desktop app.

The launchers take `DATASET_ROOT OUTPUT_DIR BATCH_SIZE MAX_STEPS SAVE_STEPS` as positional arguments, followed by any extra `lerobot-train` flags. The dataset id defaults to `local/<folder name of DATASET_ROOT>`; set `REPO_ID` to use another one. A complete run is also written as a config file in [`experiments/261024-turbovla-nexarm-parity/train_full.yaml`](experiments/261024-turbovla-nexarm-parity/train_full.yaml), which you start with `uv run lerobot-train --config_path <file>`.

The former script's flags map to `--policy.optimizer_lr` (`--lr`), `--policy.optimizer_vision_lr` (`--dinov3-lr`), `--policy.scheduler_warmup_steps` (`--warmup-steps`), `--policy.camera_keys` (`--cameras`), `--policy.chunk_size` (`--horizon`), `--steps` (`--max-steps`), `--save_freq` (`--save-steps`) and `--batch_size` (`--batch-size`). `--precision bf16` becomes `--accelerator.mixed_precision=bf16` plus `--policy.vision_compute_precision=bf16_autocast --policy.interaction_compute_precision=bf16_autocast`, and EMA becomes `--ema.enable=true --ema.decay=0.999`. `--policy.freeze_text/freeze_vision/freeze_interaction`, `--policy.loss_type` and `--policy.gripper_loss_weight` keep their meaning. Keep gradient accumulation at 1, because lerobot-train steps the scheduler once per batch.

Retiring the script dropped three features: sim/real co-training that mixes two datasets with `--real-ratio`, partial weight loading with `--pretrained-checkpoint` (fine-tuning from the official TurboVLA weights), and the offline gripper-transition report of `diagnose_turbovla.py`. The EMA also differs slightly: diffusers' EMA copies the weights on its first update, where the script blended them. If you need any of these, check out the tag `pre-reorg` or an earlier commit. Checkpoints trained with the former script still run after a one-time conversion; see [`turbovla_guide.md`](docs/source/nexarm/turbovla_guide.md#checkpoints-from-the-former-train_turbovlapy).

### Option 1: Single-GPU Training

Ideal for a workstation or single GPU (RTX 3090, 4090, A5000, A100):

```bash
# Using the preconfigured launcher script:
bash scripts/nexarm/train_turbovla_single_gpu.sh \
    outputs/datasets/nexarm_stack_bowls \
    outputs/train/nexarm_turbovla \
    16 50000 5000

# Or using lerobot-train directly:
CUDA_VISIBLE_DEVICES=0 uv run lerobot-train --policy.type=turbovla \
    --dataset.repo_id=local/nexarm_stack_bowls --dataset.root=outputs/datasets/nexarm_stack_bowls \
    --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=16 --policy.optimizer_lr=5e-5 --policy.optimizer_vision_lr=5e-5 \
    --policy.vision_compute_precision=bf16_autocast --policy.interaction_compute_precision=bf16_autocast \
    --accelerator.mixed_precision=bf16 --ema.enable=true --ema.decay=0.999 \
    --batch_size=16 --steps=50000 --save_freq=5000 \
    --output_dir=outputs/train/nexarm_turbovla
```

### Option 2: Multi-GPU Training with Accelerate

Distributes batches across all available GPUs on the node. `--batch_size` is per GPU:

```bash
# Uses all detected GPUs automatically (or set NUM_GPUS=4):
bash scripts/nexarm/train_turbovla_multi_gpu.sh \
    outputs/datasets/nexarm_stack_bowls \
    outputs/train/nexarm_turbovla \
    16 50000 5000

# Or explicitly, with the same flags as Option 1:
uv run accelerate launch --multi_gpu --num_processes=4 --main_process_port=29500 \
    -m lerobot.scripts.lerobot_train --policy.type=turbovla \
    --dataset.repo_id=local/nexarm_stack_bowls --dataset.root=outputs/datasets/nexarm_stack_bowls \
    --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=16 --policy.optimizer_lr=5e-5 --policy.optimizer_vision_lr=5e-5 \
    --policy.vision_compute_precision=bf16_autocast --policy.interaction_compute_precision=bf16_autocast \
    --accelerator.mixed_precision=bf16 --ema.enable=true --ema.decay=0.999 \
    --batch_size=16 --steps=50000 --save_freq=5000 \
    --output_dir=outputs/train/nexarm_turbovla
```

### Option 3: HPC / SLURM Cluster Job

Submit a batch job to a SLURM cluster:

```bash
sbatch scripts/nexarm/train_turbovla.slurm
```

The job trains on 4 GPUs with `--dataset.root=outputs/datasets/nexarm_stack_bowls` and writes to `outputs/train/nexarm_turbovla`, like the launchers above; edit the script to change them. To train on real data instead, point `--dataset.repo_id` and `--dataset.root` at the real dataset from section 2 (for example `thanhkt/nexarm_stack_bowls_top` and its root).

To monitor your SLURM job:

```bash
squeue -u $USER
tail -f logs/turbovla_<job_id>.out
```

---

## 4. Monitoring & Metrics

### Terminal Progress

Every `--log_freq` steps (200 by default), lerobot-train logs the step, the samples and episodes seen, the epoch, the loss, the gradient norm, the learning rate and the timings (`step`, `smpl`, `ep`, `epch`, `loss`, `grdn`, `lr`, `data_s`, `updt_s`).

### Weights & Biases (WandB)

Enable online logging by adding the WandB flags to any of the commands above (the launchers pass extra flags through):

```bash
bash scripts/nexarm/train_turbovla_single_gpu.sh \
    outputs/datasets/nexarm_stack_bowls outputs/train/nexarm_turbovla 16 50000 5000 \
    --wandb.enable=true --wandb.project=nexarm-turbovla
```

---

## 5. Artifacts & Checkpoints Produced

Every `--save_freq` steps the training run writes `checkpoints/<step>/` in your `--output_dir` (e.g. `outputs/train/nexarm_turbovla/`), and `checkpoints/last` points to the newest:

| Path                                       | Purpose                                                           |
| ------------------------------------------ | ----------------------------------------------------------------- |
| `checkpoints/<step>/pretrained_model_ema/` | **Recommended**: Exponential Moving Average weights for inference |
| `checkpoints/<step>/pretrained_model/`     | Live weights from that step, plus `train_config.json`             |
| `checkpoints/<step>/training_state/`       | Optimizer, scheduler, RNG and EMA state for resuming              |

Each model folder holds the weights, the policy `config.json` (cameras and their order, chunk size, backbones) and the pre/post-processors with the dataset's normalization statistics, so `--policy.path` alone loads it. To resume an interrupted run, pass `--config_path=<output_dir>/checkpoints/last/pretrained_model/train_config.json --resume=true`.

---

## 6. Deploying the Checkpoint to the Robot

Copy the checkpoint directory back to your robot controller or local machine:

```bash
# Transfer the EMA weights of the last checkpoint from the server to the local robot machine:
rsync -avzP user@server-ip:~/lerobot-nexarm/outputs/train/nexarm_turbovla/checkpoints/last/pretrained_model_ema/ \
    ./outputs/train/nexarm_turbovla/pretrained_model_ema/
```

### A. Test in MuJoCo Simulation:

```bash
MUJOCO_GL=egl uv run lerobot-eval \
    --policy.path=outputs/train/nexarm_turbovla/pretrained_model_ema \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=20
```

The environment gives each episode the prompt of its sampled bowl order, so no task needs to be passed.

### B. Deploy on Physical NexArm Follower:

```bash
uv run lerobot-rollout \
    --strategy.type=base \
    --policy.path=outputs/train/nexarm_turbovla/pretrained_model_ema \
    --robot.type=nexarm_follower \
    --robot.port=/dev/ttyUSB1 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"top":{"type":"intelrealsense","serial_number_or_name":"<REALSENSE_SERIAL>","width":640,"height":480,"fps":30}}' \
    --task="Stack the bowls with red on bottom, blue in middle, and black on top."
```

Give the policy exactly the cameras it was trained on; for a 2-view checkpoint, remove the `top` entry. Keep `--task` identical to the training prompt. See [`turbovla_guide.md`](docs/source/nexarm/turbovla_guide.md#4-rollout--evaluation) for checkpoints from the former script.

---

## 7. Hyperparameter Recommendations

| Parameter                         | Recommended Value | Description                                                               |
| --------------------------------- | ----------------- | ------------------------------------------------------------------------- |
| `--batch_size`                    | `16` (per GPU)    | Total batch size across GPUs: 16–64                                       |
| `--policy.chunk_size`             | `16`              | Action chunk size; the first 8 actions of each chunk run at 30 FPS        |
| `--policy.optimizer_lr`           | `5e-5`            | Base AdamW learning rate for action head and interaction                  |
| `--policy.optimizer_vision_lr`    | `5e-5`            | Learning rate for DINOv3 backbone (if unfrozen)                           |
| `--policy.scheduler_warmup_steps` | `1000`            | Linear warmup steps before cosine decay                                   |
| `--ema.decay`                     | `0.999`           | Exponential moving average decay rate (with `--ema.enable=true`)          |
| `--accelerator.mixed_precision`   | `bf16`            | BFloat16 mixed precision on Ampere/Ada/Hopper (RTX 30xx/40xx, A100, H100) |
| `--steps`                         | `30k–50k`         | Convergence usually reached between 30k–50k steps on ~50 episodes         |

With `bf16`, also pass `--policy.vision_compute_precision=bf16_autocast --policy.interaction_compute_precision=bf16_autocast`, as the launchers do.
