# Training Hiwonder NexArm with TurboVLA

This guide covers setting up, training, and deploying **TurboVLA** (Vision-Language-Action model) for the 6-DOF **Hiwonder NexArm** platform using LeRobot datasets.

---

## What is TurboVLA?

[TurboVLA](https://github.com/H-EmbodVis/TurboVLA) reformulates the conventional $V \to L \to A$ pathway as a direct $V + L \to A$ mapping:

- **Vision**: DINOv3 ViT-B over the dataset's camera views (`front`, `wrist` and, when present, `top`) at 224×224 by default (`--image-size`).
- **Language**: BERT (`google-bert/bert-base-uncased`) encoding task instructions.
- **Interaction**: Lightweight 6-layer bidirectional cross-attention module.
- **Action Decoder**: Fast ACT-style transformer predicting continuous action chunks (default horizon: 16 steps) conditioned on multimodal representations and 6-DOF robot proprioception.
- **Efficiency**: Operates at 32 Hz with < 1 GB VRAM at inference time on consumer GPUs.

---

## 1. Environment & Asset Setup

Verify your environment dependencies and cache base foundation models:

```bash
# 1. Install the full NexArm profile (run.md, Dependency Profiles) plus the TurboVLA extras
uv sync --locked --extra test --extra dev --extra core_scripts --extra training --extra nexarm --extra intelrealsense --extra smolvla --extra groot

# 2. Check health and download vision/language backbones
uv run python examples/nexarm/setup_turbovla.py --download-backbones

# 3. (Optional) Download official pretrained TurboVLA weights from Hugging Face:
uv run python examples/nexarm/setup_turbovla.py --download-pretrained
```

`setup_turbovla.py` and `scripts/nexarm/setup_server.sh` mention only `--extra smolvla --extra training --extra groot`; on a machine that also records or rolls out on the real arm, keep the full line above, because `uv sync` removes extras you do not list.

---

## 2. Generating or Collecting Data

### Option A: Simulation (3-Bowl Stacking Task)

Generate synthetic demonstration episodes using the scripted simulation task with language annotations:

```bash
MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --repo-id local/nexarm_stack_bowls \
    --root outputs/datasets/nexarm_stack_bowls \
    --episodes 50
```

Each episode automatically records:

- Camera views: `observation.images.front`, `observation.images.wrist` and `observation.images.top` (`--cameras front,wrist` for two views)
- 6-DOF robot state: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
- 6-DOF action targets
- Language instruction for the episode's bowl order, e.g. _"Stack the bowls with red on bottom, blue in middle, and black on top."_

Domain randomization is on by default (`--no-dr` disables it). Generation flags (`--gpus`, `--resume`, `--calibration`, ...) are listed in [`pipeline.md`, section 1a](pipeline.md#1a-simulation-scripted-demonstrations).

### Option B: Real Hardware Demonstration Recording

Collect real stack-bowls demonstrations with the collection launcher, which merges sessions into the local root `thanhkt/nexarm_stack_bowls_top` (front, wrist, top). See [`DATA_COLLECTION_GUIDE.md`](../../../DATA_COLLECTION_GUIDE.md):

```bash
./scripts/nexarm/collect.sh --num-episodes 50
```

The merged root lives at `~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top`; pass that directory as `--real-dataset-root` below. Its prompt is _"Stack the bowls with red on bottom, blue in middle, and black on top."_

---

## 3. Training TurboVLA

### Standard Training (from scratch with DINOv3 + BERT)

```bash
uv run python examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --output-dir outputs/train/nexarm_turbovla \
    --batch-size 16 \
    --max-steps 50000 \
    --save-steps 5000 \
    --lr 5e-5 \
    --horizon 16
```

### Fine-Tuning from Pretrained Weights (Fastest Convergence)

If you downloaded pretrained TurboVLA release weights, freeze the visual backbone and fine-tune the action head:

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

### Co-Training with Simulation and Real Data

Run from the repository root, or use absolute paths:

```bash
CUDA_VISIBLE_DEVICES=3,4 uv run torchrun --nproc_per_node=2 --master_port=29500 \
    examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --real-dataset-root ~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top \
    --real-ratio 0.5 \
    --pretrained-checkpoint pretrained/TurboVLA/checkpoints/robotwin/steps_55000_ema_model.safetensors \
    --output-dir outputs/train/nexarm_turbovla_cotrain \
    --batch-size 16 \
    --max-steps 50000 \
    --save-steps 5000 \
    --lr 5e-5 \
    --horizon 16 \
    --num-workers 8 \
    --wandb --wandb-project nexarm-turbovla
```

`--cameras` defaults to the camera keys common to every dataset, so 3-camera sim and real data train on three views. To co-train with the legacy 2-camera Hub dataset, use `--real-repo-id thanhkt/nexarm_stack_bowls` instead of `--real-dataset-root`; the common views are then `front,wrist`, and the checkpoint needs `--no-top-cam` on the real robot. `--front-cam-key` / `--wrist-cam-key` select exactly two views and cannot be combined with `--cameras`. To use a Hub simulation dataset, replace `--sim-dataset-root` with `--sim-repo-id <owner/dataset>`.

Each local root must already contain the complete `meta/` directory, the recorded `data/`, and any referenced `videos/`. Datasets with an explicit local root are loaded with `local_files_only=True`; incomplete metadata, episodes, or videos produce a local file error without contacting the Hub, and local roots are checked before distributed initialization or model loading. If startup reports missing local metadata, locate the actual dataset root:

```bash
find outputs/datasets -path '*/meta/info.json' -print
```

Pass the directory above `meta/` as `--sim-dataset-root`; for example, `/path/to/dataset/meta/info.json` requires `--sim-dataset-root /path/to/dataset`. If no metadata exists, generate or copy the complete LeRobot dataset first.

For real-only training, use `--no-sim --real-dataset-root <root>` and omit `--real-ratio`.

### Outputs Produced:

- `steps_XXXX_ema_pytorch_model.pt` & `steps_XXXX_ema_model.safetensors`: Exponential Moving Average weights (recommended for inference); `final_ema_pytorch_model.pt` at the end of training.
- `steps_XXXX_model.pt`: Raw optimizer weights.
- `stats_turbovla.json`: Min/max dataset normalization statistics (automatically consumed by rollout).
- `config.json`: Architecture parameters plus the training contract read by rollout: `task_type` (`--task-type auto` picks `stack_bowls` or `pick_place` from the prompts), ordered `camera_keys`, `image_size`, `fps` and `task`.

---

## 4. Rollout & Evaluation

`rollout_turbovla.py` reads the task type, cameras and their order, image size, fps and default prompt from the checkpoint's `config.json`. Checkpoints trained before these fields existed fall back to front/wrist[/top] by view count, 224 px, 30 fps and a task type guessed from the file names (override with `--task-type`). `stats_turbovla.json` is auto-detected from the checkpoint folder.

### In MuJoCo Simulation:

```bash
uv run python examples/nexarm/rollout_turbovla.py \
    --robot sim \
    --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt
```

In the stack-bowls scene the prompt follows the sampled bowl order, so leave `--task` unset. For a pick-and-place checkpoint without a saved task type, add `--task-type pick_place`; its training prompt is _"Pick up the red cube, place it in the green target zone, and release it."_

### On Physical Hardware:

```bash
uv run python examples/nexarm/rollout_turbovla.py \
    --robot real \
    --follower-port /dev/ttyUSB1 \
    --front-cam 0 \
    --wrist-cam 1 \
    --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt \
    --task "Stack the bowls with red on bottom, blue in middle, and black on top."
```

The `top` RealSense is opened by default. A 2-view checkpoint needs `--no-top-cam`; the script exits at startup if the opened cameras differ from the checkpoint's. Keep `--task` identical to the training prompt (omit it to use the saved one).
