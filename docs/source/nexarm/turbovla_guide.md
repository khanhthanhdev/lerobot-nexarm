# Training Hiwonder NexArm with TurboVLA

This guide covers setting up, training, and deploying **TurboVLA** (Vision-Language-Action model) for the 6-DOF **Hiwonder NexArm** platform using LeRobot datasets.

---

## What is TurboVLA?

[TurboVLA](https://github.com/H-EmbodVis/TurboVLA) reformulates the conventional $V \to L \to A$ pathway as a direct $V + L \to A$ mapping:

- **Vision**: DINOv3 ViT-B (2 views: `front` + `wrist` at 224×224).
- **Language**: BERT (`google-bert/bert-base-uncased`) encoding task instructions.
- **Interaction**: Lightweight 6-layer bidirectional cross-attention module.
- **Action Decoder**: Fast ACT-style transformer predicting continuous action chunks (default horizon: 16 steps) conditioned on multimodal representations and 6-DOF robot proprioception.
- **Efficiency**: Operates at 32 Hz with < 1 GB VRAM at inference time on consumer GPUs.

---

## 1. Environment & Asset Setup

Verify your environment dependencies and cache base foundation models:

```bash
# 1. Ensure required training extras are synced
uv sync --extra smolvla --extra training --extra groot

# 2. Check health and download vision/language backbones
uv run python examples/nexarm/setup_turbovla.py --download-backbones

# 3. (Optional) Download official pretrained TurboVLA weights from Hugging Face:
uv run python examples/nexarm/setup_turbovla.py --download-pretrained
```

---

## 2. Generating or Collecting Data

### Option A: Simulation (3-Bowl Stacking Task)

Generate synthetic demonstration episodes using the scripted simulation task with language annotations:

```bash
uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --repo-id local/nexarm_stack_bowls \
    --root outputs/datasets/nexarm_stack_bowls \
    --episodes 50
```

Each episode automatically records:

- Camera views: `observation.images.front` and `observation.images.wrist`
- 6-DOF robot state: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
- 6-DOF action targets
- Language instruction: e.g. _"Stack the red bowl on the blue bowl and then stack the black bowl on top."_

### Option B: Real Hardware Demonstration Recording

Record demonstrations using leader-follower teleoperation:

```bash
uv run python examples/nexarm/record.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port /dev/ttyUSB1 \
    --front-cam 0 \
    --wrist-cam 1 \
    --repo-id local/nexarm_real_task \
    --task "Pick up the red block and place it into the green tray" \
    --num-episodes 50
```

---

## 3. Training TurboVLA

### Standard Training (from scratch with DINOv3 + BERT)

```bash
uv run python examples/nexarm/train_turbovla.py \
    --dataset-root outputs/datasets/nexarm_stack_bowls \
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
    --dataset-root outputs/datasets/nexarm_stack_bowls \
    --pretrained-checkpoint pretrained/TurboVLA/checkpoints/robotwin/steps_55000_ema_model.safetensors \
    --output-dir outputs/train/nexarm_turbovla_ft \
    --freeze-vision \
    --batch-size 16 \
    --max-steps 30000 \
    --save-steps 5000
```

### Co-Training with Local Simulation and Hugging Face Real Data

Run from the repository root, or use an absolute path for the local simulation dataset:

```bash
CUDA_VISIBLE_DEVICES=3,4 uv run torchrun --nproc_per_node=2 --master_port=29500 \
    examples/nexarm/train_turbovla.py \
    --sim-dataset-root outputs/datasets/nexarm_stack_bowls \
    --real-repo-id thanhkt/nexarm_stack_bowls \
    --real-ratio 0.5 \
    --cameras front,wrist \
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

The simulation directory must already contain the complete `meta/` directory, the recorded `data/`, and any referenced `videos/`. Datasets with an explicit local root are loaded with `local_files_only=True`; incomplete metadata, episodes, or videos produce a local file error without contacting the Hub. The real dataset is downloaded from Hugging Face because `--real-dataset-root` is omitted. To use a Hub simulation dataset too, replace `--sim-dataset-root` with `--sim-repo-id <owner/dataset>`.

If startup reports missing local metadata, locate the actual dataset root:

```bash
find outputs/datasets -path '*/meta/info.json' -print
```

Pass the directory above `meta/` as `--sim-dataset-root`; for example, `/path/to/dataset/meta/info.json` requires `--sim-dataset-root /path/to/dataset`. If no metadata exists, generate or copy the complete LeRobot dataset first. Earlier versions of this script fell back to downloading the placeholder `local/nexarm_dataset` when local metadata was missing, producing a misleading Hub 404. Local roots are now checked before distributed initialization or model loading.

For real-only training, use `--no-sim --real-repo-id thanhkt/nexarm_stack_bowls` and omit `--real-ratio`.

### Outputs Produced:

- `steps_XXXX_ema_pytorch_model.pt` & `steps_XXXX_ema_model.safetensors`: Exponential Moving Average weights (recommended for inference).
- `steps_XXXX_model.pt`: Raw optimizer weights.
- `stats_turbovla.json`: Min/max dataset normalization statistics (automatically consumed by rollout).
- `config.json`: Architecture parameters.

---

## 4. Rollout & Evaluation

### In MuJoCo Simulation:

```bash
uv run python examples/nexarm/rollout_turbovla.py \
    --robot sim \
    --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt \
    --task "Stack the red bowl on the blue bowl and then stack the black bowl on top."
```

_(Note: `stats_turbovla.json` is auto-detected from the checkpoint folder if omitted)._

### On Physical Hardware:

```bash
uv run python examples/nexarm/rollout_turbovla.py \
    --robot real \
    --follower-port /dev/ttyUSB1 \
    --front-cam 0 \
    --wrist-cam 1 \
    --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt \
    --task "Pick up the red cube, place it in the green target zone, and release it."
```
