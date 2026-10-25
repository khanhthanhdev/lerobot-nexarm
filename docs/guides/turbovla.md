# Training Hiwonder NexArm with TurboVLA

This guide covers setting up, training, and deploying **TurboVLA** (Vision-Language-Action model) for the 6-DOF **Hiwonder NexArm** platform using LeRobot datasets.

---

## What is TurboVLA?

[TurboVLA](https://github.com/H-EmbodVis/TurboVLA) reformulates the conventional $V \to L \to A$ pathway as a direct $V + L \to A$ mapping:

- **Vision**: DINOv3 ViT-B over the dataset's camera views (`front`, `wrist` and, when present, `top`) at 224×224 by default (`--policy.image_size`).
- **Language**: BERT (`google-bert/bert-base-uncased`) encoding task instructions.
- **Interaction**: Lightweight 6-layer bidirectional cross-attention module.
- **Action Decoder**: Fast ACT-style transformer predicting continuous action chunks (default horizon: 16 steps) conditioned on multimodal representations and 6-DOF robot proprioception.
- **Efficiency**: Operates at 32 Hz with < 1 GB VRAM at inference time on consumer GPUs.

---

## 1. Environment & Asset Setup

Verify your environment dependencies and cache base foundation models:

```bash
# 1. Install the full NexArm profile (docs/getting-started/install.md) plus the TurboVLA extra
uv sync --locked --extra test --extra dev --extra core_scripts --extra training --extra nexarm --extra intelrealsense --extra turbovla

# 2. Check health and download vision/language backbones
uv run python examples/nexarm/setup_turbovla.py --download-backbones

# 3. (Optional) Download official pretrained TurboVLA weights (lerobot-train cannot partially load them into a NexArm model, see section 3):
uv run python examples/nexarm/setup_turbovla.py --download-pretrained
```

`--extra turbovla` installs the `turbovla` model package from <https://github.com/khanhthanhdev/TurboVLA>, pinned to one commit in the root `pyproject.toml` (`[tool.uv.sources]`). Keep the full line above, because `uv sync` removes extras you do not list.

### Developing TurboVLA locally

To test changes to TurboVLA itself, clone it next to this repo and point the source at the clone for the time being:

```toml
# pyproject.toml, [tool.uv.sources]: temporary, never commit this line
turbovla = { path = "../TurboVLA", editable = true }
```

Then run `uv sync` with the same extras. An editable install can put the whole TurboVLA checkout on `sys.path`, which makes its repo-only folders (`experiments`, `deployment`) importable. Check with `uv run python -I -c "import experiments"`; it must fail. If it does not, reinstall with `uv sync --reinstall-package turbovla --config-settings editable_mode=strict` (same extras). When the change works, push it to TurboVLA, set `rev` to the pushed commit, restore the `git` source, run `uv lock` and commit `pyproject.toml` and `uv.lock`.

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

Domain randomization is on by default (`--no-dr` disables it). Generation flags (`--gpus`, `--resume`, `--calibration`, ...) are listed in [the pipeline guide, section 1a](pipeline.md#1a-simulation-scripted-demonstrations).

### Option B: Real Hardware Demonstration Recording

Collect real stack-bowls demonstrations with the collection launcher, which merges sessions into the local root `thanhkt/nexarm_stack_bowls_top` (front, wrist, top). See [Data collection](data-collection.md):

```bash
./scripts/nexarm/collect.sh --num-episodes 50
```

The merged root lives at `~/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top`; pass that directory as `--dataset.root` (with `--dataset.repo_id=thanhkt/nexarm_stack_bowls_top`) below. Its prompt is _"Stack the bowls with red on bottom, blue in middle, and black on top."_

---

## 3. Training TurboVLA

TurboVLA trains through LeRobot's trainer with the policy plugin (`--policy.type=turbovla`), and the checkpoint loads directly in `lerobot-eval`, `lerobot-rollout` and the desktop app. The loss, parameter groups, frozen BERT, AdamW settings and warmup-cosine schedule are those of the former `train_turbovla.py`; a 2k-step GPU run gave loss curves within 4% of it ([`experiments/261024-turbovla-nexarm-parity`](../../experiments/261024-turbovla-nexarm-parity/README.md)).

```bash
uv run lerobot-train --policy.type=turbovla \
    --dataset.repo_id=local/nexarm_stack_bowls --dataset.root=outputs/datasets/nexarm_stack_bowls \
    --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=16 --policy.optimizer_lr=5e-5 --policy.optimizer_vision_lr=5e-5 \
    --policy.vision_compute_precision=bf16_autocast --policy.interaction_compute_precision=bf16_autocast \
    --accelerator.mixed_precision=bf16 --ema.enable=true --ema.decay=0.999 \
    --batch_size=16 --steps=50000 --save_freq=5000 \
    --output_dir=outputs/train/nexarm_turbovla
```

`scripts/nexarm/train_turbovla_single_gpu.sh` runs the same command. For several GPUs, use `scripts/nexarm/train_turbovla_multi_gpu.sh` or `uv run accelerate launch --multi_gpu --num_processes=<N> -m lerobot.scripts.lerobot_train <same flags>`. To train on real data, pass the real dataset instead, for example `--dataset.repo_id=thanhkt/nexarm_stack_bowls_top --dataset.root=$HOME/.cache/huggingface/lerobot/thanhkt/nexarm_stack_bowls_top`. `--policy.freeze_vision=true` trains only the interaction layers and the action head. `--policy.camera_keys` sets the views and their order; by default the model uses every camera of the dataset. A legacy 2-camera dataset such as `thanhkt/nexarm_stack_bowls` gives a 2-view checkpoint, which runs on the real robot without the `top` camera. The mapping of the former script's flags is in [Server training](server-training.md#3-training-on-the-server).

`--dataset.root` must be the folder that contains `meta/info.json`, the recorded `data/` and any referenced `videos/`. If startup reports missing local metadata, locate the actual dataset root:

```bash
find outputs/datasets -path '*/meta/info.json' -print
```

Retiring the script dropped three features: sim/real co-training that mixes two datasets with `--real-ratio`, partial weight loading with `--pretrained-checkpoint` (fine-tuning from the official TurboVLA weights), and the offline gripper-transition report of `diagnose_turbovla.py`. The EMA also differs slightly: diffusers' EMA copies the weights on its first update, where the script blended them. If you need any of these, check out the tag `pre-reorg` or an earlier commit.

### Outputs Produced:

- `checkpoints/<step>/pretrained_model_ema/`: Exponential Moving Average weights (recommended for inference), every `--save_freq` steps; `checkpoints/last` points to the newest.
- `checkpoints/<step>/pretrained_model/`: Live weights plus `train_config.json`.
- `checkpoints/<step>/training_state/`: Optimizer, scheduler and EMA state for `--resume=true`.
- Each model folder holds the policy `config.json` (ordered `camera_keys`, chunk size, image size, backbones) and the pre/post-processors with the dataset's normalization statistics.

---

## 4. Rollout & Evaluation

Every LeRobot command takes the checkpoint folder as `--policy.path`. The policy reads its cameras and their order, image size and normalization statistics from the folder.

### In MuJoCo Simulation:

```bash
MUJOCO_GL=egl uv run lerobot-eval \
    --policy.path=outputs/train/nexarm_turbovla/checkpoints/last/pretrained_model_ema \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=20
```

In the stack-bowls scene the environment gives each episode the prompt of its sampled bowl order, so no task needs to be passed. Up to 10 episode videos are saved under `--output_dir`.

### On Physical Hardware:

```bash
uv run lerobot-rollout \
    --strategy.type=base \
    --policy.path=outputs/train/nexarm_turbovla/checkpoints/last/pretrained_model_ema \
    --robot.type=nexarm_follower \
    --robot.port=/dev/ttyUSB1 \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30},"top":{"type":"intelrealsense","serial_number_or_name":"<REALSENSE_SERIAL>","width":640,"height":480,"fps":30}}' \
    --task="Stack the bowls with red on bottom, blue in middle, and black on top."
```

Give the policy exactly the cameras it was trained on; a 2-view checkpoint needs the `top` entry removed. Keep `--task` identical to the training prompt.

### Checkpoints from the former `train_turbovla.py`

The `turbovla` extra installs the policy plugin [`policies/lerobot_policy_turbovla`](../../policies/lerobot_policy_turbovla/README.md) (`--policy.type=turbovla`). Checkpoints trained with `lerobot-train --policy.type=turbovla` load directly. Convert a checkpoint from the former `train_turbovla.py` once, then pass the folder as `--policy.path`:

```bash
uv run python -m lerobot_policy_turbovla.convert_checkpoint \
    --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt \
    --output-dir outputs/train/nexarm_turbovla/lerobot_policy
```

Add `--vision-model` / `--text-model` if the checkpoint was trained with other backbones than DINOv3 ViT-B and BERT-base. The converter reads `config.json` and `stats_turbovla.json` next to the checkpoint and keeps the saved cameras and their order, chunk size, image size and training prompt, so a converted policy falls back to that prompt when no task is given. The plugin's pre/post-processors include gripper snapping and 8 executed actions per chunk.
