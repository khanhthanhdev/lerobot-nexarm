# ACT on real NexArm stack-bowl data

Run from the repository root. The launcher uses [thanhkt/nexarm_stack_bowls](https://huggingface.co/datasets/thanhkt/nexarm_stack_bowls), pins its Hub revision, and checks the six joint/gripper channels and both front/wrist cameras. At preparation time it contains 113 episodes, 34,365 frames, and two task labels at 30 FPS. ACT is conditioned on images and joint state; it does not consume the task text.

```bash
uv sync --locked --extra training --extra evaluation

# Verify actual AV1 video decoding and action padding using episode 0.
uv run python examples/nexarm/train_stack_bowls_act.py --check-data

# Inspect the setup without starting training.
uv run python examples/nexarm/train_stack_bowls_act.py --dry-run

# Initial ACT training: ImageNet-pretrained ResNet18, new ACT transformer.
uv run python examples/nexarm/train_stack_bowls_act.py

# Fine-tune an existing compatible NexArm ACT checkpoint in a new run.
uv run python examples/nexarm/train_stack_bowls_act.py \
  --pretrained /path/to/checkpoints/last/pretrained_model \
  --output-dir outputs/train/nexarm_stack_bowls_act_finetune
```

`--pretrained` also accepts a Hugging Face ACT model ID. The checkpoint must have matching camera names/shapes and six-dimensional state/actions; ensure its joint order and physical units match this dataset. Checkpoint architecture, including action chunk size, is preserved. LeRobot replaces the normalization statistics with those from the stack-bowl dataset and starts a new optimizer. A complete LeRobot checkpoint includes the saved pre/post processors.

The YAML starts with batch size 2 and eight gradient accumulation steps (effective batch 16) for an 8 GB GPU. Memory use depends on checkpoint architecture; lower batch size if necessary. Fresh ACT uses 100-action chunks, approximately 3.33 seconds at 30 FPS. The trainer holds out the last 10% of episodes per task for loss evaluation, evaluates up to 256 samples every 5,000 steps, and saves resumable checkpoints every 5,000 steps. Dataset normalization statistics are provided by the dataset, rather than recomputed on the training split. Held-out loss does not measure physical stacking success.

```bash
# Short training smoke test; writes a separate checkpoint.
uv run python examples/nexarm/train_stack_bowls_act.py \
  --output-dir outputs/train/nexarm_stack_bowls_act_smoke \
  --steps=1 --batch_size=1 --gradient_accumulation_steps=1 --num_workers=0

# Adjust the duration and batch settings.
uv run python examples/nexarm/train_stack_bowls_act.py \
  --steps=50000 --batch_size=1 --gradient_accumulation_steps=16

# Resume weights AND optimizer state from an interrupted run.
uv run lerobot-train \
  --config_path=outputs/train/nexarm_stack_bowls_act/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

Use `--dataset-root /absolute/path/to/dataset` for an existing local copy, or `--revision COMMIT_SHA` to reproduce a particular Hub version. Relative output paths are resolved from your current directory. Hub data is cached under `HF_LEROBOT_HOME/prepared/thanhkt/nexarm_stack_bowls/COMMIT_SHA` to avoid reusing stale recordings. Video files contain multiple episodes, so even `--check-data` can download hundreds of megabytes. Model publishing and W&B are disabled by default.
