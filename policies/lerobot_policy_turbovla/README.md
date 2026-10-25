# lerobot_policy_turbovla

[TurboVLA](https://github.com/H-EmbodVis/TurboVLA) as a LeRobot policy plugin. It registers the policy type `turbovla`; the model code comes from the `turbovla` package (<https://github.com/khanhthanhdev/TurboVLA>, pinned by commit in the root `pyproject.toml`).

Status: training and inference. `lerobot-train --policy.type=turbovla` trains with the loss, parameter groups, frozen modules, AdamW settings and warmup-cosine schedule of the former `examples/nexarm/train_turbovla.py`. GPU parity is done: over 2k steps the loss curves of both pipelines stay within 4% (see [`experiments/261024-turbovla-nexarm-parity/README.md`](../../experiments/261024-turbovla-nexarm-parity/README.md)). `lerobot-eval`, `lerobot-rollout` and the desktop app run its checkpoints, or a converted checkpoint from the former script.

## Install

The package is a workspace member pulled by the `turbovla` extra; plain `uv sync` does not install it.

```bash
uv sync --locked --extra turbovla   # plus the extras of your profile, see docs/getting-started/install.md
```

## Convert a checkpoint from the former `train_turbovla.py`

```bash
uv run python -m lerobot_policy_turbovla.convert_checkpoint \
    --checkpoint outputs/train/nexarm_turbovla/final_ema_pytorch_model.pt \
    --output-dir outputs/train/nexarm_turbovla/lerobot_policy
uv run lerobot-eval --policy.path=outputs/train/nexarm_turbovla/lerobot_policy --env.type=nexarm ...
```

The converter reads `config.json` and `stats_turbovla.json` next to the checkpoint. Pass `--text-model` / `--vision-model` when the checkpoint was trained with other backbones than `google-bert/bert-base-uncased` and `facebook/dinov3-vitb16-pretrain-lvd1689m`. By default the backbones load from the local Hugging Face cache only (`--allow-hf-download` to change that); the choice is saved in the policy config. Loading a converted policy still loads both backbones' configs, tokenizer and image processor before the checkpoint weights replace theirs.

## Training

```bash
uv run lerobot-train --policy.type=turbovla     --dataset.repo_id=local/nexarm_stack_bowls --dataset.root=outputs/datasets/nexarm_stack_bowls     --policy.device=cuda --policy.push_to_hub=false --ema.enable=true --ema.decay=0.999     --batch_size=16 --steps=50000 --output_dir=outputs/train/nexarm_turbovla_lerobot
```

| former `train_turbovla.py`                                   | `lerobot-train`                                                                                           |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `--lr`, `--dinov3-lr`                                        | `--policy.optimizer_lr`, `--policy.optimizer_vision_lr`                                                   |
| `--weight-decay`, `--max-grad-norm`                          | `--policy.optimizer_weight_decay`, `--policy.optimizer_grad_clip_norm`                                    |
| `--warmup-steps`, `--max-steps`                              | `--policy.scheduler_warmup_steps`, `--steps`                                                              |
| `--freeze-text` / `--freeze-vision` / `--freeze-interaction` | `--policy.freeze_text` / `freeze_vision` / `freeze_interaction`                                           |
| `--loss-type`, `--gripper-loss-weight`                       | `--policy.loss_type`, `--policy.gripper_loss_weight` (on `--policy.gripper_index`)                        |
| `--horizon`, `--cameras`                                     | `--policy.chunk_size`, `--policy.camera_keys`                                                             |
| `--precision bf16`                                           | `--accelerator.mixed_precision=bf16` plus `--policy.{vision,interaction}_compute_precision=bf16_autocast` |
| `--ema-decay`                                                | `--ema.enable=true --ema.decay=...`                                                                       |
| `--seed`                                                     | `--seed`                                                                                                  |

The scheduler (`turbovla_cosine_with_warmup`) is stepped once per batch by lerobot-train, where the former script stepped it once per optimizer update, so keep gradient accumulation at 1. Not supported through lerobot-train: sim/real co-training with `--real-ratio` and partial weight loading with `--pretrained-checkpoint`. diffusers' EMA copies the weights on its first update, where the script blended them.

## Processing

| Step   | What it does                                                                                                                                                                       |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| State  | min-max to [-1, 1] with the dataset's statistics (`stats_turbovla.json` for a converted checkpoint)                                                                                |
| Images | uint8 frames in the trained camera order through the backbone's image processor, resized to `image_size`                                                                           |
| Task   | passed as text; the model tokenizes it (an empty task falls back to `--policy.default_task`, which a converted checkpoint sets to its training prompt)                             |
| Action | min-max back to raw units, then the gripper (index 5) is snapped to the dataset's open or closed extreme around `gripper_threshold` (`--policy.binarize_gripper=false` to disable) |

`select_action` predicts a `chunk_size` chunk and executes the first `n_action_steps` (default 8).

## Tests

```bash
uv run pytest policies/lerobot_policy_turbovla -q
```

`test_convert_checkpoint.py` saves a small randomly initialised TurboVLA (BERT-tiny, DINOv2-small with registers) in the former script's checkpoint format, converts it, and checks that the policy loads with identical weights and the saved training contract. It needs the two backbones in the local Hugging Face cache, or network access, but no dataset.

`test_training.py` uses the same backbones plus episode 143 of the `thanhkt/nexarm_stack_bowls` dataset (from the local cache, or `TURBOVLA_PARITY_DATASET_ROOT`) and checks the plugin on its own: action chunks built from delta timestamps, the masked L1 loss with gripper weight 3 and padded steps excluded, the three parameter groups and the frozen modules, the AdamW preset and the warmup-cosine schedule values, and that two lerobot-train updates move the trained weights and leave the frozen ones unchanged. Both tests are skipped when their backbones or dataset are unavailable.
