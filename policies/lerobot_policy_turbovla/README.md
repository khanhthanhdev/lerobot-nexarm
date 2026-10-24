# lerobot_policy_turbovla

[TurboVLA](https://github.com/H-EmbodVis/TurboVLA) as a LeRobot policy plugin. It registers the policy type `turbovla`; the model code comes from the `turbovla` package (<https://github.com/khanhthanhdev/TurboVLA>, pinned by commit in the root `pyproject.toml`).

Status: training and inference. `lerobot-train --policy.type=turbovla` trains with the loss, parameter groups, frozen modules, AdamW settings and warmup-cosine schedule of `examples/nexarm/train_turbovla.py` (CPU parity checked; the GPU parity run is pending, see `experiments/261024-turbovla-nexarm-parity/`). `lerobot-eval`, `lerobot-rollout` and the desktop app run its checkpoints, or a converted `train_turbovla.py` checkpoint.

## Install

The package is a workspace member pulled by the `turbovla` extra; plain `uv sync` does not install it.

```bash
uv sync --locked --extra turbovla   # plus the extras of your profile, see run.md
```

## Convert a `train_turbovla.py` checkpoint

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

| `train_turbovla.py`                                          | `lerobot-train`                                                                                           |
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

The scheduler (`turbovla_cosine_with_warmup`) is stepped once per batch by lerobot-train and once per optimizer update by the script, so keep gradient accumulation at 1 to compare runs. Not supported through lerobot-train: sim/real co-training with `--real-ratio` and `--pretrained-checkpoint`.

## Processing (same as `rollout_turbovla.py`)

| Step   | What it does                                                                                                                                                                       |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| State  | min-max to [-1, 1] with the checkpoint's `stats_turbovla.json`                                                                                                                     |
| Images | uint8 frames in the trained camera order through the backbone's image processor, resized to `image_size`                                                                           |
| Task   | passed as text; the model tokenizes it (empty task falls back to the training prompt)                                                                                              |
| Action | min-max back to raw units, then the gripper (index 5) is snapped to the dataset's open or closed extreme around `gripper_threshold` (`--policy.binarize_gripper=false` to disable) |

`select_action` predicts a `chunk_size` chunk and executes the first `n_action_steps` (default 8, as `--open-loop-steps`).

## Tests

```bash
uv run pytest policies/lerobot_policy_turbovla -q
```

`test_parity.py` builds a small randomly initialised TurboVLA (BERT-tiny, DINOv2-small with registers), saves it the way `train_turbovla.py` does, converts it, and checks that the plugin and `rollout_turbovla.py` produce the same actions on 20 recorded NexArm frames. It needs the `thanhkt/nexarm_stack_bowls` dataset (episode 143) and the two backbones in the local Hugging Face cache, or network access; otherwise it is skipped. Set `TURBOVLA_PARITY_CHECKPOINT` (plus `TURBOVLA_PARITY_TEXT_MODEL` / `TURBOVLA_PARITY_VISION_MODEL` if it was trained with other backbones) to also check a trained checkpoint.

`test_training_parity.py` uses the same backbones and dataset: on identical weights and frames (including padded action chunks), the preprocessed batch, the loss, two optimizer updates through lerobot-train's `update_policy`, the parameter groups and the learning-rate schedule must equal those of `train_turbovla.py`.
