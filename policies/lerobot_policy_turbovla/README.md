# lerobot_policy_turbovla

[TurboVLA](https://github.com/H-EmbodVis/TurboVLA) as a LeRobot policy plugin. It registers the policy type `turbovla`; the model code comes from the `turbovla` package (<https://github.com/khanhthanhdev/TurboVLA>, pinned by commit in the root `pyproject.toml`).

Status: inference only. `lerobot-eval`, `lerobot-rollout` and the desktop app can run a converted checkpoint; `lerobot-train --policy.type=turbovla` raises `NotImplementedError`. Train with `examples/nexarm/train_turbovla.py`.

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
