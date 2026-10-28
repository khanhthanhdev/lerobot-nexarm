# lerobot_policy_dem

[DEM, the Decoupled Embodiment Model](https://github.com/Apollo-Lab-Yale/decoupled-embodiment-model), as a LeRobot policy plugin. It registers the policy type `dem`; the model code is the paper's `dem` package, pinned by commit in the root `pyproject.toml`.

DEM is made of three networks that never see each other: a DINOv3 ConvNeXt-B vision encoder (stage-3 tokens, 16 × 16 × 512 per 256 px camera), a frozen NeoBERT language encoder (32 instruction tokens) and a 107M MeanFlow action head that reads both through cross-attention and maps noise to a 16-step action chunk in one forward pass.

Status: training and inference run through `lerobot-train`, `lerobot-eval` and `lerobot-rollout`. The paper's trained weights are not public, so everything trains from the public DINOv3 and NeoBERT checkpoints. Results on the stack-bowls sim:

| Run (local, not kept unless linked)                                                        | Data                      | Steps | Sim success | Finding                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------ | ------------------------- | ----- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| v1: paper recipe, three cameras, batch 16                                                  | 1000 episodes, six orders | 100k  | 0/10        | The head ignored the instruction and the state (chunk change 0.3 across the six prompts and 1.2 for a swapped state, against 25 between noise draws); `meanflow_mse` spiked up to 12,554 until the learning rate decayed. |
| v2: the defaults below, two cameras, batch 32, EMA                                         | the same                  | 5k    | not run     | Stable (largest `meanflow_mse` 1.9) and the state is used (chunk change 282), but the instruction is still ignored (0.7). Stopped.                                                                                        |
| [single order](../../experiments/261026-dem-stack-bowls-single-order/README.md): v2 recipe | 500 episodes, one order   | 50k   | 5/20 (EMA)  | Grasp, carry and nest work; every failure picks the black (top) bowl first.                                                                                                                                               |

## Install

The package is a workspace member pulled by the `dem` extra; plain `uv sync` does not install it.

```bash
uv sync --locked --extra dem   # plus the extras of your profile, see docs/getting-started/install.md
```

`facebook/dinov3-convnext-base-pretrain-lvd1689m` is gated: accept its terms on the Hub and run `hf auth login` once. `chandar-lab/NeoBERT` is public. `dem` asks for `transformers>=5.13`; the root `pyproject.toml` overrides that with lerobot's own range (`<5.6`), on which the `dem` test suite passes.

## Training

```bash
uv run lerobot-train --config_path experiments/261026-dem-stack-bowls-single-order/train.yaml
```

The model and loss defaults are the paper's; the conditioning and the learning rates are not:

| Change from the paper                               | Why                                                                                                                                                                                                                                                  | Paper behaviour                                                                                  |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `--policy.global_conditioning=true` (`head_dem.py`) | The instruction (4 learned queries pool the NeoBERT tokens) and the state also enter every head block through AdaLN, so they cannot be drowned out by hundreds of vision tokens. The new output layer starts at zero, so step 0 is the paper's head. | `--policy.global_conditioning=false`                                                             |
| lr 5e-5 head / 2.5e-5 vision, 2000 warmup steps     | At 1e-4 / 5e-5 without warmup the MeanFlow error spiked up to 12,554 until the learning rate had decayed.                                                                                                                                            | `--policy.optimizer_lr=1e-4 --policy.optimizer_vision_lr=5e-5 --policy.scheduler_warmup_steps=0` |

The single-order config also trains on the paper's two views (front + wrist, `--policy.camera_keys`), at batch 32 with EMA weights (`--ema.enable=true --ema.decay=0.999`; evaluate `pretrained_model_ema`).

The rest follows the paper:

| Paper (`dem` README)                                        | `lerobot-train`                                                                                                                    |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| weight decay 1e-4, betas (0.9, 0.95), clipping 1.0          | `--policy.optimizer_weight_decay`, `--policy.optimizer_betas`, `--policy.optimizer_grad_clip_norm`                                 |
| cosine schedule                                             | `cosine_decay_with_warmup`, `--policy.scheduler_decay_steps` (cut to `--steps` when larger)                                        |
| dispersive weight 0.25, `ratio_r_eq_t = 0.5`                | `--policy.dispersive_weight`, `--policy.ratio_r_eq_t`                                                                              |
| vision under bf16 autocast, head and loss in fp32           | `--policy.vision_bf16_autocast` (the head stays fp32 even under `--accelerator.mixed_precision`)                                   |
| vision fine-tuned with the head (final configuration)       | `--policy.freeze_vision=false`                                                                                                     |
| chunk 16, 256 px, 32 language tokens, `new100m` head, 1 NFE | `--policy.chunk_size`, `--policy.image_size`, `--policy.max_language_tokens`, `--policy.head_size`, `--policy.num_inference_steps` |

On one RTX A5000, three cameras at batch 16 use 12.7 GB and run at 2.9 steps/s; two cameras at batch 32 use 18.1 GB and run at 1.7 steps/s.

Differences from the paper code:

- **Normalization.** LeRobot z-scores state and action (`MEAN_STD`) and the model's own normalizers stay the identity. The paper's two stages (q01/q99 scaling to [-1, 1], then z-scoring) form one affine map per dimension, which equals z-scoring the raw values.
- **Padding.** As in `dem`, the loss covers every step of the chunk; past the episode end LeRobot repeats the last action.
- **Multi-GPU.** The MeanFlow target is a `torch.func.jvp` through the head; it is checked on a single GPU only.
- **Gripper.** No gripper snapping (TurboVLA snaps to open or closed): DEM's output is used as predicted.

The total loss can go below zero: the dispersive term is a log-mean-exp, which is at most 0. It also hides trouble, because the adaptive weight divides out large errors: watch `meanflow_mse`, and on a checkpoint check that the chunk changes with the instruction (the v1 run's changed by 0.3 raw units across the six stacking orders, against 25 between noise draws).

## Processing

| Step   | What it does                                                                                                                                                                                                  |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| State  | z-scored with the dataset statistics                                                                                                                                                                          |
| Images | float frames in [0, 1], in `--policy.camera_keys` order (default: the dataset's order), resized to `image_size` inside the policy with `dem.preprocess_images`; ImageNet normalization happens in the encoder |
| Task   | passed as text and encoded by NeoBERT once per distinct string (an empty task falls back to `--policy.default_task`)                                                                                          |
| Action | z-scoring undone                                                                                                                                                                                              |

`select_action` predicts a `chunk_size` chunk and executes the first `n_action_steps` (default 8).

A checkpoint holds every weight, frozen NeoBERT included (1.6 GB), so `from_pretrained` builds the towers without downloading the Hub weights and only fetches the NeoBERT tokenizer.

## Tests

```bash
uv run pytest policies/lerobot_policy_dem -q
```

The tests build DEM with randomly initialised towers and a small head on CPU (about 3 minutes). They check plugin discovery, the defaults, the processors, that the loss reaches the head, the vision encoder and the global conditioning branch (and not the frozen NeoBERT), the action queue, a save/load round trip, and that the globally conditioned head starts as the paper's head, reads the instruction and the state through AdaLN alone, ignores padded instruction tokens and supports the forward-mode JVP. They need the NeoBERT tokenizer in the local Hugging Face cache, or network access.
