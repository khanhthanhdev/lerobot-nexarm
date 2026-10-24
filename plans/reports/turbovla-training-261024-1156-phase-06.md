# TurboVLA training through lerobot-train: execution report (CPU part)

Date: 2026-10-24. Branch `chore/upstream-sync` (not pushed). Host: Windows Server 2022, CPU only, no OpenGL. Scope: plan steps 1-2, CPU-scale parity, experiment folders and docs; the GPU steps (2k-step parity, full run + sim eval, LIBERO, 2-GPU run) and retiring the scripts stay open.

## Commits

| Commit      | Summary                                                                                     |
| ----------- | ------------------------------------------------------------------------------------------- |
| `9b526f26d` | `refactor(examples): expose the TurboVLA loss, parameter groups and schedule as functions`  |
| `bd7768423` | `feat(policies): train TurboVLA with lerobot-train`                                         |
| `82830473c` | `test(policies): check TurboVLA training against train_turbovla.py`                         |
| `45cfcc854` | `docs(experiments): add TurboVLA parity and LIBERO experiments with lerobot-train commands` |

This report, the phase-06 execution notes and the plan row are committed separately as `docs(plans)`. Nothing under `src/lerobot/` changed.

## What changed

- `modeling_turbovla.py`: `forward(batch)` returns `(loss, {"action_loss": ...})`, the loss of `train_turbovla.py`: per-element L1 (`loss_type` also `mse`, `smooth_l1`), gripper weight 3.0 on `gripper_index` 5, `action_is_pad` steps zeroed, sum divided by valid steps x action dim. `get_optim_params` returns the script's groups in its order: action head (`optimizer_lr`), everything except the action head and vision encoder (`optimizer_lr`), the vision encoder (`optimizer_vision_lr`, left out when frozen). `freeze_text` (default true; the model freezes BERT, its projection still trains), `freeze_vision`, `freeze_interaction` set `requires_grad=False`; `train()` is overridden so these modules stay in eval mode, since `update_policy` calls `policy.train()` every step (the script calls it once).
- `configuration_turbovla.py`: training fields with the script's defaults; `get_optimizer_preset()` = `AdamWConfig(lr=5e-5, weight_decay=1e-10, grad_clip_norm=1.0)` (betas/eps equal torch defaults, as in the script); `get_scheduler_preset()` = new plugin scheduler `turbovla_cosine_with_warmup` (`TurboVLASchedulerConfig`, registered on `LRSchedulerConfig`). No LeRobot scheduler matches: `cosine_decay_with_warmup` starts warmup at `1/(warmup+1)` and decays over `num_decay_steps` from step 0; the script uses `(step+1)/warmup` and a 5% cosine floor over `steps - warmup`. `action_delta_indices = range(chunk_size)` was already in place; the test checks `resolve_delta_timestamps` equals the script's `delta_timestamps`.
- Image conversion: the plugin's rounding is used in training and inference. LeRobot decodes video as `uint8 / 255` in float32; for all 256 levels truncation (script) and rounding (plugin) give back the same byte (checked for the float32 and the float64-then-float32 paths), and the test asserts the pixel values are bitwise equal. Rounding stays because it is the nearest byte when frames are not uint8-derived (image augmentation) and keeps one processor for training and inference.
- `train_turbovla.py`: `build_param_groups`, `lr_multiplier`, `action_loss` moved out of `main()` unchanged; `--seed` (default off) calls LeRobot's `set_seed`. Launchers, `train_turbovla.py` and `rollout_turbovla.py` stay.
- Tests: `tests/conftest.py` (shared dataset-root and tiny-backbone fixtures; `test_parity.py` now uses them), `tests/test_training_parity.py`.
- Docs: `SERVER_TRAINING_GUIDE.md` and `docs/source/nexarm/turbovla_guide.md` list the `lerobot-train --policy.type=turbovla` and `accelerate launch --num_processes=N .venv/bin/lerobot-train` commands before the script launchers, marked as their replacement once the GPU parity passes, with the flag mapping. Plugin README status updated.
- Experiments: `experiments/261024-turbovla-nexarm-parity/` (`README.md`, `train_parity.yaml`, `train_full.yaml`) and `experiments/261024-turbovla-libero/` (`README.md`, `train.yaml`). `train_parity.yaml` was loaded by `lerobot-train` with CPU overrides for 2 steps (EMA on): it trains and writes `pretrained_model` and `pretrained_model_ema`. LIBERO ids verified with `HfApi().dataset_info`: `lerobot/libero` at `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` (1693 episodes, state 8-D, action 7-D, cameras `image`, `image2`), `HuggingFaceVLA/libero` at `86958911c0f959db2bbbdb107eb3e17c5f9c798e`. `train_full.yaml` and the LIBERO `train.yaml` were only checked by `check-yaml`, not loaded by `lerobot-train`; every field they add over `train_parity.yaml` (`gripper_index`, `binarize_gripper`, `dataset.root`) is an existing config field. The NexArm Hub head is `64ce8aeb1d3968293c0290c4110bcea7424325b1`, the same as the local snapshot.

## Parity (CPU, fp32, seeded)

Same seeded TurboVLA (BERT-tiny with the bert-base tokenizer, DINOv2-small-with-registers, 224 px) built by `make_policy`, deep-copied for the script side; 6 frames of `thanhkt/nexarm_stack_bowls` episode 143, 2 of them with padded chunks; torch RNG seeded before each forward (dropout and drop-path active). Script side: its `compute_or_load_normalization_stats`, `TurboVLANexArmCollator`, `action_loss`, `build_param_groups`, `lr_multiplier`, AdamW, `clip_grad_norm_` 1.0, `LambdaLR`. LeRobot side: `default_collate`, `make_pre_post_processors(dataset_stats=meta.stats)`, `policy.forward`, the presets, `lerobot.scripts.lerobot_train.update_policy` with a CPU `Accelerator`.

| Check                                                                   | Result                                                                                                        |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Script stats equal `meta.stats` min/max (action, state)                 | exact                                                                                                         |
| Pixel values / state / action / `action_is_pad` / task after processing | equal (pixels bitwise; state, action atol 1e-6)                                                               |
| Loss, seed 1                                                            | 0.63976169 vs 0.63976169, diff **0.0**                                                                        |
| Loss, seed 2                                                            | 0.63685453 vs 0.63685453, diff **0.0**                                                                        |
| Same batch without the padding mask                                     | loss differs by more than 1e-4                                                                                |
| Two optimizer updates (lr 1e-3, vision 5e-4, warmup 2)                  | max weight diff **0.0** (tol 1e-6); max weight change 1.5e-3; frozen weights unchanged                        |
| Param groups (names, lr), frozen sets, module train/eval states         | equal; with `freeze_vision` + `freeze_interaction`: 2 groups, both modules frozen and in eval after `train()` |
| Schedule, 12 steps                                                      | LRs equal (rel 1e-12), ends at 5% of peak                                                                     |
| Mutations: no padding mask; frozen modules back in train mode           | 2 tests fail each (reverted)                                                                                  |

## 30-step smoke runs

Episodes 143-152 (3072 frames), CPU, batch 8, lr 3e-4 for every group, warmup 5, 30 steps, seed 1000, `num_workers` 0, `HF_HUB_OFFLINE=1`. LeRobot: `lerobot-train --policy.type=turbovla --dataset.root=<snapshot> --dataset.episodes=[143..152] --policy.text_model_name_or_path=<bert-tiny-tok> --policy.vision_model_name_or_path=facebook/dinov2-with-registers-small --policy.vision_compute_precision=fp32 --policy.camera_keys=[front,wrist] --policy.optimizer_lr=3e-4 --policy.optimizer_vision_lr=3e-4 --policy.scheduler_warmup_steps=5 --batch_size=8 --steps=30 --save_freq=30`. Script: `train_turbovla.py --no-sim --real-dataset-root <snapshot> --real-episodes 143..152 --cameras front,wrist --bert-path <bert-tiny-tok> --dinov3-path facebook/dinov2-with-registers-small --device cpu --batch-size 8 --max-steps 30 --warmup-steps 5 --lr 3e-4 --dinov3-lr 3e-4 --seed 1000`. Initial weights and data order differ between the pipelines (different RNG consumption before model build; seeded generator vs global-RNG shuffle), so only trends are comparable.

| Step | lerobot-train loss | lr      | train_turbovla.py loss | lr       |
| ---- | ------------------ | ------- | ---------------------- | -------- |
| 1    | 0.687              | 1.2e-04 | 0.7072                 | 1.20e-04 |
| 2    | 0.559              | 1.8e-04 | 0.5566                 | 1.80e-04 |
| 3    | 0.475              | 2.4e-04 | 0.4543                 | 2.40e-04 |
| 4    | 0.545              | 3.0e-04 | 0.4936                 | 3.00e-04 |
| 5    | 0.576              | 3.0e-04 | 0.5065                 | 3.00e-04 |
| 6    | 0.568              | 3.0e-04 | 0.4973                 | 2.99e-04 |
| 7    | 0.530              | 3.0e-04 | 0.4773                 | 2.96e-04 |
| 8    | 0.531              | 2.9e-04 | 0.4554                 | 2.90e-04 |
| 9    | 0.496              | 2.8e-04 | 0.4920                 | 2.82e-04 |
| 10   | 0.635              | 2.7e-04 | 0.4529                 | 2.73e-04 |
| 11   | 0.422              | 2.6e-04 | 0.5277                 | 2.61e-04 |
| 12   | 0.425              | 2.5e-04 | 0.5261                 | 2.48e-04 |
| 13   | 0.432              | 2.3e-04 | 0.5136                 | 2.34e-04 |
| 14   | 0.412              | 2.2e-04 | 0.4898                 | 2.18e-04 |
| 15   | 0.550              | 2.0e-04 | 0.5115                 | 2.02e-04 |
| 16   | 0.421              | 1.8e-04 | 0.5311                 | 1.84e-04 |
| 17   | 0.517              | 1.7e-04 | 0.4577                 | 1.66e-04 |
| 18   | 0.556              | 1.5e-04 | 0.4992                 | 1.49e-04 |
| 19   | 0.544              | 1.3e-04 | 0.4538                 | 1.31e-04 |
| 20   | 0.469              | 1.1e-04 | 0.4410                 | 1.13e-04 |
| 21   | 0.483              | 9.7e-05 | 0.5335                 | 9.68e-05 |
| 22   | 0.452              | 8.1e-05 | 0.4516                 | 8.11e-05 |
| 23   | 0.425              | 6.7e-05 | 0.5324                 | 6.67e-05 |
| 24   | 0.538              | 5.4e-05 | 0.4954                 | 5.36e-05 |
| 25   | 0.508              | 4.2e-05 | 0.4691                 | 4.22e-05 |
| 26   | 0.506              | 3.3e-05 | 0.4002                 | 3.26e-05 |
| 27   | 0.382              | 2.5e-05 | 0.5175                 | 2.50e-05 |
| 28   | 0.374              | 1.9e-05 | 0.4629                 | 1.95e-05 |
| 29   | 0.411              | 1.6e-05 | 0.4426                 | 1.61e-05 |
| 30   | 0.479              | 1.5e-05 | 0.3506                 | 1.50e-05 |

Mean of steps 1-5 / 26-30: lerobot-train 0.568 / 0.430, script 0.544 / 0.435. The learning-rate sequences are identical (LeRobot logs one fewer digit). LeRobot checkpoint `checkpoints/000030/pretrained_model` (plus `training_state`) saved; reloaded with `PreTrainedConfig.from_pretrained` + `make_policy(cfg, ds_meta)` + `make_pre_post_processors(pretrained_path=...)`, `select_action` on a frame of episode 150 returned 8 actions of shape (6,) in raw units, gripper snapped to 2833. The script wrote `final_model.pt`, `final_ema_*` and `stats_turbovla.json`.

## Verification

| Check                                                                   | Result                                                                                                                                                               |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv run pytest policies -q`                                             | 28 passed, 2 skipped (the `trained` inference-parity param without its env var)                                                                                      |
| `tests/scripts/test_{train,rollout}_turbovla.py`                        | 38 passed                                                                                                                                                            |
| `uv run pytest tests -k nexarm -q --ignore=tests/utils/test_process.py` | 338 passed, 10 failed (the known OpenGL/env failures: nexarm env/sim render, gym integration, Isaac assets CRLF, sim dataset generator), 5 skipped; same as baseline |
| `pre-commit run --files <changed files>`                                | clean (Prettier reformatted the new READMEs' tables once; mypy has no files in scope)                                                                                |

## Open items

- GPU/Linux: the 2k-step parity in both pipelines, the full 50k runs with the 50-episode `NexArmStackBowls-v0` eval, LIBERO training and the 4 x 50-episode eval, the 100-step 2-GPU run. Commands are in the two experiment READMEs.
- After parity: delete `train_turbovla.py` / `rollout_turbovla.py`, switch `scripts/nexarm/train_turbovla_*.sh` and `.slurm` to `accelerate launch lerobot-train --policy.type=turbovla`, decide on `setup_turbovla.py` / `diagnose_turbovla.py`.
- Documented exceptions (only in the script): sim/real co-training with `--real-ratio`, `--pretrained-checkpoint` partial loading of TurboVLA release weights. With gradient accumulation lerobot-train's scheduler counts batches, the script's counts updates. diffusers' EMA copies the weights on its first update where the script blends. A batch with no task gets "Complete the robotic manipulation task." in the script and `default_task` (empty) in the plugin.
- `AGENTS.md` (`CLAUDE.md` links to it) still says the plugin is "inference only"; left for the user to update.

## Notes

- `plans/261024-0016-research-repo-reorganization/phase-08-nexarm-plugins-on-upstream-lerobot.md` gained a "Scope decision" section during this session from someone else; it is not part of these commits.
- Scratch outputs (smoke runs, logs) are in the session scratchpad, not in the repo.

Status: DONE_WITH_CONCERNS
Summary: TurboVLA trains through `lerobot-train --policy.type=turbovla` with the script's loss, parameter groups, frozen modules, AdamW and schedule; CPU parity against `train_turbovla.py`'s own code is exact (loss and weight differences 0.0) and both 30-step smokes decrease. The GPU parity, sim/LIBERO evals and retiring the scripts remain open.
Concerns/Blockers: co-training ratio and `--pretrained-checkpoint` stay script-only; the GPU-scale parity (bf16, DINOv3/BERT-base, EMA) is unverified.
