# 261024-turbovla-libero

**Status: pending (needs a Linux GPU machine with EGL and the `libero` extra).** Nothing below has been run yet; the results table is empty on purpose.

## Hypothesis

TurboVLA trained with `lerobot-train --policy.type=turbovla` on LeRobot-format LIBERO data gives a benchmark number for the plugin that can be compared with the other policies in LeRobot's LIBERO tables, scored by `lerobot-eval --env.type=libero` instead of TurboVLA's own RLDS evaluation. This checks that the plugin is not tied to NexArm (7-D actions, 8-D state, other cameras).

## Data

| Field             | Value                                                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Dataset `repo_id` | `lerobot/libero` (the dataset the [LeRobot LIBERO docs](https://huggingface.co/docs/lerobot/libero) recommend; MP4 video, 2 cameras) |
| Revision          | `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` (Hub head on 2026-10-24)                                                                  |
| Episodes used     | all 1693 (273,465 frames at 10 fps; `observation.state` 8-D, `action` 7-D)                                                           |

`HuggingFaceVLA/libero` (same demonstrations, PNG frames in parquet) also exists, at `86958911c0f959db2bbbdb107eb3e17c5f9c798e`; use it only if no video decoder works.

## Command

```bash
uv sync --locked --extra turbovla --extra libero
uv run lerobot-train --config_path experiments/261024-turbovla-libero/train.yaml
```

Evaluate on the four suites. `--eval.n_episodes` counts episodes per task, and each suite has 10 tasks, so 5 gives 50 episodes per suite:

```bash
EXP=experiments/261024-turbovla-libero/outputs
MUJOCO_GL=egl uv run lerobot-eval \
    --policy.path=$EXP/train/checkpoints/last/pretrained_model_ema \
    --env.type=libero --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --eval.batch_size=1 --eval.n_episodes=5 --env.max_parallel_tasks=1 \
    --output_dir=$EXP/eval
```

## Checkpoint

Hub id, step and W&B run link: pending.

## Results

| Policy         | Suite            | Episodes | Success rate | 95% CI | Hardware | Notes |
| -------------- | ---------------- | -------- | ------------ | ------ | -------- | ----- |
| TurboVLA (EMA) | `libero_spatial` | 50       |              |        |          |       |
| TurboVLA (EMA) | `libero_object`  | 50       |              |        |          |       |
| TurboVLA (EMA) | `libero_goal`    | 50       |              |        |          |       |
| TurboVLA (EMA) | `libero_10`      | 50       |              |        |          |       |

## Notes

- `gripper_index: 6` puts the gripper loss weight on LIBERO's gripper action; `binarize_gripper: false` because the snapping thresholds are NexArm raw positions.
- `--eval.n_episodes` counting per task comes from reading `lerobot_eval.py` (it evaluates each task separately); confirm the per-task episode count in the eval output before trusting the 50.
- TurboVLA's own mixed-suite RLDS training stays in the TurboVLA repo; this run does not reproduce it.
