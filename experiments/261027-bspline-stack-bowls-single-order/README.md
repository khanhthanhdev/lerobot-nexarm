# 261027-bspline-stack-bowls-single-order

## Hypothesis

On single-order stack-bowls (red bottom, blue middle, black top) the instruction carries no information, so the task needs only vision and state. B-spline Policy (diffusion over adaptive-knot B-spline chunks) should stack reliably here, and gives a non-language baseline for DEM on the same task ([261026-dem-stack-bowls-single-order](../261026-dem-stack-bowls-single-order/README.md)).

## Data

| Field             | Value                                                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dataset `repo_id` | `local/nexarm_stack_bowls_red_blue_black_1k` (1000 episodes, 427,929 frames, one order; front, wrist and top cameras)                                              |
| Revision          | local; merge of the two 500-episode sets, generation commands and seeds in [261026-dem-stack-bowls-single-order](../261026-dem-stack-bowls-single-order/README.md) |
| Episodes used     | all                                                                                                                                                                |

## Command

```bash
setsid nohup env CUDA_VISIBLE_DEVICES=0 uv run lerobot-train \
    --config_path experiments/261027-bspline-stack-bowls-single-order/train.yaml \
    > experiments/261027-bspline-stack-bowls-single-order/train.log 2>&1 < /dev/null &

# Held-out layouts (eval seeds start at 1000000), the trained order.
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run lerobot-nexarm-eval \
    --policy.path=thanhkt/bspline-nexarm-stack-bowls-single-order \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --env.stack_order=red,blue,black \
    --eval.n_episodes=20 --eval.batch_size=1 --seed=1000000 \
    --output_dir=experiments/261027-bspline-stack-bowls-single-order/outputs/eval_bspline
```

- 10.5 GB on a 24 GB RTX A5000 at batch 64 with three cameras at 120x160; about 0.5 s/step, so 100k steps (about 15 epochs) take about 14 hours.
- `bspline_fit_workers: 4` and `num_workers: 4` keep RAM use low on the shared server.

## Checkpoint

[`thanhkt/bspline-nexarm-stack-bowls-single-order`](https://huggingface.co/thanhkt/bspline-nexarm-stack-bowls-single-order) (private), step 100k; 13.4 h on GPU 0, final loss about 0.009. W&B run `9ubho2mx`.

## Results

| Policy  | Setting                   | Episodes | Success rate | Notes |
| ------- | ------------------------- | -------- | ------------ | ----- |
| bspline | sim, order red,blue,black | 20       |              |       |

## Notes

- The DEM single-order run trained on the first 500 of these episodes; compare per-data-size carefully.
