# 261026-dem-stack-bowls-single-order

## Hypothesis

DEM failed on six-order stack-bowls (0/10 with the paper's recipe, results in the [plugin README](../../policies/lerobot_policy_dem/README.md)) partly because only the prompt names the order, and its head ignored the prompt. With a single order (red bottom, blue middle, black top) the prompt carries no information, so the task needs only vision and state. If DEM, with the v2 head (state through AdaLN) and the stabilised recipe, stacks reliably here, its manipulation works, and instruction following is the remaining problem for the six-order task.

## Data

| Field             | Value                                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| Dataset `repo_id` | `local/nexarm_stack_bowls_red_blue_black` (500 episodes, order `red,blue,black`, train seeds from 0) |
| Revision          | local; provenance in `outputs/datasets/nexarm_stack_bowls_red_blue_black/generation_report.json`     |
| Episodes used     | all                                                                                                  |
| Merged 1k set     | `local/nexarm_stack_bowls_red_blue_black_1k` (1000 episodes, 427,929 frames; the above + `_part2`)   |

### Generate the dataset

The same generator and settings as `local/nexarm_stack_bowls` (front, wrist and top cameras at 640x480, 30 fps, physical grasps, domain randomization, 4 cm position jitter), with one fixed order:

```bash
setsid nohup env MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --repo-id local/nexarm_stack_bowls_red_blue_black \
    --root outputs/datasets/nexarm_stack_bowls_red_blue_black \
    --episodes 500 --order red,blue,black --position-jitter-m 0.04 \
    --gpus 3,5 --workers 2 \
    > outputs/logs/generate_stack_bowls_red_blue_black.log 2>&1 < /dev/null &
grep -c 'Accepted episode' outputs/logs/generate_stack_bowls_red_blue_black.log   # progress
```

- One worker takes 75-90 s an episode, so 2 workers write 500 episodes in about 5.5 hours; 2 keeps RAM use low on the shared server. The dataset used here came from 8 workers on GPUs 3, 5 and 6 (about 3.5 episodes a minute, 2.5 hours). Each worker owns a block of 100,000 seeds and training seeds must stay below 1,000,000, so a training set allows at most 10 workers; with more the generator stops with `train seeds must stay below 1000000`. 10 workers ran no faster than 8: each encodes three AV1 streams, and the encoders timed out under the load.
- Run it detached (`setsid nohup`, or in `tmux`). A first run was killed with the session after 258 episodes, and a killed run leaves shard folders (`<root>_shard_<i>`) whose episode metadata was never written, so they cannot be loaded or resumed: delete them and start again.
- The generator refuses an existing `--root`. If it stops, rerun with `--resume` (and the same flags) to continue after the last logged seed.
- A second 500 (`_part2`, 2 workers on GPU 0) starts at seed 800,000, after the first set's eight seed blocks, so no layout repeats. Merge the two into one 1000-episode dataset with LeRobot's merge, which renumbers the episodes and keeps every source intact; each part keeps its own `generation_report.json` (seeds and settings):

  ```bash
  setsid nohup env MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
      --repo-id local/nexarm_stack_bowls_red_blue_black_part2 \
      --root outputs/datasets/nexarm_stack_bowls_red_blue_black_part2 \
      --episodes 500 --order red,blue,black --position-jitter-m 0.04 \
      --seed-start 800000 --gpus 0 --workers 2 \
      > outputs/logs/generate_stack_bowls_red_blue_black_part2.log 2>&1 < /dev/null &

  uv run lerobot-edit-dataset --operation.type=merge \
      --operation.repo_ids='["local/nexarm_stack_bowls_red_blue_black", "local/nexarm_stack_bowls_red_blue_black_part2"]' \
      --operation.roots='["outputs/datasets/nexarm_stack_bowls_red_blue_black", "outputs/datasets/nexarm_stack_bowls_red_blue_black_part2"]' \
      --new_repo_id=local/nexarm_stack_bowls_red_blue_black_1k \
      --new_root=outputs/datasets/nexarm_stack_bowls_red_blue_black_1k
  ```

- Check the result:

  ```bash
  uv run python -c "
  import json, pandas as pd
  root = 'outputs/datasets/nexarm_stack_bowls_red_blue_black'
  info = json.load(open(f'{root}/meta/info.json'))
  print(info['total_episodes'], 'episodes,', info['total_frames'], 'frames')
  print(list(pd.read_parquet(f'{root}/meta/tasks.parquet').index))  # one prompt
  "
  ```

## Command

```bash
CUDA_VISIBLE_DEVICES=3 uv run lerobot-train --config_path experiments/261026-dem-stack-bowls-single-order/train.yaml

# Held-out layouts (eval seeds start at 1000000), the trained order, EMA weights.
CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl uv run lerobot-nexarm-eval \
    --policy.path=thanhkt/dem-nexarm-stack-bowls-single-order \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --env.stack_order=red,blue,black \
    --eval.n_episodes=20 --eval.batch_size=1 --seed=1000000 \
    --output_dir=experiments/261026-dem-stack-bowls-single-order/outputs/eval_dem
```

Without `--env.stack_order` the env cycles the six orders by seed, and a single-order policy can succeed on at most one seed in six. Keep `--eval.batch_size=1` (the async vector env cannot pickle the MuJoCo EGL renderer).

## Checkpoint

[`thanhkt/dem-nexarm-stack-bowls-single-order`](https://huggingface.co/thanhkt/dem-nexarm-stack-bowls-single-order) (private): EMA weights at the root, raw weights in `raw/` (`hf download thanhkt/dem-nexarm-stack-bowls-single-order --include "raw/*" --local-dir <dir>`, then `--policy.path=<dir>/raw`). Step 50k on the first 500 episodes, 8.5 h on GPU 7; final loss -0.747. W&B run `9bchqoat`.

## Results

Sim, 20 held-out layouts (seeds 1000000-1000019), order red,blue,black, 900 steps max:

| Policy | Weights                        | Episodes | Success rate (95% CI) | Picked blue (middle) | Picked black (top) | Nested black on blue |
| ------ | ------------------------------ | -------- | --------------------- | -------------------- | ------------------ | -------------------- |
| DEM    | EMA 50k                        | 20       | 25% (11-47%)          | 25%                  | 90%                | 25%                  |
| DEM    | raw 50k                        | 20       | 20% (8-42%)           | 25%                  | 90%                | 20%                  |
| DEM    | EMA 50k, rerun with all videos | 20       | 20% (8-42%)           | 25%                  | 95%                | 20%                  |
| DEM    | raw 50k, rerun with all videos | 20       | 15% (5-36%)           | 25%                  | 85%                | 15%                  |

The reruns use the same seeds; GPU nondeterminism changes which episodes succeed (EMA 6, 11, 12, 19; raw 6, 17, 18), not the rate. Every episode's video is in `outputs/eval_dem_50k_{ema,raw}_videos/videos/`.

Open-loop probes (50k EMA, 32 training frames, raw servo units): MAE 20.1 against a hold-state baseline of 86.9 (gripper 64, arm joints 3-20); chunk change 244 when the state is swapped, 6.2 between two noise draws. EMA and raw weights give the same numbers (the learning rate had decayed to zero).

- **Every success is a clean stack; every failure is the same mistake.** In all 10 recorded EMA episodes the arm goes to the **black** bowl first, wherever it is (left, centre or right), lifts it and nests it on red, then stops. Blue is picked only in the successful episodes. Grasping, carrying and nesting work.
- **Not a position shortcut.** Reconstructing each seed's layout, the first pick follows the black bowl across all three slots. The training set has blue in each slot about equally (right 374, centre 319, left 307 of 1000).
- Success by blue's slot (EMA / raw): left 2/5 / 2/5, centre 3/9 / 2/9, right 0/6 / 0/6.
- Likely cause: with `n_obs_steps=1` the head has no memory of the phase. At the start of phase 2 of a demonstration (blue already on red) the target is the black bowl, and the policy applies that phase-2 target from the first frame, while still placing on red as in phase 1. Next to try: more data (the 1k set), a longer observation history, or a phase cue in the state.

## Notes

- Compare with the v1 run (six orders, paper recipe): 0/10, stacks in the wrong order.
- If this works, the next step is the six-order task with a mechanism that makes the head use the prompt.
