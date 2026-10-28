# 261028-plan-bind-d3-pusht

**Status: ready to run, not run yet.** The stack-bowls experiments ([D1](../261027-plan-bind-two-step/README.md), [D2](../261028-plan-bind-d2-latency/README.md)) use one simulated task with scripted demonstrations, which are close to unimodal. D3 moves Plan-Bind to Push-T, the simulation benchmark of the [B-spline Policy paper](https://arxiv.org/abs/2607.09648), where the 206 demonstrations (`lerobot/pusht`) are human and multimodal.

## Questions

1. **Does the B-spline baseline reproduce?** The paper reports a Push-T score of 72 for diffusion policy and 75 for B-spline Policy (its Table 2a). D3 trains both in this repo: LeRobot's diffusion policy and the `bsp10` port. They should land near those numbers. The match cannot be exact: the paper used its own codebase and reported the mean of the best three checkpoints.
2. **Does Plan-Bind hold on human, multimodal data?** `onestep` and `planbind` regress the trajectory instead of sampling it, so on multimodal demonstrations they may average between modes. If Plan-Bind stays near `bsp10` here, the D1 result is not an artifact of scripted data. If both regression modes drop, they need a generative head.
3. **Seed and checkpoint variance.** Two training seeds per policy (`SEEDS`). Every saved checkpoint is also evaluated, because Push-T scores are known to vary between checkpoints.
4. **Call rate and latency at 10 fps.** Push-T runs at 10 fps, so a frame of latency is 100 ms. The cells cover latency 0, 100 and 200 ms, a call every 2 frames (default) or every 8 frames (diffusion's rate), and commit `latency` or `none`. `bsp10` also runs through the same spline runtime.

## Policies

All four use LeRobot's diffusion policy recipe for Push-T: one 96x96 camera with an 84x84 random crop, two observation frames, the same UNet and ResNet-18 encoder, batch 64 and 100k steps.

| Name        | Config                 | What it is                                                                                                                           |
| ----------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `diffusion` | `train_diffusion.yaml` | LeRobot's diffusion policy, DDIM with 10 steps. 16-action chunks, 8 executed per call                                                |
| `bsp10`     | `train_bsp10.yaml`     | B-spline Policy, DDIM with 10 steps, fit tolerance 0.004 in normalized units (the paper's 1 px), at most 8 actions executed per call |
| `onestep`   | `train_onestep.yaml`   | One-step regression of the fine control points: knots every 2 frames over a 32-frame window                                          |
| `planbind`  | `train_planbind.yaml`  | Plan (coarse knots every 8 frames) and Bind (residual on its exact lift), one shared UNet; plans replaced when older than 16 frames  |

**Knot spacing.** Push-T's actions are not smooth at 10 fps. A uniform cubic spline with knots every 4 frames (D1's spacing) misses the demonstrated actions by a median 6 px and a p95 of 20 px, out of a ~500 px range. Every 2 frames, the miss is 3 px and 11 px. The paper found Push-T scores flat for fit errors up to 8 px (its Table 3), so D3 uses 2 frames.

## Eval cells

Every cell runs `lerobot-eval --env.type=pusht` with eval seed 1000, so cells share episodes. A cell is named `<policy>_L<latency frames>[_<commit>][_every<frames>]_s<training seed>[_step<checkpoint>]`.

| Group            | Cells (per training seed)                                                                                         | Episodes |
| ---------------- | ----------------------------------------------------------------------------------------------------------------- | -------- |
| Main comparison  | the four policies at latency 0, last checkpoint (`onestep`, `planbind` through the runtime with commit `latency`) | 200      |
| Checkpoint curve | the main cells at every saved checkpoint (25k, 50k, 75k, 100k steps)                                              | 100      |
| Call rate        | `onestep`, `planbind` × commit `latency`, `none` × a call every 2 or 8 frames, latency 0                          | 200      |
| Latency          | `bsp10` through the runtime at latency 0, 100, 200 ms; `onestep`, `planbind` at 100 and 200 ms; commit `latency`  | 200      |

`results.md` reports the paper's two metrics. The score is the mean over episodes of each episode's best target coverage, in percent. The paper's success counts an episode whose best coverage reaches 0.9. The table also gives the env's own success (coverage 0.95).

## Run it

**Box:** Ubuntu 22.04 or 24.04, one NVIDIA GPU, driver 570 or newer. Batch 64 is expected to fit in 8 GB, and the smoke run checks it. About 60 GB of free disk when `HF_REPO` is set. Without uploads, every checkpoint is kept (about 3 GB each, 32 in all), so allow about 110 GB. It also needs `ffmpeg` (the dataset's videos are AV1), `git`, `git-lfs`, `curl`, `libegl1` and `libgl1`; `run_all.sh` prints the `apt-get` line if any are missing.

```bash
git clone https://github.com/khanhthanhdev/lerobot-nexarm.git && cd lerobot-nexarm
git checkout feat/plan-bind
tmux new -s planbind-d3
HF_REPO=<user>/plan-bind-d3 experiments/261028-plan-bind-d3-pusht/run_all.sh   # smoke run, then the full run
```

- The dataset is public. Uploads need `uv run hf auth login` once; the first run stops and says so if `HF_REPO` is set and you are not logged in.
- `run_all.sh smoke` trains each policy for 200 steps at the full batch size, so an out-of-memory error shows up within minutes. It then runs one eval cell of each kind with 2 episodes. `run_all.sh full` skips it. Rerunning the same command resumes the run.
- Each trained policy is uploaded after its checkpoints are evaluated, then its local checkpoints except the last are deleted.
- Environment overrides: `SEEDS` (2), `STEPS` (100000), `BATCH` (64), `SAVE_FREQ` (25000), `NUM_WORKERS` (8), `N_EVAL` (200), `N_SWEEP` (100), `EVAL_BATCH` (50).

**Time.** Eight trainings of 100k steps dominate. Read `step_s` in the smoke run's training log and multiply by 100,000 for one training. Evaluation is cheap by comparison: Push-T is a 2-D simulator, and the 66 cells (33 per training seed) take a few hours in all. Lower `STEPS` or `SEEDS` to fit a shorter rental.

## Outputs

- `outputs/full/train_<policy>_s<seed>/` — checkpoints (only `last` after an upload).
- `outputs/full/eval/<cell>/eval_info.json` — one directory per cell, with up to 10 videos.
- `outputs/full/results.md` — written by `summarize.py`: the main comparison against the paper's numbers, the checkpoint curve, the call rate and commit table, the latency table and every cell.

## Results

Pending.

## Not in this experiment

- **The paper's speed-up test.** It used a 200 Hz Push-T variant with 100 newly collected demonstrations, which are not public.
- **RoboCasa.** LeRobot ships a RoboCasa365 env, but not the RoboCasa version the paper used. That would be a separate experiment.
- **ACT-style regression baselines** (the paper's "Reg." rows).
