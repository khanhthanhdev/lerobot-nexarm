# 261028-plan-bind-d2-latency

**Status: ready to run, not run yet.** Follows [D1](../261027-plan-bind-two-step/README.md), where Plan-Bind beat the one-step control in all eight latency and commit cells. D2 reuses D1's checkpoints (step 60000 from the private Hub repo `thanhkt/plan-bind-d1`), so no training is needed unless `SEEDS` is above 1.

## Questions

D1 simulated latencies of 0 to 300 ms but did not measure how long a call takes, and it left open why the one-step control improved with latency. D2 asks:

1. **Measured latency.** How long does one inference call of `bsp10`, `onestep` and `planbind` take on a GPU and on a CPU, and how does each policy do at its own measured latency? `bsp10` runs through the same spline runtime as the other two (`--policy.bspline_runtime=true`), so all three are compared under the same rules.
2. **Update rate versus latency.** In D1, calls did not overlap, so latency 0, 3, 6 and 9 frames also meant one call every 4, 4, 8 and 12 frames. D2 varies the call period (`--policy.planbind_call_every=4|8|12`) at latency 0. If the one-step control improves with a longer period, its D1 gain came from replanning less often, not from latency.
3. **Design-space gaps.** Commit `now` (D1 ran only `latency` and `none`), replan thresholds 0.5 and 1.0 (0.3 cost 24 points in D1), and `bsp10` across latencies 0 to 300 ms.
4. **Reactivity.** When the middle bowl is lifted, the bottom bowl slides 3 cm (`--env.bowl_shift_m=0.03`) in a seeded random direction that keeps it clear of the top bowl. A policy has to notice and place the middle bowl at the new spot. This is the case where a slow or stale plan should hurt.
5. **Seed variance (optional).** `SEEDS=2` or more trains `onestep` and `planbind` again with seeds 1001, ... and evaluates them at latency 0 and 6.

## Inference benchmark

`bench_inference.py` loads each checkpoint, feeds it synthetic observations shaped like the real ones, and times a call at batch 1, stage by stage:

- preprocessing (device copy, normalization);
- the observation encoders;
- the head: DDIM denoising and spline decoding for `bsp10`, one UNet pass for `onestep`, Plan and Bind for `planbind`.

CUDA is synchronized around each stage. A Plan-Bind call is timed as plan plus bind, the slower of its two kinds of call; bind alone is reported too. `latency_frames` is the p95 of a call rounded up to whole frames at 30 fps, the value passed to `--policy.planbind_latency`. A second loop times `select_action` frame by frame, as `lerobot-eval` calls it.

`run_all.sh` runs it three times: CUDA fp32 (`bench/cuda.json`), CUDA with fp16 autocast (`bench/cuda_amp.json`, timing only) and CPU with `CPU_THREADS` threads (`bench/cpu.json`). The script needs no simulator, so it also runs on a robot's own computer:

```bash
uv run python experiments/261028-plan-bind-d2-latency/bench_inference.py --device cpu --cpu-threads 4 \
    --policy bsp10=<dir> --policy onestep=<dir> --policy planbind=<dir> --out bench_robot.json
```

## Eval cells

All cells use 50 episodes with eval seed 1000, the same episodes as D1, so any two cells compare episode by episode. They also use the stacking order `red,blue,black` and the step-60000 checkpoints. A cell is named `<policy>_L<latency frames>_<commit>[_<option>...]`.

| Group               | Cells                                                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Measured latency    | each policy at its CUDA and its CPU `latency_frames`, commit `latency`                                                  |
| Update rate         | `onestep`, `planbind` × commit `latency`, `none` × a call every 4, 8, 12 frames (`_every8`, `_every12`), latency 0      |
| Commit modes        | `onestep`, `planbind` × latency 0 and 6 × commit `latency`, `now`, `none` (the latency 6 cells repeat D1's, as a check) |
| Replan threshold    | `planbind` at latency 6 with thresholds 0.5 and 1.0 (`_tau0.5`, `_tau1.0`)                                              |
| `bsp10` runtime     | classic `bsp10_L0` (D1's) and the runtime at latency 0, 3, 6, 9 with commit `latency`, plus latency 0 with `none`       |
| Bowl moves          | `bsp10_L0_shift3cm`, and all three policies through the runtime at latency 0 and 6 (`_shift3cm`)                        |
| Seeds (`SEEDS > 1`) | `onestep`, `planbind` × latency 0 and 6 × commit `latency`, `none`, each extra seed (`_s1001`, ...)                     |

Cells with the same name are run once: a measured latency of 0 or 6 frames reuses the matching cell.

## Run it

**Box:** Ubuntu 22.04 or 24.04, one GPU (tested target: RTX 3090), NVIDIA driver 570 or newer. About 40 GB of free disk, or 80 GB with `SEEDS=2`. It also needs `ffmpeg`, `git`, `git-lfs`, `curl`, `libegl1` and `libgl1`; `run_all.sh` prints the `apt-get` line if any are missing. Log in to the Hub with read access to `thanhkt/plan-bind-d1`.

```bash
git clone https://github.com/khanhthanhdev/lerobot-nexarm.git && cd lerobot-nexarm
git checkout feat/plan-bind
tmux new -s planbind-d2
HF_REPO=<user>/plan-bind-d2 experiments/261028-plan-bind-d2-latency/run_all.sh   # smoke run, then the full run
```

- The first run installs `uv` and the environment, then stops if you are not logged in to the Hub: run `uv run hf auth login` and the same command again.
- `run_all.sh smoke` runs the short check only, `run_all.sh full` skips it. Rerunning the same command after a disconnect resumes the run: finished stages and cells are kept.
- `HF_REPO` is optional. When set, the full run uploads the timings, the eval cells, `results.md` and any newly trained seeds to that private repo.
- Environment overrides: `N_EVAL` (50), `EVAL_BATCH` (10), `SEEDS` (1), `CPU_THREADS` (4), `SHIFT_M` (0.03), `HF_SOURCE_REPO` (`thanhkt/plan-bind-d1`), `CKPT_STEP` (`060000`). With `SEEDS > 1`, `STEPS`, `BATCH` and `NUM_WORKERS` set the training as in D1, and `HF_SOURCE_DATASET` names the dataset to download.

| Stage                   | Smoke                   | Full (estimate, RTX 3090)                                                                              |
| ----------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------ |
| Setup                   | 5–10 min                | same                                                                                                   |
| Checkpoint download     | about 5 min             | same                                                                                                   |
| Benchmarks              | 2–5 min                 | 10–20 min (CPU is the slow part)                                                                       |
| Extra seeds (`SEEDS=2`) | 200 steps each          | two trainings of about 4 h each                                                                        |
| Evaluation              | about 20 cells × 2 eps. | about 40 cells × 50 eps.: 6–9 h (`bsp10` runtime cells, which denoise every 4 frames, are the slowest) |

## Outputs

- `outputs/full/d1/` — the downloaded D1 checkpoints.
- `outputs/full/bench/{cuda,cuda_amp,cpu}.json` — the timings.
- `outputs/full/eval/<cell>/eval_info.json` — one directory per cell, as in D1, with up to 10 videos.
- `outputs/full/results.md` — written by `summarize.py`: the timings, then one table per question with paired differences ("+a/-b" counts the episodes only one of the two cells succeeded in), then every cell with its stage rates.

## Results

Pending.

## Not in this experiment

- **Real arm.** The benchmark can time the policies on the robot's computer, but the evals are in simulation.
- **Instruction following.** The policies are not given the stacking order, so the other five orders are not evaluated.
- **Training changes** such as mixed precision or `torch.compile`; the CUDA fp16 timing only shows what they could gain at inference.
- **Temporal ensembling and RTC baselines.**
