# 261029-plan-bind-d1b-seeds

**Status: run (3 A5000 GPUs shared server, 2026-10-29 to 2026-11-03).** D1 ([261027-plan-bind-two-step](../261027-plan-bind-two-step/README.md)) showed Plan-Bind ahead of the one-step control in all 8 cells, but with one training seed per policy, 50 episodes per cell and only the last checkpoint. D1b repeats it with the statistics needed to call the result.

## Question

Does Plan-Bind's lead over the one-step control in D1 survive (a) other training seeds, (b) more eval episodes and (c) a stronger one-step control?

- **If it survives,** the claim "two-step beats one-step on this task" is supported by 3 seeds × 100 episodes, and D1's 8-cell table can be quoted.
- **If the control closes the gap** (`onestep_k2` or `onestep_long`), D1 compared against a weak control, and the research case rests on the interface instead.

**Decision rule for RQ1, fixed before the run.** D1's `onestep` feeds its UNet an all-zeros sequence, and that UNet has no positional encoding along the sequence, while Bind receives the lifted plan, which varies along it. D1's gap may therefore partly measure "position-bearing input" rather than "two-step structure". `onestep_pos` gives the one-step control a position-bearing input without a plan. If `onestep_pos` closes most of Plan-Bind's lead at latency 0 across seeds, RQ1 is reframed to the interface (RQ2, RQ3). If the lead holds against `onestep_pos`, RQ1 stands with a matched-input control.

## Design

Same data (D1's 500 scripted episodes, one stacking order), configs, batch size (32), 60k steps and eval as D1. What changes:

| Change            | D1                   | D1b                                                                                                                                                                                                                                                                                         |
| ----------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training seeds    | 1 (1000)             | 3 (1000–1002) for `bsp10`, `onestep`, `onestep_pos`, `planbind`                                                                                                                                                                                                                             |
| Episodes per cell | 50                   | 100 per seed (95% interval per seed about ±10 points; paired tests per seed)                                                                                                                                                                                                                |
| Checkpoints       | last only            | last only (the others are deleted after training, to save disk), so no checkpoint curve                                                                                                                                                                                                     |
| One-step controls | `onestep`            | plus `onestep_pos` (a trained `(n_fine, action_dim)` input initialised to a linear ramp from −1 to 1 instead of zeros: 114 extra numbers on a UNet of tens of millions; 3 seeds), `onestep_k2` (fine knots every 2 frames, calls twice as often) and `onestep_long` (120k steps), seed 1000 |
| Latency / commit  | 0–300 ms, both modes | 0, 200 and 300 ms with latency commit, and naive swapping at 0 ms                                                                                                                                                                                                                           |

The D1 policies are trained again, not reused: D1's checkpoints are on the Hub, but a repeat with other seeds needs new trainings anyway. `summarize.py` reports each seed's success with its Wilson interval and the mean over seeds; no interval is pooled over seeds, because every seed runs the same 100 eval episodes. Plan-Bind is compared with `onestep` and `onestep_pos` per seed on the shared episodes: the difference in points, the episodes solved by only one of the two, and the exact McNemar p. Success is the env's per-episode success flag, the one behind `pc_success`.

**Seeds for `onestep_pos`.** Three seeds add about 11 GPU-hours on a 3090-class card (D1 measured 3.7 h per `onestep` training); one seed adds about 4 h. Three is the default because this control decides how RQ1 is framed. For one seed, make `job_list` in `run_all.sh` emit only `onestep_pos_s1000`; to pay for three seeds instead, drop the `onestep_k2_s1000` line.

Not covered: other tasks and stacking orders, human or noisy demonstrations, the real arm. The six-order task needs instruction conditioning, which these policies lack.

## Run it

Same box as D1, with 1 to 3 GPUs. Check which GPUs are free with `nvidia-smi` first.

```bash
git clone https://github.com/khanhthanhdev/lerobot-nexarm.git && cd lerobot-nexarm
git checkout feat/plan-bind
tmux new -s planbind-d1b
# Smoke run first (about 30 min), then the full run. Download D1's dataset instead of generating it; the run
# stops unless the dataset has D1's 500 episodes and one stacking order:
GPUS="0 1 2" NUM_WORKERS=4 HF_DATASET_REPO=thanhkt/nexarm-stack-bowls-sim-red-blue-black-500 \
  HF_REPO=<user>/plan-bind-d1b experiments/261029-plan-bind-d1b-seeds/run_all.sh
```

In a second terminal: `watch -n 30 experiments/261029-plan-bind-d1b-seeds/status.sh` shows the GPUs, finished stages, each training's latest checkpoint, eval cells done and each worker's last log line. Rerunning the same command after a disconnect resumes.

How the work is split: with `GPUS` set, each GPU worker takes the next unclaimed training (`onestep_long` first, then the three `onestep_pos` so the decisive control reports early, then the three `bsp10`, three `planbind`, three `onestep`, `onestep_k2`), and evaluates that run's last checkpoint at latency 0 (the headline number). When all are trained, the workers split the 49 final-checkpoint cells between them and upload the policies. Each training keeps only its last checkpoint. Env overrides: `SEEDS`, `STEPS`, `BATCH`, `SAVE_FREQ`, `NUM_WORKERS`, `N_EVAL`, `EVAL_BATCH`.

**Time (estimate).** D1 on an RTX 3090 took 5.7 h (`bsp10`), 3.7 h (`onestep`) and 3.9 h (`planbind`) per training. The 14 trainings add up to about 62 GPU-hours on that card, so about 21 h on three GPUs of similar speed (more if the A5000 is slower), plus about 3–5 h of eval. Measure the smoke run's `step_s` and scale.

## Outputs

`outputs/full/train_<variant>_s<seed>/`, `outputs/full/eval/<cell>/eval_info.json`, `outputs/full/results.md`, and `outputs/full/logs/worker_<n>.log`.

## Results

Stack bowls, one stacking order, 100 episodes per cell (seed 1000, the same episodes in every cell), last checkpoint of each training, 3 training seeds (1000 to 1002) unless noted. The full tables, with Wilson intervals, per-seed McNemar tests and every cell, are in `results.md` (regenerate with `summarize.py`).

Success in percent, mean over seeds (`bsp10rt` is `bsp10` run through the same latency runtime as the one-step modes):

| Policy                     | 0 ms | 0 ms, naive swap | 200 ms | 300 ms |
| -------------------------- | ---- | ---------------- | ------ | ------ |
| `planbind`                 | 96   | 92               | 98     | 98     |
| `onestep`                  | 71   | 33               | 68     | 80     |
| `bsp10rt`                  | 89   | -                | 58     | 45     |
| `bsp10` (own action queue) | 100  | -                | -      | -      |

One seed (1000): `onestep_long` (120k steps) 72, `onestep_k2` (calls twice as often) 43, `onestep_k2` at D1's call rate 44.

Plan-Bind minus a control, mean of per-seed differences in points (ahead on 3 of 3 seeds in every row):

| Control   | 0 ms                      | 0 ms, naive | 200 ms | 300 ms |
| --------- | ------------------------- | ----------- | ------ | ------ |
| `onestep` | +25                       | +59         | +31    | +18    |
| `bsp10rt` | +7 (p = 0.04, 0.02, 0.58) | -           | +40    | +53    |

Measured inference time of one call, batch 1, RTX A5000 and 4 CPU threads of a shared server (`bench_inference.py`, seed-1000 checkpoints); `frames` is the p95 rounded up at 30 fps:

| Policy                   | GPU p50 / p95 (ms) | GPU frames | CPU p50 / p95 (ms) | CPU frames |
| ------------------------ | ------------------ | ---------- | ------------------ | ---------- |
| `bsp10`                  | 80.6 / 81.6        | 3          | 1305 / 1849        | 56         |
| `onestep`                | 18.6 / 24.8        | 1          | 213 / 242          | 8          |
| `planbind` (plan + bind) | 26.3 / 31.5        | 1          | 246 / 277          | 9          |

Reading it:

- **RQ1 holds across seeds.** Plan-Bind beats the one-step control on every seed and setting, by 12 to 60 points. The one-step control is also unstable across training seeds (59 to 84 at 0 ms). Training it twice as long or calling it twice as often does not close the gap.
- **Latency.** Plan-Bind stays at 92 to 99 across 0 to 300 ms. `bsp10rt` falls from 89 to 45, in line with its measured call time of 3 frames on the GPU. At 0 ms, `bsp10rt` is within 7 points of Plan-Bind, and `bsp10` with its own action queue reaches 100.
- **Latency-aware commit** matters much more for one-step (71 against 33) than for Plan-Bind (96 against 92).
- **One-step improves at 300 ms** (80 against 68 at 200 ms), as in D1. A longer latency also lowers the call rate here; [D2](../261028-plan-bind-d2-latency/README.md) separates the two.

Limits: one stacking order and scripted demonstrations, simulation only. `onestep_k2` and `onestep_long` have one seed each. Only the last checkpoint was kept, so there is no training curve. Inference times come from a shared server and are indicative. The server's per-account limits (60 GB memory, 8 cores) froze concurrent evals, so the cells were run two at a time with batch size 5. The 4 trainings that first used the six-order dataset by mistake were discarded and rerun.

Hub: checkpoints and eval cells are in `thanhkt/plan-bind-d1b`; the dataset is `thanhkt/nexarm_stack_bowls_red_blue_black_1k`.
