# 261029-plan-bind-d1b-seeds

**Status: not run yet.** D1 ([261027-plan-bind-two-step](../261027-plan-bind-two-step/README.md)) showed Plan-Bind ahead of the one-step control in all 8 cells, but with one training seed per policy, 50 episodes per cell and only the last checkpoint. D1b repeats it with the statistics needed to call the result.

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

Not run yet.
