# 261027-plan-bind-two-step

**Status: run once (RTX 3090, 2026-10-27).** Plan-Bind beat the one-step control in all eight latency and commit cells, by 16 to 56 points. Results are below. The checkpoints, eval cells and `results.md` are in the private Hub repo `thanhkt/plan-bind-d1`, and the dataset is in `thanhkt/nexarm-stack-bowls-sim-red-blue-black-500`.

## Question (RQ1)

Does generating a spline in two steps pay for itself? Plan-Bind makes a coarse plan, then adds a residual on the plan's exact lift. It is compared with a one-step control that has the same network, the same replanning rate, the same latency handling and the same observation freshness.

- **If the control matches Plan-Bind** on success, the two-level structure adds nothing for control, and the research case rests on the interface: certified limits and edits.
- **Secondary question:** how both policies degrade with inference latency (0 to 300 ms), with latency-aware commit and with naive swapping.

## Policies

All three use the same data (500 scripted episodes, all with one stacking order: red, blue, black from bottom to top), image size (120x160, three cameras), 64-frame window, seed, batch size and number of steps.

| Name       | `bspline_mode` | What it is                                                                                                                                         |
| ---------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bsp10`    | `diffusion`    | B-spline Policy with DDIM and 10 denoising steps. This is the quality ceiling, evaluated at latency 0 only, because it has no latency machinery    |
| `onestep`  | `onestep`      | **Control.** One-step regression of the fine control points (knots every 4 frames)                                                                 |
| `planbind` | `planbind`     | Plan (coarse spline, knots every 16 frames) and Bind (residual on its exact lift), one shared UNet. The plan is replaced when older than 32 frames |

Notes for reading the results:

- **What the control is.** `onestep` is a regression head, not B-spline Policy with one diffusion step. Regression is acceptable here because the scripted stack-bowls demonstrations are expected to be close to unimodal. If that assumption fails, both `onestep` and `planbind` suffer in the same way.
- **Shared runtime.** Both new modes run under the same `SplineRuntime`: one call per 4 frames (7.5 Hz at 30 fps). At episode start the first call is used at once, without waiting for the latency.
- **Simulated latency.** Latency is simulated inside the policy. A call made at frame k uses the observation of frame k, and its result is used from frame k + L. Calls do not overlap, so a large L also lowers the update rate.
- **One stacking order.** The task has six orders; this experiment uses one (`STACK_ORDER`, default `red,blue,black`) for the dataset and every eval episode (`--env.stack_order`), so the policies are not asked to follow an instruction.
- **Statistics.** One seed per policy and 50 seeded episodes per cell (seed 1000, the same episodes for every cell). Differences under about 15 points are within noise.

Details are in [the plugin README](../../policies/lerobot_policy_bspline/README.md#one-step-and-plan-bind-modes-research).

## Run it on a rented GPU

**Box:**

- Ubuntu 22.04 or 24.04, one RTX 3090 (24 GB).
- NVIDIA driver 570 or newer (the cu128 PyTorch wheels need it).
- About 100 GB of free disk; ask for 120 GB. Each checkpoint (weights and optimizer state) is about 3.5 GB, and the full run keeps 18 of them, plus 3 from the smoke run.
- `ffmpeg`, `git`, `git-lfs`, `curl`, `libegl1` and `libgl1`. `run_all.sh` checks for them and prints the `apt-get` line if any are missing.

```bash
git clone https://github.com/khanhthanhdev/lerobot-nexarm.git && cd lerobot-nexarm
git checkout feat/plan-bind
tmux new -s planbind                                      # survives a dropped ssh session
experiments/261027-plan-bind-two-step/run_all.sh          # smoke run, then the full run
```

`run_all.sh smoke` runs only the short pipeline check. `run_all.sh full` skips it. Rerunning the same command after a disconnect resumes the run:

- finished stages are skipped;
- training resumes from its last checkpoint;
- eval cells that are already done are kept.

To save the full run to the Hugging Face Hub, log in once (`uv run hf auth login`) and set the repos (both are created private):

```bash
HF_REPO=<user>/plan-bind-d1 HF_DATASET_REPO=<user>/nexarm-stack-bowls-sim experiments/261027-plan-bind-two-step/run_all.sh
```

The dataset is uploaded after it is generated. Each policy is uploaded after it is trained, then its local checkpoints except the last one are deleted, so the run fits a 100 GB disk. The eval cells and `results.md` go last.

Override the size of the full run with environment variables, for example `STEPS=30000 N_EVAL=30 run_all.sh full`. W&B runs offline; logs stay in `wandb/` under each training directory.

| Stage                                 | Smoke                         | Full (estimate, RTX 3090)             |
| ------------------------------------- | ----------------------------- | ------------------------------------- |
| Setup (`uv sync`, EGL and CUDA check) | 5–10 min                      | same                                  |
| Sim dataset                           | 4 episodes, about 2 min       | 500 episodes, 8 workers, under 1 h    |
| Training, three policies              | 200 steps each, a few minutes | 60k steps each, roughly 3–5 h each    |
| Evaluation                            | 10 cells × 2 episodes         | 18 cells × 50 episodes, roughly 2–4 h |

The full-run times are estimates; measure the first 1,000 steps and scale. Plan for a 12–20 h rental, or lower `STEPS` and `N_EVAL`.

## Outputs

- `outputs/full/train_<policy>/` — checkpoints (`checkpoints/last/pretrained_model`).
- `outputs/full/eval/<cell>/eval_info.json` — one directory per cell, named `<policy>_L<latency frames>_<commit>`. Each holds `pc_success` and the per-stage rates (`pc_picked_middle`, ...) from `lerobot-nexarm-eval`.
- `outputs/full/results.md` — written by `summarize.py`: every cell, plus the paired Plan-Bind minus one-step table for RQ1.

The smoke run writes the same layout under `outputs/smoke/`. Its dataset stays inside that folder. The full dataset goes to `outputs/datasets/nexarm_stack_bowls`; an existing complete dataset there is reused, and a partial one stops the run.

## Results

RTX 3090 (Vast.ai), `run_all.sh full` on `feat/plan-bind` at `a1bfdc1f1`, 2026-10-27. Success in percent over 50 episodes (seed 1000, the same episodes in every cell); 95% intervals are in `results.md`.

| Latency | Commit       | `bsp10` | `onestep` | `planbind` | Plan-Bind minus one-step |
| ------- | ------------ | ------- | --------- | ---------- | ------------------------ |
| 0 ms    | latency      | 100     | 68        | 98         | +30                      |
| 0 ms    | none (naive) | –       | 36        | 92         | +56                      |
| 100 ms  | latency      | –       | 36        | 86         | +50                      |
| 100 ms  | none (naive) | –       | 48        | 96         | +48                      |
| 200 ms  | latency      | –       | 72        | 96         | +24                      |
| 200 ms  | none (naive) | –       | 72        | 100        | +28                      |
| 300 ms  | latency      | –       | 72        | 98         | +26                      |
| 300 ms  | none (naive) | –       | 84        | 100        | +16                      |

Global re-planning on a residual trigger (`planbind_replan_threshold=0.3`, 200 ms, latency commit) dropped Plan-Bind from 96 to 72.

Reading it:

- **RQ1.** The two-level structure pays off here: Plan-Bind is near the diffusion ceiling (`bsp10`, 100) at every latency, and the one-step control is not.
- **Latency hardly hurts Plan-Bind** in this task (86 to 100 across 0 to 300 ms). Latency-aware commit showed no clear gain over naive swapping for either policy; the differences are mostly within noise.
- **One-step improves with latency** (36–68 at 0 ms, 72–84 at 300 ms). A plausible cause is that a larger latency also lowers the update rate (calls do not overlap), so the control replans less often and jitters less. [D2](../261028-plan-bind-d2-latency/README.md) tests this, together with measured inference times.
- **Limits.** One training seed per policy, one task and stacking order, and scripted demonstrations.
- **Paired test (added 2026-10-31).** Every cell runs the same 50 episodes, so `summarize.py` now also prints the episodes solved by only one of the two policies and the exact McNemar p. Plan-Bind's lead is significant in all 8 cells (p ≤ 0.008; for example 29 against 1 episodes at 0 ms, naive). The table above is unchanged.
- **The two identical 18/50 one-step cells are a coincidence.** `onestep_L0_none` and `onestep_L3_latency` agree on 26 of 50 episodes (12 succeed only in the first, 12 only in the second), and their videos differ, so both settings were applied.
- **Open confound.** The one-step control maps an all-zeros sequence through a UNet with no positional encoding, while Bind receives the lifted plan, which carries sequence position. [D1b](../261029-plan-bind-d1b-seeds/README.md) adds `onestep_pos`, a one-step control with a position-bearing input, to separate the two.

Measured times: dataset 67 min (10 workers); training 5.7 h (`bsp10`), 3.7 h (`onestep`) and 3.9 h (`planbind`); eval 18 cells in 2.8 h.

## Not in this experiment

- **Real arm and certified limits (RQ2).** The projection options exist (`planbind_position_limit`, `planbind_velocity_limit`) but are not in the eval matrix.
- **Edit preservation (RQ3).**
- **Moving the bowl mid-episode.** In D2.
- **Temporal ensembling and RTC baselines.**
- **Adaptive knots.**
- **Latency conditioning of Bind.** Bind is conditioned on the plan's age, not on the latency.
