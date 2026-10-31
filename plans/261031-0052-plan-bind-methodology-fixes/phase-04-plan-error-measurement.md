---
title: "Phase 4: Plan-error measurement"
status: in-progress
priority: P2
effort: 3h
dependencies: []
---

# Phase 4: Plan-error measurement

## Context

- Review §4: [`review-261031-0736-plan-bind-methodology.md`](../reports/review-261031-0736-plan-bind-methodology.md)
- Training code: `PlanBindModel.compute_loss` (`planbind.py` lines 101–138): Bind trains on `coarse_fit @ stale_window + N(0, planbind_plan_noise²)`, never on `model.plan(...)`
- Trigger statistic: `SplineRuntime._bind` returns `residual.abs().amax(dim=(1, 2))`; `planbind_replan_threshold` compares against it
- Loader pattern to reuse: `experiments/261028-plan-bind-d2-latency/bench_inference.py` lines 105–120 (`PreTrainedConfig.from_pretrained`, `BsplinePolicy.from_pretrained`, `make_pre_post_processors`)
- Data: `thanhkt/nexarm-stack-bowls-sim-red-blue-black-500` (with videos; the images are needed for `global_cond`, so this is not the parquet-only path of `sfe-fit-lag`); held-out layouts come from `examples/nexarm/generate_stack_bowls_dataset.py --split eval`
- Checkpoint: `thanhkt/plan-bind-d1`, `train_planbind/checkpoints/060000/pretrained_model` (private; `uv run hf auth login` on the server)
- D2: `experiments/261028-plan-bind-d2-latency/README.md`, thresholds 0.5 and 1.0

## Overview

Bind is trained with teacher forcing on oracle coarse plans plus Gaussian noise of 0.02. At inference it corrects the network's own plan, whose error has never been measured. The residual trigger at τ = 0.3 cut success from 96 to 72 % in D1; if typical plan errors already exceed 0.3, the trigger fires on healthy plans. This script measures, on D1's `planbind` checkpoint, (a) the plan error against the coarse least-squares fit of the demonstrated future, and (b) Bind's trigger statistic, at plan ages 0 to 32 frames, on training episodes and on held-out layouts. Its output places τ on percentiles so D2's thresholds are chosen from data.

## Requirements

- [x] `experiments/261028-plan-bind-d2-latency/plan_error.py` (D2 owns the thresholds), CLI: `--policy <pretrained_model dir>`, `--dataset-root`/`--repo-id`, `--episodes <n>` (sampled episodes), `--frames-per-episode <k>`, `--ages 0,8,16,24,32`, `--device`, `--out <json>`.
- [x] Per sampled frame `t`: build the policy batch with the policy's `delta_timestamps` through `LeRobotDataset` and the preprocessor; compute `global_cond`; `coarse_true = coarse_fit @ actions[t : t + window]`; `plan = model.plan(global_cond)`; `plan_error = (plan − coarse_true)` in normalised units; for each age `a`, `lifted = lifted_plan(plan_at(t − a), a // spacing)` and `trigger = bind(lifted, a, global_cond).abs().amax()`. Use `policy.diffusion._prepare_global_conditioning` on the stacked queues as `select_action` does.
- [x] Report, per age and split: percentiles 50, 90, 95, 99 and max of `|plan_error|` (max over points and joints, and RMS), and of the trigger statistic; the fraction of frames whose trigger exceeds 0.3, 0.5 and 1.0; the training noise 0.02 for reference.
- [x] Write JSON and print a Markdown table; D2's README gets a "Plan error" subsection pointing at the output (text only, no result numbers until the run).
- [ ] Held-out layouts: generate 20 episodes with `generate_stack_bowls_dataset.py --split eval --episodes 20 --order red,blue,black` into `outputs/datasets/nexarm_stack_bowls_eval_20` on the server, and run the script on both roots.

## Implementation steps

1. Copy the loader from `bench_inference.py` into a `load_policy(path, device)` helper in the new script (two experiment scripts; a shared module is not warranted yet).
2. Build the dataset with `LeRobotDataset(repo_id, root=root, delta_timestamps=...)` from the policy config's `observation_delta_indices` and `action_delta_indices` divided by fps (see how `lerobot-train` builds them in `lerobot.datasets.factory`; mirror it rather than hand-rolling).
3. Sample frames at least `planbind_max_plan_age` into each episode so the stale window exists, and at least `bspline_window` before its end. For every sampled `t` and every age `a`, also load the observation at `t − a` and run `model.plan` **there**: a stale plan is the plan the policy would have made `a` frames earlier, not the current plan shifted. Only Bind runs on the observation at `t`.
4. Compute the quantities above under `torch.no_grad()`; normalise actions with the preprocessor so the units match the training loss.
5. Aggregate and write the JSON; print the table.
6. Add the D2 README subsection and the server command line.

## Todo

- [x] Script with loader, sampling and metrics
- [x] Held-out generation command
- [x] D2 README subsection
- [x] Ruff clean; a `--dry-run` that builds one batch on CPU with a tiny fake policy is optional

## Verification

Windows: `uv run ruff check experiments/261028-plan-bind-d2-latency --no-fix`; `uv run python experiments/261028-plan-bind-d2-latency/plan_error.py --help`. The dataset download and video decoding need ffmpeg; a full run here is possible only on CPU and slow, so treat it as optional.

Server (`vishc-server-2`, GPU 6):

- `uv run hf download thanhkt/plan-bind-d1 --include "train_planbind/checkpoints/060000/*" --local-dir outputs/d1`
- `uv run hf download thanhkt/nexarm-stack-bowls-sim-red-blue-black-500 --repo-type dataset --local-dir outputs/datasets/nexarm_stack_bowls`
- `uv run python examples/nexarm/generate_stack_bowls_dataset.py --root outputs/datasets/nexarm_stack_bowls_eval_20 --episodes 20 --order red,blue,black --split eval --gpus 6`
- `uv run python experiments/261028-plan-bind-d2-latency/plan_error.py --policy outputs/d1/train_planbind/checkpoints/060000/pretrained_model --dataset-root <root> --episodes 50 --frames-per-episode 10 --device cuda --out outputs/plan_error_<split>.json` for both roots.

## Success criteria

- The JSON places τ = 0.3, 0.5 and 1.0 on the percentiles of the trigger statistic at each age, on training and held-out layouts.
- The D2 thresholds (and, if needed, `planbind_plan_noise`) can be justified from these numbers in D2's README.

## Risks

- The held-out generator needs EGL and a GPU; GPU 6 was free on 2026-10-31, the others were not.
- If the plan error is far above 0.02, the finding is also an argument for training Bind on detached self-generated plans; that change is a later experiment, not this script.
