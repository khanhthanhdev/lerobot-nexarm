# Plan-Bind on LIBERO (D4)

Status: proposed. Date: 2026-11-03. Depends on: D1b results written up, D3 (Push-T) smoke run read.

## Outcome

A LIBERO result for Plan-Bind that can sit next to B-spline Policy (`bsp10`), the one-step control and published LIBERO policies, scored by `lerobot-eval --env.type=libero` on the four suites. It answers: does the D1/D1b lead of Plan-Bind over one-step survive on a standard, language-conditioned, multi-task benchmark with delta-end-effector actions?

## Why this is not just "run D1 again"

| Gap                                                                      | Why it blocks                                                                                                                            | Decision                                                                                                |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| No language input                                                        | LIBERO suites hold 10 tasks per scene (`libero_goal` shares one scene across all 10), so a policy without the instruction cannot succeed | Add a task-text condition to `global_cond` (phase 1). It is the only code change in the policy          |
| Actions are delta-EE velocities (7-D, gripper in the last dim) at 10 fps | Splines are fitted to jittery deltas and to a step-like gripper; fit error may cap success                                               | Measure fit error first (phase 0); fall back to a larger knot density or a separate gripper channel     |
| 2 cameras (`lerobot/libero`), 8-D state, other image size                | D1 configs assume 3 cameras at 120x160                                                                                                   | New configs; keep each policy's input identical                                                         |
| Long horizons, 4 suites                                                  | Eval is slow and the shared server freezes under memory pressure                                                                         | Reuse the D1b findings: at most 2 eval jobs at once, `batch_size` 5, stall watchdog, work in `/dev/shm` |

## Phases

### Phase 0: feasibility (about half a day, one GPU, no training)

- Fit cubic B-splines to `lerobot/libero` action windows at `planbind_fine_spacing` 2, 4 and the coarse spacing; report RMS error per action dim and the gripper's error. Decide knot spacing from this, not from D1's 4/16.
- Run `bench_inference.py`-style timing on a LIBERO-shaped checkpoint at init weights (batch 1), so latency numbers are known before training.
- Pass: spline fit RMS error well below the per-dim action std (set the threshold from the numbers; do not pick it after seeing results).

### Phase 1: language conditioning (code, about 1 day)

- Add `task_condition` to `BsplineConfig`: `none` (default, D1 behaviour), `embed` (frozen text encoder output, projected and added to `global_cond`). Use the encoder already in the stack for SmolVLA or TurboVLA if one is installed; otherwise a small frozen sentence encoder. Cache embeddings per task string.
- The same condition feeds `bsp10`, `onestep` and `planbind`, so the comparison stays fair.
- Tests: one config test, and one test that two different instructions give different `global_cond` while `none` leaves D1 outputs bit-identical.

### Phase 2: smoke (30 minutes)

`--steps=200`, four episodes, one suite task, `eval.n_episodes=1`. Checks data loading, the text condition, memory at the chosen batch size, `step_s`, and that eval runs. Gate for phase 3.

### Phase 3: training

Policies and settings (same data, steps, batch, image size, seeds for all):

| Name          | What                                                            | Seeds |
| ------------- | --------------------------------------------------------------- | ----- |
| `planbind`    | plan + bind                                                     | 3     |
| `onestep`     | one-step control (same net, same call rate)                     | 3     |
| `onestep_pos` | position-bearing one-step control from the methodology-fix plan | 3     |
| `bsp10`       | B-spline Policy, DDIM 10                                        | 3     |
| `diffusion`   | LeRobot diffusion policy, same backbone                         | 1     |

One model over all four suites, as LeRobot's LIBERO tables do (check their protocol first and match it), `lerobot/libero` at the pinned revision from `261024-turbovla-libero`. Steps: set from phase 2 `step_s`; plan for 60k to 100k. 3 seeds x 4 policies is 12 trainings plus controls, so run two at a time and prune all but the last checkpoint after the final eval.

### Phase 4: evaluation (verifying Plan-Bind)

- Main: latency 0, all four suites, 10 tasks x 10 episodes per suite (the unit LeRobot reports), same seeds for every policy.
- Latency: 0, 100, 200, 300 ms with latency commit for `planbind`, `onestep`, `bsp10` through the runtime (`bspline_runtime=true`), for the best seed of each. Naive swapping at 200 ms.
- Update rate separated from latency: calls every 2, 4, 8 frames at latency 0 (D2 design).
- Measured inference time per call on GPU and CPU (`bench_inference.py`), reported next to success so the latency axis is real.
- Statistics: pooled success with Wilson intervals, paired per-seed difference and McNemar (already in `summarize.py`). Per suite and overall. No claim from a gap under about 10 points with fewer than 3 seeds.

### Phase 5: comparison with related work

- Same harness, same eval: `act`, `smolvla`, `pi0` or `pi05` and TurboVLA, using released checkpoints where they exist and the LIBERO eval code in LeRobot. Say plainly when a baseline was trained by someone else with a different recipe or data.
- Published numbers (OpenVLA, pi0, SmolVLA, and so on): copy only from their papers or from LeRobot's LIBERO docs, cite the source and the eval protocol, and mark them as not run by us. Do not mix them into paired tests.
- Report parameters, per-call latency and success in one table, since Plan-Bind's claim is quality at low latency and small size, not the highest absolute LIBERO score.

## Risks

- Plan-Bind may not beat `onestep` here: deltas are not smooth and tasks are shorter-horizon, which removes the coarse-plan advantage. That is a valid result; report it and keep the D1 claim scoped to NexArm stack bowls.
- Language conditioning may dominate the results. Include a `task_condition=none` run on `libero_spatial` as a check.
- Small tasks across four suites make per-task success noisy; use the suite mean as the unit.
- Server limits found in D1b: 60 GB memory and 8 CPU cores for the account. Two concurrent jobs, `NUM_WORKERS=2`.

## Acceptance

- Phase 0 numbers and the knot-spacing decision written down before training.
- Language-conditioned configs for all policies, with tests passing.
- Four-suite results for at least 3 seeds of `planbind`, `onestep`, `bsp10`, plus latency and call-rate tables.
- Related-work table with provenance for every number.
- A README at `experiments/<date>-plan-bind-libero/` with command, data revision, results and limits, in the repo's experiment format.
