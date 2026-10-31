# Review: code and methodology of the Plan-Bind experiments (D1, D1b, D2), with notes on the steering and state-feedback work

Date: 2026-10-31. Read-only review of `policies/lerobot_policy_bspline/src/lerobot_policy_bspline/{planbind,uniform_spline,modeling_bspline,configuration_bspline}.py`, `experiments/261027-plan-bind-two-step/`, `experiments/261029-plan-bind-d1b-seeds/`, `experiments/261028-plan-bind-d2-latency/`, the D1 eval outputs under `experiments/261027-plan-bind-two-step/outputs/full/eval/`, and the plans of the steering and state-feedback repos. Two probe scripts were run on CPU against the installed LeRobot UNet; they live in the session scratchpad and are reproduced below in outline. Nothing in the repo was changed.

## Summary

- **D1's result is statistically solid but its control is not matched on input information.** Every Plan-Bind cell beats the one-step cell on the same 50 episodes with exact McNemar p ≤ 0.03 (table in §2). But LeRobot's 1D UNet has no positional encoding along the control-point axis. The one-step head feeds it an all-zeros sequence, so the only way that network can tell control point 3 from control point 12 is through zero-padding at the sequence ends. Bind instead receives the lifted plan, which already varies along the sequence. Part of the +16 to +56 point gap may therefore be "a position-bearing input beats a constant input", not "two steps beat one". **D1b does not test this, and it should before its ~17 GPU-hours are spent** (§3).
- **Bind is trained with teacher forcing.** It only ever sees the coarse least-squares fit of the true future plus Gaussian noise of 0.02, never the network's own plan. The size of the real plan error at inference was never measured. This is a plausible reason the residual trigger (τ = 0.3) hurt (§4).
- **Two bookkeeping points.** D1's unresolved question about the identical 18/50 cells is closed: the two cells agree on only 23 of 50 episodes, so it is a coincidence. D1b's `summarize.py` pools 300 episodes across seeds as if independent, but every seed reuses the same 100 eval episodes, so its Wilson intervals will be too narrow (§5).
- **The steering study's code was redesigned on the server on 2026-10-30 and is uncommitted.** The new design is stronger than the plan in this repo describes, and only one copy of it exists (§7).

## 1. What the code does, and what checks out

| Claim in the README or proposal                                    | Code                                                                                                                                                                                       | Verdict                                                                                |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| Same parameter count for `onestep` and `planbind`                  | One `PlanBindModel` UNet serves `plan`, `onestep` and `bind`; the role goes in through the timestep embedding (`planbind.py:79-99`)                                                        | Holds                                                                                  |
| The lift is exact                                                  | `lift_matrix` solves least squares on 8 samples per frame over nested uniform knots; nesting makes the coarse curve a member of the fine space (`uniform_spline.py:51-68`)                 | Holds                                                                                  |
| Latency commit gives a C² join                                     | `n_frozen("latency") = 3 + ceil(latency / spacing)`; a cubic span depends on 4 points, so freezing 3 past the swap knot fixes value, slope and curvature there (`uniform_spline.py:80-92`) | Holds                                                                                  |
| Same episodes in every cell                                        | `--seed 1000`, `n_episodes 50`, one stacking order, stage logging per episode                                                                                                              | Holds; per-episode outcomes are in `eval_info.json → per_task[0].metrics.stages`       |
| "One network call per update" for Plan-Bind                        | The first call and every re-plan (plan older than 32 frames, so every 8th call) run Plan and Bind: about 1.13 UNet passes per call on average                                              | Minor overstatement; D2's benchmark times plan + bind as the slow case, which is right |
| Hard limits on position, velocity and acceleration (proposal §3.3) | `project` clamps position and first differences only (`uniform_spline.py:95-117`)                                                                                                          | Acceleration is not implemented; RQ2 cannot yet be run as written                      |
| Commit modes `latency`, `now`, `none`                              | All three exist                                                                                                                                                                            | D1 ran only `latency` and `none`; D2 adds `now`                                        |

Other code observations:

- `SplineRuntime.step` only issues a call when `frame % call_period == 0` and no call is pending. With call period 4 this makes the update period 4, 4, 8, 12 frames at latencies 0, 3, 6, 9: the latency/update-rate confound D1 acknowledges. D2's `planbind_call_every` sweep is the right fix.
- Bind's plan age at inference (`age_frames // fine_spacing` spans, re-plan when age > 32) matches the ages it is trained on (0 to 32 frames in steps of 4). Consistent.
- D1b's `run_full` defaults `DATA` to `outputs/datasets/nexarm_stack_bowls_red_blue_black_1k`, and `get_data` downloads D1's 500-episode set into it when missing. If a real 1,000-episode dataset already sits at that path, D1b silently trains on different data than D1 while its README says "same data". Low severity; worth a name or an episode-count check.

## 2. D1 re-analysed with the per-episode outcomes

D1 reported per-cell Wilson intervals and a rule of thumb ("differences under 15 points are noise"). The saved stage logs allow the paired test the design deserves. Exact two-sided McNemar on the 50 shared episodes:

| Latency | Commit  | One-step wins | Plan-Bind wins | Only Plan-Bind | Only one-step | McNemar p |
| ------- | ------- | ------------- | -------------- | -------------- | ------------- | --------- |
| 0 ms    | latency | 34            | 49             | 16             | 1             | 0.0003    |
| 0 ms    | none    | 18            | 46             | 29             | 1             | < 0.0001  |
| 100 ms  | latency | 21            | 43             | 25             | 3             | < 0.0001  |
| 100 ms  | none    | 24            | 49             | 26             | 1             | < 0.0001  |
| 200 ms  | latency | 39            | 50             | 11             | 0             | 0.0010    |
| 200 ms  | none    | 36            | 50             | 14             | 0             | 0.0001    |
| 300 ms  | latency | 38            | 49             | 12             | 1             | 0.0034    |
| 300 ms  | none    | 44            | 50             | 6              | 0             | 0.0312    |

Two more facts from the same logs:

- **The identical 18/50 cells are a coincidence.** `onestep_L0_none` and `onestep_L3_latency` agree on 23 of 50 episodes (12 succeed only in the first, 15 only in the second), and their eval videos differ. The settings were applied differently. D1's unresolved question can be closed.
- **One-step's failures are not "hard episodes".** Over its 8 cells, no episode fails in every cell; 14 episodes fail in exactly one cell, 10 in two, and only 5 in six. Policy and environment are deterministic, so this spread comes entirely from the latency and commit settings. For Plan-Bind, 39 of 50 episodes never fail. This supports D1's jitter hypothesis, which D2's update-rate sweep tests.

So the gap is real and large on this run. The question is what it measures.

## 3. The one-step control receives less information than Bind

**Architecture.** `DiffusionConditionalUnet1d` (LeRobot, `modeling_diffusion.py:657-795`) is Conv1d blocks with GroupNorm and FiLM from the global condition, two stride-2 downsamples and two stride-2 transposed convolutions. The only positional embedding is the sinusoidal one on the _diffusion timestep_, a scalar per sample. Nothing encodes the index along the sequence.

**Consequence for a constant input.** Every layer is equivariant to shifts of 4 along the sequence, so for an all-zeros input the output is exactly periodic with period 4 away from the ends, and all other position dependence comes from zero padding at the boundaries. A probe on a randomly initialised UNet with the D1 widths confirms this: on a 256-row zeros input, rows 4 apart differ by 2 × 10⁻⁵ of the output scale, rows 1 or 3 apart by about 1.0. On the 24-row input `onestep` uses (19 control points padded to 24), every row deviates from that periodic pattern by 0.2 to 1.4 of the output scale, so the boundary does reach every row and the network _can_ learn a position-dependent output, but only by reading how far each row is from the padded ends.

**The asymmetry.** `onestep` maps zeros → 19 fine control points. `plan` maps zeros → 7 coarse points, a much easier target for the same pathway. `bind` maps the lifted plan → a residual, and the lifted plan carries the sequence position directly. The diffusion ceiling `bsp10` starts from noise, which also breaks the symmetry. So of the three policies, only the control is asked to produce a long, position-dependent sequence from an input with no position signal. RQ1 as run compares "two-step structure plus a structured input" against "one step with a constant input".

**What would settle it** (cheap, before D1b): a one-step variant whose input carries position but no plan, for example a learned `(n_fine, D)` input tensor (114 parameters), a fixed linear ramp, or the current normalised joint state repeated per row. If that variant closes most of the gap, the two-step claim shrinks to the interface (RQ2 and RQ3), which the proposal already names as the fallback. If it does not, RQ1 stands on much firmer ground. D1b's two extra controls do not address this: `onestep_long` only trains longer, and `onestep_k2` (knots every 2 frames, 35 control points) makes the sequence longer, which should make the padding pathway weaker, not stronger.

The proposal (§2, RQ1) describes the control as "one-step BSP", a generative model run once from noise. The implemented control is a regression from zeros, which the D1 README states openly. A one-step flow or consistency head from noise would be closer to the proposal and would not have the positional handicap; the learned-input variant above is the cheaper route to the same question.

## 4. Bind is trained on oracle plans

`PlanBindModel.compute_loss` (`planbind.py:125-138`) builds Bind's input as the coarse least-squares fit of the _demonstration's_ future window, shifted back by a random age, plus N(0, 0.02²) noise. The target is the current window. At inference Bind receives `model.plan(...)`, whose error is structured and of unknown size.

- This is standard teacher forcing and the 92 to 100 % results show it generalises on this task. It is still unmeasured: a one-off script could compare `coarse_fit` of held-out demonstration windows against `model.plan` on the same observations and report the error distribution next to the 0.02 noise.
- It offers a candidate explanation for the τ = 0.3 result (96 → 72 %). The residual is Bind's correction of the plan error; if typical plan errors already exceed 0.3 in normalised units, the trigger fires on healthy plans and the fresh plan then has no committed context. D2's thresholds 0.5 and 1.0 are the right next step, and the plan-error measurement would say where they sit on the distribution.
- Ablation worth adding later: Bind trained on detached self-generated plans, or with noise matched to the measured plan error.

## 5. Statistics in the D1b and D2 scripts

- D1b `summarize.py` pools wins over seeds and gives a Wilson interval on 300 episodes. The three seeds share the same 100 eval episodes, so the outcomes are correlated across seeds and the interval is too narrow. The per-seed difference table with "wins in k of 3 seeds" is the honest unit. Better still: per seed, count discordant episodes and run McNemar as above, then report the three p-values or the mean paired difference with a bootstrap over episodes. The per-episode outcomes are already in each `eval_info.json`, and D2's `summarize.py` reads them for its "+a/−b" columns, so D1b can reuse that code.
- D1b trains three seeds for all three policies and evaluates 100 episodes per cell; that design is proportionate. Deleting all but the last checkpoint is a sound disk trade-off and the README says so.
- D2 reuses D1's single seed, which is fine for timing and the update-rate sweep but means the "bowl shift" reactivity cells inherit D1's seed luck.

## 6. State-feedback ensembling (sfe)

Already audited in its own session; the README records the negative results and the untested adapter. The one methodological point that still applies: the real-arm case (a heavier payload, a pose-dependent sag model in the sim) has not started, and the LeRobot adapter has only been exercised with stand-in objects.

## 7. Steering study

- The plan in `plans/261026-0556-steering-gate-study/plan.md` is methodologically careful: paired episodes, fixed seed splits, a random gate at matched firing rate, harm and rescue reported separately, McNemar with Holm correction, and pre-registered go/no-go gates.
- On `vishc-server-2`, `~/code/policy-steering` holds an **uncommitted redesign** (10-30): `pipeline.py`, `jobs.py`, `scores.py`, `steerability.py` are new; the phase shell scripts are deleted; `gates.py`, `lerobot_io.py`, `rollout.py`, `store.py` changed (605 insertions, 532 deletions). Its README describes a steerability map (rescue and harm by steering onset, stage and time since the bowl shift), an oracle best-onset bound, exact offline gate replay on recorded base trajectories, and a random-direction control. That is a stronger design than the phase plan, and it ran end to end as a smoke test (`runs-smoke2/REPORT.md`, labels faked). None of it is committed, synced to this machine, or reflected in `plan.md`.
- The local copy at `C:\Users\Administrator\code\policy-steering` is the older design (commit `257df66`).

## Recommendations, in order

1. **Before D1b runs:** add a one-step control with a position-bearing input (§3) to its job list. It costs one extra training (about 4 GPU-hours) and decides how RQ1 can be framed.
2. **Measure the plan error** (§4) with a short script on held-out demonstrations; use it to pick D2's replan thresholds and to decide whether Bind's training noise needs matching.
3. **Fix D1b's statistics** (§5): per-seed discordant counts and McNemar from the stage logs instead of pooled Wilson.
4. **Close D1's open question** in its README: the 18/50 cells are a coincidence (23 of 50 agreeing episodes).
5. **Implement the acceleration bound** in `project` before RQ2, or restate RQ2 as position and velocity only.
6. **Commit the steering redesign on vishc-server-2** and update `plan.md` to match it, or the only copy of a day's design work stays at risk.

## Unresolved questions

- Should the position-bearing control be a learned input, a fixed ramp, or a one-step flow from noise as the proposal originally described? The learned input is the cheapest; the flow is closest to the proposal.
- Is the steering redesign on the server the intended direction, or an experiment the user has not yet accepted? The review assumes it is intended because it is the newer and better-controlled design.
