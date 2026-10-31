---
title: "Plan-Bind methodology fixes before D1b"
description: "Fix the confounded one-step control, the teacher-forced Bind, the paired statistics and the missing acceleration bound found by the 2026-10-31 review, so D1b answers RQ1 cleanly."
status: in-progress
priority: P1
effort: 11h
branch: feat/plan-bind
tags: [experimental, bugfix, refactor]
blockedBy: []
blocks: []
created: 2026-10-31
---

# Plan-Bind methodology fixes before D1b

## Overview

The review [`review-261031-0736-plan-bind-methodology.md`](../reports/review-261031-0736-plan-bind-methodology.md) found that D1's one-step control feeds an all-zeros sequence to a UNet that has no positional encoding along the control-point axis, while Bind receives the lifted plan. D1's +16 to +56 point gap is statistically solid (McNemar p ≤ 0.03 in every cell) but may partly measure "position-bearing input" rather than "two-step structure". D1b, as written, does not test this. This plan adds the missing control, measures the plan error Bind is never trained on, makes the D1b statistics paired, implements the acceleration bound RQ2 claims, and guards the D1b dataset path. It changes code and experiment scripts only; the D1 results stay as recorded.

Experiments run elsewhere: unit tests, ruff and the summarizers run on this Windows box; anything touching MuJoCo or a checkpoint runs on `vishc-server-2` (GPU 6 free, `uv` in `~/.local/bin`, no `.venv` yet) or marinelab.

## Goals

| #   | Goal                                                                                                                      | Priority |
| --- | ------------------------------------------------------------------------------------------------------------------------- | -------- |
| 1   | D1b trains a one-step control whose input carries sequence position, with a pre-registered decision rule for RQ1          | P1       |
| 2   | D1b reports per-seed paired differences (discordant counts, exact McNemar) instead of pooled Wilson intervals             | P1       |
| 3   | The plan error and Bind's trigger statistic are measured on D1's checkpoint, so D2's thresholds are chosen from data      | P2       |
| 4   | `project` bounds acceleration as well as position and velocity, and states what happens at a joint limit                  | P3       |
| 5   | D1b refuses a dataset that is not D1's 500-episode set, and the READMEs record the new control and the closed D1 question | P2       |

## Phases

| #   | Phase                                                                                | Depends on | Status                              |
| --- | ------------------------------------------------------------------------------------ | ---------- | ----------------------------------- |
| 1   | [Position-bearing one-step control](./phase-01-position-bearing-one-step-control.md) | —          | Code done; server smoke run pending |
| 2   | [Acceleration bound in `project`](./phase-02-acceleration-bound.md)                  | —          | Done                                |
| 3   | [Paired statistics for D1b](./phase-03-paired-statistics.md)                         | 1          | Done                                |
| 4   | [Plan-error measurement](./phase-04-plan-error-measurement.md)                       | —          | Script done; server run pending     |
| 5   | [D1b guards and documentation](./phase-05-d1b-guards-and-documentation.md)           | 1, 3       | Done                                |

File ownership: phase 1 owns `experiments/261029-plan-bind-d1b-seeds/run_all.sh`; phase 5 edits it only after phase 1 is merged. Phase 3 owns `experiments/261029-plan-bind-d1b-seeds/summarize.py`, including the new variant name from phase 1. Phases 1 and 2 both touch `configuration_bspline.py` and `planbind.py` in disjoint places (one new field each); do them in order or in one branch.

## Decisions taken in this plan

- The new control is `onestep` with a **learned input tensor** of shape `(n_fine, action_dim)` (114 numbers for D1's grid), initialised to a linear ramp so position is in the input from step 0. A one-step generative head from noise, the proposal's original control, is deferred (see phase 1).
- The acceleration bound uses second differences of the control points over `spacing²`. When the position wall makes the three intervals empty, **position and velocity win** (a point inside the box always meets the velocity interval) and only the acceleration bound is violated at that point; the plugin README says so, because the proposal's "exactly by construction" only holds away from the joint limits.
- D2's `summarize.py` is **not** touched: D2 may be mid-run on marinelab. The shared statistics module is new code that D1b uses; D2 can adopt it after its run.

## Success Criteria

- [ ] `uv run pytest policies/lerobot_policy_bspline/tests -svv` passes on Windows with the new config fields, and D1's checkpoints still load with default settings.
- [ ] `experiments/261029-plan-bind-d1b-seeds/run_all.sh smoke` on `vishc-server-2` trains and evaluates `onestep_pos` next to the other variants.
- [ ] D1b's `results.md` shows, per seed, discordant counts and exact McNemar p for Plan-Bind against `onestep` and `onestep_pos`.
- [ ] `plan_error.py` writes a JSON and a table that place τ = 0.3, 0.5 and 1.0 on the percentiles of Bind's trigger statistic.
- [ ] D1b's README states the RQ1 decision rule before the run starts.
- [ ] `uv run pre-commit run --all-files --show-diff-on-failure` is clean.

## Dependencies and notes

- D1's checkpoints and dataset are in the private Hub repos `thanhkt/plan-bind-d1` and `thanhkt/nexarm-stack-bowls-sim-red-blue-black-500`; the server needs `uv run hf auth login` once.
- The steering redesign on `vishc-server-2` (`~/code/policy-steering`, uncommitted) is outside this plan. Committing it is a user action noted in the review.
- Not in this plan: re-running D1, the real-arm RQ2 run, temporal-ensembling baselines.

<!-- slug: plan-bind-methodology-fixes -->
