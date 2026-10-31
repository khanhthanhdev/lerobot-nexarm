---
title: "Phase 5: D1b guards and documentation"
status: done
priority: P2
effort: 1h
dependencies: [1, 3]
---

# Phase 5: D1b guards and documentation

## Context

- Review §1 (dataset path), §2 (closed question), §3 (decision rule): [`review-261031-0736-plan-bind-methodology.md`](../reports/review-261031-0736-plan-bind-methodology.md)
- `experiments/261029-plan-bind-d1b-seeds/run_all.sh`: `get_data` (checks `total_tasks` only), `run_full` (`DATA` defaults to `outputs/datasets/nexarm_stack_bowls_red_blue_black_1k`)
- `experiments/261029-plan-bind-d1b-seeds/README.md`, `experiments/261027-plan-bind-two-step/README.md`, `policies/lerobot_policy_bspline/README.md`
- `plans/README.md` (open-work table)

## Overview

Three small changes that make the D1b run reproducible and pre-registered: a guard so D1b cannot silently train on a different dataset than D1, the RQ1 decision rule and the new control written into the D1b README before the run, and the closure of D1's open question. This phase edits `run_all.sh` after phase 1 has merged its changes to the same file.

## Requirements

- [x] `get_data` checks `total_episodes` against `EPISODES` (500) as well as `total_tasks == 1`, and dies with the counts when they differ.
- [x] The default `DATA` folder name says what it holds: `outputs/datasets/nexarm_stack_bowls_red_blue_black_500`. No D1b run exists yet on any machine, so the rename breaks nothing.
- [x] D1b README: add `onestep_pos` to the controls table with the parameter delta (114 numbers) and the ramp initialisation; add the seed trade-off (3 seeds, about 12 GPU-hours, or 1 seed); state the decision rule: "If `onestep_pos` closes most of Plan-Bind's lead at latency 0 across seeds, RQ1 is reframed to the interface (RQ2, RQ3). If the lead holds against `onestep_pos`, RQ1 stands with a matched-input control." Replace "pooled success with Wilson intervals" with the per-seed paired description from phase 3. Update the time estimate.
- [x] D1 README: under "Unresolved questions", close the 18/50 question with the fact (by the success flag: 26 of 50 agreeing episodes, 12 and 12 discordant; videos differ; the review's 23/12/15 counted stage flags) and keep the results table as recorded.
- [x] Plugin README: document `planbind_onestep_input` in the modes table (phase 1) and the acceleration limit with its position-wins caveat (phase 2), if those phases have not already done so.
- [ ] `plans/README.md`: when the plan closes, move its row from "Open work" to "Closed" (the row was added when the plan was created).

## Implementation steps

1. Edit `get_data` to read `total_episodes` and `total_tasks` in one `python -c` call and compare both.
2. Rename the default in `run_full`; update the comment.
3. Write the README changes; keep D1's numbers untouched.

## Todo

- [x] `get_data` guard and `DATA` rename
- [x] D1b README: control, trade-off, decision rule, statistics, time
- [x] D1 README: closed question
- [x] Plugin README check

## Verification

Windows: `bash -n experiments/261029-plan-bind-d1b-seeds/run_all.sh` through PowerShell; `uv run pre-commit run --all-files --show-diff-on-failure` (markdown and shell hooks).

Server: `run_all.sh smoke` still passes after the rename (the smoke run uses its own `outputs/smoke/data`); a deliberate run with `DATA` pointing at a 4-episode smoke dataset and `MODE=full` must die with the episode counts.

## Success criteria

- The D1b README states the control, the decision rule and the statistics before any full run starts.
- A wrong dataset stops the run with a message that names both counts.
