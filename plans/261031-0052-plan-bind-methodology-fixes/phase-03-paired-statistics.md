---
title: "Phase 3: Paired statistics for D1b"
status: done
priority: P1
effort: 2h
dependencies: [1]
---

# Phase 3: Paired statistics for D1b

## Context

- Review §2 and §5: [`review-261031-0736-plan-bind-methodology.md`](../reports/review-261031-0736-plan-bind-methodology.md)
- Code: `experiments/261029-plan-bind-d1b-seeds/summarize.py` (reads `overall` only; pools wins across seeds), `experiments/261027-plan-bind-two-step/summarize.py` (D1; no paired test), `experiments/261028-plan-bind-d2-latency/summarize.py` (has `paired()` with "+a/−b" counts; **not touched**, D2 may be running)
- Data: every `eval_info.json` from `lerobot-nexarm-eval` carries per-episode outcomes at `per_task[0]["metrics"]["stages"]`, a list of `{"success": bool, "stages": {...}}` in episode order (verified on D1's 18 cells: 50 entries each)
- Local fixture: `experiments/261027-plan-bind-two-step/outputs/full/eval/<cell>/eval_info.json` (18 cells)
- Test layout: script tests live in `tests/scripts/`; `pyproject.toml` sets `pythonpath = ["."]`

## Overview

Every D1b seed evaluates the same 100 episodes (eval seed 1000), so outcomes are correlated across seeds and a Wilson interval on the 300 pooled episodes is too narrow. The honest unit is the per-seed paired comparison on shared episodes: discordant counts and an exact McNemar test, as in the review's table. This phase adds a small statistics module next to D1 (D1b already sources D1's `common.sh`, so cross-folder reuse is the existing pattern) and rewrites D1b's summarizer around it.

## Requirements

- [x] New module `experiments/261027-plan-bind-two-step/episode_stats.py` with pure functions: `episode_successes(eval_info: dict) -> list[bool] | None`, `discordant(a, b) -> tuple[int, int]` (only-a, only-b), `mcnemar_exact(b, c) -> float` (two-sided, binomial), `wilson(wins, n) -> tuple[float, float]`, and `paired_difference(a, b) -> dict` returning the difference in points, the discordant counts and the p-value.
- [x] D1b's `summarize.py` imports the module by file path (`importlib.util.spec_from_file_location`, since the folder name is not an importable package name), reads per-episode outcomes, and reports:
  - main table: per-seed success with a per-seed Wilson interval and the mean over seeds, replacing the pooled interval;
  - "Plan-Bind minus one-step": per seed and setting, the difference in points, `only Plan-Bind / only one-step`, McNemar p; the same rows against `onestep_pos`; the mean paired difference over seeds;
  - the control table extended with `onestep_pos` (`VARIANTS` and the controls tuple).
- [x] When a cell lacks per-episode outcomes (older `eval_info.json`), fall back to the aggregate with a visible `(unpaired)` mark instead of failing.
- [x] D1's `summarize.py` gains a McNemar column in its RQ1 table (optional, small; the D1 README's table is a dated record and is not regenerated).
- [x] Unit test `tests/scripts/test_plan_bind_episode_stats.py`: known values (`mcnemar_exact(1, 16) ≈ 0.0003`, `mcnemar_exact(0, 6) = 0.03125`, `mcnemar_exact(0, 0) = 1`), `episode_successes` on a minimal dict and on a dict without stages (`None`), and a summarizer run on a fixture built from two D1 cells copied with `_s1000` suffixes.

## Implementation steps

1. Write `episode_stats.py` with docstrings; no third-party imports (`math.comb` for the binomial).
2. Rewrite `load_cells` in D1b's `summarize.py` to keep the `eval_info.json` path and the per-episode list; compute `wins` from the list when present, else from `pc_success` as now.
3. Replace `pooled()` with `per_seed()` (per-seed value and interval, mean over seeds).
4. Build the paired table from `paired_difference` per seed and setting, for `onestep` and `onestep_pos`.
5. Add the McNemar column to D1's `summarize.py` RQ1 table using the same module (same folder, plain import).
6. Add the unit test; build the fixture in a `tmp_path` by copying `onestep_L0_none/eval_info.json` to `onestep_L0_none_s1000/` and `planbind_L0_none/eval_info.json` to `planbind_L0_none_s1000/`; assert the output contains `29` and `1` discordant counts and a p-value below 0.001. Skip the fixture test when the D1 outputs are absent (CI has no outputs).

## Todo

- [x] `episode_stats.py`
- [x] D1b summarizer rewrite
- [x] D1 summarizer McNemar column
- [x] Unit test and fixture
- [x] Ruff clean

## Verification

Windows:

- `uv run pytest tests/scripts/test_plan_bind_episode_stats.py -svv`
- `uv run python experiments/261027-plan-bind-two-step/summarize.py experiments/261027-plan-bind-two-step/outputs/full/eval` reproduces the review's table (0 ms / none: 29 vs 1, p < 0.0001; 300 ms / none: 6 vs 0, p = 0.0312).
- D1b summarizer on the renamed fixture prints per-seed rows without error.
- `uv run ruff check experiments tests/scripts --no-fix`

No server step.

## Success criteria

- D1b's `results.md` layout is decided before the run and verified on real D1 data.
- No pooled interval across seeds remains in the output.

## Implementation notes

- Per-episode success comes from `per_task[0]["metrics"]["successes"]`, the env's success flag behind `pc_success`. The `stages` entries hold stage flags only, and "all stages reached" differs from the success flag by 2 to 3 episodes in some one-step cells. The review's table used the stage flags, so its counts differ from the summarizer's in four of eight rows (for example 300 ms, naive: 8 against 0, p = 0.0078, not 6 against 0). The 0 ms rows agree (16/1 and 29/1). Every cell stays significant.

## Risks

- `per_task[0]` assumes one task per eval; D1b uses one stacking order, so one task. Guard with a clear error if several tasks appear.
