---
phase: 1
title: "Housekeeping and baseline"
status: completed
priority: P1
effort: "2h"
dependencies: []
---

# Phase 01: Housekeeping and baseline

## Goal
Start from a clean, tagged state with a recorded test baseline, so every later phase can be compared and rolled back.

## Context
- The `TurboVLA` submodule is checked out at `98ee1be3`, but the repo records `5082ece6`.
- `bspline-policy` is a submodule that was never initialized.
- There are six untracked reports in `plans/reports/` plus this plan.
- There is no `upstream` remote. The fork branched at `3e538352` (2026-07-08).
- On this Windows box, `tests -k nexarm` gives 338 passed and 10 failed. All 10 are environment problems: MuJoCo has no OpenGL here, and `nexarm_scene.json` gets CRLF line endings. `tests/utils/test_process.py` cannot even be collected on Windows (`signal.SIGHUP`).

## Files to Create / Modify
- Modify: `.gitmodules` and the `TurboVLA` gitlink (commit the current pointer)
- Create: `plans/reports/baseline-261024-<time>-nexarm-tests.md`
- Commit: the untracked `plans/reports/*.md` and this plan folder

## Tasks & Steps
1. Check that `98ee1be3` exists on the TurboVLA remote: `git -C TurboVLA branch -r --contains 98ee1be3`. If it does, commit the pointer (`chore(turbovla): bump submodule`). If it does not, push it from the TurboVLA checkout first.
2. Commit the untracked reports and this plan (`docs(plans): ...`).
3. Add the upstream remote: `git remote add upstream https://github.com/huggingface/lerobot.git && git fetch upstream`.
4. Tag the rollback point: `git tag pre-reorg && git push origin pre-reorg`.
5. Run the baseline on the Linux GPU server, where the MuJoCo rendering tests can run:
   - `uv run pytest tests -k nexarm -q`
   - `uv run pytest tests/nexarm_service -q`
   - `make test-nexarm-ete`
   Record the pass/fail lists in the baseline report, and note the Windows-only failures separately.

## Verification
- `git status` shows no untracked reports and no submodule drift.
- `git remote -v` lists `upstream`.
- The baseline report exists and has results from both Linux and Windows.

## Risk / Rollback
None. This phase only adds commits, a remote and a tag.
