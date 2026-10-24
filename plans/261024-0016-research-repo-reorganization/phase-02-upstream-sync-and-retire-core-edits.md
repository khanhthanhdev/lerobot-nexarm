---
phase: 2
title: "Upstream sync and retire core edits"
status: pending
priority: P1
effort: "2-3d"
dependencies: [1]
---

# Phase 02: Upstream sync and retire core edits

## Goal
Merge current upstream LeRobot and shrink this fork's edits to upstream files down to what NexArm still needs before phase 08. New policy code is then written against the latest LeRobot API.

## Context
Measured against fork point `3e538352`: upstream has 332 newer commits touching 488 files. Outside the NexArm folders, this fork modifies 23 upstream files, deletes 2 and adds 5. The user does not intend to keep custom changes to train or record.

### What each edit is, and what happens to it

| Edit | Where | Disposition |
|---|---|---|
| Registration imports of `nexarm_*` robots/teleops | `lerobot_record/rollout/teleoperate/calibrate/replay/find_*`, `envs/__init__.py`, `envs/configs.py` | Keep for now; they disappear in phase 08 when the NexArm code becomes plugins |
| Pruned robots, motors, `lerobot_setup_can.py`, `lerobot_setup_motors.py` | deletions | Accept upstream's files (user decision): stop pruning |
| `gradient_accumulation_steps` | `configs/train.py`, `lerobot_train.py` | Take upstream's: upstream `d3793b7e` already has it |
| Per-stage success rates (`info["stage/<name>"]`) | `lerobot_eval.py` | Generic. Keep and open an upstream PR. Fallback if not merged when phase 08 starts: compute stage rates from the per-episode `eval_info.json` in a small NexArm tool, and drop the patch |
| `local_files_only` (no silent Hub fallback for incomplete local datasets) | `datasets/dataset_metadata.py`, `lerobot_dataset.py` | Generic. Check upstream first; otherwise keep and open an upstream PR. Fallback if not merged when phase 08 starts: check that local files are complete in the record wrapper and in `nexarm_service` before loading, and drop the patch |
| PyAV fallback when torchcodec is missing | `datasets/streaming_dataset.py`, `utils/import_utils.py` | Check upstream first; keep only if still needed |
| Lazy `groot` imports | `policies/groot/__init__.py` | Check upstream first; keep only if still needed |
| `--merge_root`, `--rerun_save_path`, `utils/recording_session.py`, `datasets/collection_merge.py` | `lerobot_record.py` | Move to a NexArm record wrapper with the same flags (user decision); `lerobot_record.py` returns to upstream |
| `datasets/mobile_nexarm_artifact.py`, `scripts/lerobot_nexarm_sim_benchmark.py`, `envs/nexarm.py` | new files | No conflict; they move in phase 08 |

## Files to Create / Modify
- Modify: the files in the table above, by merge resolution
- Modify: `pyproject.toml` and `uv.lock` (upstream dependency changes)
- Create: a NexArm record wrapper that calls `lerobot.scripts.lerobot_record.record` and then merges, keeping the same flags. It lives in `examples/nexarm/record.py` for now and moves in phase 08/09.
- Modify: `CLAUDE.md` "Scope of the fork": hardware layers are no longer pruned

## Tasks & Steps
1. `git switch -c chore/upstream-sync && git merge upstream/main`.
2. Resolve conflicts using the table. Where an edit is now upstream, take upstream's version.
3. Move the record extras into the wrapper. Keep the flag names and behaviour so `DATA_COLLECTION_GUIDE.md` commands only change the command name.
4. `uv lock`, then `uv sync --locked --extra all --extra test --extra dev`.
5. Open upstream PRs for the generic fixes (stage rates, `local_files_only`, others still needed). Record the PR links in this phase file. Any PR not merged when phase 08 starts uses the fallback in the table, so phase 08 never waits on upstream review.
6. Write down the remaining diff against upstream: `git diff upstream/main -- src/lerobot ':!src/lerobot/**/nexarm*'`. It should only contain the registration imports and any patches waiting on PRs.

## Verification
- Full test suite and NexArm tests on Linux are at least as good as the phase-01 baseline.
- `make test-nexarm-ete` passes.
- Sim dataset generation smoke test: `examples/nexarm/generate_stack_bowls_dataset.py` with 2 episodes.
- Real-arm smoke test before merging to `main`: teleoperate, then record 2 episodes (with merge into a collection), then replay 1 episode.
- `uv run pre-commit run --all-files` is clean.

## Risk / Rollback
High conflict count. The work stays on a branch until the real-arm smoke test passes. Roll back with `git reset --hard pre-reorg` on the branch.
