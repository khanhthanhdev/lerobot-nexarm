---
phase: 3
title: "Workspace and experiments layout"
status: completed
priority: P1
effort: "3h"
dependencies: [2]
---

# Phase 03: Workspace and experiments layout

## Goal
Create the uv workspace that the policy and NexArm plugin packages install from, and a fixed shape for experiments.

## Files to Create / Modify
- Modify: `pyproject.toml`: add `[tool.uv.workspace] members = ["policies/*", "packages/*"]`
- Create: `experiments/README.md`: folder convention and README template
- Create: `experiments/_template/` with `README.md` (hypothesis, dataset repo_id and revision, command, checkpoint, results table, notes) and `train.yaml`
- Modify: `.gitignore`: ignore `experiments/*/outputs/` and `outputs/`

## Tasks & Steps
1. Add the workspace table. With no members yet, check that `uv lock` still succeeds. If uv rejects an empty glob, add the table in phase 05 together with the first member.
2. Write the experiment convention:
   - one folder per experiment, named `YYMMDD-<slug>`;
   - configs are draccus YAML passed with `--config_path`;
   - results report n_episodes (at least 50 in sim, at least 10 on the real arm), the success rate and the checkpoint Hub id;
   - training outputs stay out of git, and runs log to W&B (`--wandb.enable=true`).
3. Leave the existing YAMLs in `examples/nexarm/` where they are; phase 09 moves them.

## Verification
- `uv lock` and `uv sync --locked` succeed.
- `experiments/_template/train.yaml` loads: `uv run lerobot-train --config_path experiments/_template/train.yaml --help`.

## Risk / Rollback
Low. Revert the commit.

## Execution notes (2026-10-24, branch `chore/upstream-sync`)
- uv 0.13.0 accepts `members = ["policies/*", "packages/*"]` with no members yet; `uv lock` and `uv sync --locked` succeed with no lockfile change.
- `.gitignore` needed no change: its existing `outputs/` rule (no leading slash) already ignores `outputs/` at any depth, including `experiments/*/outputs/` (checked with `git check-ignore`).
- `experiments/_template/train.yaml` parses into `TrainPipelineConfig` (policy `act`, W&B on), and `lerobot-train --config_path experiments/_template/train.yaml --help` runs.
