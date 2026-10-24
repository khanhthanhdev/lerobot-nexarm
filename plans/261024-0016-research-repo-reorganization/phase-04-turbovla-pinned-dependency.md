---
phase: 4
title: "TurboVLA as a pinned dependency"
status: pending
priority: P1
effort: "0.5-1d"
dependencies: [3]
---

# Phase 04: TurboVLA as a pinned dependency

## Goal
Replace the `TurboVLA` submodule with a pinned `turbovla` package dependency, so a fresh clone gets TurboVLA from `uv sync`.

## Context
- TurboVLA's `pyproject.toml` (package `turbovla`) pins `numpy>=1.26,<2` and `transformers>=4.56,<5`. This repo needs `numpy>=2.0,<2.3` and `transformers>=5.4,<5.6`, so `uv lock` will fail until TurboVLA's pins change. Fork commit `e13ab44c` says TurboVLA already works with Transformers 5, so the pins look stale.
- Today the TurboVLA scripts reach the package by adding the repo-root `TurboVLA/` folder to `sys.path` (`examples/nexarm/setup_turbovla.py:18`), not by installing it.

- TurboVLA's `[tool.setuptools.packages.find]` installs top-level packages `turbovla`, `experiments`, `deployment`, `starVLA` and `vla_adapter`, from `.`, `third_party/vla_adapter` and `third_party/starvla_runtime`. A top-level package named `experiments` would shadow or collide with other code in site-packages, and the `third_party/` folders must be present in a git install.

## Files to Create / Modify
- In the TurboVLA repo: limit the installed packages to what the model needs (`turbovla*` plus any `vla_adapter`/`starVLA` modules that `turbovla.models` imports). Keep `experiments` and `deployment` as repo-only scripts, or rename them under a `turbovla_` prefix;
- In the TurboVLA repo: `pyproject.toml` pins; move LIBERO/RoboTwin-only dependencies into extras if they force old numpy (for example `tensorflow`, `robosuite==1.4.1`)
- Modify: `pyproject.toml`: add a `turbovla` extra, with `[tool.uv.sources] turbovla = { git = "https://github.com/khanhthanhdev/TurboVLA", rev = "<sha>" }`
- Delete: the `TurboVLA` submodule entry in `.gitmodules` and its gitlink
- Modify: `examples/nexarm/setup_turbovla.py`, `train_turbovla.py`, `rollout_turbovla.py`, `diagnose_turbovla.py`: import the installed package instead of using the `sys.path` hack
- Modify: `scripts/nexarm/setup_server.sh` and other places that run `git submodule update` for TurboVLA

## Tasks & Steps
1. In TurboVLA: set `numpy>=2`, `transformers>=5.4,<5.6`; narrow the package list; check that `third_party/` code needed at import time is real files, not nested submodules. If it is nested submodules, confirm uv's git install fetches them, or vendor them. Run its own tests or a forward pass; commit and push. Record the sha.
2. Here: add the git source at that sha. `uv lock`.
3. `git submodule deinit TurboVLA && git rm TurboVLA`.
4. Update the four scripts and the setup script. Document how to develop TurboVLA locally: temporarily set `turbovla = { path = "../TurboVLA", editable = true }` in `[tool.uv.sources]`, and never commit that change.

## Verification
- `uv lock` resolves; `uv run python -c "import turbovla; print(turbovla.__file__)"` points into the venv.
- `uv run python examples/nexarm/setup_turbovla.py` (environment check part) passes.
- 20-step smoke run of `examples/nexarm/train_turbovla.py` on a small NexArm dataset, with the loss going down.
- A fresh clone plus `uv sync --extra turbovla` works without any submodule command, and `python -c "import experiments"` fails in that venv (no stray top-level package).

## Risk / Rollback
If TurboVLA does not work with numpy 2, keep the submodule for now and solve the blocker inside TurboVLA first. Phases 05-06 wait on this; phase 07 does not.
