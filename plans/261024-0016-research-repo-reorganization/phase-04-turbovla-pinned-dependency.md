---
phase: 4
title: "TurboVLA as a pinned dependency"
status: completed
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

## Scouted facts (2026-10-24, TurboVLA at `98ee1be3`)
- `turbovla/` imports none of `vla_adapter`, `starVLA`, `experiments` or `deployment`. Only the console scripts in TurboVLA's `pyproject.toml` (`turbovla-libero-eval`, `turbovla-robotwin-train/eval`) reference them. So the package list can be narrowed to `turbovla*`, with those scripts moved to repo-only use or an extra.
- `third_party/vla_adapter` and `third_party/starvla_runtime` are ordinary tracked files (46 files, no gitlinks), and TurboVLA has no `.gitmodules`. A git install needs no nested submodule fetch.
- Core pins to change: `numpy>=1.26,<2`, `transformers>=4.56,<5`. `robosuite==1.4.1` and `tensorflow` are already in the `libero` extra, not core.
- `turbovla/data/libero_rlds.py` and `turbovla/data/mixed_suite.py` import `tensorflow`, but tensorflow is not a core dependency. Keep those imports lazy, or keep them out of the import path used by `turbovla.models`.
- `turbovla.models.configuration.TurboVLAConfig` is a dataclass made of `text`, `vision`, `interaction` and `action` sub-configs (`ActionHeadConfig`: `action_dim`, `state_dim`, `horizon`, ...). Alias it as `TurboVLAModelConfig` in phase 05.

## Execution notes (2026-10-24)
- TurboVLA side: branch `chore/lerobot-plugin-deps`, commit `28bc1e7e8cf259aacee4fd928258caa669cbaaae` (on `5082ece`, Transformers-5 BERT fix). Core pins `numpy>=2.0,<2.3`, `transformers>=5.4,<5.6`; only `turbovla*` is installed.
- Here: `turbovla = ["lerobot[transformers-dep]", "turbovla"]` extra, git source pinned to that sha; `uv lock` resolves (347 packages). The submodule, its gitlink and the now-empty `.gitmodules` are gone; the old checkout was clean and its HEAD is on `origin/chore/lerobot-plugin-deps` (its git dir stays in `.git/modules/TurboVLA`).
- Scripts import the installed package; `--turbovla-repo` and the `sys.path` inserts are removed from all four. `setup_turbovla.py` checks the installed package and no longer requires `accelerate` (not a TurboVLA dependency). `train_turbovla.py` gained `--sim-episodes` / `--real-episodes` to train on part of a dataset. Docs and `setup_server.sh` now use `--extra turbovla` instead of `--extra smolvla --extra groot`; local TurboVLA development is documented in `docs/source/nexarm/turbovla_guide.md`.
- Verified on Windows: `uv sync --locked --extra turbovla --extra test --extra dev --inexact`; `turbovla.__file__` is in `.venv/Lib/site-packages`; `python -I -c "import experiments"` fails; `setup_turbovla.py --check-only` passes (no downloads: DINOv3 is gated). 20-step CPU smoke run of `train_turbovla.py` on `thanhkt/nexarm_stack_bowls` episodes 143-152 with `facebook/dinov2-with-registers-small` and `prajjwal1/bert-tiny` (+ bert-base tokenizer), batch 8, lr 3e-4: logged loss 0.545, 0.514, 0.482, 0.411, 0.381.
- Not verified here: a fresh clone on Linux (`uv sync --extra turbovla` without submodule commands).
