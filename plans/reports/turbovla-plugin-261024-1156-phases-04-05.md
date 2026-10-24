# TurboVLA phases 04 (parent side) and 05: execution report

Date: 2026-10-24. Branch `chore/upstream-sync` (not pushed). Host: Windows Server 2022, CPU only, no OpenGL.

## Commits

| Commit      | Phase | Summary                                                                        |
| ----------- | ----- | ------------------------------------------------------------------------------ |
| `140738fdb` | 04    | `build(turbovla): install TurboVLA as a pinned package instead of a submodule` |
| `84554a6ca` | 05    | `feat(policies): add the TurboVLA policy plugin for LeRobot inference`         |

This report is committed separately as `docs(plans)`.

## Phase 04: TurboVLA as a pinned dependency

- Root `pyproject.toml`: extra `turbovla = ["lerobot[transformers-dep]", "turbovla", "lerobot_policy_turbovla"]` (plugin added in phase 05); `[tool.uv.sources] turbovla = { git = "https://github.com/khanhthanhdev/TurboVLA", rev = "28bc1e7e8cf259aacee4fd928258caa669cbaaae" }`. `uv lock` resolves.
- Submodule removed: before `deinit`, `git -C TurboVLA status --short` was clean (only ignored `build/`, `turbovla.egg-info/`) and HEAD `28bc1e7` was on `origin/chore/lerobot-plugin-deps`. `git submodule deinit -f TurboVLA && git rm TurboVLA`; `.gitmodules` was empty afterwards and is deleted.
- Scripts: `setup/train/rollout/diagnose_turbovla.py` import the installed package; `--turbovla-repo` and all `sys.path` inserts (including `third_party/starvla_runtime`) are gone. `setup_turbovla.py` checks the installed package and no longer checks `accelerate` (not a TurboVLA dependency). `train_turbovla.py` gained `--sim-episodes` / `--real-episodes` (needed to train on the partly downloaded dataset).
- Docs and `scripts/nexarm/setup_server.sh`: `--extra smolvla --extra groot` replaced by `--extra turbovla` (README, run.md, AGENT_GUIDE, SERVER_TRAINING_GUIDE, pipeline.md, sim_training.md, turbovla_guide.md); no `git submodule` / `--recurse-submodules` instructions remain. AGENTS.md (CLAUDE.md) section rewritten. Local TurboVLA development (temporary `path = "../TurboVLA", editable = true`, never committed; check `python -I -c "import experiments"`, fall back to `--config-settings editable_mode=strict`) is in `docs/source/nexarm/turbovla_guide.md`.

Verification:

| Check                                                                                                                                                                                                                | Result                                                       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `uv sync --locked --extra turbovla --extra test --extra dev --inexact`                                                                                                                                               | ok                                                           |
| `import turbovla` location                                                                                                                                                                                           | `.venv\Lib\site-packages\turbovla\__init__.py`               |
| `python -I -c "import experiments"` (also `deployment`)                                                                                                                                                              | `ModuleNotFoundError`                                        |
| `setup_turbovla.py --check-only`                                                                                                                                                                                     | passes; no downloads run (DINOv3 is gated)                   |
| 20-step `train_turbovla.py`, CPU, `thanhkt/nexarm_stack_bowls` episodes 143-152 (3072 frames), `facebook/dinov2-with-registers-small` + `prajjwal1/bert-tiny` weights with the bert-base tokenizer, batch 8, lr 3e-4 | loss logged every 4 steps: 0.545, 0.514, 0.482, 0.411, 0.381 |

## Phase 05: TurboVLA policy plugin, inference

`policies/lerobot_policy_turbovla/` (deps `lerobot`, `turbovla`), workspace member pulled only by the `turbovla` extra.

- `configuration_turbovla.py`: `@PreTrainedConfig.register_subclass("turbovla")` `TurboVLAConfig`; defaults equal `rollout_turbovla.py` (chunk 16, `n_action_steps` 8, BERT-base / DINOv3 ViT-B, 224 px, min-max STATE/ACTION, IDENTITY VISUAL, gripper snap at 1800 raw on index 5). `camera_keys` keeps the trained view order.
- `modeling_turbovla.py`: `TurboVLAPolicy` builds the model from `turbovla.models` (`TurboVLAConfig as TurboVLAModelConfig`); `reset`, `select_action` (queue of `n_action_steps`), `predict_action_chunk`; `from_pretrained` defaults to `strict=True`; `forward` raises `NotImplementedError` pointing at `train_turbovla.py` + `convert_checkpoint` (no phase number in code, per the stable-code-artifacts rule).
- `processor_turbovla.py`: rename, batch, LeRobot normalizer, `turbovla_image_processor` (float frame -> uint8 by rounding -> PIL -> backbone `AutoImageProcessor` with the square resize), device; unnormalizer, `turbovla_gripper_snap`, cpu. Task stays text (the model tokenizes it).
- `convert_checkpoint.py` (`python -m lerobot_policy_turbovla.convert_checkpoint`): old checkpoint + `config.json` + `stats_turbovla.json` -> `config.json`, `model.safetensors`, `policy_{pre,post}processor*.json/.safetensors`. Checked on the phase-04 smoke checkpoint.
- `rollout_turbovla.py` gained `--vision-path` (the vision backbone was hard-coded to DINOv3).

Verification:

| Check                                                                                                                                                                                                                                                            | Result                                                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Installed name / `get_known_choices()` after `register_third_party_plugins()`                                                                                                                                                                                    | `lerobot_policy_turbovla`; `turbovla` present; `get_policy_class("turbovla")` is `TurboVLAPolicy`                    |
| Parity, seeded random TurboVLA (BERT-tiny, DINOv2-small-reg), saved by `train_turbovla.save_checkpoint`, real `TurboVLAPolicyRunner` vs convert + `from_pretrained` + `make_pre_post_processors(pretrained_path=...)`, 20 frames of episode 143, fp32, eval, CPU | max abs action diff **0.0** (tolerance 1e-4); snapped gripper equal; first 8 `select_action` outputs equal the chunk |
| Same parity on the 20-step smoke-trained checkpoint (`TURBOVLA_PARITY_CHECKPOINT`)                                                                                                                                                                               | max abs diff 0.0                                                                                                     |
| Mutations (then reverted): reversed camera order; pixel values -1                                                                                                                                                                                                | test fails (max diff 2.9 and 9.0)                                                                                    |
| Real trained TurboVLA checkpoint on this machine                                                                                                                                                                                                                 | none found (`outputs/`, `pretrained/`, caches)                                                                       |
| `uv run pytest policies -q`                                                                                                                                                                                                                                      | 22 passed, 2 skipped (the `trained` parity param without the env var)                                                |
| `tests/scripts/test_{rollout,train}_turbovla.py`                                                                                                                                                                                                                 | 38 passed                                                                                                            |
| `uv run pytest tests -k nexarm -q --ignore=tests/utils/test_process.py`                                                                                                                                                                                          | 338 passed, 10 failed (known OpenGL/env failures), 5 skipped                                                         |
| Plain `uv sync --locked --extra test` in a scratch venv                                                                                                                                                                                                          | neither `turbovla` nor the plugin installed; plugin registration works; `pytest policies`: 14 passed, 2 skipped      |
| `pre-commit run --files <changed files>`                                                                                                                                                                                                                         | clean except the known `keyboard_input.py` termios mypy errors                                                       |

## Open items

- `lerobot-eval --policy.path=<converted> --env.type=nexarm --env.task=stack_bowls --eval.n_episodes=50` on Linux (no OpenGL here).
- 10 real-arm `lerobot-rollout` episodes and one desktop-app eval session.
- Fresh Linux clone + `uv sync --extra turbovla` without submodule commands.
- Phase 06 note: training-time `to_pil` in `train_turbovla.py` truncates `x * 255` while the plugin rounds. For uint8-derived frames both give the same bytes; check again if training data ever is not uint8-derived.

## Notes for the user

- `C:\nxv` is a scratch venv from the plain-sync check; the safety check refused its deletion. Delete it by hand.
- `.git/modules/TurboVLA` still holds the old submodule's git directory (local branches, any stashes).
- `SERVER_TRAINING_GUIDE.md` still lists "Accelerate" among what the setup installs.
- The `turbovla` extra names `lerobot_policy_turbovla`, which is not on PyPI, so a published `lerobot` wheel's metadata would reference it (bspline uses a dependency group instead; the task asked for an extra).
- Converted policies keep `local_files_only=True` unless converted with `--allow-hf-download`; on an uncached machine the backbones must be fetched first.
- Loading a converted policy still loads both backbones' configs, tokenizer and image processor (and pretrained weights) before the checkpoint replaces the weights.
