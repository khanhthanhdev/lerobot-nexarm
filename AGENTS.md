This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (NexArm setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

This is the **Hiwonder NexArm** fork of LeRobot — focused exclusively on the Hiwonder NexArm 6-axis robot platform. Non-NexArm robot formats and motor drivers have been pruned to keep the codebase lean and focused. It integrates with Hugging Face Hub for model/dataset sharing.

**NexArm-specific modules:**

| Path                                       | What it does                                                                                 |
| ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `src/lerobot/motors/nexarm/`               | CommProtocol UART framing, 6-servo sync read/write, torque, bridge mode (CMD 56/68/96/97/98) |
| `src/lerobot/robots/nexarm_follower/`      | `NexArmFollowerConfig` + `NexArmFollower` — connect, observe, send_action                    |
| `src/lerobot/robots/nexarm_sim/`           | `NexArmSimConfig` + `NexArmSim` — single-arm MuJoCo simulation environment                   |
| `src/lerobot/robots/mobile_bi_nexarm_sim/` | `MobileBiNexArmSimConfig` + `MobileBiNexArmSim` — bimanual mobile robot simulation           |
| `src/lerobot/teleoperators/nexarm_leader/` | `NexArmLeaderConfig` + `NexArmLeader` — read positions, leader→follower joint mapping        |
| `src/lerobot/envs/nexarm.py`               | Gym environment for NexArm simulation                                                        |
| `sim/`                                     | MuJoCo MJCF, URDF description, Isaac Lab configs, meshes, and exporters                      |
| `examples/nexarm/`                         | Ready-to-run scripts for teleoperate, record, and rollout                                    |

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                # All tests
DEVICE=cuda make test-end-to-end                     # All E2E tests
uv run pre-commit run --all-files --show-diff-on-failure  # Full quality suite
```

## Linting and Formatting

```bash
uv run pre-commit install --install-hooks            # Once per clone; run checks before each commit
uv run pre-commit run --all-files --show-diff-on-failure  # Lint, format, type, spelling, and security checks
uv run ruff check . --no-fix                         # Check Python lint rules without modifying files
uv run ruff format . --check                         # Check Python formatting without modifying files
uv run ruff format .                                 # Apply Python formatting intentionally
```

Run the full pre-commit suite before submitting changes. `pyproject.toml` owns Ruff, Bandit, Typos, and Mypy settings; `.pre-commit-config.yaml` wires those tools into local hooks and CI.

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
