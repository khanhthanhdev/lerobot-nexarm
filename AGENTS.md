This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (NexArm setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

This is the **Hiwonder NexArm** fork of LeRobot — focused exclusively on the Hiwonder NexArm 6-axis robot platform. It tracks upstream LeRobot: upstream's robots, teleoperators and motor drivers are kept alongside the NexArm ones. It integrates with Hugging Face Hub for model/dataset sharing.

**NexArm-specific modules:**

NexArm code is LeRobot plugins in `packages/` (uv workspace members, installed by plain `uv sync`). LeRobot's CLIs import them through `register_third_party_plugins()`; the registered type names are unchanged. Paths below are under `packages/<package>/src/<package>/`.

| Path                                                           | What it does                                                                                                                |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `lerobot_robot_nexarm/motors/`                                 | CommProtocol UART framing, 6-servo sync read/write, torque, bridge mode (CMD 56/68/96/97/98)                                |
| `lerobot_robot_nexarm/nexarm_follower/`                        | `NexArmFollowerConfig` + `NexArmFollower` (`nexarm_follower`) — connect, observe, send_action                               |
| `lerobot_robot_nexarm/nexarm_sim/`                             | `NexArmSimConfig` + `NexArmSim` (`nexarm_sim`) — single-arm MuJoCo simulation                                               |
| `lerobot_robot_nexarm/mobile_bi_nexarm_sim/`                   | `MobileBiNexArmSimConfig` + `MobileBiNexArmSim` (`mobile_bi_nexarm_sim`) — bimanual mobile sim                              |
| `lerobot_robot_nexarm/{collection_merge,recording_session}.py` | Collection recording tools behind `examples/nexarm/record_collection.py`, incl. the manual-control keys                     |
| `lerobot_robot_nexarm/{local_dataset,rerun_recording}.py`      | Local-only dataset loading; `.rrd` recording of the Rerun stream and `lerobot-nexarm-rollout`                               |
| `lerobot_teleoperator_nexarm/`                                 | `NexArmLeaderConfig` + `NexArmLeader` (`nexarm_leader`) — read positions, leader→follower mapping                           |
| `lerobot_env_nexarm/`                                          | `NexArmEnv` (`--env.type=nexarm`), the gym envs, `lerobot-nexarm-eval` (per-stage rates) and `lerobot-nexarm-sim-benchmark` |
| `nexarm_service/`                                              | Desktop-app engine (`lerobot-nexarm-service`, `--extra service`)                                                            |
| `sim/`                                                         | MuJoCo MJCF, URDF description, Isaac Lab configs, meshes, and exporters                                                     |
| `examples/nexarm/`                                             | Ready-to-run scripts for teleoperate, record, and rollout                                                                   |

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
uv run pytest tests/robots/test_nexarm_follower.py::test_config_defaults -svv  # Single test
uv run pytest tests -k nexarm -svv                   # All NexArm-related tests (hardware is mocked)
make test-nexarm-ete                                 # NexArm sim gym integration test
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

## Scope of the Fork

The hardware layers (`robots/`, `motors/`, `teleoperators/`) are no longer pruned: upstream's robots, motors and teleoperators are back, and the NexArm types are registered next to them. `src/lerobot/` matches upstream: the NexArm types register through the plugins, and the former core patches live in the plugins (`local_dataset`, `rerun_recording`, the manual-control keys in `recording_session`, `lerobot-nexarm-eval` for per-stage rates). Keep it that way; add NexArm behavior to `packages/`, not to `src/lerobot/`. The NexArm collection flags for recording (`--manual_control`, `--recording_log`, `--root_repo_id`, `--merge_root`, `--rerun_save_path`) live in the wrapper `examples/nexarm/record_collection.py`, not in `lerobot-record`. Upstream policies (`act`, `diffusion`, `smolvla`, `pi0*`, `groot`, ...) and simulation envs (`aloha`, `pusht`, `libero`, `metaworld`, ...) are still registered, and the `make test-end-to-end` targets still train on upstream `aloha` datasets.

## TurboVLA and B-spline Policy

There are no git submodules. TurboVLA is the `turbovla` package from its own repo (<https://github.com/khanhthanhdev/TurboVLA>), pinned by commit in `[tool.uv.sources]` and installed by `uv sync --extra turbovla`. For local TurboVLA development, temporarily point that source at a checkout (`turbovla = { path = "../TurboVLA", editable = true }`) and never commit the change; see `docs/source/nexarm/turbovla_guide.md`. The same extra installs the workspace plugin `policies/lerobot_policy_turbovla/` (`--policy.type=turbovla`; it trains with `lerobot-train` and runs with `lerobot-eval`/`lerobot-rollout`; `convert_checkpoint.py` turns a checkpoint from the former `train_turbovla.py` into a LeRobot policy folder). B-spline Policy is the workspace plugin `policies/lerobot_policy_bspline/` (`--policy.type=bspline`), installed by plain `uv sync`. The TurboVLA workflow lives in `examples/nexarm/` (`setup_turbovla.py`) and imports the installed package. Multi-GPU and Slurm launchers are in `scripts/nexarm/`.

## User-Facing Guides

Point to these rather than re-deriving workflows: `run.md` (sim, hardware, teleop, training, sim2real commands), `AGENT_GUIDE.md`, `DATA_COLLECTION_GUIDE.md`, `SERVER_TRAINING_GUIDE.md`, the numbered hardware tutorials in `docs/`, and `sim/README.md` (MuJoCo/URDF/Isaac Lab assets and exporters).

## Notes

- **Mypy**: one ruleset (`[tool.mypy]` in `pyproject.toml`) checks the paths in its `files` setting, except `lerobot.rl`, the vendored `molmoact2_hf_model` and the generated `*_pb2` modules. Run it with `pre-commit run mypy --all-files`. Add type annotations when modifying code.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
- **NexArm conventions** (shared constants in `packages/lerobot_robot_nexarm/src/lerobot_robot_nexarm/motors/nexarm.py`; keep code and docs consistent): leader `/dev/ttyUSB0`, follower `/dev/ttyUSB1`; gripper raw 1195 = open, 2833 = closed; arm joints `rad = (raw - 2048) * 2π / 4096` in both hardware and sim; sim datasets use `robot_type="nexarm_follower"`; the RealSense `top` camera is on by default (`--no-top-cam` disables it); `lerobot-record` appends `_YYYYMMDD_HHMMSS` to `repo_id`.
