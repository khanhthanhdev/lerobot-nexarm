This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`docs/README.md`](./docs/README.md)** (start-here index: install, sim and hardware quickstarts, recording, training, picking a policy, eval — with copy-pasteable commands).

## Project Overview

This is the **Hiwonder NexArm** research repo on top of LeRobot. It contains no LeRobot sources: `lerobot` is installed from upstream (<https://github.com/huggingface/lerobot>), pinned by commit in `[tool.uv.sources]` of the root `pyproject.toml`, and this repo adds the NexArm robot, teleoperator, env and desktop-service plugins, the research policies, sim assets and tools. Upstream's robots, policies and envs stay available next to the NexArm ones. It integrates with Hugging Face Hub for model/dataset sharing.

**NexArm-specific modules:**

NexArm code is LeRobot plugins in `packages/` (uv workspace members, installed by plain `uv sync`). LeRobot's CLIs import them through `register_third_party_plugins()`; the registered type names are unchanged. Paths below are under `packages/<package>/src/<package>/`.

| Path                                                           | What it does                                                                                                                                              |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lerobot_robot_nexarm/motors/`                                 | CommProtocol UART framing, 6-servo sync read/write, torque, bridge mode (CMD 56/68/96/97/98)                                                              |
| `lerobot_robot_nexarm/nexarm_follower/`                        | `NexArmFollowerConfig` + `NexArmFollower` (`nexarm_follower`) — connect, observe, send_action                                                             |
| `lerobot_robot_nexarm/nexarm_sim/`                             | `NexArmSimConfig` + `NexArmSim` (`nexarm_sim`) — single-arm MuJoCo simulation                                                                             |
| `lerobot_robot_nexarm/mobile_bi_nexarm_sim/`                   | `MobileBiNexArmSimConfig` + `MobileBiNexArmSim` (`mobile_bi_nexarm_sim`) — bimanual mobile sim                                                            |
| `lerobot_robot_nexarm/{collection_merge,recording_session}.py` | Collection recording tools behind `examples/nexarm/record_collection.py`, incl. the manual-control keys                                                   |
| `lerobot_robot_nexarm/{local_dataset,rerun_recording}.py`      | Local-only dataset loading; `.rrd` recording of the Rerun stream and `lerobot-nexarm-rollout`                                                             |
| `lerobot_robot_nexarm/camera_config.py`                        | The `--front-cam`/`--wrist-cam`/`--top-cam` flags and camera configs shared by the examples and tools                                                     |
| `lerobot_teleoperator_nexarm/`                                 | `NexArmLeaderConfig` + `NexArmLeader` (`nexarm_leader`) — read positions, leader→follower mapping                                                         |
| `lerobot_env_nexarm/`                                          | `NexArmEnv` (`--env.type=nexarm`), the gym envs, `lerobot-nexarm-eval` (per-stage rates) and `lerobot-nexarm-sim-benchmark`                               |
| `nexarm_service/`                                              | Desktop-app engine (`lerobot-nexarm-service`, `--extra service`)                                                                                          |
| `sim/`                                                         | MuJoCo MJCF, URDF description, Isaac Lab configs, meshes, and exporters                                                                                   |
| `sim/mobile_nexarm/`                                           | Mobile bimanual NexArm assets, ManiSkill/Isaac configs and video2sim                                                                                      |
| `examples/nexarm/`                                             | Ready-to-run tutorials: teleoperate, record, rollout, the sim demos and sim dataset generators, ACT and TurboVLA launchers                                |
| `tools/nexarm/`                                                | Operational tools: arm/camera/sim calibration, joint-mapping and servo probes, dataset audits, merges, uploads, videos                                    |
| `policies/`                                                    | Research policy plugins: `lerobot_policy_turbovla` and `lerobot_policy_dem` (adapters for the pinned `turbovla`/`dem` packages), `lerobot_policy_bspline` |
| `experiments/`                                                 | One `YYMMDD-<slug>` folder per experiment: README (hypothesis, data, command, results) and its draccus configs                                            |
| `firmware/`, `hardware/`                                       | Vendor controller firmware and protocol docs; mechanical, 3D-printing and reliability guidelines                                                          |

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Pinned lerobot + NexArm plugins + B-spline policy
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything (lerobot[all])
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests policies -svv --maxfail=10       # All tests (NexArm + research policies)
uv run pytest tests/robots/test_nexarm_follower.py::test_config_defaults -svv  # Single test
uv run pytest tests -k nexarm -svv                   # All NexArm-related tests (hardware is mocked)
make test-nexarm-ete                                 # NexArm sim gym integration test
DEVICE=cuda make test-end-to-end                     # E2E tests (NexArm sim)
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

## LeRobot (upstream, pinned)

LeRobot's code lives in `.venv` (`uv run python -c "import lerobot; print(lerobot.__file__)"`); read it there or in the upstream repo at the pinned commit. The parts NexArm code builds on:

- **`lerobot.scripts`** — the `lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-rollout`, ... CLIs. They call `register_third_party_plugins()`, which imports the installed `lerobot_robot_*`, `lerobot_teleoperator_*`, `lerobot_env_*` and `lerobot_policy_*` packages.
- **`lerobot.configs`** — draccus dataclass configs; polymorphism via `draccus.ChoiceRegistry` and `@register_subclass("name")`. `TrainPipelineConfig` in `train.py`, `PreTrainedConfig` in `policies.py`.
- **`lerobot.policies`** — `PreTrainedPolicy` (`nn.Module` + `HubMixin`) in `pretrained.py`; factory with lazy imports in `factory.py`.
- **`lerobot.processor`** — `ProcessorStep` registry; `DataProcessorPipeline` / `PolicyProcessorPipeline`.
- **`lerobot.datasets`** — `LeRobotDataset` and `LeRobotDatasetMetadata`.
- **`lerobot.envs`** — `EnvConfig` and the env factory; each env defines `gym_kwargs` and `create_envs()`.
- **`lerobot.robots`, `lerobot.motors`, `lerobot.cameras`, `lerobot.teleoperators`** — hardware abstraction layers the NexArm plugins subclass.

To move to a newer LeRobot, follow "Upgrading LeRobot" in `CONTRIBUTING.md`.

## Repository Structure

- **`packages/`** — NexArm plugins (table above); **`policies/`** — research policy plugins (TurboVLA adapter, B-spline Policy). Both are uv workspace members.
- **`tests/`** — Pytest suite for the NexArm plugins, examples and scripts, organized by module; mocks in `tests/mocks/`. Policy tests sit in `policies/*/tests/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `tests.yml` (NexArm and policy tests plus `make test-nexarm-ete`), `security.yml` (TruffleHog), `mobile_nexarm_sim.yml` (manual).
- **`docs/`** — all user guides, indexed by `docs/README.md`: `getting-started/` (install, simulation, hardware), `guides/` (pipeline, data collection, training tips, TurboVLA, sim and server training, sim calibration, desktop app), `reference/` (architecture and conventions, Python API, troubleshooting), `hiwonder/` (the vendor tutorials) and `service/` (the engine's API snapshots). LeRobot's own docs are at <https://huggingface.co/docs/lerobot>.
- **`examples/nexarm/`** — ready-to-run NexArm tutorials; **`tools/nexarm/`** — calibration, probes and dataset tools; **`experiments/`** — one folder per experiment; **`scripts/`** — protocol vectors, GPU bootstrap, multi-GPU and Slurm launchers.
- **`sim/`, `firmware/`, `hardware/`** — sim assets and exporters, vendor firmware, hardware files.
- **`docker/`** — `Dockerfile.user` (CPU image of this repo).
- **Root files**: `pyproject.toml` (the non-published `lerobot-nexarm` workspace root: lerobot pin, extras, tool config), `Makefile` (`build-user`, E2E targets), `uv.lock`, `CONTRIBUTING.md` & `README.md`.

## Scope of the Fork

LeRobot is used unmodified: the NexArm types register through the plugins, and NexArm behavior that LeRobot lacks lives in the plugins (`local_dataset`, `rerun_recording`, the manual-control keys in `recording_session`, `lerobot-nexarm-eval` for per-stage rates). Keep it that way; add NexArm behavior to `packages/` and fix LeRobot bugs upstream. `lerobot-find-joint-limits` and the HIL-SERL commands do not load plugins, so they do not see the NexArm types. The NexArm collection flags for recording (`--manual_control`, `--recording_log`, `--root_repo_id`, `--merge_root`, `--rerun_save_path`) live in the wrapper `examples/nexarm/record_collection.py`, not in `lerobot-record`. Upstream policies (`act`, `diffusion`, `smolvla`, `pi0*`, `groot`, ...) and simulation envs (`aloha`, `pusht`, `libero`, `metaworld`, ...) are still registered.

## TurboVLA and B-spline Policy

There are no git submodules. TurboVLA is the `turbovla` package from its own repo (<https://github.com/khanhthanhdev/TurboVLA>), pinned by commit in `[tool.uv.sources]` and installed by `uv sync --extra turbovla`. For local TurboVLA development, temporarily point that source at a checkout (`turbovla = { path = "../TurboVLA", editable = true }`) and never commit the change; see `docs/guides/turbovla.md`. The same extra installs the workspace plugin `policies/lerobot_policy_turbovla/` (`--policy.type=turbovla`; it trains with `lerobot-train` and runs with `lerobot-eval`/`lerobot-rollout`; `convert_checkpoint.py` turns a checkpoint from the former `train_turbovla.py` into a LeRobot policy folder). B-spline Policy is the workspace plugin `policies/lerobot_policy_bspline/` (`--policy.type=bspline`), installed by plain `uv sync`. The TurboVLA workflow lives in `examples/nexarm/` (`setup_turbovla.py`) and imports the installed package. Multi-GPU and Slurm launchers are in `scripts/nexarm/`.

## User-Facing Guides

Point to these rather than re-deriving workflows. `docs/README.md` indexes them; each topic has one owning page and the others link to it, so update the owner when behaviour changes. Install and dependency profiles: `docs/getting-started/install.md`; sim and hardware quickstarts: `docs/getting-started/`; record, train, roll out, benchmark: `docs/guides/pipeline.md`; the collection rig: `docs/guides/data-collection.md`; policy choice and training length: `docs/guides/training-tips.md`; TurboVLA, multi-GPU and server training: `docs/guides/`; conventions and troubleshooting: `docs/reference/`; tests: `CONTRIBUTING.md`; MuJoCo/URDF/Isaac Lab assets and exporters: `sim/README.md`. When helping a user (not developing the repo), ask the questions in `docs/README.md`, "For AI assistants helping someone use NexArm", before suggesting commands.

## Notes

- **Mypy**: one ruleset (`[tool.mypy]` in `pyproject.toml`) checks the NexArm packages and `scripts/`. The installed `lerobot` ships no `py.typed`, so its symbols are `Any` to mypy and calls into LeRobot are not type-checked; the tests cover that boundary. Run it with `pre-commit run mypy --all-files`. Add type annotations when modifying code.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `lerobot.utils.import_utils`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
- **NexArm conventions** (shared constants in `packages/lerobot_robot_nexarm/src/lerobot_robot_nexarm/motors/nexarm.py`; keep code and docs consistent): leader `/dev/ttyUSB0`, follower `/dev/ttyUSB1`; gripper raw 1195 = open, 2833 = closed; arm joints `rad = (raw - 2048) * 2π / 4096` in both hardware and sim; sim datasets use `robot_type="nexarm_follower"`; the RealSense `top` camera is on by default (`--no-top-cam` disables it); `lerobot-record` appends `_YYYYMMDD_HHMMSS` to `repo_id`.
