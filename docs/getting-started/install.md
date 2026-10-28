# Install

Set up this repository once per machine. Every other guide assumes this install.

## Install `uv` and clone

[`uv`](https://docs.astral.sh/uv/) is the required package manager. It creates a locked virtual environment with Python 3.12+ and the exact versions in `uv.lock`.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/Hiwonder-official/lerobot-nexarm.git
cd lerobot-nexarm
```

## Dependency profiles

This is the canonical install for every NexArm guide. The **full NexArm profile** covers simulation data generation, real collection and rollout (including the RealSense `top` camera), training, Rerun visualization and tests:

```bash
uv sync --locked --extra test --extra dev --extra core_scripts --extra training --extra nexarm --extra intelrealsense
```

Add extras to that line instead of syncing a smaller set:

| Workflow                               | Add                                |
| -------------------------------------- | ---------------------------------- |
| TurboVLA (`--policy.type=turbovla`)    | `--extra turbovla`                 |
| DEM (`--policy.type=dem`)              | `--extra dem`                      |
| Benchmarking pi0 / SmolVLA checkpoints | `--extra pi --extra smolvla`       |
| Diffusion policy                       | `--extra diffusion`                |
| LIBERO benchmark                       | `--extra libero`                   |
| Desktop-app engine                     | `--extra service`                  |
| Everything (all policies and tools)    | use `uv sync --locked --extra all` |

`uv sync --locked` alone is enough for the MuJoCo viewer and keyboard demos in [Simulation](simulation.md), but not for dataset generation, recording or training. `uv sync` uninstalls every extra you do not list, so always repeat the full line.

> **Note**: Always invoke commands through `uv run <command>` (e.g. `uv run python ...` or `uv run lerobot-teleoperate`). Do not manually activate the `.venv` or install packages with `pip`.

`lerobot` itself is installed from upstream, pinned to one commit in `[tool.uv.sources]` of `pyproject.toml`; the NexArm plugins in `packages/` and the policy plugins in `policies/` are installed with it. See [Architecture](../reference/architecture.md).

## Git LFS and Hugging Face login

```bash
# Pull test fixtures and assets
git lfs install && git lfs pull

# Log into Hugging Face Hub (required for uploading datasets or downloading private or gated models)
uv run hf auth login
```

## Linux serial permissions

On Linux (Ubuntu/Debian), grant your user read/write access to the arms' serial ports:

```bash
sudo usermod -a -G dialout $USER   # then log out and back in
```

## Platform support

| Platform      | Status   | Notes                                                   |
| ------------- | -------- | ------------------------------------------------------- |
| Ubuntu 20.04+ | Verified | Port format `/dev/ttyUSB0`; add user to `dialout` group |
| Windows 10/11 | Verified | Install CH340 driver; port format `COM19`               |
| macOS         | Verified | Port format `/dev/tty.usbserial-xxx`                    |

Headless Linux machines render the simulation with `MUJOCO_GL=egl`.

## Verify the install

```bash
uv run python -c "from lerobot_robot_nexarm.nexarm_sim import NexArmSim; from lerobot_robot_nexarm.nexarm_follower import NexArmFollower; print('NexArm installation OK!')"
```

Next: [try the simulation](simulation.md) or [set up the arm](hardware.md).
