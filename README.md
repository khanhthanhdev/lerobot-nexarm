# Hiwonder NexArm LeRobot VLA Open-Source 6-Axis Robotic Arm

[NexArm](https://www.hiwonder.com/products/nexarm6-axis) is an open-source, [🤗 LeRobot](https://github.com/huggingface/lerobot)-native robotic arm designed for embodied AI research and rapid validation of imitation and reinforcement learning policies. Its dual‑chip architecture (ESP32 + AT32) enables synchronous leader‑follower teleoperation with millisecond‑level tracking latency, generating clean demonstration data that feeds directly into LeRobot training pipelines.

The robotic arm features an all‑metal chassis driven by 65 kg·cm magnetic encoder servos, delivering ±2 mm repeatability and smooth, jitter‑free motion. Combined with advanced inverse kinematics and curve smoothing, it natively computes complex trajectories while minimizing start‑stop vibrations for fluid movement.

For on‑device perception, NexArm integrates a 6 TOPS K230 vision module, allowing you to run multimodal large models and computer vision pipelines—such as YOLO tracking and hand‑eye coordinated grasping—without a PC. All schematics, firmware, and code are open source, making NexArm a transparent and modifiable platform for academic research and real‑world robotic prototyping.

<p align="center">
  <img src="./media/readme/VLA_architecture.jpg" alt="nexarm" width="600"/>
</p>

## 🎯 Compared to SO‑ARM101:

- Payload: 500 g (150% improvement over SO‑ARM101)
- Repeatability: ±2 mm (33% improvement over SO‑ARM101)
- Architecture: ESP32 + AT32 dual‑chip design
- Build: All‑metal body with rotating base and parallel‑rail gripper
- Versatility: Supports multiple control methods out of the box

## End-to-End Robotic Arm Comparison

| Model               | NexArm                                                              | SO-ARM101                                                           |
| ------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Payload             | 500 g                                                               | 200 g                                                               |
| Repeatability       | ±2 mm                                                               | ±3 mm                                                               |
| Workspace           | 0.5 m                                                               | 0.4 m                                                               |
| Joint Servo         | Dual-output-shaft magnetic encoder bus servo                        | Single-output-shaft magnetic encoder bus servo with fixed shaft     |
| Body Material       | Aerospace-grade metal structure                                     | PLA 3D-printed structural parts                                     |
| End Effector        | Parallel rail gripper                                               | 3D-printed vertical-opening gripper                                 |
| Product Positioning | Advanced end-to-end development / advanced embodied AI applications | Entry-level end-to-end development / basic embodied AI applications |

---

## Quickstart (5 minutes, no hardware)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # install uv
git clone https://github.com/Hiwonder-official/lerobot-nexarm.git
cd lerobot-nexarm
uv sync --locked                                          # pinned LeRobot + NexArm plugins
uv run python examples/nexarm/simulate.py                 # MuJoCo viewer with joint sliders
uv run python examples/nexarm/pick_place_sim.py --seed 0  # scripted pick-and-place
```

For data generation, recording, training and tests, install the full profile in [Install](docs/getting-started/install.md). Always run commands through `uv run`.

## Documentation

**Start at [`docs/README.md`](docs/README.md)**: it lists every guide by role (simulation only, hardware owner, data collector, researcher, desktop-app user, contributor).

| I want to…                                  | Read                                                                                                                                |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Install the repository                      | [Install](docs/getting-started/install.md)                                                                                          |
| Try the simulation                          | [Simulation quickstart](docs/getting-started/simulation.md)                                                                         |
| Connect and teleoperate the arms            | [Hardware quickstart](docs/getting-started/hardware.md)                                                                             |
| Record, train and roll out a policy         | [Pipeline](docs/guides/pipeline.md) and [Training tips](docs/guides/training-tips.md)                                               |
| Train TurboVLA, on several GPUs or a server | [TurboVLA](docs/guides/turbovla.md), [Sim training](docs/guides/sim-training.md), [Server training](docs/guides/server-training.md) |
| Understand the code                         | [Architecture and conventions](docs/reference/architecture.md)                                                                      |
| Fix a problem                               | [Troubleshooting](docs/reference/troubleshooting.md)                                                                                |
| Contribute                                  | [`CONTRIBUTING.md`](CONTRIBUTING.md)                                                                                                |

## What this repository is

The NexArm research repository on top of [LeRobot](https://github.com/huggingface/lerobot). It contains no LeRobot sources: `lerobot` is installed from upstream, pinned by commit, and this repository adds the NexArm robot, teleoperator, environment and desktop-engine plugins (`packages/`), research policies such as TurboVLA and the B-spline Policy (`policies/`), simulation assets (`sim/`), tools (`tools/nexarm/`), examples (`examples/nexarm/`) and experiments (`experiments/`). Upstream's robots, policies and environments stay available next to the NexArm ones.

## License

[Apache License 2.0](LICENSE).
