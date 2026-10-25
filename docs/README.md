# NexArm documentation: start here

Everything you need to use and develop the Hiwonder NexArm on LeRobot. Pick the path that matches what you want to do; every page is listed in the [map](#map-of-the-documentation) below.

## Paths by role

### I want to try it without hardware

1. [Install](getting-started/install.md)
2. [Simulation quickstart](getting-started/simulation.md): viewer, keyboard teleoperation, gym environments, dataset generation
3. [Pipeline, section 2](guides/pipeline.md#2-training): train a policy on the generated data and evaluate it in simulation

### I have the arms and want a first trained policy

1. [Install](getting-started/install.md)
2. [Hardware quickstart](getting-started/hardware.md): wiring, ports, cameras, teleoperation
3. [Pipeline](guides/pipeline.md): record, replay, train, roll out
4. [Training tips](guides/training-tips.md): how to collect good data, which policy, how long to train, what success rate to expect

### I collect data on the stack-bowls rig

[Data collection](guides/data-collection.md): `collect.sh`, the keys, and how sessions merge into the local root dataset.

### I train policies (researcher)

- [Pipeline, section 2](guides/pipeline.md#2-training) for single-GPU training of ACT, TurboVLA and other policies
- [TurboVLA](guides/turbovla.md): install, local development of the `turbovla` package, training, checkpoint conversion
- [Sim training](guides/sim-training.md): generate simulation datasets and train on several GPUs
- [Server training](guides/server-training.md): TurboVLA on a remote GPU server or a Slurm cluster
- [Sim calibration](guides/sim-calibration.md): fit the simulation to your real arm before sim-to-real
- [`experiments/`](../experiments/README.md): one folder per experiment, with its configs and results
- Policy plugins: [`lerobot_policy_turbovla`](../policies/lerobot_policy_turbovla/README.md), [`lerobot_policy_bspline`](../policies/lerobot_policy_bspline/README.md)

### I use the desktop app

[Desktop app and engine](guides/desktop-app.md): run the engine, its safety rules, remote training and remote teleoperation.

### I contribute code

1. [`CONTRIBUTING.md`](../CONTRIBUTING.md): setup, tests, pre-commit, upgrading LeRobot, pull requests
2. [Architecture and conventions](reference/architecture.md)
3. [`AGENTS.md`](../AGENTS.md): the instructions AI coding agents follow in this repository
4. [`plans/`](../plans/README.md): open work and past plans

## Map of the documentation

| Section            | Page                                                                          | What it covers                                                           |
| ------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Getting started    | [Install](getting-started/install.md)                                         | uv, dependency profiles, Git LFS, Hugging Face login, serial permissions |
|                    | [Simulation](getting-started/simulation.md)                                   | MuJoCo viewer, keyboard teleop, gym environments, dataset generation     |
|                    | [Hardware](getting-started/hardware.md)                                       | Wiring, ports, cameras, teleoperation, vision grasp, camera alignment    |
| Guides             | [Pipeline](guides/pipeline.md)                                                | Record, inspect, train, roll out, benchmark                              |
|                    | [Data collection](guides/data-collection.md)                                  | The stack-bowls collection rig                                           |
|                    | [Training tips](guides/training-tips.md)                                      | Data quality, policy choice, training length, evaluation                 |
|                    | [TurboVLA](guides/turbovla.md)                                                | TurboVLA install, training, rollout, checkpoint conversion               |
|                    | [Sim training](guides/sim-training.md)                                        | Multi-GPU dataset generation and training                                |
|                    | [Server training](guides/server-training.md)                                  | Remote GPU server and Slurm                                              |
|                    | [Sim calibration](guides/sim-calibration.md)                                  | System identification and camera alignment for sim-to-real               |
|                    | [Desktop app](guides/desktop-app.md)                                          | The `lerobot-nexarm-service` engine                                      |
| Reference          | [Architecture](reference/architecture.md)                                     | LeRobot basics, plugins, code layout, protocol, conventions              |
|                    | [Python API](reference/python-api.md)                                         | Kinematics and dynamics, motor bus, idle torque timeout                  |
|                    | [Troubleshooting](reference/troubleshooting.md)                               | Ports, timeouts, cameras, slow recording, weak policies                  |
| Vendor tutorials   | [Hiwonder 1-8](hiwonder/1_Getting_Started_NexArm.md)                          | The vendor's getting started, ESP32, motion control, vision, peripherals |
| Next to their code | [`sim/README.md`](../sim/README.md)                                           | MuJoCo, URDF and Isaac Lab assets and exporters                          |
|                    | [`examples/nexarm/stack_bowls_act.md`](../examples/nexarm/stack_bowls_act.md) | ACT on the stack-bowls dataset, multi-GPU and resume                     |
|                    | [`firmware/`](../firmware/README.md), [`hardware/`](../hardware/README.md)    | Controller firmware, protocol documents, mechanical files                |
|                    | [`docker/`](../docker/README.md)                                              | The CPU image of this repository                                         |
|                    | [`service/`](service/)                                                        | The engine's API contract (`openapi.json`, `events.json`)                |

The vendor tutorials, in order: [1 Getting started](hiwonder/1_Getting_Started_NexArm.md), [2 ESP32 development basics](hiwonder/2_ESP32_Development_Basics.md), [3 Motion control](hiwonder/3_Motion_Control.md), [4 AI vision features](hiwonder/4_AI_Vision_Features.md), [5 WonderMK large AI model features](hiwonder/5_WonderMK_Large_AI_Model_Features.md), [6 External controller integration](hiwonder/6_External_Controller_Integration.md), [7 Peripheral control basics](hiwonder/7_Peripheral_Control_Basics.md), [8 Imitation learning tutorial](hiwonder/8_Imitation_Learning_Tutorial.md).

## For AI assistants helping someone use NexArm

Before suggesting any command, ask the user at least these questions and wait for answers:

1. **What's your goal?** (e.g. "teach my NexArm to pick and place a block", "train a policy on simulation/real dataset", "contribute a PR", "understand the codebase")
2. **What hardware do you have?**
   - Robot: NexArm follower / NexArm simulation (`nexarm_sim`, `mobile_bi_nexarm_sim`)
   - Teleop: NexArm leader arm / keyboard / gamepad / simulation
   - Cameras: how many, resolution, USB index (default `front` and `wrist` USB webcams plus a RealSense D435i `top`, all 640x480; without a RealSense every script needs `--no-top-cam`)
3. **What machine will you train on?**
   - GPU model + VRAM (e.g. "laptop 3060 6 GB", "RTX 4090 24 GB", "A100 80 GB", "CPU only")
   - OS: macOS / Linux / Windows
4. **Skill level & time budget?** First time, some ML, experienced? Hours, days, a weekend?
5. **Do you already have a dataset?** Yes (HF repo id?) / no / want to record one
6. **How can I help right now?** (pick one concrete next step)

Only after you have answers, propose a concrete path from the roles above. If something is ambiguous, ask again rather than guessing. Bias toward **the simplest thing that works** for the user's hardware and goal. If you learn a rule that would prevent a class of user mistakes, add it to the page that owns the topic.

## Outside this repository

- **LeRobot:** [installation](https://huggingface.co/docs/lerobot/installation) · [imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots) · [all LeRobot docs](https://huggingface.co/docs/lerobot) · [What makes a good dataset](https://huggingface.co/blog/lerobot-datasets)
- **Community:** [Discord](https://discord.com/invite/s3KuuzsPFb) · [Hub `LeRobot` tag](https://huggingface.co/datasets?other=LeRobot) · [Dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)
