# Simulation quickstart (no hardware needed)

The repository includes MuJoCo models of the NexArm, a 6-DOF Cartesian solver and Gymnasium environments, so you can start without a physical arm. [Install](install.md) first. Model files, joint mapping and the Isaac Lab twin are described in [`sim/README.md`](../../sim/README.md).

## Interactive MuJoCo viewer

Launch the passive MuJoCo viewer to inspect the NexArm model and drag joints with interactive sliders:

```bash
uv run python examples/nexarm/simulate.py
```

Optional flags:

- `--model sim/description/mjcf/scene.xml`: Specify a custom MJCF scene model.
- `--fps 30`: Target simulation frame rate.

## 6-DOF Cartesian keyboard teleoperation

Control the simulated robot's end-effector in 3D Cartesian task space ($X, Y, Z$, Roll, Pitch, Yaw) using real-time Damped Least-Squares (DLS) Inverse Kinematics (Linux/macOS only; the script needs `termios`):

```bash
uv run python examples/nexarm/teleoperate_cartesian.py --robot sim
```

**Keybindings & Controls:**

| Key                 | Action                                                                                   |
| :------------------ | :--------------------------------------------------------------------------------------- |
| **`W` / `S`**       | Translate End-Effector **$\pm X$** (Forward / Backward)                                  |
| **`A` / `D`**       | Translate End-Effector **$\pm Y$** (Left / Right)                                        |
| **`R` / `F`**       | Translate End-Effector **$\pm Z$** (Up / Down)                                           |
| **`U` / `J`**       | Rotate End-Effector **$\pm \text{Roll}$**                                                |
| **`I` / `K`**       | Rotate End-Effector **$\pm \text{Pitch}$**                                               |
| **`O` / `L`**       | Rotate End-Effector **$\pm \text{Yaw}$**                                                 |
| **`Space`**         | **Toggle Gripper** (Open $\leftrightarrow$ Close)                                        |
| **`1` / `2` / `3`** | Step resolution: **Fine** (2 mm / 1°), **Medium** (5 mm / 2.5°), **Coarse** (15 mm / 5°) |
| **`H`**             | Return to default **Home Pose**                                                          |
| **`Q`**             | **Exit** gracefully                                                                      |

> **Headless / CI Mode**: Add `--dry-run` to run 5 headless steps without opening a window:
>
> ```bash
> uv run python examples/nexarm/teleoperate_cartesian.py --robot sim --dry-run
> ```

The same script drives the real arm with `--robot real`; see [Hardware](hardware.md#6-dof-cartesian-hardware-teleoperation).

## Interactive pick-and-place task

Run the benchmark pick-and-place simulation where the robot must grasp a red cube and place it into the green target zone:

```bash
uv run python examples/nexarm/pick_place_sim.py --seed 0
```

Add `--auto-reset` to continuously loop through randomized episodes.

## Gymnasium environment

Run the registered Gymnasium environment `NexArmPickPlace-v0` with camera rendering and step rewards:

```bash
uv run python examples/nexarm/demo_gym_env.py
```

`NexArmStackBowls-v0` is the real-robot task (stack the red, blue and black bowls in the order of the episode's
instruction) with the same actions, cameras and options. Select it with `--env.task` to evaluate, or to evaluate
while training, a policy trained on the real or sim stack-bowls datasets:

```bash
uv run lerobot-eval --policy.path=outputs/train/act_stack_bowls/checkpoints/last/pretrained_model \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=20
```

`--env.bowl_shift_m=0.03` tests reactivity: when the middle bowl is first lifted, the bottom bowl slides 3 cm in a seeded random direction that keeps it clear of the top bowl, once per episode. The default, 0, leaves the bowls in place.

The Isaac Lab twin of this task, for parallel training in Isaac Sim, is described in
[`sim/isaac_lab/README.md`](../../sim/isaac_lab/README.md).

## Generate a dataset

Generate scripted pick-and-place demonstrations. Visual, dynamics and object domain randomization (DR) is on by default; `--no-dr` disables it. Each new run needs its own `--root`: a fresh run on an existing root is an error, and `--resume` appends to it.

```bash
# Nominal dataset without domain randomization
MUJOCO_GL=egl uv run python examples/nexarm/generate_sim_dataset.py \
    --repo-id local/nexarm_sim_standard \
    --root outputs/datasets/nexarm_sim_standard \
    --episodes 20 \
    --no-dr

# Sim2Real-hardened dataset (DR + 2-step latency)
MUJOCO_GL=egl uv run python examples/nexarm/generate_sim_dataset.py \
    --repo-id local/nexarm_sim2real \
    --root outputs/datasets/nexarm_sim2real \
    --episodes 50 \
    --action-delay-steps 2
```

The 3-bowl stacking generator `generate_stack_bowls_dataset.py` takes the same flags. All generation flags, resuming and stale datasets are covered in [the pipeline guide, section 1a](../guides/pipeline.md#1a-simulation-scripted-demonstrations); generating on several GPUs in [Sim training](../guides/sim-training.md).

## Rerun 3D showcase

Visualize kinematics, joint sweeps, and multi-axis trajectories in the [Rerun](https://rerun.io) viewer:

```bash
# Single joint limits sweep
uv run python examples/nexarm/visualize_rerun_sim.py --mode joint_sweep

# Full workspace reach visualization
uv run python examples/nexarm/visualize_rerun_sim.py --mode workspace

# Complete multi-mode showcase saved to .rrd
uv run python examples/nexarm/visualize_rerun_sim.py --mode all --view save
```

Next: train a policy on the generated data with [the pipeline guide](../guides/pipeline.md#2-training).
