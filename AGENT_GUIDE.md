# AGENT_GUIDE.md — NexArm LeRobot Helper for AI Agents & Users

This file is a practical, copy-paste-friendly companion for any AI agent (Cursor, Claude, ChatGPT, Codex, etc.) helping a user work with Hiwonder NexArm on LeRobot. It complements [`AGENTS.md`](./AGENTS.md) (dev/contributor context) with **user-facing guidance**: how to start, how to teleoperate, how to record, what to train, how long, and how to rollout on the NexArm arm.

---

## 1. Start here — ask the user first (MANDATORY)

Before suggesting any command, an agent MUST ask the user at least these questions and wait for answers:

1. **What's your goal?** (e.g. "teach my NexArm to pick and place a block", "train a policy on simulation/real dataset", "contribute a PR", "understand the codebase")
2. **What hardware do you have?**
   - Robot: NexArm follower / NexArm simulation (`nexarm_sim`, `mobile_bi_nexarm_sim`)
   - Teleop: NexArm leader arm / keyboard / gamepad / simulation
   - Cameras: how many, resolution, USB index (usually `front` and `wrist` at 640x480)
3. **What machine will you train on?**
   - GPU model + VRAM (e.g. "laptop 3060 6 GB", "RTX 4090 24 GB", "A100 80 GB", "CPU only")
   - OS: macOS / Linux / Windows
4. **Skill level & time budget?** First time, some ML, experienced? Hours, days, a weekend?
5. **Do you already have a dataset?** Yes (HF repo id?) / no / want to record one
6. **How can I help right now?** (pick one concrete next step)

Only after you have answers, propose a concrete path. If something is ambiguous, ask again rather than guessing. Bias toward **the simplest thing that works** for the user's hardware and goal.

---

## 2. LeRobot in 60 seconds

LeRobot = **datasets + policies + envs + robot control**, unified by a small set of strong abstractions.

- **`LeRobotDataset`** — episode-aware dataset (video or images + actions + state), loadable from the Hub or disk.
- **Policies** (`ACT`, `Diffusion`, `SmolVLA`, `π0`, `π0.5`, `Wall-X`, `X-VLA`, `VQ-BeT`, `TD-MPC`, …) — all inherit `PreTrainedPolicy` and can be pushed/pulled from the Hub.
- **Processors** — small composable transforms between dataset → policy → robot.
- **Envs** (sim: `nexarm`, etc.) and **Robots** (real: `nexarm_follower`, `nexarm_sim`) — same action/observation contract so code swaps cleanly.
- **CLI** — `lerobot-record`, `lerobot-train`, `lerobot-eval`, `lerobot-teleoperate`, `lerobot-calibrate`, `lerobot-find-port`, `lerobot-replay`, `lerobot-rollout`.

See [`AGENTS.md`](./AGENTS.md) for repo architecture.

---

## 3. Quickstart paths (pick one)

### Path A — "I have a NexArm arm and want my first trained policy"

Go to §4 (NexArm end-to-end), then §5 (data tips), then §6 (pick a policy — likely **ACT**), then §7 (how long), then §8 (eval).

### Path B — "No physical arm, I want simulation or training on existing datasets"

Use `nexarm_sim` or simulation scripts in `examples/nexarm/`:

- `python examples/nexarm/demo_gym_env.py`
- `python examples/nexarm/pick_place_sim.py`
- `python examples/nexarm/generate_sim_dataset.py`

### Path C — "I just want to understand the codebase"

Read §2 above, then `AGENTS.md` "Architecture", then open `src/lerobot/motors/nexarm/nexarm.py`, `src/lerobot/robots/nexarm_follower/`, and `src/lerobot/policies/act/`.

---

## 4. NexArm end-to-end cheat-sheet

Minimum commands in order. Confirm follower (slave) and leader (master) arms are powered and connected via USB.

**4.1 Install**

Select the profile matching your workflow:

```bash
# Profile 1: Simulation & Benchmark (no physical robot needed)
uv sync --locked --extra dev --extra viz

# Profile 2: Hardware Teleoperation & Recording
uv sync --locked --extra nexarm --extra core_scripts

# Profile 3: Training & Evaluation (ACT baseline included)
uv sync --locked --extra training --extra evaluation
# For Diffusion: add --extra diffusion
# For SmolVLA: add --extra smolvla

# Profile 4: Full Development Stack (everything)
uv sync --locked --extra all

git lfs install && git lfs pull
uv run hf auth login                         # required to push datasets/policies
```

All repository commands should use `uv run`, which keeps Python and dependencies inside the locked project environment.

**4.2 Find USB ports** — run once per arm, unplug when prompted.

```bash
uv run lerobot-find-port
```

Linux typically assigns `/dev/ttyUSB0` and `/dev/ttyUSB1` (or check `dmesg`). Ensure port permissions (`sudo chmod 666 /dev/ttyUSB*`).

**4.3 Joint layout & hardware architecture**

- **6 DOF joints**: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.
- **Servos**: HX-30HM serial bus servos (0-4095 range, 1 Mbps).
- **Follower**: Dual-chip (ESP32 + AT32). Automatically enters bridge mode (CMD 68) when connected.
- **Leader**: Master ESP32 runs with torque disabled (CMD 98) for zero-resistance gravity-defying manipulation.
- Joint 2 (`shoulder_lift`) is mirrored (`4096 - pos`), Joint 6 (`gripper`) is mapped `[1195, 2833]`.

**4.4 Teleoperate** (sanity check, leader mirrors follower in real-time)

```bash
uv run lerobot-teleoperate \
  --robot.type=nexarm_follower --robot.port=<FOLLOWER_PORT> \
  --teleop.type=nexarm_leader  --teleop.port=<LEADER_PORT> \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

Or run the ready script:

```bash
uv run python examples/nexarm/teleoperate.py --robot-port <FOLLOWER_PORT> --leader-port <LEADER_PORT>
```

**4.5 Record a dataset** — keys: **→** next, **←** redo, **ESC** finish & upload.

```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')

uv run lerobot-record \
  --robot.type=nexarm_follower --robot.port=<FOLLOWER_PORT> \
  --teleop.type=nexarm_leader  --teleop.port=<LEADER_PORT> \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/nexarm_cube_pick \
  --dataset.single_task="Pick the cube and place it into the tray" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --display_data=true
```

**4.6 Replay an episode** (sanity check)

```bash
uv run lerobot-replay --robot.type=nexarm_follower --robot.port=<FOLLOWER_PORT> \
  --dataset.repo_id=${HF_USER}/nexarm_cube_pick --dataset.episode=0
```

**4.7 Train** (default: ACT — fastest, lowest memory).

```bash
uv run lerobot-train \
  --dataset.repo_id=${HF_USER}/nexarm_cube_pick \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_nexarm \
  --job_name=act_nexarm \
  --batch_size=8 \
  --wandb.enable=false
```

**4.8 Evaluate & Rollout on the real robot**

```bash
uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/act_nexarm/checkpoints/last/pretrained_model \
  --robot.type=nexarm_follower --robot.port=<FOLLOWER_PORT> \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --task="Pick the cube and place it into the tray" --duration=60
```

---

## 5. Data collection tips (beginner → reliable policy)

Good data beats clever models. Adopt these defaults and deviate only with evidence.

### 5.1 Setup & ergonomics

- **Fix the rig and cameras** before touching the software. If the rig vibrates or the operator gets frustrated, fix that first — more bad data won't help.
- **Lighting matters more than resolution.** Diffuse, consistent light. Avoid moving shadows.
- **"Can you do the task from the camera view alone?"** If no, your cameras are wrong. Fix before recording.
- Enable **action interpolation** for rollouts when available for smoother trajectories.

### 5.2 Practice before you record

- Do 5–10 demos without recording. Build a deliberate, repeatable strategy.
- Hesitant or inconsistent demos teach the model hesitation.

### 5.3 Quality over speed

Deliberate, high-quality execution beats fast sloppy runs. Optimize for speed only **after** strategy is dialed in — never trade quality for it.

### 5.4 Consistency within and across episodes

Same grasp, approach vector, and timing. Coherent strategies are much easier to learn than wildly varying movements.

### 5.5 Start small, then extend (the golden rule)

- **First 50 episodes = constrained version** of the task: one object, fixed position, fixed camera setup, one operator.
- Train a quick ACT model. See what fails.
- **Then add diversity** along one axis at a time: more positions → more lighting → more objects → more operators.
- Don't try to collect the "perfect dataset" on day one. Iterate.

### 5.6 Policy choice for beginners

- **Laptop / first time / want results fast → ACT.** Works surprisingly well, trains fast even on a laptop GPU.
- **Bigger GPU / language-conditioned / multi-task → SmolVLA.** Unfreezing the vision encoder (see §7) is a big win here.
- Defer π0 / π0.5 / Wall-X / X-VLA until you have a proven ACT baseline and a 20+ GB GPU.

### 5.7 Recommended defaults for your first task

| Setting          | Value                                                                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Episodes         | **50** to start, scale to 100–300 after first training                                                                                                |
| Episode length   | 20–45 s (shorter is fine for grasp/place)                                                                                                             |
| Reset time       | 10 s                                                                                                                                                  |
| FPS              | 30                                                                                                                                                    |
| Cameras          | **2 cameras recommended**: 1 fixed front + 1 wrist. Multi-view often outperforms single-view. A single fixed camera also works to keep things simple. |
| Task description | Short, specific, action-phrased sentence                                                                                                              |

### 5.8 Troubleshooting signal

- Policy fails at one specific stage → record 10–20 more episodes **targeting that stage**.
- Policy flaps / oscillates → likely inconsistent demos, or need more training; re-record worst episodes (use **←** to redo).
- Policy ignores the object → camera framing or lighting issue, not a model issue.

See also: [What makes a good dataset](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset).

---

## 6. Which policy should I train?

Match the policy to the user's **GPU memory** and **time budget**. Numbers below come from an internal profiling run (one training update per policy). They are **indicative only** — see caveats.

### 6.1 Profiling snapshot (indicative)

All policies typically train for **5–10 epochs** (see §7).

> **Human-facing version:** the [Compute Hardware Guide](./docs/source/hardware_guide.mdx) reuses the table below and adds a cloud-GPU tier guide and a Hugging Face Jobs pointer.

| Policy      | Batch | Update (ms) | Peak GPU mem (GB) | Best for                                                                                         |
| ----------- | ----: | ----------: | ----------------: | ------------------------------------------------------------------------------------------------ |
| `act`       |     4 |    **83.9** |          **0.94** | First-time users, laptops, single-task. Fast and reliable.                                       |
| `diffusion` |     4 |       168.6 |              4.94 | Multi-modal action distributions; needs mid-range GPU.                                           |
| `smolvla`   |     1 |       357.8 |              3.93 | Language-conditioned, multi-task, small VLA. **Unfreeze vision encoder for big gains** (see §7). |
| `xvla`      |     1 |       731.6 |             15.52 | Large VLA, multi-task.                                                                           |
| `wall_x`    |     1 |       716.5 |             15.95 | Large VLA with world-model objective.                                                            |
| `pi0`       |     1 |       940.3 |             15.50 | Strong large VLA baseline (Physical Intelligence).                                               |
| `pi05`      |     1 |      1055.8 |             16.35 | Newer π policy; similar footprint to `pi0`.                                                      |

**Critical caveats:**

- **Optimizer:** measured with **SGD**. LeRobot's default is **AdamW**, which keeps extra optimizer state → **peak memory will be noticeably higher** with the default, especially for `pi0`, `pi05`, `wall_x`, `xvla`.
- **Batch size:** the large policies were profiled at batch 1. In practice use a **larger batch** for stable training (see §7.4). Memory scales roughly linearly with batch.

### 6.2 Decision rules

- **< 8 GB VRAM (laptop, 3060, M-series Mac):** → `act`. Maybe `diffusion` if you have ~6–8 GB free.
- **12–16 GB VRAM (4070/4080, A4000):** → `smolvla` with defaults, or `act`/`diffusion` with larger batch. `pi0`/`pi05`/`wall_x`/`xvla` feasible only with small batch + gradient accumulation.
- **24+ GB VRAM (3090/4090/A5000):** → any policy. Prefer `smolvla` (unfrozen) for multi-task; `act` for single-task grasp-and-place (still often the best ROI). Could experiment with `pi0` or `pi05` or `xvla`
- **80 GB (A100/H100):** → any, with healthy batch. `pi05`, `xvla`, `wall_x` become comfortable.
- **CPU only:** → don't train here. Use Google Colab (see [`docs/source/notebooks.mdx`](./docs/source/notebooks.mdx)) or a rented GPU.

### 6.3 Policy Compatibility Matrix on NexArm

| Policy group  | Status on NexArm       | Requirements & Compatibility Notes                                                  |
| ------------- | ---------------------- | ----------------------------------------------------------------------------------- |
| **ACT**       | **Validated Baseline** | Fully compatible. Standard 2-camera (front + wrist) + 6-DoF raw joint positions.    |
| **Diffusion** | **Validated Baseline** | Fully compatible. Requires `uv sync --extra diffusion`.                             |
| **SmolVLA**   | **Validated Baseline** | Compatible VLA. Requires `uv sync --extra smolvla` and `--task` description.        |
| **TDMPC**     | **Incompatible**       | Rejects 640x480 dual-cam setup (requires square images `H == W` and single camera). |
| **VQ-BeT**    | **Incompatible**       | Rejects 2-camera setup (`len(image_features) == 1` enforced at config validation).  |
| **π0 / π0.5** | Experimental           | Requires 20+ GB VRAM or small batch + gradient accumulation.                        |

---

## 7. How long should I train?

Robotics imitation learning usually converges in a **few epochs over the dataset**, not hundreds of thousands of raw steps. Think **epochs first**, then translate to steps.

### 7.1 Rule of thumb

- **Typical total: 5–10 epochs.** Start at 5, eval, then decide if more helps.
- Very small datasets (< 30 episodes) may want slightly more epochs — but first, **collect more data**.
- VLAs with a pretrained vision backbone typically need **fewer** epochs than training from scratch.

### 7.2 Steps ↔ epochs conversion

```
total_frames     = sum of frames over all episodes      # e.g. 50 eps × 30 fps × 30 s ≈ 45,000
steps_per_epoch  = ceil(total_frames / batch_size)
total_steps      = epochs × steps_per_epoch
```

Examples for `--batch_size=8`:

| Dataset size            |  Frames | Steps / epoch | 5 epochs | 10 epochs |
| ----------------------- | ------: | ------------: | -------: | --------: |
| 50 eps × 30 s @ 30 fps  |  45,000 |        ~5,625 |      28k |       56k |
| 100 eps × 30 s @ 30 fps |  90,000 |       ~11,250 |      56k |      113k |
| 300 eps × 30 s @ 30 fps | 270,000 |       ~33,750 |     169k |      338k |

Pass the resulting total with `--steps=<N>`; eval at intermediate checkpoints (`outputs/train/.../checkpoints/`).

### 7.3 Per-policy starting points (single-task, ~50 episodes)

| Policy         | Batch | Steps (first run) | Notes                                                             |
| -------------- | ----: | ----------------: | ----------------------------------------------------------------- |
| `act`          |  8–16 |           30k–80k | Usually converges under 50k for single-task.                      |
| `diffusion`    |  8–16 |          80k–150k | Benefits from longer training than ACT.                           |
| `smolvla`      |   4–8 |           30k–80k | Pretrained VLM → converges fast.                                  |
| `pi0` / `pi05` |   1–4 |           30k–80k | Memory-bound; use gradient accumulation for effective batch ≥ 16! |

### 7.4 Batch size & gradient accumulation

- **Bigger batch is preferable** for stable gradients on teleop data.
- If GPU memory is the bottleneck, use **gradient accumulation** via `--gradient_accumulation_steps=<N>` to raise _effective_ batch without raising peak memory:
  ```bash
  uv run lerobot-train ... \
    --batch_size=2 \
    --gradient_accumulation_steps=4
  ```
  Effective batch size will be $2 \times 4 = 8$ samples per update step.
- Scale **learning rate** gently with batch; most LeRobot defaults work fine for a 2–4× batch change.

### 7.5 Scale LR schedule & checkpoints with `--steps`

LeRobot's default schedulers (e.g. SmolVLA's cosine decay) use `scheduler_decay_steps=30_000`, which is sized for long training runs. When you shorten training (e.g. 5k–10k steps on a small dataset), **scale the scheduler down to match** — otherwise the LR stays near the peak and never decays. Same for checkpoint frequency.

```bash
uv run lerobot-train ... \
  --steps=5000 \
  --policy.scheduler_decay_steps=5000 \
  --save_freq=5000
```

Rule of thumb: set `scheduler_decay_steps ≈ steps`, and `save_freq` to whatever granularity you want for eval (e.g. every 1k–5k steps). Match `scheduler_warmup_steps` proportionally if your run is very short.

### 7.6 SmolVLA: unfreeze the vision encoder for real gains

SmolVLA ships with `freeze_vision_encoder=True`. Unfreezing usually **improves performance substantially** on specialized tasks, at the cost of more VRAM and slower steps. Enable with:

```bash
lerobot-train ... --policy.type=smolvla \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
```

### 7.7 Signals to stop / keep going

- Train loss plateaus → stop, save a Hub checkpoint.
- Train loss still dropping and you're under 10 epochs → keep going.

---

## 8. Evaluation & benchmarks

Two flavors of evaluation:

### 8.1 Real-robot eval (SO-101, etc.)

Reuse `lerobot-record` with `--policy.path` to run the trained policy on-robot and save the run as an eval dataset. Convention: prefix the dataset with `eval_`.

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/eval_my_task \
  --dataset.single_task="<same task description used during training>" \
  --dataset.num_episodes=10 \
  --policy.path=${HF_USER}/act_my_task
```

Report success rate across episodes. Compare to a teleoperated baseline and to an earlier checkpoint to catch regressions.

### 8.2 Sim-benchmark eval

For policies trained on sim datasets (PushT, Aloha, LIBERO, MetaWorld, RoboCasa, …) use `lerobot-eval` against the matching `env.type`:

```bash
lerobot-eval \
  --policy.path=${HF_USER}/diffusion_pusht \
  --env.type=pusht \
  --eval.n_episodes=50 \
  --eval.batch_size=10 \
  --policy.device=cuda
```

- Use `--policy.path=outputs/train/.../checkpoints/<step>/pretrained_model` for local checkpoints.
- `--eval.n_episodes` should be ≥ 50 for a stable success-rate estimate.
- Available envs live in `src/lerobot/envs/`. See [`docs/source/libero.mdx`](./docs/source/libero.mdx), [`metaworld.mdx`](./docs/source/metaworld.mdx), [`robocasa.mdx`](./docs/source/robocasa.mdx), [`vlabench.mdx`](./docs/source/vlabench.mdx) for specific benchmarks.
- To add a new benchmark, see [`docs/source/adding_benchmarks.mdx`](./docs/source/adding_benchmarks.mdx) and [`envhub.mdx`](./docs/source/envhub.mdx).

### 8.2b Dockerfiles for benchmark eval

Benchmark envs have native dependencies that are painful to install locally. The repo ships **pre-baked Dockerfiles** for each supported benchmark — use these to run `lerobot-eval` in a reproducible environment:

| Benchmark   | Dockerfile                                                                             |
| ----------- | -------------------------------------------------------------------------------------- |
| LIBERO      | [`docker/Dockerfile.benchmark.libero`](./docker/Dockerfile.benchmark.libero)           |
| LIBERO+     | [`docker/Dockerfile.benchmark.libero_plus`](./docker/Dockerfile.benchmark.libero_plus) |
| MetaWorld   | [`docker/Dockerfile.benchmark.metaworld`](./docker/Dockerfile.benchmark.metaworld)     |
| RoboCasa    | [`docker/Dockerfile.benchmark.robocasa`](./docker/Dockerfile.benchmark.robocasa)       |
| RoboCerebra | [`docker/Dockerfile.benchmark.robocerebra`](./docker/Dockerfile.benchmark.robocerebra) |
| RoboMME     | [`docker/Dockerfile.benchmark.robomme`](./docker/Dockerfile.benchmark.robomme)         |
| RoboTwin    | [`docker/Dockerfile.benchmark.robotwin`](./docker/Dockerfile.benchmark.robotwin)       |
| VLABench    | [`docker/Dockerfile.benchmark.vlabench`](./docker/Dockerfile.benchmark.vlabench)       |

Build and run (adapt to your benchmark):

```bash
docker build -f docker/Dockerfile.benchmark.robomme -t lerobot-bench-robomme .
docker run --gpus all --rm -it \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  lerobot-bench-robomme \
  lerobot-eval --policy.path=<your_policy> --env.type=<env> --eval.n_episodes=50
```

See [`docker/README.md`](./docker/README.md) for base-image details.

### 8.3 Target success rates

Single-task grasp-and-place with 50 clean episodes: ACT should reach **> 70% success** on the training configuration. Less → data problem (see §5), not model problem. Expect a drop when generalizing to new positions — scale episodes or diversity to recover.

---

## 9. Further reading & resources

- **Getting started:** [`installation.mdx`](./docs/source/installation.mdx) · [`il_robots.mdx`](./docs/source/il_robots.mdx) · [What makes a good dataset](https://huggingface.co/blog/lerobot-datasets)
- **Per-policy docs:** browse [`docs/source/*.mdx`](./docs/source/) (policies, hardware, benchmarks, advanced training).
- **Community:** [Discord](https://discord.com/invite/s3KuuzsPFb) · [Hub `LeRobot` tag](https://huggingface.co/datasets?other=LeRobot) · [Dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)

> Keep this file current. If you learn a rule that would prevent a class of user mistakes, add it here and in [`AGENTS.md`](./AGENTS.md).
