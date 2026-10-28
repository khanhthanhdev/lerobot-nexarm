# Training tips: data, policy choice, training length, evaluation

Rules of thumb for getting a working policy on the NexArm. The commands themselves are in [the pipeline guide](pipeline.md); this page helps you choose what to run.

## 1. Data collection tips (beginner → reliable policy)

Good data beats clever models. Adopt these defaults and deviate only with evidence.

### 1.1 Setup & ergonomics

- **Fix the rig and cameras** before touching the software. If the rig vibrates or the operator gets frustrated, fix that first — more bad data won't help.
- **Lighting matters more than resolution.** Diffuse, consistent light. Avoid moving shadows.
- **"Can you do the task from the camera view alone?"** If no, your cameras are wrong. Fix before recording.
- Enable **action interpolation** for rollouts when available for smoother trajectories.

### 1.2 Practice before you record

- Do 5–10 demos without recording. Build a deliberate, repeatable strategy.
- Hesitant or inconsistent demos teach the model hesitation.

### 1.3 Quality over speed

Deliberate, high-quality execution beats fast sloppy runs. Optimize for speed only **after** strategy is dialed in — never trade quality for it.

### 1.4 Consistency within and across episodes

Same grasp, approach vector, and timing. Coherent strategies are much easier to learn than wildly varying movements.

### 1.5 Start small, then extend (the golden rule)

- **First 50 episodes = constrained version** of the task: one object, fixed position, fixed camera setup, one operator.
- Train a quick ACT model. See what fails.
- **Then add diversity** along one axis at a time: more positions → more lighting → more objects → more operators.
- Don't try to collect the "perfect dataset" on day one. Iterate.

### 1.6 Policy choice for beginners

- **Laptop / first time / want results fast → ACT.** Works surprisingly well, trains fast even on a laptop GPU.
- **Bigger GPU / language-conditioned / multi-task → SmolVLA.** Unfreezing the vision encoder (see section 3) is a big win here.
- Defer π0 / π0.5 / Wall-X / X-VLA until you have a proven ACT baseline and a 20+ GB GPU.

### 1.7 Recommended defaults for your first task

| Setting          | Value                                                                                                                                                                                                                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Episodes         | **50** to start, scale to 100–300 after first training                                                                                                                                                                                                                                                       |
| Episode length   | 20–45 s (shorter is fine for grasp/place)                                                                                                                                                                                                                                                                    |
| Reset time       | No limit (`--dataset.reset_time_s=inf`, see [the pipeline guide](pipeline.md#1c-any-task-recordpy-and-lerobot-record)); press **→** when the scene is reset                                                                                                                                                  |
| FPS              | 30                                                                                                                                                                                                                                                                                                           |
| Cameras          | **3 cameras by default**: fixed front + wrist + top (RealSense, auto-detected; `--top-cam <serial>` if several). 2 (front + wrist, `--no-top-cam`) also works; keep the same set for training and rollout. Multi-view often outperforms single-view. A single fixed camera also works to keep things simple. |
| Task description | Short, specific, action-phrased sentence                                                                                                                                                                                                                                                                     |

### 1.8 Troubleshooting signal

- Policy fails at one specific stage → record 10–20 more episodes **targeting that stage**.
- Policy flaps / oscillates → likely inconsistent demos, or need more training; re-record worst episodes (use **←** to redo).
- Policy ignores the object → camera framing or lighting issue, not a model issue.

See also: [What makes a good dataset](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset).

---

## 2. Which policy should I train?

Match the policy to the user's **GPU memory** and **time budget**. Numbers below come from an internal profiling run (one training update per policy). They are **indicative only** — see caveats.

### 2.1 Profiling snapshot (indicative)

All policies typically train for **5–10 epochs** (see section 3).

> **Human-facing version:** the [Compute Hardware Guide](https://huggingface.co/docs/lerobot/hardware_guide) reuses the table below and adds a cloud-GPU tier guide and a Hugging Face Jobs pointer.

| Policy      | Batch | Update (ms) | Peak GPU mem (GB) | Best for                                                                                                |
| ----------- | ----: | ----------: | ----------------: | ------------------------------------------------------------------------------------------------------- |
| `act`       |     4 |    **83.9** |          **0.94** | First-time users, laptops, single-task. Fast and reliable.                                              |
| `diffusion` |     4 |       168.6 |              4.94 | Multi-modal action distributions; needs mid-range GPU.                                                  |
| `smolvla`   |     1 |       357.8 |              3.93 | Language-conditioned, multi-task, small VLA. **Unfreeze vision encoder for big gains** (see section 3). |
| `xvla`      |     1 |       731.6 |             15.52 | Large VLA, multi-task.                                                                                  |
| `wall_x`    |     1 |       716.5 |             15.95 | Large VLA with world-model objective.                                                                   |
| `pi0`       |     1 |       940.3 |             15.50 | Strong large VLA baseline (Physical Intelligence).                                                      |
| `pi05`      |     1 |      1055.8 |             16.35 | Newer π policy; similar footprint to `pi0`.                                                             |

**Critical caveats:**

- **Optimizer:** measured with **SGD**. LeRobot's default is **AdamW**, which keeps extra optimizer state → **peak memory will be noticeably higher** with the default, especially for `pi0`, `pi05`, `wall_x`, `xvla`.
- **Batch size:** the large policies were profiled at batch 1. In practice use a **larger batch** for stable training (see section 3.4). Memory scales roughly linearly with batch.

### 2.2 Decision rules

- **< 8 GB VRAM (laptop, 3060, M-series Mac):** → `act`. Maybe `diffusion` if you have ~6–8 GB free.
- **12–16 GB VRAM (4070/4080, A4000):** → `smolvla` with defaults, or `act`/`diffusion` with larger batch. `pi0`/`pi05`/`wall_x`/`xvla` feasible only with small batch + gradient accumulation.
- **24+ GB VRAM (3090/4090/A5000):** → any policy. Prefer `smolvla` (unfrozen) for multi-task; `act` for single-task grasp-and-place (still often the best ROI). Could experiment with `pi0` or `pi05` or `xvla`
- **80 GB (A100/H100):** → any, with healthy batch. `pi05`, `xvla`, `wall_x` become comfortable.
- **CPU only:** → don't train here. Use Google Colab (see [`docs/source/notebooks.mdx`](https://huggingface.co/docs/lerobot/notebooks)) or a rented GPU.

### 2.3 Policy Compatibility Matrix on NexArm

| Policy group  | Status on NexArm       | Requirements & Compatibility Notes                                                                                                                          |
| ------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ACT**       | **Validated Baseline** | Fully compatible. front + wrist (+ top) cameras + 6-DoF raw joint positions.                                                                                |
| **TurboVLA**  | **Validated VLA**      | Fully compatible. Real-time 32 Hz VLA with <1GB VRAM. Uses DINOv3 + BERT + ACT head.                                                                        |
| **DEM**       | **Experimental**       | DINOv3 ConvNeXt-B + NeoBERT + MeanFlow head, one-step sampling. The paper's recipe failed on stack-bowls (0/10); use the v2 defaults. Gated DINOv3 weights. |
| **Diffusion** | **Validated Baseline** | Fully compatible. Requires `uv sync --extra diffusion`.                                                                                                     |
| **SmolVLA**   | **Validated Baseline** | Compatible VLA. Requires `uv sync --extra smolvla` and `--task` description.                                                                                |
| **TDMPC**     | **Incompatible**       | Rejects 640x480 dual-cam setup (requires square images `H == W` and single camera).                                                                         |
| **VQ-BeT**    | **Incompatible**       | Rejects 2-camera setup (`len(image_features) == 1` enforced at config validation).                                                                          |
| **π0 / π0.5** | Experimental           | Requires 20+ GB VRAM or small batch + gradient accumulation.                                                                                                |

---

## 3. How long should I train?

Robotics imitation learning usually converges in a **few epochs over the dataset**, not hundreds of thousands of raw steps. Think **epochs first**, then translate to steps.

### 3.1 Rule of thumb

- **Typical total: 5–10 epochs.** Start at 5, eval, then decide if more helps.
- Very small datasets (< 30 episodes) may want slightly more epochs — but first, **collect more data**.
- VLAs with a pretrained vision backbone typically need **fewer** epochs than training from scratch.

### 3.2 Steps ↔ epochs conversion

```
total_frames     = sum of frames over all episodes      # e.g. 50 eps × 30 fps × 30 s ≈ 45,000
steps_per_epoch  = ceil(total_frames / (batch_size × num_gpus))
total_steps      = epochs × steps_per_epoch
```

Examples for `--batch_size=8`:

| Dataset size            |  Frames | Steps / epoch | 5 epochs | 10 epochs |
| ----------------------- | ------: | ------------: | -------: | --------: |
| 50 eps × 30 s @ 30 fps  |  45,000 |        ~5,625 |      28k |       56k |
| 100 eps × 30 s @ 30 fps |  90,000 |       ~11,250 |      56k |      113k |
| 300 eps × 30 s @ 30 fps | 270,000 |       ~33,750 |     169k |      338k |

Pass the resulting total with `--steps=<N>`; eval at intermediate checkpoints (`outputs/train/.../checkpoints/`).

### 3.3 Per-policy starting points (single-task, ~50 episodes)

| Policy         | Batch | Steps (first run) | Notes                                                                           |
| -------------- | ----: | ----------------: | ------------------------------------------------------------------------------- |
| `act`          |  8–16 |           30k–80k | Usually converges under 50k for single-task.                                    |
| `turbovla`     |  8–16 |           30k–50k | Lightweight VLA. DINOv3 + BERT; <1GB VRAM at 32 Hz inference.                   |
| `dem`          |    32 |          50k–100k | 18 GB at batch 32 with front + wrist; EMA on. Watch `meanflow_mse`, not `loss`. |
| `diffusion`    |  8–16 |          80k–150k | Benefits from longer training than ACT.                                         |
| `smolvla`      |   4–8 |           30k–80k | Pretrained VLM → converges fast.                                                |
| `pi0` / `pi05` |   1–4 |           30k–80k | Memory-bound; use gradient accumulation for effective batch ≥ 16!               |

### 3.4 Batch size & gradient accumulation

- **Bigger batch is preferable** for stable gradients on teleop data.
- If GPU memory is the bottleneck, use **gradient accumulation** via `--accelerator.gradient_accumulation.steps=<N>` to raise _effective_ batch without raising peak memory:
  ```bash
  uv run lerobot-train ... \
    --batch_size=2 \
    --accelerator.gradient_accumulation.steps=4
  ```
  Effective batch size will be $2 \times 4 = 8$ samples per update step. `--steps` and `--save_freq` count micro-batches, not optimizer updates, so this run applies one update every 4 steps: multiply `--steps` by the accumulation factor to keep the same number of updates.
- Scale **learning rate** gently with batch; most LeRobot defaults work fine for a 2–4× batch change.

### 3.5 Scale LR schedule & checkpoints with `--steps`

LeRobot's default schedulers (e.g. SmolVLA's cosine decay) use `scheduler_decay_steps=30_000`, which is sized for long training runs. When you shorten training (e.g. 5k–10k steps on a small dataset), **scale the scheduler down to match** — otherwise the LR stays near the peak and never decays. Same for checkpoint frequency.

```bash
uv run lerobot-train ... \
  --steps=5000 \
  --policy.scheduler_decay_steps=5000 \
  --save_freq=5000
```

Rule of thumb: set `scheduler_decay_steps ≈ steps`, and `save_freq` to whatever granularity you want for eval (e.g. every 1k–5k steps). Match `scheduler_warmup_steps` proportionally if your run is very short.

### 3.6 SmolVLA: unfreeze the vision encoder for real gains

SmolVLA ships with `freeze_vision_encoder=True`. Unfreezing usually **improves performance substantially** on specialized tasks, at the cost of more VRAM and slower steps. Enable with:

```bash
lerobot-train ... --policy.type=smolvla \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
```

### 3.7 Signals to stop / keep going

- Train loss plateaus → stop, save a Hub checkpoint.
- Train loss still dropping and you're under 10 epochs → keep going.

---

## 4. Evaluation & benchmarks

Two flavors of evaluation:

### 4.1 Real-robot eval (NexArm)

Run the trained policy on the robot with a `sentry` rollout, which saves the run as an eval dataset. Convention: prefix the dataset with `eval_` (`lerobot-record` rejects `eval_` names; it is for data collection only).

```bash
uv run python examples/nexarm/rollout.py \
  --follower-port /dev/ttyUSB1 \
  --policy-path outputs/train/act_nexarm/checkpoints/last/pretrained_model \
  --strategy sentry --repo-id ${HF_USER}/eval_nexarm_cube_pick \
  --task "<same task description used during training>" --duration 120
```

Cameras default to front, wrist and the RealSense top; add `--no-top-cam` for a 2-camera policy.

Report success rate across episodes. Compare to a teleoperated baseline and to an earlier checkpoint to catch regressions.

### 4.2 Sim-benchmark eval

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
- Upstream envs live in the installed `lerobot.envs` package; the NexArm env is `--env.type=nexarm` (`packages/lerobot_env_nexarm/`). See [`docs/source/libero.mdx`](https://huggingface.co/docs/lerobot/libero), [`metaworld.mdx`](https://huggingface.co/docs/lerobot/metaworld), [`robocasa.mdx`](https://huggingface.co/docs/lerobot/robocasa), [`vlabench.mdx`](https://huggingface.co/docs/lerobot/vlabench) for specific benchmarks.
- To add a new benchmark, see [`docs/source/adding_benchmarks.mdx`](https://huggingface.co/docs/lerobot/adding_benchmarks) and [`envhub.mdx`](https://huggingface.co/docs/lerobot/envhub).

### 4.2b Dockerfiles for benchmark eval

Benchmark envs have native dependencies that are painful to install locally. Upstream LeRobot ships a pre-baked `Dockerfile.benchmark.<name>` for LIBERO, LIBERO+, MetaWorld, RoboCasa, RoboCerebra, RoboMME, RoboTwin and VLABench in [huggingface/lerobot `docker/`](https://github.com/huggingface/lerobot/tree/main/docker). Those images contain upstream LeRobot only, so use them for upstream policies; for TurboVLA on LIBERO, install `--extra libero --extra turbovla` here (see [`experiments/261024-turbovla-libero`](../../experiments/261024-turbovla-libero/README.md)).

See [`docker/README.md`](../../docker/README.md) for base-image details.

### 4.3 Target success rates

Single-task grasp-and-place with 50 clean episodes: ACT should reach **> 70% success** on the training configuration. Less → data problem (see section 1), not model problem. Expect a drop when generalizing to new positions — scale episodes or diversity to recover.
