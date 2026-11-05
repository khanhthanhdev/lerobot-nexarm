# 261105-plan-bind-libero

**Status: running (smoke run first, then the full run).** [D3](../261028-plan-bind-d3-pusht/README.md) showed that Plan-Bind's stack-bowls lead (D1, D1b) does not carry over to Push-T's human, multimodal demonstrations. D4 asks the same question on LIBERO, a standard multi-task benchmark with language instructions, to compare with B-spline Policy and published work.

## Questions

- Does Plan-Bind beat the one-step control (`onestep`, same network, same call rate) on LIBERO, and how close does it get to B-spline Policy (`bsp10`)?
- How do the three degrade with inference latency and call rate on LIBERO?

## Setup

| Item     | Value                                                                                                                                                                                                                                                                                                                                                                          |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Data     | `lerobot/libero` at `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`: 1693 episodes, all four suites in one model, 10 fps, 7-D delta-EE actions (gripper last), 8-D state, 2 cameras (resized to 128x128, random crop 120)                                                                                                                                                           |
| Policies | `bsp10` (B-spline Policy, DDIM 10), `onestep`, `planbind`, with the 32-frame window, fine knots every 2 frames and coarse plan every 8, as in D3 (not tuned for LIBERO)                                                                                                                                                                                                        |
| Language | `task_condition=hash`: the instruction's words are hashed, embedded and mean-pooled into the UNet condition. LeRobot's diffusion policy has no language input, so it is not run                                                                                                                                                                                                |
| Training | 60k steps, batch 32, 2 seeds (1000, 1001), last checkpoint only                                                                                                                                                                                                                                                                                                                |
| Eval     | `lerobot-eval --env.type=libero` on `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, 2 episodes per task (20 per suite, 80 per cell; the smoke run measured about 35 s per episode, so a cell takes about 50 minutes), same seeds in every cell. Cells: main (latency 0), commit latency and none, latency 0 and 200 ms, `bsp10` also through the spline runtime |

Not yet done: the spline-fit check of LIBERO's jittery delta actions (the knot spacing is D3's), published baselines run in this harness, and a no-language check.

## Run it

At most 2 GPUs on the shared server (its 60 GB memory and 8-core limits also froze concurrent evals in D1b):

```bash
git clone https://github.com/khanhthanhdev/lerobot-nexarm.git && cd lerobot-nexarm
GPUS="0 1" NUM_WORKERS=2 experiments/261105-plan-bind-libero/run_all.sh smoke   # about 30 minutes
GPUS="0 1" NUM_WORKERS=2 HF_REPO=<user>/plan-bind-d4 experiments/261105-plan-bind-libero/run_all.sh full
```

`status.sh` prints progress; rerunning the same command resumes. On first import LIBERO asks for its dataset folder; answer once with `uv run python -c "import libero.libero"` before a run under `nohup`.

## Results

Pending.
