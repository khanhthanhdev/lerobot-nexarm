# 261024-bspline-vs-diffusion-stack-bowls

**Status: pending (needs a Linux GPU machine with EGL for sim generation and eval, and the real arm).** Only the smoke check below has run; the results tables are empty on purpose.

## Hypothesis

B-spline Policy (`--policy.type=bspline`, plugin `policies/lerobot_policy_bspline/`) reaches a stack-bowls success rate within noise of the diffusion baseline trained on the same data for the same steps. With `--policy.bspline_speed_up` above 1 it finishes episodes faster, because the predicted spline is executed at a higher rate. The paper (arXiv 2607.09648) reports that kind of speed-up at similar success.

## Data

| Field             | Value                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------- |
| Dataset `repo_id` | `local/nexarm_stack_bowls` (sim, generated below), root `outputs/datasets/nexarm_stack_bowls` |
| Revision          | record the `generation_report.json` hash / commit of the generator used                       |
| Episodes used     | all 300 accepted episodes                                                                     |

## Command

Generate the sim dataset (Linux, EGL):

```bash
MUJOCO_GL=egl uv run python examples/nexarm/generate_stack_bowls_dataset.py \
    --repo-id local/nexarm_stack_bowls --root outputs/datasets/nexarm_stack_bowls \
    --episodes 300 --gpus 0,1,2,3
```

Train both policies with the same data, seed, batch size, image size and steps. Both configs resize the 480x640 frames to 120x160 (`resize_shape`) and use batch 64, which needs more than 8 GB. With the three sim cameras an 8 GB GPU fits batch 16 (peak 7.2 GB), not 32; pass `--batch_size=16` to both commands there. On a larger GPU you can drop or raise `resize_shape`, as long as both configs keep the same value. With four 8 GB GPUs, `accelerate launch` at `--batch_size=16` per GPU keeps the configs' batch of 64 ([Sim training, section 4](../../docs/guides/sim-training.md#4-train-b-spline-policy-on-several-gpus)); train diffusion the same way, without `--policy.bspline_fit_workers`.

```bash
uv run lerobot-train --config_path experiments/261024-bspline-vs-diffusion-stack-bowls/train_bspline.yaml
uv run lerobot-train --config_path experiments/261024-bspline-vs-diffusion-stack-bowls/train_diffusion.yaml
```

Evaluate in sim, 50 episodes each (B-spline also with a 2x speed-up):

```bash
EXP=experiments/261024-bspline-vs-diffusion-stack-bowls/outputs
MUJOCO_GL=egl uv run lerobot-eval --policy.path=$EXP/train_diffusion/checkpoints/last/pretrained_model \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=50 --output_dir=$EXP/eval_diffusion
MUJOCO_GL=egl uv run lerobot-eval --policy.path=$EXP/train_bspline/checkpoints/last/pretrained_model \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=50 --output_dir=$EXP/eval_bspline
MUJOCO_GL=egl uv run lerobot-eval --policy.path=$EXP/train_bspline/checkpoints/last/pretrained_model \
    --policy.bspline_speed_up=2.0 \
    --env.type=nexarm --env.task=NexArmStackBowls-v0 --eval.n_episodes=50 --output_dir=$EXP/eval_bspline_x2
```

Real arm, 10 episodes (same cameras as the training data):

```bash
uv run lerobot-rollout --strategy.type=base \
    --policy.path=$EXP/train_bspline/checkpoints/last/pretrained_model \
    --robot.type=nexarm_follower --robot.port=/dev/ttyUSB1 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, top: {type: intelrealsense, serial_number_or_name: <REALSENSE_SERIAL>, width: 640, height: 480, fps: 30}}" \
    --task="Stack the bowls with red on bottom, blue in middle, and black on top." --duration=60
```

Porting-fidelity check: before trusting the NexArm numbers, reproduce one upstream number on a benchmark LeRobot ships (for example robomimic-style square or PushT), comparing `bspline` against `diffusion` there.

## Checkpoint

Hub id, step and W&B run link: _pending_.

## Results

| Policy    | Setting  | Episodes | Success rate | Mean episode time | Notes                  |
| --------- | -------- | -------- | ------------ | ----------------- | ---------------------- |
| diffusion | sim      | 50       |              |                   |                        |
| bspline   | sim      | 50       |              |                   | `bspline_speed_up=1.0` |
| bspline   | sim      | 50       |              |                   | `bspline_speed_up=2.0` |
| diffusion | real arm | 10       |              |                   |                        |
| bspline   | real arm | 10       |              |                   |                        |

## Smoke check (2026-10-25)

`marinelab` (RTX 5050 8 GB), `main` at `76422ee3`, on a 4-episode dataset from the generator command (`--episodes 4`), 50 steps, W&B off:

| Batch | B-spline        | Diffusion       |
| ----- | --------------- | --------------- |
| 64    | out of memory   | out of memory   |
| 32    | out of memory   | out of memory   |
| 16    | trains (7.2 GB) | trains (7.2 GB) |
| 8     | trains (6.5 GB) | trains (6.6 GB) |

The three eval commands run on the batch-16 checkpoints with `--eval.n_episodes=2`. This checks the commands, not the policies.

## Notes

- A CPU smoke run of `lerobot-train --policy.type=bspline` (200 steps) and checkpoint reload is recorded in `plans/reports/bspline-port-261024-1156-phase-07.md`. It checks wiring only, not task performance.
