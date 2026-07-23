# ACT synthetic pretraining and real fine-tuning

Build the smoke corpus first:

```bash
MUJOCO_GL=egl uv run python sim/mobile_nexarm/data_engine/generate_mujoco.py \
  --output outputs/mobile_nexarm/smoke --episodes 100
```

The command exits nonzero below 70% scripted success, and every artifact is
revalidated for the exact 16-D schema, finite values, camera count, and 30 FPS
timestamps. Convert accepted episode directories with `data_engine/bridge.py`.
ManiSkill and Isaac should use disjoint seed ranges; `split_for_seed` assigns
each source 80/10/10 without moving seeds between splits.

LeRobot trains by update count, so compute five epochs as
`ceil(total_frames / batch_size) * 5`, then pretrain ACT:

```bash
uv run lerobot-train \
  --dataset.repo_id=<synthetic-dataset> \
  --policy.type=act \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=<five-epoch-step-count> \
  --output_dir=outputs/train/mobile_nexarm_act_synthetic
```

Select checkpoints by held-out simulation success, not loss. For real
fine-tuning, keep 20 episodes outside the dataset used below, rebuild
normalization statistics from the real dataset, load the synthetic checkpoint
with `--policy.path`, and reduce both ACT learning rates from `1e-5` to `1e-6`:

```bash
uv run lerobot-train \
  --dataset.repo_id=<real-train-dataset> \
  --policy.path=outputs/train/mobile_nexarm_act_synthetic/checkpoints/<selected>/pretrained_model \
  --policy.optimizer_lr=1e-6 \
  --policy.optimizer_lr_backbone=1e-6 \
  --batch_size=8 \
  --steps=<five-real-epoch-step-count> \
  --output_dir=outputs/train/mobile_nexarm_act_real
```

The release gate is 80% on held-out synthetic seeds and 70% over the 20 real
trials, with zero safety stops or joint-limit violations.
