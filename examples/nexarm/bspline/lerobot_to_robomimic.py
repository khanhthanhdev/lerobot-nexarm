"""Convert a NexArm LeRobot dataset into the robomimic HDF5 layout used by bspline-policy.

Output layout (matches `task/nexarm_joint_bspline.yaml`):
    data/demo_<i>/actions          (T, 6) float32  joint targets, same units as observation.state
    data/demo_<i>/obs/joint_pos    (T, 6) float32  observation.state
    data/demo_<i>/obs/front_image  (T, S, S, 3) uint8

Usage:
    uv run python examples/nexarm/bspline/lerobot_to_robomimic.py \
        --root outputs/datasets/nexarm_stack_bowls --repo-id thanhkt/nexarm_stack_bowls \
        --output data/nexarm_bspline.hdf5
"""

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

IMAGE_KEY = "observation.images.front"


_DS = None


def _load_episode(ep: int, repo_id: str, root: str | None, image_size: int):
    global _DS
    if _DS is None:
        _DS = LeRobotDataset(repo_id, root=root)
    ds = _DS
    start = int(ds.meta.episodes[ep]["dataset_from_index"])
    end = int(ds.meta.episodes[ep]["dataset_to_index"])
    states, actions, images = [], [], []
    for i in range(start, end):
        item = ds[i]
        states.append(item["observation.state"].numpy())
        actions.append(item["action"].numpy())
        img = (item[IMAGE_KEY].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        images.append(cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA))
    return ep, np.asarray(states, np.float32), np.asarray(actions, np.float32), np.asarray(images, np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None, help="Local dataset dir (default: HF cache)")
    parser.add_argument("--output", default="data/nexarm_bspline.hdf5")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    total = LeRobotDataset(args.repo_id, root=args.root).meta.total_episodes
    episodes = total if args.max_episodes is None else min(args.max_episodes, total)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    load = partial(_load_episode, repo_id=args.repo_id, root=args.root, image_size=args.image_size)
    with h5py.File(out, "w") as f, mp.get_context("spawn").Pool(args.workers) as pool:
        data = f.create_group("data")
        for ep, states, actions, images in tqdm(pool.imap(load, range(episodes)), total=episodes, desc="episodes"):
            demo = data.create_group(f"demo_{ep}")
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("obs/joint_pos", data=states)
            demo.create_dataset("obs/front_image", data=images, compression="gzip", compression_opts=1)
    print(f"wrote {episodes} episodes -> {out}")


if __name__ == "__main__":
    main()
