#!/usr/bin/env python3

"""Export playable MP4 demo videos for episodes in a LeRobot dataset.

Example usage:
    # Export episode 0 side-by-side (front + wrist)
    uv run python examples/nexarm/export_episode_video.py --episodes 0

    # Export multiple episodes
    uv run python examples/nexarm/export_episode_video.py --episodes 0 1 2 --output-dir outputs/demo_videos

    # Export only front camera view
    uv run python examples/nexarm/export_episode_video.py --episodes 0 --camera front
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.io_utils import write_video


def tensor_to_rgb(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert (C, H, W) or (H, W, C) float [0, 1] or uint8 [0, 255] to RGB uint8."""
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu().numpy()
    else:
        img = np.array(tensor)

    if img.ndim == 3 and img.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))

    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    return img


def export_episode(
    dataset: LeRobotDataset,
    episode_idx: int,
    output_path: Path,
    camera: str = "all",
    fps: int = 30,
) -> None:
    ep_info = dataset.meta.episodes[episode_idx]
    from_idx = int(ep_info["dataset_from_index"])
    to_idx = int(ep_info["dataset_to_index"])
    num_frames = to_idx - from_idx

    tasks = ep_info.get("tasks", [])
    task_prompt = tasks[0] if tasks else ""

    # Detect camera keys
    feature_cam_keys = [
        k for k in dataset.features.keys()
        if "image" in k or dataset.features[k].get("dtype") in ("image", "video")
    ]
    # Preferred order
    cam_order = ["observation.images.front", "observation.images.wrist", "observation.images.top"]
    ordered_cam_keys = [k for k in cam_order if k in feature_cam_keys] + [
        k for k in feature_cam_keys if k not in cam_order
    ]

    frames_rgb: list[np.ndarray] = []
    print(f"[INFO] Rendering episode {episode_idx} ({num_frames} frames, cameras: {ordered_cam_keys})...")

    for f_idx in tqdm(range(from_idx, to_idx), desc=f"Ep {episode_idx}", leave=False):
        item = dataset[f_idx]
        cam_imgs = {}
        for k in ordered_cam_keys:
            if k in item:
                cam_imgs[k] = tensor_to_rgb(item[k])

        if camera in ("all", "both"):
            # Horizontal stack of all detected cameras
            imgs_to_stack = list(cam_imgs.values())
            if not imgs_to_stack:
                continue
            base_h = imgs_to_stack[0].shape[0]
            resized_imgs = []
            for img in imgs_to_stack:
                if img.shape[0] != base_h:
                    w = int(img.shape[1] * base_h / img.shape[0])
                    resized = np.array(Image.fromarray(img).resize((w, base_h)))
                else:
                    resized = img
                resized_imgs.append(resized)
            frame_arr = np.hstack(resized_imgs)
        else:
            # Target specific camera (e.g. 'front', 'wrist', 'top')
            matched = [img for k, img in cam_imgs.items() if camera.lower() in k.lower()]
            if matched:
                frame_arr = matched[0]
            else:
                frame_arr = next(iter(cam_imgs.values())) if cam_imgs else None

        if frame_arr is None:
            continue

        # Draw banner with PIL
        pil_frame = Image.fromarray(frame_arr)
        draw = ImageDraw.Draw(pil_frame, "RGBA")
        w, h = pil_frame.size
        banner_h = 36
        draw.rectangle([(0, 0), (w, banner_h)], fill=(20, 20, 20, 200))

        ep_text = f"Episode {episode_idx} | Frame {f_idx - from_idx + 1}/{num_frames}"
        draw.text((12, 10), ep_text, fill=(255, 255, 255, 255))

        if task_prompt:
            draw.text((w // 4, 10), f"Task: {task_prompt}", fill=(100, 220, 255, 255))

        frames_rgb.append(np.array(pil_frame.convert("RGB")))

    if not frames_rgb:
        print(f"[WARN] No frames collected for episode {episode_idx}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(output_path, frames_rgb, fps=fps)
    print(f"[SUCCESS] Exported video to {output_path} ({len(frames_rgb)} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/nexarm_stack_bowls")
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/nexarm_stack_bowls"))
    parser.add_argument("--episodes", type=int, nargs="+", default=[0], help="Episode indices to export (e.g. 0 1 2)")
    parser.add_argument("--camera", default="all", help="Camera name ('all', 'front', 'wrist', 'top')")
    parser.add_argument("--output", type=Path, default=None, help="Specific output video file path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_videos"))
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    print(f"[INFO] Loading dataset from {args.root}...")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    print(f"[INFO] Total episodes available: {dataset.num_episodes}")

    for ep in args.episodes:
        if ep < 0 or ep >= dataset.num_episodes:
            print(f"[ERROR] Episode {ep} out of range [0, {dataset.num_episodes - 1}]")
            continue
        if args.output is not None and len(args.episodes) == 1:
            out_file = args.output
        else:
            out_file = args.output_dir / f"episode_{ep:06d}.mp4"
        export_episode(dataset, ep, out_file, camera=args.camera, fps=args.fps)


if __name__ == "__main__":
    main()
