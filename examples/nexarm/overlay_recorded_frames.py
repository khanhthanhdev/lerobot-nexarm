#!/usr/bin/env python

"""Check the real-to-sim joint mapping by overlaying sim renders on recorded camera frames.

For sampled frames of one recorded episode, this poses the MuJoCo arm at the recorded ``observation.state``
and renders the same camera. It then writes real | sim | 50% blend side by side as PNG images. The wrist camera
rides on the arm, so its view depends only on the joint mapping and the camera mount, not on where an external
camera stands. If the wrist overlays line up, the mapping is right.

Pass ``--mapping`` with the JSON written by ``probe_joint_mapping.py`` to test a measured mapping before it
is applied to the sim. Rendering needs OpenGL (on headless Linux set ``MUJOCO_GL=egl``).

Example:
    uv run python examples/nexarm/overlay_recorded_frames.py --repo-id thanhkt/nexarm_stack_bowls \
        --episode 150 --mapping outputs/nexarm/joint_mapping_probe.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from lerobot.motors.nexarm.mujoco_mapping import MUJOCO_JOINTS, raw_to_joint_position
from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim.mujoco_backend import resolve_model_path

try:  # imported as a package module (tests) or run as a script
    from examples.nexarm.probe_joint_mapping import ARM_JOINTS, mapped_qpos
except ModuleNotFoundError:
    from probe_joint_mapping import ARM_JOINTS, mapped_qpos  # type: ignore[no-redef]


def pose_model(
    model: mujoco.MjModel, data: mujoco.MjData, raw: Sequence[float], mapping: dict | None
) -> None:
    """Set the arm and gripper qpos for raw follower ticks and run forward kinematics."""
    for name, value in zip(ARM_JOINTS, mapped_qpos(raw, mapping), strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MUJOCO_JOINTS[name])
        data.qpos[model.jnt_qposadr[joint_id]] = value
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MUJOCO_JOINTS["gripper"])
    data.qpos[model.jnt_qposadr[gripper_id]] = raw_to_joint_position(
        "gripper", raw[JOINT_NAMES.index("gripper")], model.jnt_range[gripper_id]
    )
    mujoco.mj_forward(model, data)


def side_by_side(real: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """real | sim | 50% blend, all uint8 HxWx3 of the same size."""
    blend = ((real.astype(np.uint16) + sim.astype(np.uint16)) // 2).astype(np.uint8)
    return np.concatenate([real, sim, blend], axis=1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--camera", default="wrist", help="Dataset camera and sim camera with the same name")
    parser.add_argument("--frames", type=int, default=6, help="Evenly spaced frames to overlay")
    parser.add_argument("--mapping", type=Path, default=None, help="probe_joint_mapping.py output to apply")
    parser.add_argument("--model", type=Path, default=Path("sim/description/mjcf/scene.xml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/nexarm/overlay"))
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    from PIL import Image

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args(argv)
    mapping = json.loads(args.mapping.read_text())["joints"] if args.mapping else None
    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode], return_uint8=True)
    image_key = f"observation.images.{args.camera}"
    names = dataset.meta.features["observation.state"].get("names") or [f"{n}.pos" for n in JOINT_NAMES]
    order = [list(names).index(f"{n}.pos") for n in JOINT_NAMES]

    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(args.model)))
    data = mujoco.MjData(model)
    height, width = dataset.meta.features[image_key]["shape"][:2]
    renderer = mujoco.Renderer(model, height=height, width=width)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        for index in np.linspace(0, len(dataset) - 1, args.frames).round().astype(int):
            item = dataset[int(index)]
            raw = np.asarray(item["observation.state"], dtype=np.float64)[order]
            real = np.asarray(item[image_key]).transpose(1, 2, 0)
            pose_model(model, data, raw, mapping)
            renderer.update_scene(data, camera=args.camera)
            frame = side_by_side(real, renderer.render())
            path = args.out / f"episode_{args.episode:04d}_frame_{int(index):04d}.png"
            Image.fromarray(frame).save(path)
            print(f"wrote {path}  state {raw.round().astype(int).tolist()}")
    finally:
        renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
