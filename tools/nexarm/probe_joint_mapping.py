#!/usr/bin/env python

"""Measure how the real follower's raw servo angles map onto the MuJoCo model's joints.

The sim maps ``rad = (raw - 2048) * 2*pi / 4096`` onto each joint with no sign or offset, so raw 2048 lands on
the exported CAD pose, which need not match the real arm's pose at 2048 (the vendor Home pose). This guided check
records the real arm posed straight up by hand, which the sim can reproduce exactly, and after a quarter turn of
each joint. It then derives a per-joint sign and zero offset that make the sim pose match the real one at any
raw value:

    qpos = sign * (raw - 2048) * 2*pi / 4096 + offset_rad

The follower stays un-energised so you can pose it by hand, as in ``calibrate_arms.py``. Support the arm while
you move it. Directions are named from the robot's point of view:

- front: toward the work area;
- back: away from it;
- left and right: the robot's own left and right.

Example:
    uv run python tools/nexarm/probe_joint_mapping.py --port /dev/ttyUSB1
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import mujoco
import numpy as np

from lerobot_robot_nexarm.motors.mujoco_mapping import MUJOCO_JOINTS
from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES, POSITION_CENTER, TICKS_PER_REVOLUTION
from lerobot_robot_nexarm.nexarm_sim.mujoco_backend import resolve_model_path

try:  # imported as a package module (tests) or run as a script
    from tools.nexarm.servo_step_test import follower_bus
except ModuleNotFoundError:
    from servo_step_test import follower_bus  # type: ignore[no-redef]

ARM_JOINTS = JOINT_NAMES[:-1]
FLEX_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")
# The sim robot faces -y, so its left is +x.
DIRECTIONS = {"front": (0.0, -1.0), "back": (0.0, 1.0), "left": (1.0, 0.0), "right": (-1.0, 0.0)}
QUARTER_TURN_TICKS = TICKS_PER_REVOLUTION / 4
JAW_CENTER_OFFSET = np.array([0.0, 0.033, 0.0])  # jaw pad centre in gripper_frame, as in the bowl generator

PROMPTS = {
    "upright": (
        "Pose the arm straight up by hand: arm facing the work area, upper arm, forearm and gripper all "
        "vertical, gripper pointing up. This is not the vendor Home pose."
    ),
    "shoulder_lift": "From upright, tilt only the shoulder a quarter turn (upper arm horizontal), either way.",
    "elbow_flex": "From upright, bend only the elbow a quarter turn (forearm horizontal), either way.",
    "wrist_flex": "From upright, bend only the wrist a quarter turn (gripper horizontal), either way.",
    "shoulder_pan": "From upright, turn only the base a quarter turn counter-clockwise, seen from above.",
    "wrist_roll": "From upright, turn only the wrist a quarter turn counter-clockwise, seen from above.",
}
# The link whose direction the user reports after each flex move.
FLEX_LINK = {"shoulder_lift": "upper arm", "elbow_flex": "forearm", "wrist_flex": "gripper"}


class SimArm:
    """Forward kinematics of the arm joints in the MuJoCo model, in sim qpos."""

    def __init__(self, model_path: Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, MUJOCO_JOINTS[n]) for n in ARM_JOINTS
        ]
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame")
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")

    def forward(self, qpos: Sequence[float]) -> None:
        for joint_id, value in zip(self.joint_ids, qpos, strict=True):
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(self.model, self.data)

    def anchor(self, index: int) -> np.ndarray:
        return self.data.xanchor[self.joint_ids[index]].copy()

    def axis(self, index: int) -> np.ndarray:
        return self.data.xaxis[self.joint_ids[index]].copy()

    def jaw(self) -> np.ndarray:
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        return self.data.site_xpos[self.site_id] + rotation @ JAW_CENTER_OFFSET

    def upright_offsets(self, iterations: int = 3) -> np.ndarray:
        """Sim qpos that stands the arm straight up: shoulder, elbow and wrist links all vertical."""
        qpos = np.zeros(len(ARM_JOINTS))
        for _ in range(iterations):
            for index in (1, 2, 3):
                self.forward(qpos)
                tip = self.anchor(index + 1) if index < 3 else self.jaw()
                axis = self.axis(index)
                link = tip - self.anchor(index)
                # Rotate the link about the joint axis onto +z, both projected onto the plane normal to the axis.
                current = link - axis * (link @ axis)
                wanted = np.array([0.0, 0.0, 1.0]) - axis * axis[2]
                angle = np.arctan2(axis @ np.cross(current, wanted), current @ wanted)
                qpos[index] += angle
        return qpos


def mapped_qpos(raw: Sequence[float], mapping: Mapping[str, Mapping[str, float]] | None) -> np.ndarray:
    """Arm joint qpos for raw ticks; ``mapping`` is the probe's ``joints`` entry, or None for the current one."""
    radians = (np.asarray(raw[: len(ARM_JOINTS)], dtype=np.float64) - POSITION_CENTER) * 2 * np.pi
    radians /= TICKS_PER_REVOLUTION
    if mapping is None:
        return radians
    sign = np.array([mapping[name]["sign"] for name in ARM_JOINTS], dtype=np.float64)
    offset = np.array([mapping[name]["offset_rad"] for name in ARM_JOINTS], dtype=np.float64)
    return sign * radians + offset


def direction_of(vector: Sequence[float]) -> str:
    """Closest of front/back/left/right for a horizontal direction."""
    xy = np.asarray(vector[:2], dtype=np.float64)
    return max(DIRECTIONS, key=lambda name: float(np.dot(DIRECTIONS[name], xy)))


def rotate_direction(name: str, quarter_turns: int) -> str:
    """Rotate a direction counter-clockwise (seen from above) by whole quarter turns."""
    order = ("front", "left", "back", "right")  # counter-clockwise seen from above when facing -y
    return order[(order.index(name) + quarter_turns) % 4]


def solve_mapping(
    sim: SimArm,
    upright_raw: Sequence[float],
    moved_raw: Mapping[str, Sequence[float]],
    flex_directions: Mapping[str, str],
    camera_side: str,
) -> dict:
    """Per-joint sign and zero offset from the probe readings.

    ``upright_raw`` is the reading with the arm posed straight up, ``moved_raw[joint]`` the reading after the
    quarter turn of that joint, ``flex_directions[joint]`` where the moved link points, and ``camera_side``
    where the wrist camera sits in the upright pose. The offset maps ``upright_raw`` onto the sim's upright
    qpos, so it holds whatever pose the arm takes at raw 2048.
    """
    upright = sim.upright_offsets()
    sim.forward(upright)
    up = np.array([0.0, 0.0, 1.0])
    joints: dict[str, dict] = {}
    warnings: list[str] = []
    for index, name in enumerate(ARM_JOINTS):
        delta = float(moved_raw[name][index] - upright_raw[index])
        if abs(delta) < QUARTER_TURN_TICKS / 4:
            raise ValueError(f"{name} moved only {delta:+.0f} ticks; redo its quarter turn")
        axis = sim.axis(index)
        if name in FLEX_JOINTS:
            # A positive sim rotation about ``axis`` tilts a vertical link toward axis × z.
            sim_direction = direction_of(np.cross(axis, up))
            real_direction = flex_directions[name]
            if real_direction == sim_direction:
                sim_sign = 1.0
            elif real_direction == rotate_direction(sim_direction, 2):
                sim_sign = -1.0
            else:
                raise ValueError(
                    f"{name} tilts {real_direction}, but the sim joint only tilts front/back here "
                    f"({sim_direction}); the base pan is probably offset by a quarter turn"
                )
        else:
            # Counter-clockwise seen from above is a positive rotation about +z.
            sim_sign = float(np.sign(axis @ up))
        sign = int(np.sign(delta) * sim_sign)
        upright_rad = (upright_raw[index] - POSITION_CENTER) * 2 * np.pi / TICKS_PER_REVOLUTION
        offset = float(upright[index] - sign * upright_rad)
        joints[name] = {"sign": sign, "offset_rad": offset, "quarter_turn_ticks": abs(delta)}
        low, high = sim.model.jnt_range[sim.joint_ids[index]]
        if not low <= offset <= high:
            warnings.append(
                f"{name}: raw 2048 is at sim qpos {offset:+.2f} rad, outside the joint range "
                f"[{low:+.2f}, {high:+.2f}]; re-centre the range on the new zero when applying this mapping"
            )
        ratio = abs(delta) / QUARTER_TURN_TICKS
        if not 0.75 <= ratio <= 1.25:
            warnings.append(
                f"{name}: a quarter turn read {abs(delta):.0f} ticks, {ratio:.2f}x the expected "
                f"{QUARTER_TURN_TICKS:.0f}; check the pose, or the servo scale differs"
            )

    # Pan and roll both turn about vertical when upright, so the wrist camera side fixes the roll zero.
    camera = sim.data.cam_xpos[sim.camera_id] - sim.jaw()
    sim_camera_side = direction_of(camera)
    order = ("front", "left", "back", "right")
    quarter_turns = (order.index(camera_side) - order.index(sim_camera_side)) % 4
    if quarter_turns == 3:
        quarter_turns = -1
    roll_axis_up = float(np.sign(sim.axis(4) @ up))
    joints["wrist_roll"]["offset_rad"] += roll_axis_up * quarter_turns * np.pi / 2

    return {
        "formula": "qpos = sign * (raw - 2048) * 2*pi/4096 + offset_rad",
        "joints": joints,
        "upright_raw_ticks": dict(zip(ARM_JOINTS, map(float, upright_raw[: len(ARM_JOINTS)]), strict=True)),
        "sim_upright_qpos": upright.tolist(),
        "sim_camera_side_at_upright": sim_camera_side,
        "camera_side_at_upright": camera_side,
        "warnings": warnings,
    }


def read_average(bus, samples: int = 5) -> list[float]:
    readings = np.array([bus.read_positions() for _ in range(samples)], dtype=np.float64)
    return np.median(readings, axis=0).tolist()


def ask_direction(question: str, ask: Callable[[str], str]) -> str:
    keys = {name[0]: name for name in DIRECTIONS}
    while True:
        answer = ask(f"{question} [f]ront/[b]ack/[l]eft/[r]ight: ").strip().lower()[:1]
        if answer in keys:
            return keys[answer]
        print("Answer f, b, l or r.")


def run_probe(bus, ask: Callable[[str], str] = input) -> tuple[list[float], dict, dict, str]:
    """Interactive readings: upright, then a quarter turn per joint, returning to upright in between."""
    ask("Support the arm with one hand: its motors turn off when you press Enter.")
    bus.set_torque(False)
    ask(PROMPTS["upright"] + " Press Enter.")
    upright = read_average(bus)
    camera_side = ask_direction("When upright, which side of the gripper is the wrist camera on?", ask)
    moved, directions = {}, {}
    for name in ("shoulder_pan", *FLEX_JOINTS, "wrist_roll"):
        ask(PROMPTS[name] + " Press Enter.")
        moved[name] = read_average(bus)
        if name in FLEX_JOINTS:
            directions[name] = ask_direction(f"Which way does the {FLEX_LINK[name]} point now?", ask)
        ask("Return the arm to upright. Press Enter.")
    return upright, moved, directions, camera_side


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyUSB1", help="Follower serial port")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--model", type=Path, default=Path("sim/description/mjcf/scene.xml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/nexarm/joint_mapping_probe.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sim = SimArm(resolve_model_path(args.model))
    with follower_bus(args.port, args.baudrate) as bus:
        upright, moved, directions, camera_side = run_probe(bus)
    result = solve_mapping(sim, upright, moved, directions, camera_side)
    result["readings"] = {"upright": upright, "moved": moved, "flex_directions": directions}

    print(f"\n{result['formula']}")
    for name, joint in result["joints"].items():
        print(
            f"{name:<14} sign {joint['sign']:+d}  offset {np.rad2deg(joint['offset_rad']):+8.2f} deg  "
            f"quarter turn {joint['quarter_turn_ticks']:.0f} ticks"
        )
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
