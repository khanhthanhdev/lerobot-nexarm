"""Read ``nexarm_scene.json``, the scene description that ``build_isaac_assets.py`` derives from the MJCF.

Plain Python and NumPy only, so it can be checked without Isaac.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

ISAAC_LAB_DIR = Path(__file__).resolve().parents[1]
SCENE_JSON = ISAAC_LAB_DIR / "nexarm_scene.json"
SCHEMA = "nexarm_isaac_scene_v1"
# build_usd.py writes the robot and bowl USDs here (not tracked by git).
USD_DIR = ISAAC_LAB_DIR / "usd"


@dataclass(frozen=True)
class SceneSpec:
    data: dict
    path: Path

    @classmethod
    def load(cls, path: Path | str = SCENE_JSON) -> SceneSpec:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{path} has schema {data.get('schema')!r}, expected {SCHEMA!r}")
        return cls(data, path)

    def resolve(self, relative: str) -> Path:
        """A path stored in the JSON, relative to the JSON file."""
        return (self.path.parent / relative).resolve()

    @property
    def robot(self) -> dict:
        return self.data["robot"]

    @property
    def bowls(self) -> dict:
        return self.data["bowls"]

    @property
    def task(self) -> dict:
        return self.data["stack_bowls"]

    @cached_property
    def action_joints(self) -> list[str]:
        """Driven joints, one row per entry of the action map (the gripper drives both jaws)."""
        return [joint for action in self.robot["actions"] for joint in action["joints"]]

    @cached_property
    def action_map(self) -> dict[str, np.ndarray]:
        """Per follower feature (the 6 action entries): raw limits, home, and the affine raw -> joint target map."""
        actions = self.robot["actions"]
        return {
            "raw_low": np.array([a["raw_range"][0] for a in actions], dtype=float),
            "raw_high": np.array([a["raw_range"][1] for a in actions], dtype=float),
            "home_raw": np.array([a["home_raw"] for a in actions], dtype=float),
            "scale": np.array([a["scale"] for a in actions], dtype=float),
            "offset": np.array([a["offset"] for a in actions], dtype=float),
            "target_low": np.array([a["target_range"][0] for a in actions], dtype=float),
            "target_high": np.array([a["target_range"][1] for a in actions], dtype=float),
        }

    def raw_to_target(self, raw: np.ndarray) -> np.ndarray:
        """Raw servo ticks (..., 6) -> joint targets (..., 6), exactly as the MuJoCo backend maps them."""
        m = self.action_map
        raw = np.clip(raw, m["raw_low"], m["raw_high"])
        return np.clip(m["scale"] * raw + m["offset"], m["target_low"], m["target_high"])

    def target_to_raw(self, target: np.ndarray) -> np.ndarray:
        m = self.action_map
        target = np.clip(target, m["target_low"], m["target_high"])
        return np.clip((target - m["offset"]) / m["scale"], m["raw_low"], m["raw_high"])

    def home_joint_positions(self) -> dict[str, float]:
        """Every robot joint at the follower's home pose (undriven joints at 0)."""
        targets = self.raw_to_target(self.action_map["home_raw"])
        home = dict.fromkeys(self.robot["drives"], 0.0)
        for action, target in zip(self.robot["actions"], targets, strict=True):
            for joint in action["joints"]:
                home[joint] = float(target)
        return home


def camera_apertures(
    fovy_deg: float, width: int, height: int, focal_length: float = 1.0
) -> tuple[float, float]:
    """Horizontal and vertical apertures giving a pinhole camera the MuJoCo vertical field of view."""
    vertical = 2.0 * focal_length * math.tan(math.radians(fovy_deg) / 2.0)
    return vertical * width / height, vertical
