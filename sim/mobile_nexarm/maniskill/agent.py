"""Lazy ManiSkill registration for the generated mobile bimanual NexArm."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

ARM_FEATURES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
BASE_JOINTS = ("base_x", "base_y", "base_yaw")
LIFT_JOINTS = ("lift_axis",)


def resolve_joint_indices(active_joints: list[Any], required_names: list[str]) -> list[int]:
    """Resolve SAPIEN's potentially interleaved active joints by name."""
    index_by_name = {
        str(getattr(joint, "name", getattr(joint, "get_name", lambda: "")())): index
        for index, joint in enumerate(active_joints)
    }
    missing = [name for name in required_names if name not in index_by_name]
    if missing:
        raise ValueError(f"articulation is missing required joints: {missing}")
    return [index_by_name[name] for name in required_names]


def register_agent() -> type[Any]:
    """Register and return the ManiSkill BaseAgent class on demand."""
    try:
        import numpy as np
        import sapien
        from mani_skill.agents.base_agent import BaseAgent, Keyframe
        from mani_skill.agents.controllers import PDJointPosControllerConfig
        from mani_skill.agents.registration import register_agent as mani_register_agent
        from mani_skill.utils import common
    except ImportError as exc:
        raise RuntimeError(
            "ManiSkill is not installed. Run `uv sync` in sim/mobile_nexarm/maniskill."
        ) from exc

    generated_urdf = Path(__file__).resolve().parents[1] / "generated" / "mobile_bi_nexarm.urdf"
    left_arm = [f"left_{name}" for name in ARM_FEATURES]
    right_arm = [f"right_{name}" for name in ARM_FEATURES]

    @mani_register_agent()
    class MobileBiNexArmAgent(BaseAgent):
        uid = "mobile_bi_nexarm"
        urdf_path = str(generated_urdf)
        keyframes = {
            "home": Keyframe(
                qpos=np.asarray(
                    [0.0, 0.0, 0.0, 0.18] + [0.0] * 5 + [0.0] + [0.0] * 5 + [0.0],
                    dtype=np.float32,
                ),
                pose=sapien.Pose(),
            )
        }

        @property
        def _controller_configs(self):
            base = PDJointPosControllerConfig(
                list(BASE_JOINTS),
                lower=None,
                upper=None,
                stiffness=2e3,
                damping=2e2,
                force_limit=1e3,
                normalize_action=False,
            )
            lift = PDJointPosControllerConfig(
                list(LIFT_JOINTS),
                lower=0.0,
                upper=0.35,
                stiffness=2e3,
                damping=2e2,
                force_limit=1e3,
                normalize_action=False,
            )
            arm_configs = {
                side: PDJointPosControllerConfig(
                    names,
                    lower=None,
                    upper=None,
                    stiffness=1e3,
                    damping=1e2,
                    force_limit=100,
                    normalize_action=False,
                )
                for side, names in (("left_arm", left_arm), ("right_arm", right_arm))
            }
            gripper_configs = {
                side: PDJointPosControllerConfig(
                    [f"{side}_gripper"],
                    lower=-0.0255,
                    upper=0.0,
                    stiffness=1e3,
                    damping=1e2,
                    force_limit=100,
                    normalize_action=False,
                )
                for side in ("left", "right")
            }
            return common.deepcopy_dict(
                {
                    "pd_joint_pos": {
                        "base": base,
                        "lift": lift,
                        **arm_configs,
                        "left_gripper": gripper_configs["left"],
                        "right_gripper": gripper_configs["right"],
                    },
                    "fixed_base_pd_joint_pos": {
                        "lift": deepcopy(lift),
                        **{name: deepcopy(value) for name, value in arm_configs.items()},
                        "left_gripper": deepcopy(gripper_configs["left"]),
                        "right_gripper": deepcopy(gripper_configs["right"]),
                    },
                }
            )

        def _after_init(self) -> None:
            names = [
                *BASE_JOINTS,
                *LIFT_JOINTS,
                *left_arm,
                "left_gripper",
                *right_arm,
                "right_gripper",
            ]
            self.canonical_joint_indices = resolve_joint_indices(list(self.robot.get_active_joints()), names)

    return MobileBiNexArmAgent
