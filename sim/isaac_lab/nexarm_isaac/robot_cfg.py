"""Isaac Lab articulation config for the NexArm follower, built from ``nexarm_scene.json``.

The drive gains, effort limits and armature are the MJCF position-actuator values, so a joint target produces
the same servo response in Isaac as in MuJoCo. The robot USD comes from ``build_usd.py``.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .scene_spec import USD_DIR, SceneSpec

ROBOT_USD = USD_DIR / "nexarm.usd"
# Isaac Lab 2.1 renamed the simulator-side limits; accept both.
_EFFORT_FIELD = "effort_limit_sim" if hasattr(ImplicitActuatorCfg, "effort_limit_sim") else "effort_limit"


def make_robot_cfg(spec: SceneSpec | None = None, usd_path: Path | str = ROBOT_USD) -> ArticulationCfg:
    spec = spec or SceneSpec.load()
    if not Path(usd_path).is_file():
        raise FileNotFoundError(f"{usd_path} not found. Run sim/isaac_lab/build_usd.py first.")
    actuators = {}
    for joint, drive in spec.robot["drives"].items():
        actuators[joint] = ImplicitActuatorCfg(
            joint_names_expr=[joint],
            stiffness=drive["stiffness"],
            damping=drive["damping"],
            armature=drive["armature"],
            # The gear pinion is undriven in the MJCF; give it only damping, with room to apply it.
            **{_EFFORT_FIELD: drive["effort"] or 1.0},
        )
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # The MJCF arm collides with itself except for the <exclude> pairs, which the env filters.
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(spec.robot["base_pos"]),
            rot=tuple(spec.robot["base_quat"]),
            joint_pos=spec.home_joint_positions(),
        ),
        actuators=actuators,
        soft_joint_pos_limit_factor=1.0,
    )
