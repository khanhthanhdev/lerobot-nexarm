"""Isaac Lab articulation configuration for the Hiwonder NexArm robot.

This configuration defines the physics parameters, actuator models (implicit PD),
default initial state, and joint drive stiffness/damping for Isaac Lab / Isaac Sim.
"""

try:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.utils import configclass
except ImportError:
    # Allow importing on machines without Isaac Lab installed for linting/inspection
    def configclass(cls):
        return cls

    sim_utils = None
    ImplicitActuatorCfg = None
    ArticulationCfg = object


@configclass
class NexArmCfg(ArticulationCfg):
    """Configuration for the Hiwonder NexArm 5-DoF + Gripper robot."""

    def __init__(self, usd_path: str = "sim/isaac_lab/nexarm.usd"):
        if sim_utils is None:
            return

        super().__init__(
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "joint_1_base_to_link_1": 0.0,
                    "joint_2_link_1_to_link_2": 0.0,
                    "joint_3_link_2_to_link_3": 0.0,
                    "joint_4_link_3_to_link_4": 0.0,
                    "joint_5_link_4_to_link_5": 0.0,
                    "right_jaw_slide_joint": 0.0,
                    "left_jaw_slide_joint": 0.0,
                },
            ),
            actuators={
                "shoulder_lift": ImplicitActuatorCfg(
                    joint_names_expr=["joint_2_link_1_to_link_2"],
                    stiffness=80.0,
                    damping=4.0,
                    effort_limit=6.37,
                    velocity_limit=5.51,
                    armature=0.008,
                ),
                "mid_arm": ImplicitActuatorCfg(
                    joint_names_expr=[
                        "joint_1_base_to_link_1",
                        "joint_3_link_2_to_link_3",
                        "joint_4_link_3_to_link_4",
                    ],
                    stiffness=40.0,
                    damping=2.0,
                    effort_limit=2.94,
                    velocity_limit=5.51,
                    armature=0.004,
                ),
                "wrist_roll": ImplicitActuatorCfg(
                    joint_names_expr=["joint_5_link_4_to_link_5"],
                    stiffness=20.0,
                    damping=0.8,
                    effort_limit=1.18,
                    velocity_limit=5.24,
                    armature=0.001,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["right_jaw_slide_joint", "left_jaw_slide_joint"],
                    stiffness=300.0,
                    damping=10.0,
                    effort_limit=50.0,
                    velocity_limit=0.042,
                    armature=0.0005,
                ),
            },
        )
