"""LeRobot plugin for the Hiwonder NexArm.

Importing it registers the `nexarm_follower`, `nexarm_sim` and `mobile_bi_nexarm_sim` robot types. The servo
driver is in `lerobot_robot_nexarm.motors`.
"""

from .mobile_bi_nexarm_sim import MobileBiNexArmSim, MobileBiNexArmSimConfig
from .motors import NexArmMotorsBus
from .nexarm_follower import NexArmFollower, NexArmFollowerConfig
from .nexarm_sim import NexArmSim, NexArmSimConfig

__all__ = [
    "MobileBiNexArmSim",
    "MobileBiNexArmSimConfig",
    "NexArmFollower",
    "NexArmFollowerConfig",
    "NexArmMotorsBus",
    "NexArmSim",
    "NexArmSimConfig",
]
