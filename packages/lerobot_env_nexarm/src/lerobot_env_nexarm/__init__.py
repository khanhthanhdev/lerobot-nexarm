"""LeRobot plugin for the NexArm MuJoCo tasks.

Importing it registers the `nexarm` env type and the `NexArmPickPlace-v0` and `NexArmStackBowls-v0` gym ids.
"""

from .configs import NexArmEnv
from .nexarm import NexArmPickPlaceEnv, NexArmStackBowlsEnv

__all__ = ["NexArmEnv", "NexArmPickPlaceEnv", "NexArmStackBowlsEnv"]
