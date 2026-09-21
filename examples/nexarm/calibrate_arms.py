#!/usr/bin/env python

"""Calibrate Leader-Follower alignment for NexArm.

Connects to both arms, turns off torque so you can freely pose them,
and computes the joint offsets so the follower matches the leader's angle exactly.

Usage:
  uv run python examples/nexarm/calibrate_arms.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1
"""

import argparse

from lerobot.motors import MotorCalibration
from lerobot.motors.nexarm.nexarm import (
    JOINT_NAMES,
    POSITION_MAX,
    POSITION_MIN,
    map_leader_to_follower,
)
from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig
from lerobot.teleoperators.nexarm_leader import NexArmLeader, NexArmLeaderConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate NexArm Leader-Follower alignment")
    parser.add_argument("--leader-port", default="/dev/ttyUSB0", help="Serial port for Leader arm")
    parser.add_argument("--follower-port", default="/dev/ttyUSB1", help="Serial port for Follower arm")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Connecting to Leader on {args.leader_port}...")
    leader = NexArmLeader(NexArmLeaderConfig(port=args.leader_port))
    leader.connect(calibrate=False)

    print(f"Connecting to Follower on {args.follower_port}...")
    follower = NexArmFollower(NexArmFollowerConfig(port=args.follower_port))
    follower.connect(calibrate=False)

    # Disable torque on follower so user can position it by hand
    follower.bus.set_torque(False)

    print("\n" + "=" * 60)
    print("NEXARM ALIGNMENT CALIBRATION")
    print("=" * 60)
    print("1. Both arms now have torque OFF (you can freely move them by hand).")
    print("2. Move both arms so they are in the EXACT SAME physical pose")
    print("   (e.g., both straight upright, or both resting in the home position).")
    print("3. Press [ENTER] when ready to record the calibration.")
    print("=" * 60 + "\n")

    try:
        input("Press [ENTER] to record calibration...")
    except KeyboardInterrupt:
        print("\nCalibration canceled.")
        leader.disconnect()
        follower.disconnect()
        return

    # Read positions from both arms
    leader_pos = leader.bus.read_positions()
    follower_pos = follower.bus.read_positions()
    mapped_leader = map_leader_to_follower(leader_pos)

    print("\nCaptured Positions:")
    print(
        f"{'Joint':<15} | {'Leader Raw':<10} | {'Leader Mapped':<13} | {'Follower Raw':<12} | {'Offset (Diff)':<12}"
    )
    print("-" * 72)

    new_calibration = {}
    for i, name in enumerate(JOINT_NAMES):
        l_raw = leader_pos[i]
        l_map = mapped_leader[i]
        f_raw = follower_pos[i]
        diff = f_raw - l_map
        homing_offset = 2048 + diff

        new_calibration[name] = MotorCalibration(
            id=i + 1,
            drive_mode=0,
            homing_offset=homing_offset,
            range_min=POSITION_MIN,
            range_max=POSITION_MAX,
        )
        print(f"{name:<15} | {l_raw:<10} | {l_map:<13} | {f_raw:<12} | {diff:+12d}")

    print("-" * 72)

    # Save to leader's calibration path
    leader.calibration = new_calibration
    leader._save_calibration()
    print(f"\nSuccessfully saved calibration to:\n  {leader.calibration_fpath}")

    # Also save to follower's calibration path for reference
    follower.calibration = new_calibration
    follower._save_calibration()

    print("\nCalibration complete! Now when you run `lerobot-teleoperate`,")
    print("the follower arm will account for these offsets and match the leader accurately.")

    leader.disconnect()
    follower.disconnect()


if __name__ == "__main__":
    main()
