#!/usr/bin/env python

"""Check Leader-Follower alignment for NexArm at the vendor Home pose.

Zero positions live in the servos, not in software: this script no longer writes a ``homing_offset``.
It writes identity calibration files (resetting any older software offsets after backing them up) and
prints how far each joint is from the other arm. Fix any offset in hardware: the follower on the
desktop's Servo Advanced page, the leader on the Synchronizer page.

Place both arms by hand at the vendor Home pose (every joint at 2048 ticks), then:
  uv run python tools/nexarm/calibrate_arms.py --leader-port /dev/ttyUSB0 --follower-port /dev/ttyUSB1
"""

import argparse

from lerobot_robot_nexarm.motors import NexArmMotorsBus
from nexarm_service.calibration import (
    DEFAULT_TOLERANCE_TICKS,
    build_report,
    reset_non_identity_files,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Check NexArm Leader-Follower alignment at Home")
    parser.add_argument("--leader-port", default="/dev/ttyUSB0", help="Serial port for Leader arm")
    parser.add_argument("--follower-port", default="/dev/ttyUSB1", help="Serial port for Follower arm")
    parser.add_argument(
        "--tolerance", type=int, default=DEFAULT_TOLERANCE_TICKS, help="Allowed ticks per joint"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    reset = reset_non_identity_files()
    for path in reset:
        print(f"Reset software offsets in {path} (a backup sits next to it).")

    leader = NexArmMotorsBus(port=args.leader_port)
    follower = NexArmMotorsBus(port=args.follower_port)
    leader.connect()
    follower.connect()
    follower.enter_lerobot_mode()
    try:
        # Read only: the follower stays un-energised so it can be posed by hand.
        leader_raw = leader.read_positions()
        follower_raw = follower.read_positions()
    finally:
        follower.exit_lerobot_mode()
        leader.disconnect()
        follower.disconnect()

    report = build_report(leader_raw, follower_raw, reset, tolerance=args.tolerance)
    print(f"{'Joint':<15} | {'Leader Raw':<10} | {'Leader Mapped':<13} | {'Follower':<8} | {'Delta':<6}")
    print("-" * 64)
    for j in report.joints:
        flag = "" if j.ok else "  <-- fix in hardware"
        print(
            f"{j.joint:<15} | {j.leader_raw:<10} | {j.leader_mapped:<13} | {j.follower:<8} | {j.delta:+6d}{flag}"
        )
    print(f"\n{report.hint}")


if __name__ == "__main__":
    main()
