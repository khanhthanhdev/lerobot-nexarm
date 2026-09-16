#!/usr/bin/env python

"""Interactive 6-DOF Cartesian Teleoperation for NexArm (Simulation & Real Follower).

Enables intuitive end-effector Cartesian control (X/Y/Z translation, Roll/Pitch/Yaw
orientation, and gripper toggle) without requiring a physical leader arm. Solves
real-time inverse kinematics and displays live gravity compensation torques.

Usage:
    # 1. Teleoperate simulated follower in MuJoCo
    uv run python examples/nexarm/teleoperate_cartesian.py --robot sim

    # 2. Teleoperate physical follower arm
    uv run python examples/nexarm/teleoperate_cartesian.py --robot real --port /dev/ttyUSB1

    # 3. Dry-run test (headless validation)
    uv run python examples/nexarm/teleoperate_cartesian.py --robot sim --dry-run
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

import numpy as np

from lerobot.motors.nexarm import NexArmKinematicsDynamics
from lerobot.motors.nexarm.kinematics_dynamics import HOME_POSITIONS, JOINT_NAMES, RAW_RANGES
from lerobot.robots.nexarm_sim import NexArmSim, NexArmSimConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NexArm 6-DOF Cartesian Teleoperation")
    parser.add_argument("--robot", choices=["sim", "real"], default="sim", help="Target robot (sim or real)")
    parser.add_argument("--port", default="/dev/ttyUSB1", help="Serial port for real follower arm")
    parser.add_argument("--fps", type=int, default=30, help="Control loop rate (Hz)")
    parser.add_argument("--step-mm", type=float, default=5.0, help="Default translation step (mm)")
    parser.add_argument("--step-deg", type=float, default=3.0, help="Default rotation step (degrees)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Execute 5 sample steps and exit (for automated testing)"
    )
    return parser.parse_args()


class RawTerminal:
    """Context manager for non-blocking raw terminal input."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.old_settings = None

    def __enter__(self) -> RawTerminal:
        if self.fd is not None:
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        return self

    def __exit__(self, *args) -> None:
        if self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self, timeout_s: float = 0.03) -> str | None:
        if self.fd is None:
            return None
        rlist, _, _ = select.select([self.fd], [], [], timeout_s)
        if rlist:
            return sys.stdin.read(1)
        return None


def format_hud(
    target_xyz: np.ndarray,
    target_rpy_deg: np.ndarray,
    current_q: dict[str, float],
    grav_torques: dict[str, float],
    gripper_raw: float,
    step_m: float,
    step_rad: float,
    ik_ok: bool,
) -> str:
    lines = [
        "\033[2J\033[H",  # Clear screen and move cursor home
        "================================================================================",
        "                     NEXARM 6-DOF CARTESIAN TELEOPERATION                       ",
        "================================================================================",
        f" Target EE Pose (World):  X: {target_xyz[0] * 1000:+6.1f} mm | Y: {target_xyz[1] * 1000:+6.1f} mm | Z: {target_xyz[2] * 1000:+6.1f} mm",
        f" Orientation (RPY deg):   R: {target_rpy_deg[0]:+6.1f}°  | P: {target_rpy_deg[1]:+6.1f}°  | Y: {target_rpy_deg[2]:+6.1f}°",
        f" Step Resolution:         {step_m * 1000:.1f} mm / {np.rad2deg(step_rad):.1f}°   | IK Status: {'[OK]' if ik_ok else '[REACH LIMIT]'}",
        "--------------------------------------------------------------------------------",
        " Joint Positions (Raw 0..4095):",
        "   " + " | ".join(f"{name[:8]}: {current_q[f'{name}.pos']:4.0f}" for name in JOINT_NAMES[:-1]),
        " Gravity Compensation Torques g(q) [Nm]:",
        "   " + " | ".join(f"{name[:8]}: {grav_torques.get(name, 0.0):+5.2f}" for name in JOINT_NAMES[:-1]),
        f" Gripper Status:          Raw: {gripper_raw:4.0f} ({'OPEN' if gripper_raw > 2400 else 'CLOSED'})",
        "================================================================================",
        " Controls:",
        "   [W/S] : +/- X (Forward/Back)      [I/K] : +/- Pitch (Tilt Up/Down)",
        "   [A/D] : +/- Y (Left/Right)        [J/L] : +/- Yaw   (Pan Left/Right)",
        "   [R/F] : +/- Z (Up/Down)           [U/O] : +/- Roll  (Wrist Twist)",
        "   [SPACE/G] : Toggle Gripper        [1/2/3] : Fine/Medium/Coarse Step",
        "   [H]   : Return to Home            [Q]   : Quit Teleoperation",
        "================================================================================",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    # 1. Initialize Kinematics & Dynamics solver
    kd = NexArmKinematicsDynamics()
    fk_home = kd.forward_kinematics(HOME_POSITIONS)

    target_xyz = fk_home["position"].copy()
    target_rpy = fk_home["rpy_rad"].copy()
    gripper_raw = float(RAW_RANGES["gripper"][0])  # Open by default

    current_q = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
    current_q["gripper.pos"] = gripper_raw

    # 2. Connect Robot
    robot = None
    if args.robot == "sim":
        config = NexArmSimConfig(id="cartesian_teleop", fps=args.fps)
        robot = NexArmSim(config)
        robot.connect()
    else:
        from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig

        config = NexArmFollowerConfig(port=args.port, id="cartesian_teleop")
        robot = NexArmFollower(config)
        robot.connect()

    step_m = args.step_mm / 1000.0
    step_rad = np.deg2rad(args.step_deg)

    print(f"Teleoperation started on {args.robot.upper()} robot. Press Ctrl+C or 'q' to exit.")

    step_count = 0
    running = True

    try:
        with RawTerminal() as term:
            while running:
                start_loop = time.perf_counter()

                # In dry-run mode, simulate 5 moves and exit cleanly
                if args.dry_run:
                    step_count += 1
                    target_xyz[0] += 0.002
                    target_xyz[2] += 0.001
                    if step_count >= 5:
                        running = False
                else:
                    key = term.read_key(timeout_s=1.0 / args.fps)
                    if key is not None:
                        key = key.lower()
                        if key == "q":
                            break
                        elif key == "w":
                            target_xyz[0] += step_m
                        elif key == "s":
                            target_xyz[0] -= step_m
                        elif key == "a":
                            target_xyz[1] += step_m
                        elif key == "d":
                            target_xyz[1] -= step_m
                        elif key == "r":
                            target_xyz[2] += step_m
                        elif key == "f":
                            target_xyz[2] -= step_m
                        elif key == "i":
                            target_rpy[1] += step_rad
                        elif key == "k":
                            target_rpy[1] -= step_rad
                        elif key == "j":
                            target_rpy[2] += step_rad
                        elif key == "l":
                            target_rpy[2] -= step_rad
                        elif key == "u":
                            target_rpy[0] -= step_rad
                        elif key == "o":
                            target_rpy[0] += step_rad
                        elif key in (" ", "g"):
                            # Toggle gripper between open and closed
                            if gripper_raw < 2000:
                                gripper_raw = float(RAW_RANGES["gripper"][1])
                            else:
                                gripper_raw = float(RAW_RANGES["gripper"][0])
                            current_q["gripper.pos"] = gripper_raw
                        elif key == "1":
                            step_m, step_rad = 0.001, np.deg2rad(1.0)
                        elif key == "2":
                            step_m, step_rad = 0.005, np.deg2rad(3.0)
                        elif key == "3":
                            step_m, step_rad = 0.015, np.deg2rad(8.0)
                        elif key == "h":
                            target_xyz = fk_home["position"].copy()
                            target_rpy = fk_home["rpy_rad"].copy()

                # Solve IK for updated target
                ik_solution = kd.inverse_kinematics(
                    target_xyz,
                    target_rpy=target_rpy,
                    current_joint_positions=current_q,
                    tolerance_m=0.003,
                )

                ik_ok = ik_solution is not None
                if ik_ok:
                    for name in JOINT_NAMES[:-1]:
                        current_q[f"{name}.pos"] = ik_solution[name]
                    robot.send_action(current_q)

                # Compute gravity compensation torques
                grav_torques = kd.gravity_compensation_torques(current_q)

                # Render terminal HUD (skip HUD in dry-run to keep log clean)
                if not args.dry_run and sys.stdin.isatty():
                    hud_text = format_hud(
                        target_xyz,
                        np.rad2deg(target_rpy),
                        current_q,
                        grav_torques,
                        gripper_raw,
                        step_m,
                        step_rad,
                        ik_ok,
                    )
                    sys.stdout.write(hud_text + "\n")
                    sys.stdout.flush()

                elapsed = time.perf_counter() - start_loop
                sleep_t = max(0.0, (1.0 / args.fps) - elapsed)
                time.sleep(sleep_t)

    finally:
        if robot is not None and robot.is_connected:
            robot.disconnect()
        print("\nTeleoperation session ended cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
