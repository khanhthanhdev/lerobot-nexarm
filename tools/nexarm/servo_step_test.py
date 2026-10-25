#!/usr/bin/env python

"""Time single-joint step moves on the real follower to measure servo speed and response lag.

Recorded datasets stamp frames at a nominal 1/fps, so a slow record loop inflates any speed measured from
them. This test polls the follower with ``time.perf_counter`` timestamps instead. It runs each requested
``motion_speed`` (CMD 56) and reports per step:

- ``latency_s``: time until the joint has moved 2% of the step;
- ``rise_s``: 10% to 90% of the step;
- ``mean_speed_ticks_s``: step size * 0.8 / ``rise_s``;
- ``peak_speed_ticks_s``: fastest movement over any ``--speed-window-s`` window;
- ``final_error_ticks``: target minus the settled position.

Compare ``peak_speed_ticks_s`` at ``--motion-speed 2000`` and ``0`` to see whether CMD 56 limits CMD 97 moves.

The joint moves from its current position by ``+delta`` and back, ``--repeats`` times, while every other joint
holds still. Clear the space around the arm first. The follower's motion parameters are restored to the
``NexArmFollowerConfig`` defaults (the 2000/100 that ``record.yaml`` also uses) at the end, and torque stays
on so the arm keeps holding its pose. ``--motion-speed 0`` means no CMD 56 limit, so start with a small
``--delta``.

Example:
    uv run python tools/nexarm/servo_step_test.py --port /dev/ttyUSB1 --joint elbow_flex --delta 800 \
        --motion-speed 2000 0
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from lerobot_robot_nexarm.motors import NexArmMotorsBus
from lerobot_robot_nexarm.motors.nexarm import (
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    JOINT_NAMES,
    POSITION_MAX,
    POSITION_MIN,
)
from lerobot_robot_nexarm.nexarm_follower.config_nexarm_follower import NexArmFollowerConfig

# A step has started once the joint covers this share of it, which ignores servo jitter around the start.
MOVE_START_FRACTION = 0.02


@contextmanager
def follower_bus(port: str, baudrate: int) -> Iterator[NexArmMotorsBus]:
    """Connected follower bus in LeRobot bridge mode; leaves the bridge and closes the port on exit."""
    bus = NexArmMotorsBus(port=port, baudrate=baudrate)
    bus.connect()
    bus.enter_lerobot_mode()
    try:
        yield bus
    finally:
        try:
            bus.exit_lerobot_mode()
        finally:
            bus.disconnect()


def run_step(
    bus: NexArmMotorsBus,
    hold: Sequence[int],
    joint_index: int,
    target: int,
    timeout_s: float,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[np.ndarray, np.ndarray]:
    """Command one joint to ``target`` and poll until the timeout.

    Returns times in seconds since the command and the polled joint positions.
    """
    command = list(hold)
    command[joint_index] = target
    start = clock()
    bus.write_positions(command)
    times, positions = [], []
    while (now := clock() - start) < timeout_s:
        positions.append(bus.read_positions()[joint_index])
        times.append(now)
    return np.asarray(times), np.asarray(positions, dtype=np.float64)


def analyze_step(
    times: np.ndarray, positions: np.ndarray, start: float, target: float, speed_window_s: float
) -> dict[str, float | None]:
    """Latency, rise time and speeds of one step response; ``None`` where the joint never got there."""
    step = target - start
    progress = (positions - start) / step

    def first_time(fraction: float) -> float | None:
        reached = np.flatnonzero(progress >= fraction)
        return float(times[reached[0]]) if reached.size else None

    t10, t90 = first_time(0.1), first_time(0.9)
    rise = t90 - t10 if t10 is not None and t90 is not None and t90 > t10 else None
    peak = 0.0
    for i, t in enumerate(times):
        later = np.flatnonzero(times >= t + speed_window_s)
        if later.size:
            j = later[0]
            peak = max(peak, abs(positions[j] - positions[i]) / (times[j] - times[i]))
    return {
        "latency_s": first_time(MOVE_START_FRACTION),
        "rise_s": rise,
        "mean_speed_ticks_s": abs(step) * 0.8 / rise if rise else None,
        "peak_speed_ticks_s": peak if peak > 0 else None,
        "final_error_ticks": float(target - positions[-1]),
        "samples": len(times),
    }


def step_targets(joint: str, current: int, delta: int) -> tuple[int, int]:
    """Start and target of the step, refused when the target leaves the joint's raw range."""
    low, high = (GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS) if joint == "gripper" else (POSITION_MIN, POSITION_MAX)
    target = current + delta
    if not low <= target <= high:
        raise ValueError(
            f"{joint} target {target} leaves its raw range [{low}, {high}]; move the joint or flip --delta"
        )
    return current, target


def summarize(results: list[dict]) -> dict[int, dict[str, float | None]]:
    """Median of every metric per motion_speed across steps."""
    summary: dict[int, dict[str, float | None]] = {}
    for speed in sorted({r["motion_speed"] for r in results}):
        rows = [r["metrics"] for r in results if r["motion_speed"] == speed]
        summary[speed] = {}
        for key in rows[0]:
            values = [row[key] for row in rows if row[key] is not None]
            summary[speed][key] = float(np.median(values)) if values else None
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default="/dev/ttyUSB1", help="Follower serial port")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--joint", choices=JOINT_NAMES, default="elbow_flex")
    parser.add_argument("--delta", type=int, default=600, help="Step size in raw ticks (sign sets direction)")
    parser.add_argument(
        "--motion-speed", type=int, nargs="+", default=[2000, 0], help="CMD 56 speeds to test"
    )
    parser.add_argument("--motion-acc", type=int, default=NexArmFollowerConfig.motion_acc)
    parser.add_argument("--repeats", type=int, default=2, help="Out-and-back cycles per motion speed")
    parser.add_argument("--timeout", type=float, default=2.0, help="Polling time per step in seconds")
    parser.add_argument("--speed-window-s", type=float, default=0.1)
    parser.add_argument("--yes", action="store_true", help="Skip the clear-workspace confirmation")
    parser.add_argument("--out", type=Path, default=Path("outputs/nexarm/servo_step_test.json"))
    args = parser.parse_args(argv)
    if args.delta == 0 or abs(args.delta) > 1500:
        parser.error("--delta must be non-zero and at most 1500 ticks")
    if any(not 0 <= s <= 3400 for s in args.motion_speed):
        parser.error("--motion-speed values must be within 0..3400")
    if args.repeats < 1 or args.timeout <= 0 or args.speed_window_s <= 0:
        parser.error("--repeats, --timeout and --speed-window-s must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    joint_index = JOINT_NAMES.index(args.joint)
    defaults = NexArmFollowerConfig(port=args.port)
    results = []
    with follower_bus(args.port, args.baudrate) as bus:
        hold = bus.read_positions()
        start, target = step_targets(args.joint, hold[joint_index], args.delta)
        if not args.yes:
            input(
                f"{args.joint} will move {start} -> {target} -> {start}. Clear the workspace and press Enter."
            )
        try:
            for speed in args.motion_speed:
                bus.write_motion_params(acc=args.motion_acc, speed=speed)
                bus.set_torque(True)
                for _ in range(args.repeats):
                    for begin, end in ((start, target), (target, start)):
                        times, positions = run_step(bus, hold, joint_index, end, args.timeout)
                        metrics = analyze_step(times, positions, begin, end, args.speed_window_s)
                        results.append(
                            {
                                "motion_speed": speed,
                                "start": begin,
                                "target": end,
                                "metrics": metrics,
                                "times_s": times.tolist(),
                                "positions": positions.tolist(),
                            }
                        )
        finally:
            bus.write_motion_params(acc=defaults.motion_acc, speed=defaults.motion_speed)

    summary = summarize(results)
    print(f"\n{args.joint}: step {abs(args.delta)} ticks, median over {2 * args.repeats} steps")
    print(f"{'motion_speed':>12} {'latency_s':>10} {'rise_s':>8} {'mean_t/s':>9} {'peak_t/s':>9} {'err':>6}")
    for speed, row in summary.items():

        def fmt(key: str, spec: str, row: dict = row) -> str:
            return format(row[key], spec) if row[key] is not None else "-"

        print(
            f"{speed:>12} {fmt('latency_s', '10.3f'):>10} {fmt('rise_s', '8.3f'):>8} "
            f"{fmt('mean_speed_ticks_s', '9.0f'):>9} {fmt('peak_speed_ticks_s', '9.0f'):>9} "
            f"{fmt('final_error_ticks', '6.0f'):>6}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"joint": args.joint, "delta": args.delta, "motion_acc": args.motion_acc, "summary": summary}
    args.out.write_text(json.dumps(payload | {"steps": results}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
