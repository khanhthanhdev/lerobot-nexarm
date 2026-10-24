#!/usr/bin/env python

"""Audit a real NexArm LeRobot dataset before using it to tune or check the MuJoCo twin.

Reads only the joint data (``action`` and ``observation.state``, raw servo ticks) and reports:

- **Timing:** LeRobot stamps frames at a nominal 1/fps, so a record loop that ran slow ("Record loop is running
  slower ...") inflates every speed. An episode that used its whole ``--episode-time-s`` has a known
  average rate. Only those full-length episodes are used for speeds.
- **Speeds:** per-joint sustained speed over ``--speed-window-s``.
  - A joint counts as saturated when, at its fastest commands, the measured joint moves clearly slower than
    the command; only then is its fast speed a cap.
  - A teleoperated joint that never saturated was simply never moved faster, so its speed is a lower bound.
- **Lag:** the shift (in frames) that best aligns command and state, per episode. Episodes far from the median
  usually come from a session that logged differently.
- **Gripper:** where the closed gripper settles (empty vs holding an object), and how far releases open.
  Releases that stay above the stack-bowls grasp-assist release threshold never drop a bowl when replayed in
  the sim.

Example:
    uv run python examples/nexarm/audit_real_dataset.py --repo-id thanhkt/nexarm_stack_bowls --episode-time-s 15
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES
from lerobot_robot_nexarm.nexarm_sim.stack_bowls_task import GRIPPER_ASSIST_RELEASE_POS

try:  # imported as a package module (tests) or run as a script
    from examples.nexarm.calibrate_sim import Episode, load_episodes, parse_episode_list
except ModuleNotFoundError:
    from calibrate_sim import Episode, load_episodes, parse_episode_list  # type: ignore[no-redef]

GRIPPER = JOINT_NAMES.index("gripper")
ARM = slice(0, GRIPPER)
FULL_LENGTH_FRACTION = 0.97  # a full-length episode keeps at least this share of its nominal frames
CLOSE_COMMAND = 2600  # leader gripper command counted as "closed"
OPEN_COMMAND = 2300  # leader gripper command counted as "opening"
SETTLED_TICKS = 8  # gripper moves less than this per frame once it has stopped
RELEASE_FRAMES = 15  # a release goes from closed to open, and reaches its widest opening, within this
LAG_ANOMALY_FRAMES = 2
FAST_COMMAND_QUANTILE = 99.0  # "fastest commands" for the saturation check
SATURATION_RATIO = 0.9


def best_lag(episode: Episode, max_lag: int = 10) -> int:
    """Frames by which the arm state trails the command, by mean absolute error."""
    actions, states = episode.actions[:, ARM], episode.states[:, ARM]
    lags = range(min(max_lag, len(actions) - 1) + 1)
    errors = [np.mean(np.abs(actions[: len(actions) - k] - states[k:])) for k in lags]
    return int(np.argmin(errors))


def window_speeds(values: np.ndarray, window: int, fps: float) -> np.ndarray:
    """Absolute speed over ``window`` frames, ticks/s, shape (T - window, joints)."""
    return np.abs(values[window:] - values[:-window]) * fps / window


def speed_report(episodes: Sequence[Episode], fps: float, window: int) -> dict[str, dict]:
    """Per-joint sustained speed percentiles and a saturation verdict.

    Each command window is compared with the state window ``best_lag`` frames later, so response lag alone
    does not read as saturation.
    """
    state_parts, command_parts = [], []
    for e in episodes:
        lag = best_lag(e)
        state_parts.append(window_speeds(e.states[lag:], window, fps))
        command_parts.append(window_speeds(e.actions[: len(e.actions) - lag], window, fps))
    state, command = np.concatenate(state_parts), np.concatenate(command_parts)
    report = {}
    for i, name in enumerate(JOINT_NAMES):
        fast = command[:, i] >= np.percentile(command[:, i], FAST_COMMAND_QUANTILE)
        fast_command = float(np.median(command[fast, i]))
        ratio = float(np.median(state[fast, i]) / fast_command) if fast_command > 0 else 1.0
        saturated = ratio < SATURATION_RATIO
        report[name] = {
            "p95": float(np.percentile(state[:, i], 95)),
            "p99_9": float(np.percentile(state[:, i], 99.9)),
            "max": float(state[:, i].max()),
            "fast_command_tracking_ratio": ratio,
            "saturated": saturated,
            "speed_cap_ticks_s": float(np.percentile(state[:, i], 99.9)) if saturated else None,
        }
    return report


def gripper_report(episodes: Sequence[Episode]) -> dict[str, float | int | None]:
    """Where the commanded-closed gripper settles, and how far each release opens the command."""
    settled_closed, release_open = [], []
    for e in episodes:
        command, state = e.actions[:, GRIPPER], e.states[:, GRIPPER]
        still = np.r_[False, np.abs(np.diff(state)) < SETTLED_TICKS]
        settled_closed.extend(state[(command >= CLOSE_COMMAND) & still])
        # A release starts where the command drops below OPEN_COMMAND within RELEASE_FRAMES of being closed.
        for t in np.flatnonzero((command[1:] < OPEN_COMMAND) & (command[:-1] >= OPEN_COMMAND)) + 1:
            if command[max(0, t - RELEASE_FRAMES) : t].max() >= CLOSE_COMMAND:
                release_open.append(command[t : t + RELEASE_FRAMES].min())
    closed = np.asarray(settled_closed)
    releases = np.asarray(release_open)
    return {
        "closed_settled_p95": float(np.percentile(closed, 95)) if closed.size else None,
        "closed_settled_median": float(np.median(closed)) if closed.size else None,
        "closed_settled_p5": float(np.percentile(closed, 5)) if closed.size else None,
        "releases": int(releases.size),
        "release_open_command_median": float(np.median(releases)) if releases.size else None,
        "releases_above_assist_threshold": int((releases >= GRIPPER_ASSIST_RELEASE_POS).sum()),
        "assist_release_threshold": GRIPPER_ASSIST_RELEASE_POS,
    }


def audit(
    episodes: Sequence[Episode], fps: float, episode_time_s: float | None, speed_window_s: float
) -> dict:
    window = max(1, round(speed_window_s * fps))
    lengths = {e.episode_index: len(e.states) for e in episodes}
    if episode_time_s is None:
        timed = list(episodes)
    else:
        min_frames = FULL_LENGTH_FRACTION * episode_time_s * fps
        timed = [e for e in episodes if len(e.states) >= min_frames]
    lags = {e.episode_index: best_lag(e) for e in episodes}
    median_lag = int(np.median(list(lags.values())))
    anomalies = sorted(i for i, lag in lags.items() if abs(lag - median_lag) >= LAG_ANOMALY_FRAMES)
    return {
        "fps": fps,
        "episodes": len(episodes),
        "frames": {
            "min": min(lengths.values()),
            "median": float(np.median(list(lengths.values()))),
            "max": max(lengths.values()),
        },
        "timing_trusted_episodes": [e.episode_index for e in timed],
        "speed_window_s": window / fps,
        "speeds_ticks_s": speed_report(timed, fps, window) if timed else None,
        "lag_frames_median": median_lag,
        "lag_anomaly_episodes": anomalies,
        "gripper": gripper_report(episodes),
    }


def print_report(report: dict) -> None:
    trusted = report["timing_trusted_episodes"]
    print(f"{report['episodes']} episodes at {report['fps']} fps, frames {report['frames']}")
    print(
        f"Timing-trusted (full-length) episodes: {len(trusted)} {trusted[:20]}{' ...' if len(trusted) > 20 else ''}"
    )
    if report["speeds_ticks_s"] is None:
        print("No full-length episodes: speeds are not reported. Check --episode-time-s.")
    else:
        print(f"\nSustained speed over {report['speed_window_s']:.3f} s, ticks/s")
        print(f"{'joint':<15}{'p95':>8}{'p99.9':>8}{'max':>8}{'tracking':>10}  cap")
        for name, row in report["speeds_ticks_s"].items():
            cap = (
                f"{row['speed_cap_ticks_s']:.0f}" if row["saturated"] else "- (never saturated: lower bound)"
            )
            print(
                f"{name:<15}{row['p95']:>8.0f}{row['p99_9']:>8.0f}{row['max']:>8.0f}"
                f"{row['fast_command_tracking_ratio']:>10.2f}  {cap}"
            )
    print(f"\nCommand-to-state lag: median {report['lag_frames_median']} frames")
    if report["lag_anomaly_episodes"]:
        print(f"Lag anomalies (check how these were recorded): {report['lag_anomaly_episodes']}")
    gripper = report["gripper"]
    print(
        "\nCommanded-closed gripper settles at: p95 {closed_settled_p95} (about empty), "
        "median {closed_settled_median}, p5 {closed_settled_p5}".format(**gripper)
    )
    print(f"Releases: {gripper['releases']}, median opening command {gripper['release_open_command_median']}")
    if gripper["releases_above_assist_threshold"]:
        print(
            f"WARNING: {gripper['releases_above_assist_threshold']} releases never open below the sim grasp-assist "
            f"release threshold ({gripper['assist_release_threshold']}); replayed in the sim they keep holding the bowl"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="Real LeRobot dataset recorded on the NexArm")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--episode-time-s",
        type=float,
        default=None,
        help="The recording's episode time limit; enables the full-length timing check",
    )
    parser.add_argument("--exclude-episodes", default="", help="Episode indices to skip, e.g. '69-92'")
    parser.add_argument("--speed-window-s", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=Path("outputs/nexarm/real_dataset_audit.json"))
    args = parser.parse_args(argv)
    try:
        args.exclude_episodes = parse_episode_list(args.exclude_episodes)
    except ValueError:
        parser.error("--exclude-episodes must look like '3,10-12'")
    if args.speed_window_s <= 0 or (args.episode_time_s is not None and args.episode_time_s <= 0):
        parser.error("--speed-window-s and --episode-time-s must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from lerobot.datasets import LeRobotDatasetMetadata

    fps = float(LeRobotDatasetMetadata(args.repo_id, root=args.root).fps)
    episodes = load_episodes(
        args.repo_id, args.root, max_episodes=10**9, max_frames=10**9, exclude=args.exclude_episodes
    )
    if not episodes:
        raise SystemExit("No usable episodes found in the dataset")
    report = audit(episodes, fps, args.episode_time_s, args.speed_window_s)
    report["repo_id"] = args.repo_id
    print_report(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
