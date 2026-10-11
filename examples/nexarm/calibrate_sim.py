#!/usr/bin/env python

"""Fit the NexArm MuJoCo twin's action latency and joint dynamics to real recordings.

Replays the recorded real ``action`` stream of each episode through the simulator, starting from the real
initial pose, and compares the simulated joint trajectory with the real ``observation.state``. It fits
the action delay and per-joint ``kp`` / damping / frictionloss scales (and their spread, used as the
domain-randomization range) and writes a ``SimCalibration`` JSON that ``generate_sim_dataset.py``,
``generate_stack_bowls_dataset.py`` and ``NexArmSimConfig.calibration_path`` accept. The file records the
fitted ``fps`` (loaders refuse another run fps) and ``fitted_joints``.

For the cleanest fit, record a few object-free episodes that sweep each joint through its range
(e.g. with teleoperate/record). Contact-rich frames (a grasped cube) bias the gripper fit, so the gripper is
not fitted unless requested. Align cameras separately with ``calibrate_camera_alignment.py``.

Example:
    uv run python examples/nexarm/calibrate_sim.py --repo-id <user>/<real_dataset> \
        --out outputs/calibration/nexarm_sim_calibration.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim import SimCalibration
from lerobot.robots.nexarm_sim.calibration import MIN_SPREAD
from lerobot.robots.nexarm_sim.mujoco_backend import NexArmMujocoBackend, resolve_model_path

PARAMS = ("kp_scale", "damping_scale", "frictionloss_scale")
MAX_SPREAD = 0.5
SCALE_RANGE = (0.25, 4.0)  # fitted multipliers never leave 2 octaves around the model
# (log2 half-range, log2 step) per sweep: coarse global search, then local refinement.
SWEEPS = ((2.0, 0.5), (0.5, 0.25), (0.25, 0.125))


@dataclass
class Episode:
    states: np.ndarray  # (T, 6) real observation.state, raw servo units, JOINT_NAMES order
    actions: np.ndarray  # (T, 6) real action


class Simulator:
    """MuJoCo backend that can be re-parameterized and replayed cheaply."""

    def __init__(self, model_path: Path, fps: int) -> None:
        self.backend = NexArmMujocoBackend(
            model_path=model_path,
            fps=fps,
            camera_width=64,
            camera_height=48,
            camera_names=(),
        )
        model = self.backend.model
        self._base = (
            model.dof_damping.copy(),
            model.dof_frictionloss.copy(),
            model.actuator_gainprm.copy(),
            model.actuator_biasprm.copy(),
        )

    def configure(self, calibration: SimCalibration) -> None:
        model = self.backend.model
        model.dof_damping[:], model.dof_frictionloss[:] = self._base[0], self._base[1]
        model.actuator_gainprm[:], model.actuator_biasprm[:] = self._base[2], self._base[3]
        for name, joint in calibration.joints.items():
            dof = int(model.jnt_dofadr[self.backend._joint_ids[name]])
            actuator = self.backend._actuator_ids[name]
            model.dof_damping[dof] *= joint.damping_scale
            model.dof_frictionloss[dof] *= joint.frictionloss_scale
            model.actuator_gainprm[actuator, 0] *= joint.kp_scale
            model.actuator_biasprm[actuator, 1] *= joint.kp_scale
        self.backend.action_delay_steps = calibration.action_delay_steps

    def rollout(self, episode: Episode) -> np.ndarray:
        """Predicted state after each real action, aligned with ``episode.states[1:]``."""
        backend = self.backend
        backend.set_joint_positions({f"{n}.pos": episode.states[0, i] for i, n in enumerate(JOINT_NAMES)})
        predicted = np.empty((len(episode.actions) - 1, len(JOINT_NAMES)))
        for t in range(len(episode.actions) - 1):
            backend.step({f"{n}.pos": float(episode.actions[t, i]) for i, n in enumerate(JOINT_NAMES)})
            positions = backend.joint_positions()
            predicted[t] = [positions[f"{n}.pos"] for n in JOINT_NAMES]
        return predicted


def squared_errors(sim: Simulator, episodes: Sequence[Episode], calibration: SimCalibration) -> np.ndarray:
    """Mean squared error per episode and joint, shape (n_episodes, 6)."""
    sim.configure(calibration)
    errors = np.empty((len(episodes), len(JOINT_NAMES)))
    for i, episode in enumerate(episodes):
        errors[i] = ((sim.rollout(episode) - episode.states[1:]) ** 2).mean(axis=0)
    return errors


def rmse(sim: Simulator, episodes: Sequence[Episode], calibration: SimCalibration) -> np.ndarray:
    """Per-joint RMSE over all episodes (raw servo units)."""
    weights = np.array([len(e.actions) - 1 for e in episodes], dtype=np.float64)
    mse = squared_errors(sim, episodes, calibration)
    return np.sqrt((mse * weights[:, None]).sum(axis=0) / weights.sum())


def fit_delay(
    sim: Simulator,
    episodes: Sequence[Episode],
    calibration: SimCalibration,
    delays: range,
    joints: Sequence[str],
) -> int:
    columns = [JOINT_NAMES.index(j) for j in joints]
    best_delay, best_cost = calibration.action_delay_steps, float("inf")
    for delay in delays:
        calibration.action_delay_steps = delay
        cost = float(rmse(sim, episodes, calibration)[columns].mean())
        if cost < best_cost - 1e-9:
            best_delay, best_cost = delay, cost
    calibration.action_delay_steps = best_delay
    return best_delay


def fit_dynamics(
    sim: Simulator, episodes: Sequence[Episode], calibration: SimCalibration, joints: Sequence[str]
) -> None:
    """Per-joint coordinate descent over kp / damping / frictionloss scales on a log grid."""
    for half_range, step in SWEEPS:
        exponents = np.arange(-half_range, half_range + 1e-9, step)
        for name in joints:
            column = JOINT_NAMES.index(name)
            joint = calibration.joints[name]
            for param in PARAMS:
                center = getattr(joint, param) if half_range < 2.0 else 1.0
                best_value, best_cost = getattr(joint, param), float("inf")
                for exponent in exponents:
                    setattr(joint, param, float(np.clip(center * 2.0**exponent, *SCALE_RANGE)))
                    cost = float(rmse(sim, episodes, calibration)[column])
                    if cost < best_cost - 1e-9:
                        best_value, best_cost = getattr(joint, param), cost
                setattr(joint, param, best_value)


def fit_spread(
    sim: Simulator, episodes: Sequence[Episode], calibration: SimCalibration, joints: Sequence[str]
) -> None:
    """Spread of per-episode best-fit scales around the calibrated value, clamped to a sane range."""
    if len(episodes) < 2:
        return
    exponents = np.arange(-1.0, 1.0 + 1e-9, 0.25)
    for name in joints:
        column = JOINT_NAMES.index(name)
        joint = calibration.joints[name]
        for param, spread_name in (
            ("damping_scale", "damping_spread"),
            ("frictionloss_scale", "frictionloss_spread"),
        ):
            final = getattr(joint, param)
            costs = []
            for exponent in exponents:
                setattr(joint, param, float(final * 2.0**exponent))
                costs.append(squared_errors(sim, episodes, calibration)[:, column])
            setattr(joint, param, final)
            best = exponents[np.argmin(np.stack(costs), axis=0)]
            spread = float(np.std(2.0**best - 1.0))
            setattr(joint, spread_name, float(np.clip(spread, MIN_SPREAD, MAX_SPREAD)))


def fit_calibration(
    sim: Simulator,
    fit_episodes: Sequence[Episode],
    joints: Sequence[str],
    delays: range,
) -> SimCalibration:
    calibration = SimCalibration()
    # Latency is shared by all joints, while gain trades off against it per joint: estimate it from every
    # arm joint so an incorrectly modeled joint cannot masquerade as extra delay.
    arm = JOINT_NAMES[:-1]
    fit_delay(sim, fit_episodes, calibration, delays, arm)
    fit_dynamics(sim, fit_episodes, calibration, joints)
    fit_delay(sim, fit_episodes, calibration, delays, arm)
    fit_spread(sim, fit_episodes, calibration, joints)
    return calibration


def load_episodes(
    repo_id: str, root: Path | None, max_episodes: int, max_frames: int, min_frames: int = 20
) -> list[Episode]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root)
    names = dataset.meta.features["action"].get("names") or [f"{n}.pos" for n in JOINT_NAMES]
    order = [list(names).index(f"{n}.pos") for n in JOINT_NAMES]
    table = dataset.hf_dataset.with_format("numpy")
    actions = np.asarray(table["action"], dtype=np.float64)[:, order]
    states = np.asarray(table["observation.state"], dtype=np.float64)[:, order]
    index = np.asarray(table["episode_index"])
    episodes = []
    for episode_index in np.unique(index)[:max_episodes]:
        rows = np.flatnonzero(index == episode_index)[:max_frames]
        if len(rows) >= min_frames:
            episodes.append(Episode(states=states[rows], actions=actions[rows]))
    return episodes


def split_episodes(episodes: list[Episode], holdout_frac: float) -> tuple[list[Episode], list[Episode]]:
    n_holdout = int(round(len(episodes) * holdout_frac)) if len(episodes) >= 3 else 0
    return episodes[: len(episodes) - n_holdout], episodes[len(episodes) - n_holdout :]


def print_table(title: str, columns: dict[str, np.ndarray]) -> None:
    print(f"\n{title}")
    print(f"{'joint':<15}" + "".join(f"{name:>14}" for name in columns))
    for i, joint in enumerate(JOINT_NAMES):
        print(f"{joint:<15}" + "".join(f"{values[i]:>14.2f}" for values in columns.values()))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, help="Real LeRobot dataset recorded on the NexArm")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/scene.xml"))
    parser.add_argument("--fps", type=int, default=None, help="Defaults to the dataset fps")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=600, help="Frames used per episode")
    parser.add_argument(
        "--joints",
        default=",".join(JOINT_NAMES[:-1]),
        help="Comma-separated joints to fit (gripper is contact-dominated; add it only for free-air data)",
    )
    parser.add_argument("--delay-range", type=int, nargs=2, default=(0, 6), metavar=("MIN", "MAX"))
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=Path("outputs/calibration/nexarm_sim_calibration.json"))
    args = parser.parse_args(argv)
    args.joints = [j.strip() for j in args.joints.split(",") if j.strip()]
    unknown = set(args.joints) - set(JOINT_NAMES)
    if unknown or not args.joints:
        parser.error(f"--joints must be a non-empty subset of {list(JOINT_NAMES)}")
    if not 0 <= args.delay_range[0] <= args.delay_range[1]:
        parser.error("--delay-range must satisfy 0 <= MIN <= MAX")
    if not 0 <= args.holdout_frac < 1 or args.episodes <= 0 or args.max_frames < 20:
        parser.error("invalid --holdout-frac, --episodes or --max-frames")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    episodes = load_episodes(args.repo_id, args.root, args.episodes, args.max_frames)
    if not episodes:
        raise SystemExit("No usable episodes found in the dataset")
    fit_episodes, holdout = split_episodes(episodes, args.holdout_frac)

    fps = args.fps
    if fps is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        fps = int(LeRobotDataset(args.repo_id, root=args.root).fps)
    sim = Simulator(resolve_model_path(args.model), fps)
    print(f"Fitting on {len(fit_episodes)} episode(s), holdout {len(holdout)}, fps {fps}")

    before = SimCalibration()
    columns = {"fit_before": rmse(sim, fit_episodes, before)}
    if holdout:
        columns["holdout_before"] = rmse(sim, holdout, before)

    calibration = fit_calibration(
        sim, fit_episodes, args.joints, range(args.delay_range[0], args.delay_range[1] + 1)
    )
    columns["fit_after"] = rmse(sim, fit_episodes, calibration)
    if holdout:
        columns["holdout_after"] = rmse(sim, holdout, calibration)

    calibration.source = {
        "repo_id": args.repo_id,
        "fit_episodes": len(fit_episodes),
        "holdout_episodes": len(holdout),
        "fps": fps,
        "fitted_joints": args.joints,
        "rmse_raw_units": {name: values.tolist() for name, values in columns.items()},
    }
    # Loaders refuse a calibration fitted at another fps and only re-centre DR on the fitted joints.
    calibration.fps = fps
    calibration.fitted_joints = list(args.joints)
    calibration.save(args.out)

    print(f"\naction_delay_steps = {calibration.action_delay_steps}")
    print_table("RMSE (raw servo units)", columns)
    print("\nscales (kp, damping, frictionloss) and DR spread")
    for name, joint in calibration.joints.items():
        print(
            f"{name:<15} kp={joint.kp_scale:.2f} damping={joint.damping_scale:.2f} "
            f"friction={joint.frictionloss_scale:.2f} spread=({joint.damping_spread:.2f}, {joint.frictionloss_spread:.2f})"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
