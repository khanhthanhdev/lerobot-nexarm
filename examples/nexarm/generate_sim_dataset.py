#!/usr/bin/env python

"""Generate successful scripted NexArm MuJoCo episodes as a LeRobot dataset.

Each episode samples its own cube/target layout, trajectory timing and offsets, domain randomization and
(optionally) action latency, all derived from the episode seed. Episodes are kept only if the physical
task gate and the quality gates pass. Provenance goes to ``generation_report.json`` and one line per
attempt to ``generation_episodes.jsonl``.

Example:
    MUJOCO_GL=egl uv run python examples/nexarm/generate_sim_dataset.py \
        --repo-id local/nexarm_sim_pick_place \
        --root outputs/datasets/nexarm_sim_pick_place \
        --episodes 20

Calibrated to a real arm (see calibrate_sim.py), sharded over 4 GPUs, held-out eval split:
    uv run python examples/nexarm/generate_sim_dataset.py --calibration outputs/calibration/nexarm.json \
        --episodes 200 --workers 4 --gpus 0,1,2,3 --split eval
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:  # imported as a package module (tests) or run as a script
    from examples.nexarm import sim_dataset_utils as utils
except ModuleNotFoundError:
    import sim_dataset_utils as utils  # type: ignore[no-redef]

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot.motors.nexarm.mujoco_mapping import reachable_raw_range
from lerobot.motors.nexarm.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS, JOINT_NAMES
from lerobot.robots.nexarm_sim import NexArmPickPlaceTask, NexArmSim, NexArmSimConfig
from lerobot.robots.nexarm_sim.mujoco_backend import (
    HOME_POSITIONS,
    RAW_RANGES,
    NexArmMujocoBackend,
    resolve_model_path,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

DEFAULT_TASK = "Pick up the red cube, place it in the green target zone, and release it."
OPEN_GRIPPER = float(GRIPPER_OPEN_POS)
CLOSED_GRIPPER = float(GRIPPER_CLOSED_POS)
ARM_JOINTS = JOINT_NAMES[:-1]
NUM_STAGES = 10
SETTLE_TOLERANCE = 15.0  # raw servo units: 15 * 360 / 4096 ticks per revolution ~= 1.3 degrees
MIN_SETTLE_STEPS = 3
# Min-jerk has ~1.9x the peak speed of linear interpolation, which shakes the cube out of the jaws while
# carrying it, so it is only used for the free-space stages (approach, descend, release, retreat).
SMOOTH_STAGES = frozenset({0, 1, 8, 9})


@dataclass(frozen=True)
class TrajectoryVariation:
    """Per-episode perturbation of the scripted pick/place waypoints. The default is the nominal path."""

    step_scale: tuple[float, ...] = (1.0,) * NUM_STAGES
    grasp_xy: tuple[float, float] = (0.0, 0.0)
    grasp_z: float = 0.0
    approach_dz: float = 0.0
    lift_dz: float = 0.0
    transfer_dz: float = 0.0
    release_xy: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def sample(cls, seed: int) -> TrajectoryVariation:
        rng = np.random.default_rng([seed, 2])
        return cls(
            step_scale=tuple(float(x) for x in rng.uniform(0.75, 1.3, size=NUM_STAGES)),
            grasp_xy=(float(rng.uniform(-0.003, 0.003)), float(rng.uniform(-0.003, 0.003))),
            grasp_z=float(rng.uniform(-0.002, 0.002)),
            approach_dz=float(rng.uniform(-0.02, 0.02)),
            lift_dz=float(rng.uniform(-0.02, 0.02)),
            transfer_dz=float(rng.uniform(-0.02, 0.02)),
            release_xy=(float(rng.uniform(-0.005, 0.005)), float(rng.uniform(-0.005, 0.005))),
        )


class ActionNoise:
    """Temporally correlated (AR(1)) noise on the arm joints, in raw servo units."""

    def __init__(self, std: float, rng: np.random.Generator, rho: float = 0.9) -> None:
        self.std = std
        self.rng = rng
        self.rho = rho
        self._state = np.zeros(len(ARM_JOINTS))
        # Last clean waypoint: stages start from it so the recorded labels stay continuous despite noise.
        self.anchor: dict[str, float] | None = None

    def sample(self) -> dict[str, float]:
        innovation = self.rng.normal(0.0, self.std * np.sqrt(1 - self.rho**2), size=self._state.shape)
        self._state = self.rho * self._state + innovation
        return {f"{name}.pos": float(v) for name, v in zip(ARM_JOINTS, self._state, strict=True)}


@dataclass
class EpisodeMetrics:
    frames: int = 0
    max_action_delta: float = 0.0  # largest per-step arm command change (raw units)
    max_action_jerk: float = 0.0  # largest per-step change of that change
    min_limit_margin: float = float("inf")  # closest arm command to a reachable joint limit (raw units)
    final_place_error_m: float = float("nan")
    final_cube_speed: float = float("nan")


@dataclass
class QualityThresholds:
    min_frames: int = 150
    max_frames: int = 700
    max_action_delta: float = 300.0
    max_action_jerk: float = 150.0
    min_limit_margin: float = 50.0
    max_place_error_m: float = 0.02
    max_cube_speed: float = 0.05


def arm_raw_limits(backend: NexArmMujocoBackend) -> dict[str, tuple[float, float]]:
    """Raw interval each arm joint can actually reach; commands beyond it saturate at the joint limit."""
    return {
        name: reachable_raw_range(name, backend.model.jnt_range[backend._joint_ids[name]])
        for name in ARM_JOINTS
    }


class FrameRecorder:
    """Tracks quality metrics for each recorded (observation, command) pair and forwards them to a sink.

    ``raw_limits`` are the per-joint reachable raw ranges (see :func:`arm_raw_limits`) used for the
    joint-limit margin; they default to the full servo command range.
    """

    def __init__(
        self,
        sink: Callable[[dict[str, object], dict[str, float]], None] | None = None,
        raw_limits: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.sink = sink
        self.metrics = EpisodeMetrics()
        self._prev: np.ndarray | None = None
        self._prev_delta: np.ndarray | None = None
        limits = raw_limits or RAW_RANGES
        self._low = np.array([limits[name][0] for name in ARM_JOINTS], dtype=np.float64)
        self._high = np.array([limits[name][1] for name in ARM_JOINTS], dtype=np.float64)

    def __call__(self, observation: dict[str, object], action: dict[str, float]) -> None:
        arm = np.array([action[f"{name}.pos"] for name in ARM_JOINTS])
        metrics = self.metrics
        metrics.frames += 1
        if self._prev is not None:
            delta = arm - self._prev
            metrics.max_action_delta = max(metrics.max_action_delta, float(np.abs(delta).max()))
            if self._prev_delta is not None:
                metrics.max_action_jerk = max(
                    metrics.max_action_jerk, float(np.abs(delta - self._prev_delta).max())
                )
            self._prev_delta = delta
        self._prev = arm
        metrics.min_limit_margin = min(
            metrics.min_limit_margin, float(np.minimum(arm - self._low, self._high - arm).min())
        )
        if self.sink is not None:
            self.sink(observation, action)


def check_quality(metrics: EpisodeMetrics, limits: QualityThresholds) -> str | None:
    """Return the reason an otherwise successful episode must be rejected, or None."""
    if metrics.frames < limits.min_frames:
        return "quality_too_short"
    if metrics.frames > limits.max_frames:
        return "quality_too_long"
    if metrics.max_action_delta > limits.max_action_delta:
        return "quality_velocity_spike"
    if metrics.max_action_jerk > limits.max_action_jerk:
        return "quality_jerk_spike"
    if metrics.min_limit_margin < limits.min_limit_margin:
        return "quality_joint_limit"
    if not metrics.final_place_error_m <= limits.max_place_error_m:
        return "quality_placement_error"
    if not metrics.final_cube_speed <= limits.max_cube_speed:
        return "quality_cube_moving"
    return None


def _min_jerk(alpha: np.ndarray) -> np.ndarray:
    return 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5


def _clip_action(action: dict[str, float]) -> dict[str, float]:
    return {
        key: float(np.clip(value, *RAW_RANGES[key.removesuffix(".pos")])) for key, value in action.items()
    }


def _send(
    robot: NexArmSim,
    command: dict[str, float],
    noise: ActionNoise | None,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None,
) -> None:
    """Send ``command`` (plus optional noise) and record the clean command as the action label."""
    observation = robot.get_observation() if record_frame is not None else {}
    executed = command
    if noise is not None:
        jitter = noise.sample()
        executed = _clip_action({key: command[key] + jitter.get(key, 0.0) for key in command})
    robot.send_action(executed)
    if record_frame is not None:
        record_frame(observation, command)


def _interpolate_stage(
    robot: NexArmSim,
    task: NexArmPickPlaceTask,
    target_xyz: np.ndarray,
    gripper: float,
    *,
    seed: int,
    steps: int,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None,
    settle_steps: int = 10,
    noise: ActionNoise | None = None,
    smooth: bool = True,
) -> bool:
    solution = robot.backend.solve_ik(
        target_xyz,
        seed=seed,
        tolerance_m=0.001,
        restarts=0,
    )
    if solution is None:
        return False

    start = (
        noise.anchor if noise is not None and noise.anchor is not None else robot.backend.joint_positions()
    )
    target = {f"{name}.pos": HOME_POSITIONS[name] for name in JOINT_NAMES}
    target.update({f"{name}.pos": value for name, value in solution.items()})
    target["gripper.pos"] = gripper
    target = _clip_action(target)
    if noise is not None:
        noise.anchor = target

    alphas = np.linspace(0.0, 1.0, steps, endpoint=True)
    for alpha in _min_jerk(alphas) if smooth else alphas:
        command = _clip_action({key: float((1 - alpha) * start[key] + alpha * target[key]) for key in target})
        _send(robot, command, noise, record_frame)
        if task.observe().terminated:
            break
    # Hold the waypoint only until the arm has converged; idle frames teach a policy to stall.
    for step in range(settle_steps):
        _send(robot, target, noise, record_frame)
        if task.observe().terminated:
            break
        if step + 1 >= MIN_SETTLE_STEPS:
            current = robot.backend.joint_positions()
            if (
                max(abs(current[f"{name}.pos"] - target[f"{name}.pos"]) for name in ARM_JOINTS)
                < SETTLE_TOLERANCE
            ):
                break
    return True


def generate_episode(
    robot: NexArmSim,
    task: NexArmPickPlaceTask,
    *,
    seed: int,
    record_frame: Callable[[dict[str, object], dict[str, float]], None] | None = None,
    trace: bool = False,
    variation: TrajectoryVariation | None = None,
    action_noise: float = 0.0,
) -> tuple[bool, str]:
    """Run one pick/place attempt and return its accepted status.

    ``variation=None`` runs the nominal path. ``action_noise`` (raw units, std) perturbs the executed
    arm commands while the recorded action stays the clean target.
    """
    variation = variation or TrajectoryVariation()
    noise = ActionNoise(action_noise, np.random.default_rng([seed, 4])) if action_noise > 0 else None

    task.reset(seed=seed, settle_steps=25)
    cube = robot.backend.body_position("cube")
    target = robot.backend.body_position("target_zone")

    # Account for gripper_frame site offset relative to the jaw collision
    # boxes to ensure vertical overlap with the 20 mm cube.
    cube_grasp = cube + np.array([variation.grasp_xy[0], variation.grasp_xy[1], -0.020 + variation.grasp_z])
    target_release = target + np.array([variation.release_xy[0], variation.release_xy[1], -0.018])
    base_stages = (
        (cube_grasp + [0.0, 0.0, 0.12 + variation.approach_dz], OPEN_GRIPPER, 16),
        (cube_grasp, OPEN_GRIPPER, 18),
        (cube_grasp, CLOSED_GRIPPER, 80),
        (cube_grasp + [0.0, 0.0, 0.055], CLOSED_GRIPPER, 30),
        (cube_grasp + [0.0, 0.0, 0.09], CLOSED_GRIPPER, 50),
        (cube_grasp + [0.0, 0.0, 0.13 + variation.lift_dz], CLOSED_GRIPPER, 50),
        (target_release + [0.0, 0.0, 0.12 + variation.transfer_dz], CLOSED_GRIPPER, 60),
        (target_release, CLOSED_GRIPPER, 18),
        (target_release, OPEN_GRIPPER, 24),
        (target_release + [0.0, 0.0, 0.14], OPEN_GRIPPER, 18),
    )

    for stage_index, (waypoint, gripper, base_steps) in enumerate(base_stages):
        steps = max(4, round(base_steps * variation.step_scale[stage_index]))
        if not _interpolate_stage(
            robot,
            task,
            np.asarray(waypoint),
            gripper,
            seed=seed * 100 + stage_index,
            steps=steps,
            record_frame=record_frame,
            noise=noise,
            smooth=stage_index in SMOOTH_STAGES,
        ):
            return False, f"ik_failed_stage_{stage_index}"
        status = task.status()
        if trace:
            jaws = np.stack(
                [
                    robot.backend.geom_position("link_6_left_jaw_collision_0"),
                    robot.backend.geom_position("link_6_right_jaw_collision_0"),
                ]
            )
            print(
                f"seed={seed} stage={stage_index} site={robot.backend.site_position('gripper_frame')} "
                f"jaws={jaws} cube={robot.backend.body_position('cube')} grasped={status.is_grasped}"
            )
        if status.terminated:
            return status.success, status.reason or "terminated"

    # Hold after release so the task's stability gate can accept the placement.
    hold_action = _clip_action(robot.backend.joint_positions())
    for _ in range(35):
        _send(robot, hold_action, None, record_frame)
        status = task.observe()
        if status.terminated:
            return status.success, status.reason or "terminated"

    status = task.status()
    return status.success, status.reason or "task_gate_failed"


def _build_dataset(robot: NexArmSim, args: argparse.Namespace) -> LeRobotDataset:
    features = {
        **hw_to_dataset_features(robot.action_features, ACTION, args.video),
        **hw_to_dataset_features(robot.observation_features, OBS_STR, args.video),
    }
    return utils.open_dataset(
        args,
        features,
        robot.robot_type,
        encoder_queue_maxsize=120,
        encoder_threads=2 if args.video else None,
        image_writer_threads=4 if not args.video else 0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", default="local/nexarm_sim_pick_place")
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/nexarm_sim_pick_place"))
    parser.add_argument("--episodes", type=int, default=20, help="Number of accepted episodes to write")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/scene.xml"))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--no-variation", dest="variation", action="store_false", help="Use the nominal trajectory"
    )
    parser.add_argument("--layout-scale", type=float, default=1.0, help="Scale the cube/target spawn area")
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
        help="Std (raw servo units, try 3-5) of correlated noise on executed arm commands; labels stay clean",
    )
    utils.add_generation_args(parser)
    args = parser.parse_args(argv)
    if args.action_noise < 0 or args.layout_scale <= 0:
        parser.error("--action-noise must be >= 0 and --layout-scale must be positive")
    utils.finalize_generation_args(parser, args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)

    if args.workers > 1:
        return utils.run_sharded(
            Path(__file__).resolve(),
            raw_argv,
            repo_id=args.repo_id,
            root=args.root,
            episodes=args.episodes,
            seed_start=args.seed_start,
            max_attempts=args.max_attempts,
            workers=args.workers,
            gpus=args.gpus,
            split=args.split,
            resume=args.resume,
        )
    if args.gpus:
        return utils.run_on_gpu(Path(__file__).resolve(), raw_argv, args.gpus[0])

    robot = NexArmSim(
        NexArmSimConfig(
            id="synthetic_generator",
            model_path=args.model,
            fps=args.fps,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            settle_steps=0,
            action_delay_steps=args.action_delay_steps,
            enable_domain_randomization=args.domain_randomization,
            calibration_path=args.calibration,
        )
    )
    previous_report = utils.read_report(args.root) if args.resume else None
    dataset = _build_dataset(robot, args)
    existing_episodes = dataset.meta.total_episodes
    robot.connect()
    task = NexArmPickPlaceTask(robot.backend, layout_scale=args.layout_scale)
    raw_limits = arm_raw_limits(robot.backend)
    base_delay, delay_range = utils.delay_schedule(args)

    limits = QualityThresholds()
    episodes_path = args.root / utils.EPISODES_NAME
    if not args.resume:
        episodes_path.unlink(missing_ok=True)
    accepted = 0
    attempts = 0
    accepted_seeds: list[int] = []
    rejected_attempts: list[dict[str, int | str]] = []
    accepted_metrics: list[EpisodeMetrics] = []

    def sink(observation: dict[str, object], action: dict[str, float]) -> None:
        observation_frame = build_dataset_frame(dataset.features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
        dataset.add_frame({**observation_frame, **action_frame, "task": args.task})

    try:
        while accepted < args.episodes and attempts < args.max_attempts:
            seed = args.seed_start + attempts
            delay = utils.episode_delay(seed, base_delay, delay_range)
            robot.backend.action_delay_steps = delay
            variation = TrajectoryVariation.sample(seed) if args.variation else TrajectoryVariation()
            recorder = FrameRecorder(sink, raw_limits)

            success, reason = generate_episode(
                robot,
                task,
                seed=seed,
                record_frame=recorder,
                trace=args.trace,
                variation=variation,
                action_noise=args.action_noise,
            )
            attempts += 1
            metrics = recorder.metrics
            metrics.final_place_error_m = float(
                np.linalg.norm(task.cube_position[:2] - task.target_position[:2])
            )
            metrics.final_cube_speed = task.cube_speed
            if success:
                quality_reason = check_quality(metrics, limits)
                if quality_reason is not None:
                    success, reason = False, quality_reason

            row = {
                "seed": seed,
                "accepted": success,
                "reason": reason,
                "episode_index": existing_episodes + accepted if success else None,
                "action_delay_steps": delay,
                "variation": variation,
                "dr_params": robot.backend.last_domain_params,
                "metrics": metrics,
            }
            utils.append_jsonl(episodes_path, row)
            if success:
                dataset.save_episode(parallel_encoding=False)
                accepted += 1
                accepted_seeds.append(seed)
                accepted_metrics.append(metrics)
                print(f"accepted seed={seed} ({accepted}/{args.episodes})")
            else:
                dataset.clear_episode_buffer()
                rejected_attempts.append({"seed": seed, "reason": reason})
                print(f"rejected seed={seed}: {reason}")
    finally:
        robot.disconnect()
        dataset.finalize()

    model_path = resolve_model_path(args.model)
    report = utils.base_report(
        "examples/nexarm/generate_sim_dataset.py",
        args,
        model_path=model_path,
        attempts=attempts,
        accepted_seeds=accepted_seeds,
        rejected_attempts=rejected_attempts,
    )
    report["quality_thresholds"] = limits
    report["stats"] = {
        "frames": utils.summarize([m.frames for m in accepted_metrics]),
        "max_action_jerk": utils.summarize([m.max_action_jerk for m in accepted_metrics]),
        "place_error_m": utils.summarize([m.final_place_error_m for m in accepted_metrics]),
    }
    if previous_report is not None:
        report = utils.combine_reports([previous_report, report])
    total_episodes = existing_episodes + accepted
    if total_episodes:
        report["validation"] = utils.validate_dataset(args.repo_id, args.root, total_episodes)
    utils.write_report(args.root, report)
    print(f"wrote {accepted} episode(s) from {attempts} attempt(s) to {args.root}")
    return 0 if accepted == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
