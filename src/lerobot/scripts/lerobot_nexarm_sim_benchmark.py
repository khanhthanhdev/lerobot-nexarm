#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark trained LeRobot policies on the single-arm NexArm MuJoCo task."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

# MuJoCo chooses its OpenGL backend during import. Headless Linux GPU servers
# normally need EGL, while local machines should keep MuJoCo's normal choice.
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim import NexArmPickPlaceTask, NexArmSim, NexArmSimConfig
from lerobot.rollout import BaseStrategyConfig, RolloutConfig, SyncInferenceConfig, build_rollout_context
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

DEFAULT_TASK = "Pick up the red cube, place it in the green target zone, and release it."


@dataclass(frozen=True)
class PolicySpec:
    """A user-facing benchmark label and local or Hub checkpoint path."""

    label: str
    path: str


@dataclass(frozen=True)
class EpisodeMetrics:
    seed: int
    success: bool
    reason: str
    steps: int
    simulation_time_s: float
    wall_time_s: float
    control_hz: float
    inference_mean_ms: float
    inference_p50_ms: float
    inference_p95_ms: float


def parse_policy_spec(value: str) -> PolicySpec:
    """Parse ``LABEL=PATH`` while also accepting a bare checkpoint path."""

    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
    else:
        path = value.strip()
        label = Path(path.rstrip("/")).name or "policy"
    if not label:
        raise argparse.ArgumentTypeError("policy label cannot be empty")
    if not path:
        raise argparse.ArgumentTypeError("policy path cannot be empty")
    return PolicySpec(label=label, path=path)


def _latency_summary(latencies_s: list[float]) -> dict[str, float]:
    if not latencies_s:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    latency_ms = np.asarray(latencies_s, dtype=np.float64) * 1000
    return {
        "mean_ms": float(latency_ms.mean()),
        "p50_ms": float(np.percentile(latency_ms, 50)),
        "p95_ms": float(np.percentile(latency_ms, 95)),
    }


def summarize_episodes(episodes: list[EpisodeMetrics]) -> dict[str, Any]:
    """Aggregate task and runtime metrics for one policy."""

    if not episodes:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
            "termination_reasons": {},
            "mean_episode_wall_time_s": 0.0,
            "mean_control_hz": 0.0,
            "inference_mean_ms": 0.0,
            "inference_p50_ms": 0.0,
            "inference_p95_ms": 0.0,
        }

    return {
        "episodes": len(episodes),
        "successes": sum(episode.success for episode in episodes),
        "success_rate": sum(episode.success for episode in episodes) / len(episodes),
        "termination_reasons": dict(Counter(episode.reason for episode in episodes)),
        "mean_episode_wall_time_s": float(np.mean([episode.wall_time_s for episode in episodes])),
        "mean_control_hz": float(np.mean([episode.control_hz for episode in episodes])),
        "inference_mean_ms": float(np.mean([episode.inference_mean_ms for episode in episodes])),
        "inference_p50_ms": float(np.mean([episode.inference_p50_ms for episode in episodes])),
        "inference_p95_ms": float(np.mean([episode.inference_p95_ms for episode in episodes])),
    }


def validate_policy_contract(
    policy_cfg: PreTrainedConfig,
    *,
    camera_names: list[str],
    camera_height: int,
    camera_width: int,
) -> None:
    """Fail before model loading when a checkpoint is not a NexArm checkpoint."""

    input_features = policy_cfg.input_features or {}
    output_features = policy_cfg.output_features or {}
    expected_action_names = [f"{name}.pos" for name in JOINT_NAMES]

    action_feature = output_features.get(ACTION)
    if action_feature is None or action_feature.type is not FeatureType.ACTION:
        raise ValueError("Policy checkpoint has no canonical 'action' feature")
    if action_feature.shape != (len(expected_action_names),):
        raise ValueError(
            f"Policy action shape is {action_feature.shape}; single-arm NexArm requires "
            f"({len(expected_action_names)},)"
        )

    state_feature = input_features.get(OBS_STATE)
    if state_feature is None or state_feature.type is not FeatureType.STATE:
        raise ValueError("Policy checkpoint has no canonical 'observation.state' feature")
    if state_feature.shape != (len(expected_action_names),):
        raise ValueError(
            f"Policy state shape is {state_feature.shape}; single-arm NexArm requires "
            f"({len(expected_action_names)},)"
        )

    expected_visuals = {f"observation.images.{name}" for name in camera_names}
    actual_visuals = {name for name, feature in input_features.items() if feature.type is FeatureType.VISUAL}
    if actual_visuals != expected_visuals:
        raise ValueError(
            f"Policy cameras are {sorted(actual_visuals)}; benchmark cameras are "
            f"{sorted(expected_visuals)}. Pass matching --cameras values."
        )
    expected_image_shape = (3, camera_height, camera_width)
    bad_shapes = {
        name: input_features[name].shape
        for name in actual_visuals
        if input_features[name].shape != expected_image_shape
    }
    if bad_shapes:
        raise ValueError(
            f"Policy camera shapes are {bad_shapes}; benchmark renders {expected_image_shape}. "
            "Pass matching --camera-height and --camera-width values."
        )

    action_names = getattr(policy_cfg, "action_feature_names", None)
    if action_names is not None and list(action_names) != expected_action_names:
        raise ValueError(
            f"Policy action names are {list(action_names)}; NexArm requires {expected_action_names}"
        )


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peak_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**3


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _run_control_step(ctx, task: NexArmPickPlaceTask, device: torch.device) -> tuple[Any, float]:
    robot = ctx.hardware.robot_wrapper
    obs_raw = robot.get_observation()
    obs_processed = ctx.processors.robot_observation_processor(obs_raw)
    ctx.policy.inference.notify_observation(obs_processed)
    obs_frame = build_dataset_frame(ctx.data.dataset_features, obs_processed, prefix=OBS_STR)

    _cuda_synchronize(device)
    inference_started = time.perf_counter()
    action_tensor = ctx.policy.inference.get_action(obs_frame)
    _cuda_synchronize(device)
    inference_s = time.perf_counter() - inference_started
    if action_tensor is None:
        raise RuntimeError("Synchronous inference returned no action")

    ordered_keys = ctx.data.ordered_action_keys
    if len(action_tensor) != len(ordered_keys):
        raise ValueError(
            f"Policy returned {len(action_tensor)} actions, but NexArm requires {len(ordered_keys)}: "
            f"{ordered_keys}"
        )
    action = {key: float(action_tensor[index]) for index, key in enumerate(ordered_keys)}
    action = ctx.processors.robot_action_processor((action, obs_raw))
    robot.send_action(action)
    return task.observe(), inference_s


def _run_episode(
    ctx,
    task: NexArmPickPlaceTask,
    *,
    seed: int,
    fps: int,
    settle_steps: int,
    realtime: bool,
    device: torch.device,
) -> EpisodeMetrics:
    task.reset(seed=seed, settle_steps=settle_steps)
    ctx.policy.inference.reset()
    latencies: list[float] = []
    started_wall = time.perf_counter()
    started_sim = float(task.backend.data.time)
    status = task.status()
    steps = 0
    max_steps = math.ceil((task.timeout_s + task.success_hold_s + 1) * fps)

    while not status.terminated and steps < max_steps:
        step_started = time.perf_counter()
        status, inference_s = _run_control_step(ctx, task, device)
        latencies.append(inference_s)
        steps += 1
        if realtime:
            precise_sleep(max(0.0, 1 / fps - (time.perf_counter() - step_started)))

    wall_time_s = time.perf_counter() - started_wall
    simulation_time_s = float(task.backend.data.time) - started_sim
    reason = status.reason or "step_limit"
    latency = _latency_summary(latencies)
    return EpisodeMetrics(
        seed=seed,
        success=status.success,
        reason=reason,
        steps=steps,
        simulation_time_s=simulation_time_s,
        wall_time_s=wall_time_s,
        control_hz=steps / wall_time_s if wall_time_s > 0 else 0.0,
        inference_mean_ms=latency["mean_ms"],
        inference_p50_ms=latency["p50_ms"],
        inference_p95_ms=latency["p95_ms"],
    )


def benchmark_policy(spec: PolicySpec, args: argparse.Namespace) -> dict[str, Any]:
    """Load one policy, run all seeds, and release its GPU memory."""

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    _reset_cuda_peak(device)
    load_started = time.perf_counter()
    policy_cfg = PreTrainedConfig.from_pretrained(spec.path)
    policy_cfg.pretrained_path = Path(spec.path)
    policy_cfg.device = str(device)
    validate_policy_contract(
        policy_cfg,
        camera_names=args.cameras,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
    )
    rollout_cfg = RolloutConfig(
        robot=NexArmSimConfig(
            id=f"benchmark_{spec.label}",
            model_path=args.model,
            fps=args.fps,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_names=tuple(args.cameras),
            settle_steps=0,
        ),
        policy=policy_cfg,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=args.fps,
        device=str(device),
        task=args.task,
        return_to_initial_position=False,
        use_torch_compile=args.torch_compile,
        compile_warmup_inferences=args.warmup_steps,
    )

    ctx = None
    try:
        ctx = build_rollout_context(rollout_cfg, Event())
        _cuda_synchronize(device)
        load_time_s = time.perf_counter() - load_started
        load_peak_gpu_gb = _cuda_peak_gb(device)
        parameter_count = sum(parameter.numel() for parameter in ctx.policy.policy.parameters())

        robot = ctx.hardware.robot_wrapper.inner
        if not isinstance(robot, NexArmSim):
            raise TypeError(f"Expected NexArmSim, got {type(robot).__name__}")
        task = NexArmPickPlaceTask(robot.backend, timeout_s=args.episode_timeout)
        engine = ctx.policy.inference
        engine.reset()
        engine.start()
        engine.resume()

        if args.warmup_steps:
            task.reset(seed=args.seed_start, settle_steps=args.settle_steps)
            for _ in range(args.warmup_steps):
                _run_control_step(ctx, task, device)
            engine.reset()

        _reset_cuda_peak(device)
        benchmark_started = time.perf_counter()
        episodes = [
            _run_episode(
                ctx,
                task,
                seed=args.seed_start + index,
                fps=args.fps,
                settle_steps=args.settle_steps,
                realtime=args.realtime,
                device=device,
            )
            for index in range(args.episodes)
        ]
        _cuda_synchronize(device)
        benchmark_wall_time_s = time.perf_counter() - benchmark_started
        rollout_peak_gpu_gb = _cuda_peak_gb(device)
        summary = summarize_episodes(episodes)
        return {
            "label": spec.label,
            "path": spec.path,
            "policy_type": policy_cfg.type,
            "status": "completed",
            "parameter_count": parameter_count,
            "load_time_s": load_time_s,
            "load_peak_gpu_gb": load_peak_gpu_gb,
            "rollout_peak_gpu_gb": rollout_peak_gpu_gb,
            "benchmark_wall_time_s": benchmark_wall_time_s,
            **summary,
            "episode_results": [asdict(episode) for episode in episodes],
        }
    finally:
        if ctx is not None:
            try:
                ctx.policy.inference.stop()
            finally:
                robot = ctx.hardware.robot_wrapper.inner
                if robot.is_connected:
                    robot.disconnect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment_metadata(device: str) -> dict[str, Any]:
    torch_device = torch.device(device)
    gpu_name = None
    if torch_device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch_device)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(torch_device),
        "gpu_name": gpu_name,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }


def _write_results(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark.json"
    csv_path = output_dir / "benchmark.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fields = [
        "label",
        "path",
        "policy_type",
        "status",
        "error",
        "episodes",
        "successes",
        "success_rate",
        "parameter_count",
        "load_time_s",
        "benchmark_wall_time_s",
        "mean_episode_wall_time_s",
        "mean_control_hz",
        "inference_mean_ms",
        "inference_p50_ms",
        "inference_p95_ms",
        "load_peak_gpu_gb",
        "rollout_peak_gpu_gb",
        "termination_reasons",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in payload["results"]:
            row = dict(result)
            row["termination_reasons"] = json.dumps(row.get("termination_reasons", {}), sort_keys=True)
            writer.writerow(row)
    return json_path, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ACT, pi0, SmolVLA, or other trained LeRobot checkpoints on identical "
            "single-arm NexArm MuJoCo pick-and-place seeds."
        )
    )
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        type=parse_policy_spec,
        metavar="LABEL=PATH",
        help="Repeat for each local checkpoint or Hugging Face repo, e.g. --policy act=outputs/act",
    )
    parser.add_argument("--model", type=Path, default=Path("sim/fusion_export/scene.xml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nexarm_sim_benchmark"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--episode-timeout", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--cameras", nargs="+", default=["front", "wrist"])
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--settle-steps", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--realtime", action="store_true", help="Throttle simulation to the configured FPS")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.episode_timeout <= 0:
        parser.error("--episode-timeout must be positive")
    if args.settle_steps < 0 or args.warmup_steps < 0:
        parser.error("--settle-steps and --warmup-steps cannot be negative")
    labels = [spec.label for spec in args.policy]
    if len(labels) != len(set(labels)):
        parser.error("policy labels must be unique")
    return args


def _print_result(result: dict[str, Any]) -> None:
    if result["status"] == "failed":
        print(f"{result['label']}: FAILED - {result['error']}")
        return
    print(
        f"{result['label']}: success={result['success_rate']:.1%} "
        f"p50={result['inference_p50_ms']:.1f} ms "
        f"p95={result['inference_p95_ms']:.1f} ms "
        f"control={result['mean_control_hz']:.1f} Hz "
        f"GPU={result['rollout_peak_gpu_gb'] or 0:.2f} GiB"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results: list[dict[str, Any]] = []
    for spec in args.policy:
        logger.info("Benchmarking %s from %s", spec.label, spec.path)
        try:
            result = benchmark_policy(spec, args)
        except Exception as error:
            logger.exception("Benchmark failed for %s", spec.label)
            result = {
                "label": spec.label,
                "path": spec.path,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            results.append(result)
            if args.fail_fast:
                break
        else:
            results.append(result)
        _print_result(result)

    payload = {
        "environment": _environment_metadata(args.device),
        "config": {
            "model": str(args.model),
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "seeds": list(range(args.seed_start, args.seed_start + args.episodes)),
            "episode_timeout": args.episode_timeout,
            "fps": args.fps,
            "camera_width": args.camera_width,
            "camera_height": args.camera_height,
            "cameras": args.cameras,
            "task": args.task,
            "realtime": args.realtime,
            "torch_compile": args.torch_compile,
        },
        "results": results,
    }
    json_path, csv_path = _write_results(args.output_dir, payload)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 1 if any(result["status"] == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
