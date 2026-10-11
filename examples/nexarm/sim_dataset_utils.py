"""Shared helpers for NexArm simulation dataset generators: provenance, JSONL logs, sharding, validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.motors.nexarm.mujoco_mapping import JOINT_MAPPING_VERSION
from lerobot.robots.nexarm_sim.calibration import (
    SimCalibration,
    check_calibration_fps,
    resolve_action_delay_steps,
)

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPORT_NAME = "generation_report.json"
EPISODES_NAME = "generation_episodes.jsonl"
EVAL_SEED_OFFSET = 1_000_000  # train seeds are < this, held-out eval seeds are >= this
SHARD_SEED_STRIDE = 100_000

REPO_ROOT = Path(__file__).resolve().parents[2]


def to_jsonable(value: Any) -> Any:
    """Convert numpy / Path / dataclass-like values to JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable({name: getattr(value, name) for name in value.__dataclass_fields__})
    return value


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def git_sha() -> str | None:
    git_bin = shutil.which("git")
    if not git_bin:
        return None
    try:
        result = subprocess.run(  # nosec B603
            [git_bin, "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def collect_provenance(args_dict: dict[str, Any], model_path: Path, calibration_path: Path | None) -> dict:
    """Everything needed to trace which code, model and settings produced a dataset."""
    import mujoco

    from lerobot import __version__ as lerobot_version

    return {
        "git_sha": git_sha(),
        "lerobot_version": lerobot_version,
        "mujoco_version": mujoco.__version__,
        "uv_lock_sha256": sha256_file(REPO_ROOT / "uv.lock"),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "calibration_path": None if calibration_path is None else str(calibration_path),
        "calibration_sha256": None if calibration_path is None else sha256_file(calibration_path),
        "args": to_jsonable(args_dict),
    }


def summarize(values: Sequence[float]) -> dict[str, float] | None:
    if not len(values):
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(row)) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_seed_range(split: str, seed_start: int, max_attempts: int) -> None:
    """Keep train and held-out eval seeds disjoint."""
    seed_end = seed_start + max_attempts
    if split == "train" and seed_end > EVAL_SEED_OFFSET:
        raise ValueError(f"train seeds must stay below {EVAL_SEED_OFFSET}; got {seed_start}..{seed_end}")
    if split == "eval" and seed_start < EVAL_SEED_OFFSET:
        raise ValueError(f"eval seeds must start at or above {EVAL_SEED_OFFSET}; got {seed_start}")


def next_seed(root: Path, fallback: int) -> int:
    """First seed after every attempt logged in ``root`` (an attempt that was never logged was never kept)."""
    seeds = [int(row["seed"]) for row in read_jsonl(root / EPISODES_NAME) if row.get("seed") is not None]
    return max([fallback, *(seed + 1 for seed in seeds)])


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    """CLI flags shared by every simulation dataset generator."""
    parser.add_argument("--max-attempts", type=int, default=None, help="Defaults to 3x --episodes")
    parser.add_argument(
        "--split",
        choices=("train", "eval"),
        default="train",
        help=f"train seeds are < {EVAL_SEED_OFFSET}; eval seeds start at {EVAL_SEED_OFFSET}",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Defaults to the start of the split, or after the last logged seed with --resume",
    )
    parser.add_argument(
        "--no-dr",
        dest="domain_randomization",
        action="store_false",
        help="Disable per-episode visual, dynamics and object domain randomization (on by default)",
    )
    parser.add_argument(
        "--dr",
        "--domain-randomization",
        dest="domain_randomization",
        action="store_true",
        help="Enable domain randomization explicitly (the default)",
    )
    parser.add_argument(
        "--calibration", type=Path, default=None, help="SimCalibration JSON from calibrate_sim.py"
    )
    parser.add_argument(
        "--action-delay-steps",
        type=int,
        default=None,
        help="Action transport delay in control steps (e.g. 1=33ms). Defaults to the calibration, else 0",
    )
    parser.add_argument(
        "--action-delay-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Sample the delay per episode in [MIN, MAX]. With --calibration defaults to calibrated +-1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel shard processes. Defaults to one per --gpus id, else 1",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids (e.g. 0,1) for EGL rendering; workers are assigned round-robin",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append episodes to an existing dataset; new seeds continue after the logged ones",
    )
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.set_defaults(video=True, domain_randomization=True)


def finalize_generation_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate the shared flags and fill in defaults that depend on other flags."""
    if args.episodes <= 0 or args.fps <= 0:
        parser.error("--episodes and --fps must be positive")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("camera dimensions must be positive")
    if args.action_delay_steps is not None and args.action_delay_steps < 0:
        parser.error("--action-delay-steps cannot be negative")
    if (
        args.action_delay_range is not None
        and not 0 <= args.action_delay_range[0] <= args.action_delay_range[1]
    ):
        parser.error("--action-delay-range must satisfy 0 <= MIN <= MAX")

    if args.gpus is not None:
        args.gpus = [gpu.strip() for gpu in args.gpus.split(",")]
        if any(not gpu.isdecimal() for gpu in args.gpus) or len(set(args.gpus)) != len(args.gpus):
            parser.error("--gpus must be distinct non-negative GPU ids, e.g. 0,1")
    if args.workers is None:
        args.workers = len(args.gpus) if args.gpus else 1
    if args.workers <= 0:
        parser.error("--workers must be positive")

    if args.max_attempts is None:
        args.max_attempts = args.episodes * 3
    if args.max_attempts < args.episodes:
        parser.error("--max-attempts cannot be smaller than --episodes")
    check_dataset_root(parser, args.root, args.resume)
    if args.seed_start is None:
        split_start = EVAL_SEED_OFFSET if args.split == "eval" else 0
        # A single process continues after every logged seed; shards resume inside their own seed blocks.
        args.seed_start = (
            next_seed(args.root, split_start) if args.resume and args.workers == 1 else split_start
        )
    if args.seed_start < 0:
        parser.error("--seed-start cannot be negative")
    try:
        validate_seed_range(args.split, args.seed_start, args.max_attempts)
    except ValueError as error:
        parser.error(str(error))

    # Check the calibration before any dataset directory is created or any shard is launched.
    args.sim_calibration = None
    if args.calibration is not None:
        try:
            args.sim_calibration = SimCalibration.load(args.calibration)
            check_calibration_fps(args.sim_calibration, args.fps)
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"--calibration {args.calibration}: {error}")


def delay_schedule(args: argparse.Namespace) -> tuple[int, tuple[int, int] | None]:
    """Run delay and optional per-episode delay range. Explicit CLI value > calibration > 0."""
    calibration: SimCalibration | None = args.sim_calibration
    base = resolve_action_delay_steps(args.action_delay_steps, calibration)
    if args.action_delay_range is not None:
        return base, (args.action_delay_range[0], args.action_delay_range[1])
    if calibration is not None and args.action_delay_steps is None:
        return base, (max(0, calibration.action_delay_steps - 1), calibration.action_delay_steps + 1)
    return base, None


def episode_delay(seed: int, base: int, delay_range: tuple[int, int] | None) -> int:
    if delay_range is None:
        return base
    return int(np.random.default_rng([seed, 3]).integers(delay_range[0], delay_range[1] + 1))


def _set_render_gpu(env: MutableMapping[str, str], gpu: str) -> None:
    if platform.system() == "Linux":
        env.setdefault("MUJOCO_GL", "egl")
    env["MUJOCO_EGL_DEVICE_ID"] = gpu
    env["CUDA_VISIBLE_DEVICES"] = gpu


def base_report(
    generator: str,
    args: argparse.Namespace,
    *,
    model_path: Path,
    attempts: int,
    accepted_seeds: list[int],
    rejected_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report fields every generator writes. Merges add ``attempts`` and concatenate the seed lists."""
    provenance_args = {key: value for key, value in vars(args).items() if key != "sim_calibration"}
    return {
        "generator": generator,
        "provenance": collect_provenance(provenance_args, model_path, args.calibration),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "repo_id": args.repo_id,
        "joint_mapping": JOINT_MAPPING_VERSION,
        "split": args.split,
        "fps": args.fps,
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "video": args.video,
        "domain_randomization": args.domain_randomization,
        "requested_episodes": args.episodes,
        "attempts": attempts,
        "acceptance_rate": len(accepted_seeds) / max(1, attempts),
        "accepted_seeds": accepted_seeds,
        "rejected_attempts": rejected_attempts,
    }


def combine_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge reports of shards or resumed runs: counts add up, the last report supplies everything else."""
    mappings = {r.get("joint_mapping") for r in reports}
    if len(mappings) != 1:
        raise ValueError(
            f"Cannot combine reports written with different joint mappings: {sorted(map(str, mappings))}"
        )
    merged = dict(reports[-1])
    merged["requested_episodes"] = sum(r["requested_episodes"] for r in reports)
    merged["attempts"] = sum(r["attempts"] for r in reports)
    merged["accepted_seeds"] = [s for r in reports for s in r["accepted_seeds"]]
    merged["rejected_attempts"] = [a for r in reports for a in r["rejected_attempts"]]
    merged["acceptance_rate"] = len(merged["accepted_seeds"]) / max(1, merged["attempts"])
    return merged


def read_report(root: Path) -> dict[str, Any] | None:
    path = root / REPORT_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_report(root: Path, report: dict[str, Any]) -> None:
    (root / REPORT_NAME).write_text(json.dumps(to_jsonable(report), indent=2) + "\n", encoding="utf-8")


def validate_dataset(repo_id: str, root: Path, expected_episodes: int) -> dict[str, Any]:
    """Reload the finished dataset and check episode count and value ranges."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root)
    if dataset.meta.total_episodes != expected_episodes:
        raise RuntimeError(
            f"Dataset has {dataset.meta.total_episodes} episodes, expected {expected_episodes} ({root})"
        )
    summary: dict[str, Any] = {"episodes": expected_episodes, "frames": dataset.meta.total_frames}
    for column in ("action", "observation.state"):
        if column not in dataset.meta.features:
            continue
        values = np.asarray(dataset.hf_dataset.with_format("numpy")[column], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values found in column {column!r}")
        summary[column] = {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}
    return summary


# Flags consumed by the launchers themselves; stripped from the child command line.
_PARALLEL_FLAGS = ("--workers", "--gpus")
_SHARD_FLAGS = ("--repo-id", "--root", "--episodes", "--seed-start", "--max-attempts", *_PARALLEL_FLAGS)


# Shards always write fresh datasets; the parent merges them into the resumed one.
_SHARD_SWITCHES = ("--resume",)


def _strip_flags(
    argv: list[str], flags: Sequence[str] = _SHARD_FLAGS, switches: Sequence[str] = _SHARD_SWITCHES
) -> list[str]:
    stripped: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        name, has_value, _ = token.partition("=")
        if name in flags:
            skip = not has_value
            continue
        if token in switches:
            continue
        stripped.append(token)
    return stripped


def _wait_all(procs: Sequence[subprocess.Popen]) -> list[int]:
    try:
        return [proc.wait() for proc in procs]
    except BaseException:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            proc.wait()
        raise


def run_on_gpu(script: Path, argv: list[str], gpu: str) -> int:
    """Re-run ``script`` in one child process whose renderer is pinned to ``gpu``.

    MuJoCo picks its GL backend when it is imported, so the device must be set before the child starts.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    _set_render_gpu(env, gpu)
    cmd = [sys.executable, "-u", str(script), *_strip_flags(argv, _PARALLEL_FLAGS, ())]
    return _wait_all([subprocess.Popen(cmd, env=env)])[0]


def plan_shard_seeds(
    logged_seeds: Sequence[int], seed_start: int, attempts_per_worker: Sequence[int], split: str
) -> list[int]:
    """First seed of each shard. Shard ``i`` owns ``[seed_start + i * stride, seed_start + (i + 1) * stride)``.

    When resuming, each shard continues after the highest seed already logged inside its own block, so new
    attempts never reuse a seed of the existing dataset, whatever worker count produced it.
    """
    starts: list[int] = []
    for index, attempts in enumerate(attempts_per_worker):
        block_start = seed_start + index * SHARD_SEED_STRIDE
        block_end = block_start + SHARD_SEED_STRIDE
        used = [seed for seed in logged_seeds if block_start <= seed < block_end]
        start = max(used) + 1 if used else block_start
        if start + attempts > block_end:
            raise ValueError(
                f"shard {index} needs seeds {start}..{start + attempts} but its block ends at {block_end}; "
                "lower --max-attempts or pass a fresh --seed-start"
            )
        validate_seed_range(split, start, attempts)
        starts.append(start)
    return starts


def resume_backup_path(root: Path) -> Path:
    return root.parent / f"{root.name}_before_resume"


def recover_interrupted_resume(root: Path) -> None:
    """Put the original dataset back if a sharded resume stopped between its two directory renames."""
    backup = resume_backup_path(root)
    if not root.exists() and backup.exists():
        backup.rename(root)


def check_dataset_root(parser: argparse.ArgumentParser, root: Path, resume: bool) -> None:
    """A new run needs a fresh root; a resumed run needs a dataset written with the per-attempt log."""
    if not resume:
        if root.exists():
            parser.error(f"Dataset already exists: {root}. Use --resume to append or choose a new --root.")
        return
    recover_interrupted_resume(root)
    if not root.is_dir():
        parser.error(f"Cannot resume missing dataset: {root}")
    report = read_report(root)
    if not (root / EPISODES_NAME).exists() or report is None:
        parser.error(
            f"{root} has no {EPISODES_NAME} and {REPORT_NAME}; it predates the shared provenance layout, "
            "so regenerate it instead of resuming"
        )
    if report.get("joint_mapping") != JOINT_MAPPING_VERSION:
        parser.error(
            f"{root} was generated with joint mapping {report.get('joint_mapping') or 'servo-range scaling'}; "
            f"this code uses {JOINT_MAPPING_VERSION}. Appending would mix two angle conventions, so regenerate it"
        )


def open_dataset(
    args: argparse.Namespace, features: dict[str, dict], robot_type: str, **writer_kwargs: Any
) -> LeRobotDataset:
    """Create the output dataset, or reopen it with ``--resume`` after checking its features still match."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not args.resume:
        return LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=args.root,
            robot_type=robot_type,
            features=features,
            use_videos=args.video,
            streaming_encoding=args.video,
            **writer_kwargs,
        )
    print(f"[INFO] Resuming existing dataset at {args.root}...")
    dataset = LeRobotDataset.resume(
        repo_id=args.repo_id, root=args.root, streaming_encoding=args.video, **writer_kwargs
    )
    try:
        if dataset.fps != args.fps:
            raise ValueError(f"Existing dataset uses {dataset.fps} FPS; requested {args.fps}")
        if dataset.meta.robot_type != robot_type:
            raise ValueError(
                f"Existing dataset robot_type is {dataset.meta.robot_type!r}; this run writes {robot_type!r}"
            )
        for key, feature in features.items():
            actual = dataset.features.get(key, {})
            for field in ("dtype", "shape", "names"):
                if actual.get(field) != feature.get(field):
                    raise ValueError(f"Existing dataset feature {key} has incompatible {field}")
        recorded = {key for key in dataset.features if key.startswith(("observation.", "action"))}
        if recorded != set(features):
            raise ValueError("Existing dataset cameras or action/state features do not match")
    except BaseException:
        dataset.finalize()
        raise
    return dataset


def run_sharded(
    script: Path,
    argv: list[str],
    *,
    repo_id: str,
    root: Path,
    episodes: int,
    seed_start: int,
    max_attempts: int,
    workers: int,
    gpus: list[str] | None,
    split: str,
    resume: bool = False,
) -> int:
    """Run ``workers`` copies of ``script`` over disjoint seed blocks, then merge the shards into ``root``.

    With ``resume`` the existing dataset at ``root`` is merged first and each shard continues after the
    seeds already logged in its block (see :func:`plan_shard_seeds`).
    """
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if resume and not (root / EPISODES_NAME).exists():
        raise FileNotFoundError(f"Cannot resume {root}: {EPISODES_NAME} is missing")
    if not resume and root.exists():
        raise FileExistsError(f"{root} already exists; choose a new --root or pass --resume")
    workers = min(workers, episodes)
    per_worker = [episodes // workers + (1 if i < episodes % workers else 0) for i in range(workers)]
    attempts = [max(n, -(-max_attempts // workers)) for n in per_worker]
    existing_rows = read_jsonl(root / EPISODES_NAME) if resume else []
    starts = plan_shard_seeds(
        [int(row["seed"]) for row in existing_rows if row.get("seed") is not None],
        seed_start,
        attempts,
        split,
    )
    child_args = _strip_flags(argv)

    shards: list[tuple[str, Path]] = []
    procs: list[subprocess.Popen] = []
    for index, (worker_episodes, worker_attempts, start) in enumerate(
        zip(per_worker, attempts, starts, strict=True)
    ):
        shard_repo = f"{repo_id}_shard_{index}"
        shard_root = root.parent / f"{root.name}_shard_{index}"
        shutil.rmtree(shard_root, ignore_errors=True)
        shards.append((shard_repo, shard_root))
        cmd = [
            sys.executable,
            "-u",
            str(script),
            *child_args,
            "--repo-id",
            shard_repo,
            "--root",
            str(shard_root),
            "--episodes",
            str(worker_episodes),
            "--seed-start",
            str(start),
            "--max-attempts",
            str(worker_attempts),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if gpus:
            _set_render_gpu(env, gpus[index % len(gpus)])
        elif platform.system() == "Linux":
            env.setdefault("MUJOCO_GL", "egl")
        print(f"[shard {index}] {worker_episodes} episodes, seeds from {start}")
        procs.append(subprocess.Popen(cmd, env=env))

    codes = _wait_all(procs)
    # A worker exits non-zero when it accepted fewer episodes than requested; its shard is still usable
    # only if it is complete, so any failure aborts the merge and keeps the shards for inspection.
    if any(codes):
        print(f"[ERROR] shard exit codes: {codes}; shards kept for inspection: {[str(r) for _, r in shards]}")
        return 1

    reports = [json.loads((r / REPORT_NAME).read_text(encoding="utf-8")) for _, r in shards]
    sources = [LeRobotDataset(repo, root=shard_root) for repo, shard_root in shards]
    previous_report = None
    offset = 0
    if resume:
        existing = LeRobotDataset(repo_id, root=root)
        offset = existing.meta.total_episodes
        sources.insert(0, existing)
        previous_report = read_report(root)
    # Build the merged dataset next to ``root`` and only replace the original once it is complete.
    output = root.parent / f"{root.name}_merged" if resume else root
    if resume:
        shutil.rmtree(output, ignore_errors=True)
    merge_datasets(sources, output_repo_id=repo_id, output_dir=output)

    rows: list[dict[str, Any]] = list(existing_rows)
    for (_, shard_root), report in zip(shards, reports, strict=True):
        for row in read_jsonl(shard_root / EPISODES_NAME):
            if row.get("episode_index") is not None:
                row["episode_index"] += offset
            rows.append(row)
        offset += len(report["accepted_seeds"])
    for row in rows:
        append_jsonl(output / EPISODES_NAME, row)

    merged = combine_reports(reports)
    merged.update({"repo_id": repo_id, "requested_episodes": episodes, "workers": workers})
    if previous_report is not None:
        merged = combine_reports([previous_report, merged])
    merged["validation"] = validate_dataset(repo_id, output, offset)
    write_report(output, merged)

    if resume:
        backup = resume_backup_path(root)
        shutil.rmtree(backup, ignore_errors=True)
        root.rename(backup)
        try:
            output.rename(root)
        except BaseException:
            backup.rename(root)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    for _, shard_root in shards:
        shutil.rmtree(shard_root, ignore_errors=True)
    print(f"[INFO] merged {len(merged['accepted_seeds'])} episodes into {root}")
    return 0
