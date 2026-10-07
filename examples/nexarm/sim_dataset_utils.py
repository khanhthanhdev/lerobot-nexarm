"""Shared helpers for NexArm simulation dataset generators: provenance, JSONL logs, sharding, validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

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


# Flags consumed by the sharding launcher itself; stripped from the child command line.
_VALUE_FLAGS = ("--repo-id", "--root", "--episodes", "--seed-start", "--max-attempts", "--workers", "--gpus")


def _strip_flags(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        name, has_value, _ = token.partition("=")
        if name in _VALUE_FLAGS:
            skip = not has_value
            continue
        stripped.append(token)
    return stripped


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
) -> int:
    """Run ``workers`` copies of ``script`` over disjoint seed ranges, then merge shards into ``root``."""
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        raise FileExistsError(f"{root} already exists; choose a new --root")
    workers = min(workers, episodes)
    if split == "train" and seed_start + workers * SHARD_SEED_STRIDE > EVAL_SEED_OFFSET:
        raise ValueError("Too many workers for the train seed range")
    per_worker = [episodes // workers + (1 if i < episodes % workers else 0) for i in range(workers)]
    child_args = _strip_flags(argv)

    shards: list[tuple[str, Path]] = []
    procs: list[subprocess.Popen] = []
    for index, worker_episodes in enumerate(per_worker):
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
            str(seed_start + index * SHARD_SEED_STRIDE),
            "--max-attempts",
            str(max(worker_episodes, -(-max_attempts // workers))),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("MUJOCO_GL", "egl")
        if gpus:
            gpu = gpus[index % len(gpus)]
            env["MUJOCO_EGL_DEVICE_ID"] = gpu
            env["CUDA_VISIBLE_DEVICES"] = gpu
        print(
            f"[shard {index}] {worker_episodes} episodes, seeds from {seed_start + index * SHARD_SEED_STRIDE}"
        )
        procs.append(subprocess.Popen(cmd, env=env))

    codes = [proc.wait() for proc in procs]
    # A worker exits non-zero when it accepted fewer episodes than requested; its shard is still usable
    # only if it is complete, so any failure aborts the merge and keeps the shards for inspection.
    if any(codes):
        print(f"[ERROR] shard exit codes: {codes}; shards kept for inspection: {[str(r) for _, r in shards]}")
        return 1

    reports = [json.loads((r / REPORT_NAME).read_text(encoding="utf-8")) for _, r in shards]
    merge_datasets(
        [LeRobotDataset(repo, root=shard_root) for repo, shard_root in shards],
        output_repo_id=repo_id,
        output_dir=root,
    )

    offset = 0
    rows: list[dict[str, Any]] = []
    for (_, shard_root), report in zip(shards, reports, strict=True):
        for row in read_jsonl(shard_root / EPISODES_NAME):
            if row.get("episode_index") is not None:
                row["episode_index"] += offset
            rows.append(row)
        offset += len(report["accepted_seeds"])
    for row in rows:
        append_jsonl(root / EPISODES_NAME, row)

    merged = dict(reports[0])
    merged.update(
        {
            "repo_id": repo_id,
            "requested_episodes": episodes,
            "attempts": sum(r["attempts"] for r in reports),
            "accepted_seeds": [s for r in reports for s in r["accepted_seeds"]],
            "rejected_attempts": [a for r in reports for a in r["rejected_attempts"]],
            "workers": workers,
        }
    )
    merged["acceptance_rate"] = len(merged["accepted_seeds"]) / max(1, merged["attempts"])
    merged["validation"] = validate_dataset(repo_id, root, len(merged["accepted_seeds"]))
    (root / REPORT_NAME).write_text(json.dumps(to_jsonable(merged), indent=2) + "\n", encoding="utf-8")
    for _, shard_root in shards:
        shutil.rmtree(shard_root, ignore_errors=True)
    print(f"[INFO] merged {len(merged['accepted_seeds'])} episodes into {root}")
    return 0
