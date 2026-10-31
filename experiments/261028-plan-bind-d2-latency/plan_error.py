"""Measure Plan's error and Bind's re-plan trigger statistic on a `planbind` checkpoint and a dataset.

usage:
  uv run python experiments/261028-plan-bind-d2-latency/plan_error.py \
      --policy <planbind pretrained_model dir> --dataset-root <dataset dir> [--repo-id local/x] \
      --episodes 50 --frames-per-episode 10 --ages 0,8,16,24,32 --device cuda --out plan_error.json

Bind is trained on oracle plans: the coarse least-squares fit of the demonstrated window, plus Gaussian noise
of std `planbind_plan_noise`. At run time it corrects the network's own plan, made `age` frames earlier.
For sampled frames t of the dataset and every age a, this script measures, in normalized action units:

- the plan error: ``model.plan(obs[t - a]) - coarse_fit(actions[t - a : t - a + window])``, max over
  points and joints and RMS, next to the training noise;
- the trigger statistic `SplineRuntime` compares with `planbind_replan_threshold`: the largest absolute
  Bind residual on that stale plan with the observation at t, and the share of frames above each threshold.

The JSON holds, per age, the percentiles 50, 90, 95, 99 and the max of each quantity; a Markdown table is
printed. Run it once on the training dataset and once on held-out layouts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable
from pathlib import Path

import torch
from lerobot_policy_bspline.modeling_bspline import BsplinePolicy
from lerobot_policy_bspline.planbind import PlanBindModel
from torch.utils.data import default_collate

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION

PERCENTILES = (50, 90, 95, 99)
THRESHOLDS = (0.3, 0.5, 1.0)


def percentile(values: list[float], q: float) -> float:
    """The `q` quantile (0-1) of `values`, linearly interpolated."""
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def describe(values: list[float]) -> dict[str, float]:
    """Percentiles and max of `values`."""
    out = {f"p{q}": percentile(values, q / 100) for q in PERCENTILES}
    out["max"] = max(values)
    return out


def load_policy(path: Path, device: str) -> tuple[BsplinePolicy, Callable[[dict], dict]]:
    """A `planbind` checkpoint and its preprocessor (device copy and normalization)."""
    config = PreTrainedConfig.from_pretrained(path)
    config.device = device
    if getattr(config, "bspline_mode", None) != "planbind":
        raise SystemExit(f"{path} is not a `planbind` checkpoint.")
    policy = BsplinePolicy.from_pretrained(path, config=config).to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        config, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, preprocessor


@torch.no_grad()
def measure_frame(policy: BsplinePolicy, batch_at: Callable[[int], dict], ages: list[int]) -> dict[int, dict]:
    """Plan error and trigger statistic at one frame t, for every age.

    `batch_at(a)` returns the preprocessed, batched sample of frame ``t - a``: its observations and its
    normalized actions over the policy's `action_delta_indices` (`planbind_offset` frames before the frame,
    then `bspline_window`).
    """
    model = policy.diffusion
    assert isinstance(model, PlanBindModel)
    cfg = policy.config
    offset, window = cfg.planbind_offset, cfg.bspline_window
    now = batch_at(0)
    cond_now = policy._global_cond(now)
    out = {}
    for age in ages:
        stale = now if age == 0 else batch_at(age)
        cond_stale = cond_now if age == 0 else policy._global_cond(stale)
        plan = model.plan(cond_stale)
        truth = torch.einsum("cw,bwd->bcd", model.coarse_fit, stale[ACTION][:, offset : offset + window])
        error = plan - truth
        age_frames = torch.full((plan.shape[0],), age, device=plan.device)
        lifted = model.lifted_plan(plan, age_frames // cfg.planbind_fine_spacing)
        residual = model.bind(lifted, age_frames, cond_now)
        out[age] = {
            "plan_error_max": error.abs().amax().item(),
            "plan_error_rms": error.pow(2).mean().sqrt().item(),
            "trigger": residual.abs().amax().item(),
        }
    return out


def aggregate(records: list[dict[int, dict]], ages: list[int]) -> dict[str, dict]:
    """Per age: the distribution of each quantity and the share of frames whose trigger exceeds each τ."""
    result = {}
    for age in ages:
        rows = [r[age] for r in records]
        triggers = [r["trigger"] for r in rows]
        result[str(age)] = {
            "n_frames": len(rows),
            **{
                key: describe([r[key] for r in rows])
                for key in ("plan_error_max", "plan_error_rms", "trigger")
            },
            "trigger_above": {str(t): sum(v > t for v in triggers) / len(triggers) for t in THRESHOLDS},
        }
    return result


def sample_frames(
    dataset: LeRobotDataset, episodes: int, per_episode: int, before: int, after: int, seed: int
):
    """Global indices of `per_episode` frames from each of `episodes` sampled episodes.

    Every frame has `before` frames of history and `after` frames of future inside its episode.
    """
    rng = random.Random(seed)
    n_episodes = dataset.meta.total_episodes
    chosen = sorted(rng.sample(range(n_episodes), min(episodes, n_episodes)))
    frames = []
    for ep in chosen:
        start = int(dataset.meta.episodes[ep]["dataset_from_index"]) + before
        stop = int(dataset.meta.episodes[ep]["dataset_to_index"]) - after
        if stop <= start:
            continue
        frames += sorted(rng.sample(range(start, stop), min(per_episode, stop - start)))
    return frames


def table(summary: dict[str, dict], noise: float) -> str:
    """The Markdown table printed at the end."""
    lines = [
        f"Normalized action units; Bind's training noise std is {noise}.\n",
        "| Age (frames) | Frames | Plan error max p50 / p95 | Plan error RMS p50 / p95 "
        "| Trigger p50 / p90 / p95 / p99 | Trigger > 0.3 / 0.5 / 1.0 |",
        "|---|---|---|---|---|---|",
    ]
    for age, s in summary.items():
        e, r, t, above = s["plan_error_max"], s["plan_error_rms"], s["trigger"], s["trigger_above"]
        lines.append(
            f"| {age} | {s['n_frames']} | {e['p50']:.3f} / {e['p95']:.3f} | {r['p50']:.3f} / {r['p95']:.3f} "
            f"| {t['p50']:.3f} / {t['p90']:.3f} / {t['p95']:.3f} / {t['p99']:.3f} "
            f"| {' / '.join(f'{100 * above[str(x)]:.0f}%' for x in THRESHOLDS)} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Parse the arguments, run the measurement and write the JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", type=Path, required=True, help="planbind pretrained_model directory")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/plan_error", help="dataset repo id (local datasets: any)")
    parser.add_argument("--episodes", type=int, default=50, help="episodes sampled from the dataset")
    parser.add_argument("--frames-per-episode", type=int, default=10)
    parser.add_argument("--ages", default="0,8,16,24,32", help="plan ages in frames, comma-separated")
    parser.add_argument("--seed", type=int, default=0, help="frame sampling seed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    args = parser.parse_args()

    ages = [int(a) for a in args.ages.split(",")]
    policy, preprocessor = load_policy(args.policy, args.device)
    cfg = policy.config
    if any(a < 0 or a > cfg.planbind_max_plan_age or a % cfg.planbind_fine_spacing for a in ages):
        raise SystemExit(
            f"Ages must be multiples of {cfg.planbind_fine_spacing} in [0, {cfg.planbind_max_plan_age}]."
        )
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(
        args.repo_id, root=args.dataset_root, delta_timestamps=resolve_delta_timestamps(cfg, meta)
    )
    # Each stale sample at t - age needs its own history inside the episode.
    before = max(ages) + cfg.planbind_offset + cfg.n_obs_steps
    frames = sample_frames(
        dataset, args.episodes, args.frames_per_episode, before, cfg.bspline_window, args.seed
    )
    if not frames:
        raise SystemExit("No episode is long enough for the requested ages and window.")

    records = []
    for i, t in enumerate(frames):
        records.append(
            measure_frame(policy, lambda age, t=t: preprocessor(default_collate([dataset[t - age]])), ages)
        )
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(frames)} frames", flush=True)

    summary = aggregate(records, ages)
    result = {
        "policy": str(args.policy),
        "dataset_root": str(args.dataset_root),
        "frames": len(frames),
        "planbind_plan_noise": cfg.planbind_plan_noise,
        "thresholds": list(THRESHOLDS),
        "by_age": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(table(summary, cfg.planbind_plan_noise))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
