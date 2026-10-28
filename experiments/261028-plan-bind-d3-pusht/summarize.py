"""Collect the Push-T eval cells written by run_all.sh into Markdown results tables.

usage: uv run python experiments/261028-plan-bind-d3-pusht/summarize.py <eval dir> > results.md

A cell is named ``<policy>_L<latency frames>[_<commit>][_every<frames>]_s<training seed>[_step<checkpoint>]``
and holds the ``eval_info.json`` of ``lerobot-eval``. A cell without a commit mode runs the policy's own
action queue (diffusion, classic bsp10). Scores are the B-spline Policy paper's Push-T metrics: the mean over
episodes of the episode's best target coverage (``avg_max_reward``), and the share of episodes whose best
coverage reaches 0.9. ``Env success`` is the env's own threshold (coverage 0.95).
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

FPS = 10
CALL_EVERY = 2  # the default call period: one fine knot span
CELL = re.compile(
    r"^(?P<policy>[a-z0-9]+)_L(?P<latency>\d+)(?:_(?P<commit>latency|now|none))?(?:_every(?P<every>\d+))?"
    r"_s(?P<seed>\d+)(?:_step(?P<step>\d+))?$"
)
POLICIES = ("diffusion", "bsp10", "onestep", "planbind")
# B-spline Policy paper, Table 2a (Push-T, average score, their codebase and best three checkpoints).
PAPER = {"diffusion": "72", "bsp10": "75"}


def load_cells(eval_dir: Path) -> list[dict]:
    """Every eval cell under `eval_dir` with its settings and Push-T metrics."""
    cells = []
    for info_path in sorted(eval_dir.glob("*/eval_info.json")):
        match = CELL.match(info_path.parent.name)
        if not match:
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        max_rewards = [r for task in info["per_task"] for r in task["metrics"]["max_rewards"]]
        cells.append(
            {
                "policy": match["policy"],
                "latency": int(match["latency"]),
                "commit": match["commit"] or "-",
                "every": int(match["every"] or CALL_EVERY),
                "seed": int(match["seed"]),
                "step": int(match["step"]) if match["step"] else None,
                "episodes": len(max_rewards),
                "score": 100 * statistics.fmean(max_rewards),
                "success90": 100 * sum(r >= 0.9 for r in max_rewards) / len(max_rewards),
                "env_success": info["overall"]["pc_success"],
            }
        )
    return cells


def find(cells: list[dict], **settings: object) -> list[dict]:
    """The final-checkpoint cells (one per seed) matching `settings`."""
    return sorted(
        (c for c in cells if c["step"] is None and all(c[k] == v for k, v in settings.items())),
        key=lambda c: c["seed"],
    )


def mean(found: list[dict], key: str = "score") -> str:
    """Mean of `key` over seeds, with the per-seed values when there are several."""
    if not found:
        return "-"
    values = [c[key] for c in found]
    text = f"{statistics.fmean(values):.1f}"
    return text if len(values) == 1 else f"{text} ({', '.join(f'{v:.0f}' for v in values)})"


def table(header: list[str], rows: list[list[str]]) -> None:
    """Print a Markdown table."""
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def main_settings(policy: str, latency: int = 0) -> dict:
    """The settings of a policy's main cell: the runtime with latency commit for the one-step modes."""
    commit = "latency" if policy in ("onestep", "planbind") else "-"
    return {"policy": policy, "latency": latency, "commit": commit, "every": CALL_EVERY}


def main() -> None:
    """Print the results tables for the eval directory given as the first argument."""
    eval_dir = Path(sys.argv[1])
    cells = load_cells(eval_dir)
    print("# Plan-Bind D3 results: Push-T\n")
    if not cells:
        print(f"No eval cells found under {eval_dir}.")
        return
    seeds = sorted({c["seed"] for c in cells})
    print(
        f"Source: `{eval_dir}`. Training seeds {', '.join(map(str, seeds))}; values are the mean over seeds, "
        "per-seed values in brackets. Score: mean best coverage in percent.\n"
    )

    print("## Main comparison (latency 0, last checkpoint)\n")
    rows = []
    for p in POLICIES:
        found = find(cells, **main_settings(p))
        rows.append(
            [
                p,
                mean(found),
                mean(found, "success90"),
                mean(found, "env_success"),
                PAPER.get(p, "-"),
                str(found[0]["episodes"]) if found else "-",
            ]
        )
    table(["Policy", "Score", "Coverage >= 0.9 (%)", "Env success (%)", "Paper score", "Episodes"], rows)

    sweep = [c for c in cells if c["step"] is not None]
    if sweep:
        print("## Checkpoint curve (latency 0, score)\n")
        steps = sorted({c["step"] for c in sweep})
        rows = []
        for p in POLICIES:
            for seed in seeds:
                by_step = {c["step"]: c for c in sweep if c["policy"] == p and c["seed"] == seed}
                if by_step:
                    rows.append(
                        [p, str(seed)]
                        + [f"{by_step[s]['score']:.1f}" if s in by_step else "-" for s in steps]
                    )
        table(["Policy", "Seed", *[str(s) for s in steps]], rows)

    print("## Call rate and commit (latency 0, score)\n")
    print(
        f"A call every 2 frames is the default (0.2 s); every 8 is diffusion's rate (8 actions at {FPS} fps).\n"
    )
    rows = []
    for p in ("onestep", "planbind"):
        for commit in ("latency", "none"):
            row = [p, commit]
            for every in (2, 8):
                row.append(mean(find(cells, policy=p, latency=0, commit=commit, every=every)))
            rows.append(row)
    table(["Policy", "Commit", "Every 2", "Every 8"], rows)

    print("## Latency (commit latency, score)\n")
    print("bsp10 runs through the spline runtime here; its classic action queue is in the main comparison.\n")
    latencies = sorted({c["latency"] for c in cells})
    rows = []
    for p in ("bsp10", "onestep", "planbind"):
        row = [p]
        for latency in latencies:
            row.append(mean(find(cells, policy=p, latency=latency, commit="latency", every=CALL_EVERY)))
        rows.append(row)
    table(["Policy", *[f"{round(lat * 1000 / FPS)} ms" for lat in latencies]], rows)

    print("## All cells\n")
    rows = []
    for c in sorted(
        cells,
        key=lambda c: (c["policy"], c["step"] or 10**9, c["latency"], c["commit"], c["every"], c["seed"]),
    ):
        rows.append(
            [
                c["policy"],
                str(c["latency"]),
                c["commit"],
                str(c["every"]),
                str(c["seed"]),
                "last" if c["step"] is None else str(c["step"]),
                str(c["episodes"]),
                f"{c['score']:.1f}",
                f"{c['success90']:.1f}",
                f"{c['env_success']:.1f}",
            ]
        )
    table(
        [
            "Policy",
            "Latency",
            "Commit",
            "Every",
            "Seed",
            "Checkpoint",
            "Episodes",
            "Score",
            ">= 0.9",
            "Env success",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
