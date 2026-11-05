"""Collect the LIBERO eval cells written by run_all.sh into Markdown results tables.

usage: uv run python experiments/261105-plan-bind-libero/summarize.py <eval dir> > results.md

A cell is named ``<policy>_L<latency frames>[_<commit>][_every<frames>]_s<training seed>`` and holds the
``eval_info.json`` of ``lerobot-eval`` on the four LIBERO suites. A cell without a commit mode runs the policy's
own action queue (bsp10). Success is the share of successful episodes, per suite and over all episodes; each
cell's intervals are in its ``eval_info.json``.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

FPS = 10
CALL_EVERY = 2  # the default call period: one fine knot span
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CELL = re.compile(
    r"^(?P<policy>[a-z0-9]+)_L(?P<latency>\d+)(?:_(?P<commit>latency|now|none))?(?:_every(?P<every>\d+))?"
    r"_s(?P<seed>\d+)$"
)
POLICIES = ("bsp10", "onestep", "planbind")


def load_cells(eval_dir: Path) -> list[dict]:
    """Every eval cell under `eval_dir` with its settings and per-suite success."""
    cells = []
    for info_path in sorted(eval_dir.glob("*/eval_info.json")):
        match = CELL.match(info_path.parent.name)
        if not match:
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        by_suite: dict[str, list[bool]] = {}
        for task in info["per_task"]:
            by_suite.setdefault(task["task_group"], []).extend(bool(s) for s in task["metrics"]["successes"])
        all_episodes = [s for episodes in by_suite.values() for s in episodes]
        cells.append(
            {
                "policy": match["policy"],
                "latency": int(match["latency"]),
                "commit": match["commit"] or "-",
                "every": int(match["every"] or CALL_EVERY),
                "seed": int(match["seed"]),
                "episodes": len(all_episodes),
                "success": 100 * sum(all_episodes) / len(all_episodes),
                "suites": {k: 100 * sum(v) / len(v) for k, v in by_suite.items()},
            }
        )
    return cells


def find(cells: list[dict], **settings: object) -> list[dict]:
    """The cells (one per seed) matching `settings`."""
    return sorted((c for c in cells if all(c[k] == v for k, v in settings.items())), key=lambda c: c["seed"])


def mean(found: list[dict], get=lambda c: c["success"]) -> str:
    """Mean over seeds, with the per-seed values when there are several."""
    if not found:
        return "-"
    values = [get(c) for c in found]
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
    print("# Plan-Bind D4 results: LIBERO\n")
    if not cells:
        print(f"No eval cells found under {eval_dir}.")
        return
    seeds = sorted({c["seed"] for c in cells})
    print(
        f"Source: `{eval_dir}`. Training seeds {', '.join(map(str, seeds))}; success in percent, mean over "
        "seeds with the per-seed values in brackets.\n"
    )

    print("## Main comparison (latency 0, last checkpoint)\n")
    rows = []
    for p in POLICIES:
        found = find(cells, **main_settings(p))
        row = [
            p,
            mean(found),
            *[mean(found, lambda c, s=s: c["suites"].get(s, float("nan"))) for s in SUITES],
        ]
        rows.append([*row, str(found[0]["episodes"]) if found else "-"])
    table(["Policy", "All suites", *SUITES, "Episodes"], rows)

    print("## Call rate and commit (latency 0, success)\n")
    rows = []
    for p in ("onestep", "planbind"):
        for commit in ("latency", "none"):
            rows.append([p, commit, mean(find(cells, policy=p, latency=0, commit=commit, every=CALL_EVERY))])
    table(["Policy", "Commit", "Every 2 frames"], rows)

    print("## Latency (commit latency, success)\n")
    print("bsp10 runs through the spline runtime here; its classic action queue is in the main comparison.\n")
    latencies = sorted({c["latency"] for c in cells})
    rows = []
    for p in POLICIES:
        row = [p]
        for latency in latencies:
            row.append(mean(find(cells, policy=p, latency=latency, commit="latency", every=CALL_EVERY)))
        rows.append(row)
    table(["Policy", *[f"{round(lat * 1000 / FPS)} ms" for lat in latencies]], rows)

    print("## All cells\n")
    rows = [
        [
            c["policy"],
            str(c["latency"]),
            c["commit"],
            str(c["every"]),
            str(c["seed"]),
            str(c["episodes"]),
            f"{c['success']:.1f}",
        ]
        for c in sorted(cells, key=lambda c: (c["policy"], c["latency"], c["commit"], c["every"], c["seed"]))
    ]
    table(["Policy", "Latency", "Commit", "Every", "Seed", "Episodes", "Success"], rows)


if __name__ == "__main__":
    main()
