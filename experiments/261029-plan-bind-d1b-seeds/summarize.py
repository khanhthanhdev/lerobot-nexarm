"""Collect the D1b eval cells written by run_all.sh into Markdown results tables.

usage: uv run python experiments/261029-plan-bind-d1b-seeds/summarize.py <eval dir> > results.md

A cell is named ``<variant>_L<latency frames>[_<commit>][_every<frames>]_s<training seed>[_step<checkpoint>]``
and holds the ``eval_info.json`` of ``lerobot-nexarm-eval``. A cell without a commit mode runs the policy's own
action queue (bsp10). Every seed's cell uses the same eval episodes, so seeds and policies compare episode for
episode. Intervals are 95% Wilson intervals on the pooled episodes of all training seeds.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

CELL = re.compile(
    r"^(?P<variant>[a-z0-9]+(?:_[a-z0-9]+)*?)_L(?P<latency>\d+)(?:_(?P<commit>latency|now|none))?"
    r"(?:_every(?P<every>\d+))?_s(?P<seed>\d+)(?:_step(?P<step>\d+))?$"
)
VARIANTS = ("bsp10", "onestep", "onestep_k2", "onestep_long", "planbind")
# (latency in frames at 30 fps, commit mode, column label)
COLUMNS = [
    (0, "latency", "0 ms, latency"),
    (0, "none", "0 ms, naive"),
    (6, "latency", "200 ms, latency"),
    (9, "latency", "300 ms, latency"),
]


def load_cells(eval_dir: Path) -> list[dict]:
    """Every eval cell under `eval_dir` with its settings and success count."""
    cells = []
    for info_path in sorted(eval_dir.glob("*/eval_info.json")):
        match = CELL.match(info_path.parent.name)
        if not match:
            continue
        overall = json.loads(info_path.read_text(encoding="utf-8"))["overall"]
        n = int(overall["n_episodes"])
        cells.append(
            {
                "variant": match["variant"],
                "latency": int(match["latency"]),
                "commit": match["commit"] or "-",
                "every": int(match["every"] or 0),
                "seed": int(match["seed"]),
                "step": int(match["step"]) if match["step"] else None,
                "n": n,
                "wins": round(overall["pc_success"] * n / 100),
            }
        )
    return cells


def settings(variant: str, latency: int, commit: str = "latency", every: int = 0) -> dict:
    """The cell settings of a variant; bsp10 runs its own action queue, so its cells carry no commit mode."""
    return {
        "variant": variant,
        "latency": latency,
        "commit": "-" if variant == "bsp10" else commit,
        "every": every,
    }


def find(cells: list[dict], **wanted: object) -> list[dict]:
    """The final-checkpoint cells (one per seed) matching `wanted`."""
    return sorted(
        (c for c in cells if c["step"] is None and all(c[k] == v for k, v in wanted.items())),
        key=lambda c: c["seed"],
    )


def pct(c: dict) -> float:
    """A cell's success rate in percent."""
    return 100 * c["wins"] / c["n"]


def wilson(wins: int, n: int) -> tuple[float, float]:
    """95% Wilson interval of a success rate, in percent."""
    p, z = wins / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def pooled(found: list[dict]) -> str:
    """Pooled success over the seeds with its interval, then the per-seed values."""
    if not found:
        return "-"
    wins, n = sum(c["wins"] for c in found), sum(c["n"] for c in found)
    low, high = wilson(wins, n)
    text = f"{100 * wins / n:.0f} [{low:.0f}-{high:.0f}]"
    return text if len(found) == 1 else f"{text} ({', '.join(f'{pct(c):.0f}' for c in found)})"


def table(header: list[str], rows: list[list[str]]) -> None:
    """Print a Markdown table."""
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def main() -> None:
    """Print the results tables for the eval directory given as the first argument."""
    eval_dir = Path(sys.argv[1])
    cells = load_cells(eval_dir)
    print("# Plan-Bind D1b results: stack bowls, several training seeds\n")
    if not cells:
        print(f"No eval cells found under {eval_dir}.")
        return
    seeds = sorted({c["seed"] for c in cells})
    print(
        f"Source: `{eval_dir}`. Training seeds {', '.join(map(str, seeds))}. Success in percent: pooled over "
        "the seeds, [95% Wilson interval], then the per-seed values in brackets.\n"
    )
    variants = [v for v in VARIANTS if any(c["variant"] == v for c in cells)]

    print("## Main comparison (last checkpoint)\n")
    rows = []
    for v in variants:
        row = [v, str(len(find(cells, **settings(v, 0))))]
        for latency, commit, _ in COLUMNS:
            # bsp10 has no latency machinery: only its latency-0 cell exists.
            only_latency_0 = v == "bsp10" and (latency, commit) != (0, "latency")
            row.append("-" if only_latency_0 else pooled(find(cells, **settings(v, latency, commit))))
        rows.append(row)
    table(["Policy", "Seeds", *[label for *_, label in COLUMNS]], rows)

    print("## Plan-Bind minus one-step (same training seed, same eval episodes)\n")
    print("Difference in percentage points per seed; `wins` counts the seeds where Plan-Bind is ahead.\n")
    rows = []
    for latency, commit, label in COLUMNS:
        diffs = []
        for c in find(cells, **settings("planbind", latency, commit)):
            control = [
                o for o in find(cells, **settings("onestep", latency, commit)) if o["seed"] == c["seed"]
            ]
            if control:
                diffs.append(pct(c) - pct(control[0]))
        if diffs:
            per_seed = ", ".join(f"{d:+.0f}" for d in diffs)
            rows.append(
                [
                    label,
                    per_seed,
                    f"{statistics.fmean(diffs):+.1f}",
                    f"{sum(d > 0 for d in diffs)} of {len(diffs)}",
                ]
            )
    table(["Setting", "Per seed", "Mean", "wins"], rows)

    controls = [v for v in ("onestep", "onestep_k2", "onestep_long") if v in variants]
    if len(controls) > 1:
        print("## Is the one-step control strong enough? (commit latency)\n")
        print(
            "`onestep_k2` calls twice as often; the `every 4` column runs it at D1's call rate. "
            "`onestep_long` trains twice as long.\n"
        )
        rows = []
        for v in (*controls, "planbind"):
            at_rate = pooled(find(cells, **settings(v, 0, every=4))) if v == "onestep_k2" else "-"
            rows.append(
                [v, pooled(find(cells, **settings(v, 0))), at_rate, pooled(find(cells, **settings(v, 6)))]
            )
        table(["Policy", "0 ms", "0 ms, every 4 frames", "200 ms"], rows)

    sweep = [c for c in cells if c["step"] is not None]
    if sweep:
        print("## Checkpoint curve (latency 0, success)\n")
        steps = sorted({c["step"] for c in sweep})
        rows = []
        for v in variants:
            for seed in seeds:
                by_step = {c["step"]: c for c in sweep if c["variant"] == v and c["seed"] == seed}
                if by_step:
                    rows.append(
                        [v, str(seed)] + [f"{pct(by_step[s]):.0f}" if s in by_step else "-" for s in steps]
                    )
        table(["Policy", "Seed", *[f"{s // 1000}k" for s in steps]], rows)

    print("## All cells\n")
    ordered = sorted(
        cells,
        key=lambda c: (c["variant"], c["step"] or 10**9, c["latency"], c["commit"], c["every"], c["seed"]),
    )
    rows = [
        [
            c["variant"],
            str(c["latency"]),
            c["commit"],
            str(c["every"] or "-"),
            str(c["seed"]),
            "last" if c["step"] is None else str(c["step"]),
            str(c["n"]),
            f"{pct(c):.0f}",
        ]
        for c in ordered
    ]
    table(
        ["Policy", "Latency (frames)", "Commit", "Every", "Seed", "Checkpoint", "Episodes", "Success"], rows
    )


if __name__ == "__main__":
    main()
