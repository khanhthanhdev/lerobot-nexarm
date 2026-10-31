"""Collect the D1b eval cells written by run_all.sh into Markdown results tables.

usage: uv run python experiments/261029-plan-bind-d1b-seeds/summarize.py <eval dir> > results.md

A cell is named ``<variant>_L<latency frames>[_<commit>][_every<frames>]_s<training seed>[_step<checkpoint>]``
and holds the ``eval_info.json`` of ``lerobot-nexarm-eval``. A cell without a commit mode runs the policy's own
action queue (bsp10). Every seed's cell uses the same eval episodes, so seeds and policies compare episode for
episode, so the unit of comparison is one training seed: per-seed success with its 95% Wilson interval,
and per-seed paired differences (episodes solved by one policy only, exact McNemar p). No interval is pooled
over seeds: the seeds share their eval episodes, so pooled episodes are not independent.
"""

from __future__ import annotations

import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

_STATS_PATH = Path(__file__).resolve().parents[1] / "261027-plan-bind-two-step" / "episode_stats.py"
_spec = importlib.util.spec_from_file_location("episode_stats", _STATS_PATH)
assert _spec is not None and _spec.loader is not None
stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stats)

CELL = re.compile(
    r"^(?P<variant>[a-z0-9]+(?:_[a-z0-9]+)*?)_L(?P<latency>\d+)(?:_(?P<commit>latency|now|none))?"
    r"(?:_every(?P<every>\d+))?_s(?P<seed>\d+)(?:_step(?P<step>\d+))?$"
)
VARIANTS = ("bsp10", "onestep", "onestep_pos", "onestep_k2", "onestep_long", "planbind")
# One-step controls Plan-Bind is paired against in the RQ1 table.
PAIRED_CONTROLS = ("onestep", "onestep_pos")
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
        info = json.loads(info_path.read_text(encoding="utf-8"))
        successes = stats.episode_successes(info)
        if successes is not None:
            n, wins = len(successes), sum(successes)
        else:
            n = int(info["overall"]["n_episodes"])
            wins = round(info["overall"]["pc_success"] * n / 100)
        cells.append(
            {
                "variant": match["variant"],
                "latency": int(match["latency"]),
                "commit": match["commit"] or "-",
                "every": int(match["every"] or 0),
                "seed": int(match["seed"]),
                "step": int(match["step"]) if match["step"] else None,
                "n": n,
                "wins": wins,
                "successes": successes,
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


def per_seed(found: list[dict]) -> str:
    """Success per seed with its interval; with several seeds, their mean first."""
    if not found:
        return "-"
    values = []
    for c in found:
        low, high = stats.wilson(c["wins"], c["n"])
        values.append(f"{pct(c):.0f} [{low:.0f}-{high:.0f}]")
    if len(found) == 1:
        return values[0]
    return f"{statistics.fmean(pct(c) for c in found):.0f} ({', '.join(values)})"


def paired_row(ours: dict, control: dict) -> list[str]:
    """Plan-Bind minus a control on one seed: difference, discordant episodes and McNemar p."""
    if ours["successes"] is None or control["successes"] is None:
        return [f"{pct(ours) - pct(control):+.0f} (unpaired)", "-", "-"]
    result = stats.paired_difference(ours["successes"], control["successes"])
    return [
        f"{result['difference']:+.0f}",
        f"{result['only_a']} / {result['only_b']}",
        stats.format_p(result["p"]),
    ]


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
        f"Source: `{eval_dir}`. Training seeds {', '.join(map(str, seeds))}. Success in percent: the mean over "
        "the seeds, then each seed's success with its [95% Wilson interval].\n"
    )
    variants = [v for v in VARIANTS if any(c["variant"] == v for c in cells)]

    print("## Main comparison (last checkpoint)\n")
    rows = []
    for v in variants:
        row = [v, str(len(find(cells, **settings(v, 0))))]
        for latency, commit, _ in COLUMNS:
            # bsp10 has no latency machinery: only its latency-0 cell exists.
            only_latency_0 = v == "bsp10" and (latency, commit) != (0, "latency")
            row.append("-" if only_latency_0 else per_seed(find(cells, **settings(v, latency, commit))))
        rows.append(row)
    table(["Policy", "Seeds", *[label for *_, label in COLUMNS]], rows)

    print("## Plan-Bind minus one-step (same training seed, same eval episodes)\n")
    print(
        "Per seed: the difference in percentage points, the episodes solved only by Plan-Bind / only by the "
        "control, and the exact McNemar p on them. The `mean` row averages the per-seed differences and counts "
        "the seeds where Plan-Bind is ahead. `(unpaired)` marks cells without per-episode outcomes.\n"
    )
    rows = []
    for control_name in (c for c in PAIRED_CONTROLS if c in variants):
        for latency, commit, label in COLUMNS:
            diffs = []
            for ours in find(cells, **settings("planbind", latency, commit)):
                control = [
                    o
                    for o in find(cells, **settings(control_name, latency, commit))
                    if o["seed"] == ours["seed"]
                ]
                if control:
                    diffs.append(pct(ours) - pct(control[0]))
                    rows.append([control_name, label, str(ours["seed"]), *paired_row(ours, control[0])])
            if len(diffs) > 1:
                rows.append(
                    [
                        control_name,
                        label,
                        "mean",
                        f"{statistics.fmean(diffs):+.1f}",
                        f"ahead on {sum(d > 0 for d in diffs)} of {len(diffs)} seeds",
                        "-",
                    ]
                )
    table(["Control", "Setting", "Seed", "Difference", "Only Plan-Bind / only control", "McNemar p"], rows)

    controls = [v for v in ("onestep", "onestep_pos", "onestep_k2", "onestep_long") if v in variants]
    if len(controls) > 1:
        print("## Is the one-step control strong enough? (commit latency)\n")
        print(
            "`onestep_pos` feeds the UNet a trained ramp instead of zeros, so its input carries sequence "
            "position. `onestep_k2` calls twice as often; the `every 4` column runs it at D1's call rate. "
            "`onestep_long` trains twice as long.\n"
        )
        rows = []
        for v in (*controls, "planbind"):
            at_rate = per_seed(find(cells, **settings(v, 0, every=4))) if v == "onestep_k2" else "-"
            rows.append(
                [v, per_seed(find(cells, **settings(v, 0))), at_rate, per_seed(find(cells, **settings(v, 6)))]
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
