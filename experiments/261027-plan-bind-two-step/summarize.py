"""Collect the eval cells written by run_all.sh into a Markdown results table.

usage: uv run python experiments/261027-plan-bind-two-step/summarize.py <eval dir> > results.md

Each cell directory is named ``<policy>_L<latency frames>_<commit>[_<suffix>]`` and holds the
``eval_info.json`` of ``lerobot-nexarm-eval``. The second table pairs Plan-Bind with the one-step control
at the same latency and commit mode (RQ1); episodes share seeds, so the difference is paired: the episodes
solved by one policy only and the exact McNemar p on them.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from episode_stats import episode_successes, format_p, paired_difference

FPS = 30
CELL = re.compile(r"^(?P<policy>[a-z0-9]+)_L(?P<latency>\d+)(?:_(?P<commit>[a-z]+))?(?:_(?P<suffix>.+))?$")


def load_cells(eval_dir: Path) -> list[dict]:
    """Parse every `<cell>/eval_info.json` under `eval_dir` whose name matches the cell pattern."""
    cells = []
    for info_path in sorted(eval_dir.glob("*/eval_info.json")):
        match = CELL.match(info_path.parent.name)
        if not match:
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        cells.append(
            {
                "cell": info_path.parent.name,
                "policy": match["policy"],
                "latency": int(match["latency"]),
                "commit": match["commit"] or "-",
                "suffix": match["suffix"] or "",
                "overall": info["overall"],
                "successes": episode_successes(info),
            }
        )
    return cells


def fmt(value: float | None) -> str:
    """Format a percentage, or a dash when missing."""
    return "-" if value is None else f"{value:.1f}"


def main() -> None:
    """Print the results tables for the eval directory given as the first argument."""
    eval_dir = Path(sys.argv[1])
    cells = load_cells(eval_dir)
    if not cells:
        print(f"No eval cells found under {eval_dir}.")
        return
    stages = sorted(
        {
            k
            for c in cells
            for k in c["overall"]
            if k.startswith("pc_") and k not in ("pc_success", "pc_success_ci95")
        }
    )

    print("# Plan-Bind D1 results\n")
    print(f"Source: `{eval_dir}`. Latency in frames at {FPS} fps. Success and stage rates in percent.\n")
    header = [
        "Cell",
        "Policy",
        "Latency (ms)",
        "Commit",
        "Episodes",
        "Success",
        "95% CI",
        *[s[3:] for s in stages],
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for c in cells:
        o = c["overall"]
        ci = o.get("pc_success_ci95")
        ci_text = f"{ci[0]:.0f}-{ci[1]:.0f}" if ci else "-"
        row = [
            c["cell"],
            c["policy"],
            str(round(c["latency"] * 1000 / FPS)),
            c["commit"],
            str(o.get("n_episodes", "-")),
            fmt(o.get("pc_success")),
            ci_text,
            *[fmt(o.get(s)) for s in stages],
        ]
        print("| " + " | ".join(row) + " |")

    by_key = {(c["policy"], c["latency"], c["commit"], c["suffix"]): c for c in cells}
    pairs = [
        (k, by_key[("onestep", *k[1:])])
        for k in by_key
        if k[0] == "planbind" and ("onestep", *k[1:]) in by_key
    ]
    if pairs:
        print("\n## RQ1: Plan-Bind minus one-step control (same seeds)\n")
        print(
            "| Latency (ms) | Commit | One-step success | Plan-Bind success | Difference "
            "| Only Plan-Bind / only one-step | McNemar p |"
        )
        print("|---|---|---|---|---|---|---|")
        for key, control in sorted(pairs, key=lambda p: (p[0][1], p[0][2])):
            ours = by_key[key]["overall"]["pc_success"]
            base = control["overall"]["pc_success"]
            counts, p_value = "-", "-"
            if by_key[key]["successes"] is not None and control["successes"] is not None:
                result = paired_difference(by_key[key]["successes"], control["successes"])
                counts, p_value = f"{result['only_a']} / {result['only_b']}", format_p(result["p"])
            print(
                f"| {round(key[1] * 1000 / FPS)} | {key[2]} | {base:.1f} | {ours:.1f} | {ours - base:+.1f} "
                f"| {counts} | {p_value} |"
            )


if __name__ == "__main__":
    main()
