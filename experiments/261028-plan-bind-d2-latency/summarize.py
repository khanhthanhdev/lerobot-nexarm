"""Collect D2's inference timings and eval cells into Markdown results tables.

usage: uv run python experiments/261028-plan-bind-d2-latency/summarize.py <output dir> > results.md

`<output dir>/bench/*.json` are the timings of `bench_inference.py`; `<output dir>/eval/<cell>/eval_info.json`
are the eval cells of `run_all.sh`. A cell is named ``<policy>_L<latency frames>[_<commit>][_<option>...]``
where an option is ``every<frames>`` (call period), ``tau<threshold>`` (Plan-Bind replan threshold),
``shift<cm>cm`` (the bottom bowl slides when the middle bowl is lifted) or ``s<seed>`` (training seed).
A cell without a commit mode is the classic action-queue policy (bsp10 only). Every cell runs the same 50
seeded episodes, so two cells compare episode by episode: "+a/-b" counts the episodes only the second
cell succeeded / failed in.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FPS = 30
CALL_EVERY = 4  # the default call period: one fine knot span
CELL = re.compile(
    r"^(?P<policy>[a-z0-9]+)_L(?P<latency>\d+)(?:_(?P<commit>latency|now|none))?(?P<rest>(?:_.+)?)$"
)
OPTION = re.compile(r"^(?:every(?P<every>\d+)|tau(?P<tau>[\d.]+)|(?P<shift>shift[\d.]+cm)|s(?P<seed>\d+))$")


def parse_cell(name: str) -> dict | None:
    """The settings encoded in a cell name, or None when the name is not a cell."""
    match = CELL.match(name)
    if not match:
        return None
    cell = {
        "policy": match["policy"],
        "latency": int(match["latency"]),
        "commit": match["commit"] or "-",
        "every": CALL_EVERY,
        "tau": None,
        "shift": "",
        "seed": None,
    }
    for token in filter(None, match["rest"].split("_")):
        option = OPTION.match(token)
        if not option:
            return None
        if option["every"]:
            cell["every"] = int(option["every"])
        elif option["tau"]:
            cell["tau"] = option["tau"]
        elif option["shift"]:
            cell["shift"] = option["shift"]
        else:
            cell["seed"] = int(option["seed"])
    return cell


def load_cells(eval_dir: Path) -> dict[str, dict]:
    """Every eval cell under `eval_dir`, by name, with its overall metrics and per-episode successes."""
    cells = {}
    for info_path in sorted(eval_dir.glob("*/eval_info.json")):
        name = info_path.parent.name
        cell = parse_cell(name)
        if cell is None:
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        cell["overall"] = info["overall"]
        cell["successes"] = [bool(s) for task in info["per_task"] for s in task["metrics"]["successes"]]
        cells[name] = cell
    return cells


def ms(frames: int) -> int:
    """A latency in frames, in ms."""
    return round(frames * 1000 / FPS)


def success(cell: dict | None) -> str:
    """A cell's success rate, or a dash when the cell was not run."""
    return "-" if cell is None else f"{cell['overall']['pc_success']:.0f}"


def paired(base: dict | None, other: dict | None) -> str:
    """Success difference of `other` over `base` in points, with its discordant episodes."""
    if base is None or other is None:
        return "-"
    gained = sum(o and not b for b, o in zip(base["successes"], other["successes"], strict=False))
    lost = sum(b and not o for b, o in zip(base["successes"], other["successes"], strict=False))
    diff = other["overall"]["pc_success"] - base["overall"]["pc_success"]
    return f"{diff:+.0f} (+{gained}/-{lost})"


def table(header: list[str], rows: list[list[str]]) -> None:
    """Print a Markdown table."""
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def cell_name(policy: str, latency: int, commit: str = "latency", *options: str) -> str:
    """The name run_all.sh gives a cell; empty options are left out."""
    parts = [policy, f"L{latency}"] + ([] if commit == "-" else [commit]) + [o for o in options if o]
    return "_".join(parts)


def bench_tables(bench_dir: Path) -> dict[str, dict]:
    """Print the timing table; returns the bench results by tag."""
    benches = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(bench_dir.glob("*.json"))}
    if not benches:
        return benches
    print("## Inference time of one call (batch 1)\n")
    print(
        "A call: preprocess, encoders on the observation frames, then the head (bsp10: DDIM denoise and "
        "spline decode; planbind: plan and bind, the slower call). Latency frames: the call's p95 rounded up "
        f"to frames at {FPS} fps, the latency each policy is evaluated at below.\n"
    )
    rows = []
    for tag, bench in benches.items():
        for policy, rec in bench["policies"].items():
            stages = ", ".join(f"{k} {v['p50_ms']:.1f}" for k, v in rec["stages"].items())
            bind = rec.get("bind_only_call")
            rows.append(
                [
                    tag,
                    bench["hardware"],
                    policy,
                    f"{rec['call']['p50_ms']:.1f}",
                    f"{rec['call']['p95_ms']:.1f}",
                    "-" if bind is None else f"{bind['p50_ms']:.1f}",
                    str(rec["latency_frames"]),
                    f"{rec['select_action']['mean_ms']:.1f}",
                    stages,
                ]
            )
    table(
        [
            "Bench",
            "Hardware",
            "Policy",
            "Call p50 (ms)",
            "Call p95 (ms)",
            "Bind-only p50 (ms)",
            "Latency frames",
            "select_action mean (ms)",
            "Stage p50 (ms)",
        ],
        rows,
    )
    return benches


def main() -> None:
    """Print the results tables for the output directory given as the first argument."""
    out_dir = Path(sys.argv[1])
    cells = load_cells(out_dir / "eval")
    print("# Plan-Bind D2 results\n")
    print(
        f"Source: `{out_dir}`. Success and stage rates in percent; 50 seeded episodes per cell unless noted.\n"
    )
    benches = bench_tables(out_dir / "bench")
    if not cells:
        print(f"No eval cells found under {out_dir / 'eval'}.")
        return
    get = cells.get

    if benches:
        print("## Each policy at its measured latency\n")
        rows = []
        for tag, bench in benches.items():
            if tag not in ("cuda", "cpu"):
                continue
            for policy, rec in bench["policies"].items():
                frames = rec["latency_frames"]
                rows.append(
                    [tag, policy, str(frames), str(ms(frames)), success(get(cell_name(policy, frames)))]
                )
        table(["Bench", "Policy", "Latency (frames)", "Latency (ms)", "Success"], rows)

    print("## Update rate without latency\n")
    print(
        "Latency 0; a call every 4 frames is D1's rate. In D1 a call could not start before the previous one"
    )
    print("swapped in, so latency 0, 3, 6 and 9 also meant a call every 4, 4, 8 and 12 frames: these cells")
    print("tell rate from latency apart.\n")
    rows = []
    for policy in ("onestep", "planbind"):
        for commit in ("latency", "none"):
            row = [policy, commit]
            for every in (4, 8, 12):
                row.append(success(get(cell_name(policy, 0, commit, "" if every == 4 else f"every{every}"))))
            rows.append(row)
    table(["Policy", "Commit", "Every 4", "Every 8", "Every 12"], rows)

    print("## Commit modes\n")
    rows = []
    for policy in ("onestep", "planbind"):
        for latency in sorted({c["latency"] for c in cells.values() if c["commit"] == "now"}):
            base = get(cell_name(policy, latency, "latency"))
            now = get(cell_name(policy, latency, "now"))
            none = get(cell_name(policy, latency, "none"))
            rows.append(
                [policy, str(ms(latency)), success(base), success(now), success(none), paired(base, now)]
            )
    table(["Policy", "Latency (ms)", "latency", "now", "none", "now vs latency"], rows)

    taus = sorted({c["tau"] for c in cells.values() if c["tau"]}, key=float)
    if taus:
        print("## Plan-Bind replan threshold (latency 6, commit latency)\n")
        base = get(cell_name("planbind", 6))
        rows = [["off", success(base), "-"]]
        for t in taus:
            c = get(cell_name("planbind", 6, "latency", f"tau{t}"))
            rows.append([t, success(c), paired(base, c)])
        table(["Threshold", "Success", "vs off"], rows)

    print("## bsp10 through the spline runtime\n")
    print(
        "`classic` is the action-queue policy (D1's bsp10_L0): it plans 32 frames and replans when they run out.\n"
    )
    classic = get("bsp10_L0")
    rows = [["classic", "0", "-", success(classic), "-"]]
    for c in sorted((c for c in cells.values() if c["policy"] == "bsp10" and c["commit"] != "-"), key=_order):
        if c["shift"] or c["seed"] is not None:
            continue
        rows.append(["runtime", str(ms(c["latency"])), c["commit"], success(c), paired(classic, c)])
    table(["Mode", "Latency (ms)", "Commit", "Success", "vs classic"], rows)

    shifted = sorted((c for c in cells.values() if c["shift"]), key=_order)
    if shifted:
        print("## The bottom bowl moves when the middle bowl is lifted\n")
        rows = []
        for c in shifted:
            base_name = cell_name(c["policy"], c["latency"], c["commit"])
            base = get(base_name)
            o = c["overall"]
            rows.append(
                [
                    base_name,
                    c["shift"],
                    success(base),
                    success(c),
                    paired(base, c),
                    f"{o.get('pc_nested_middle', 0):.0f}",
                ]
            )
        table(["Cell", "Shift", "Static", "Shifted", "Shifted vs static", "Middle nested (shifted)"], rows)

    seeded = sorted((c for c in cells.values() if c["seed"] is not None), key=_order)
    if seeded:
        print("## Training seeds\n")
        print(
            "Seed 1000 is D1's training run; each row compares one checkpoint with it on the same episodes.\n"
        )
        rows = []
        for c in seeded:
            base = get(cell_name(c["policy"], c["latency"], c["commit"]))
            rows.append(
                [c["policy"], str(ms(c["latency"])), c["commit"], str(c["seed"]), success(base), success(c)]
            )
        table(["Policy", "Latency (ms)", "Commit", "Seed", "Seed 1000", "This seed"], rows)

    stages = sorted(
        {k for c in cells.values() for k in c["overall"] if k.startswith("pc_") and k != "pc_success_ci95"}
        - {"pc_success"}
    )
    print("## All cells\n")
    rows = []
    for name, c in sorted(cells.items(), key=lambda item: _order(item[1])):
        o = c["overall"]
        ci = o.get("pc_success_ci95")
        rows.append(
            [
                name,
                str(o.get("n_episodes", "-")),
                f"{o['pc_success']:.0f}",
                f"{ci[0]:.0f}-{ci[1]:.0f}" if ci else "-",
                *[f"{o.get(s, 0):.0f}" for s in stages],
            ]
        )
    table(["Cell", "Episodes", "Success", "95% CI", *[s[3:] for s in stages]], rows)


def _order(c: dict) -> tuple:
    return (c["policy"], c["shift"], c["seed"] or 0, c["latency"], c["commit"], c["every"], c["tau"] or "")


if __name__ == "__main__":
    main()
