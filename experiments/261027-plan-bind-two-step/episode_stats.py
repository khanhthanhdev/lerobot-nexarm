"""Paired statistics on the per-episode outcomes of ``lerobot-nexarm-eval`` cells.

Cells evaluated with the same eval seed run the same episodes, so two policies compare episode by episode:
the discordant episodes (solved by one policy only) and an exact McNemar test on them. The Plan-Bind
summarizers (D1 here, D1b by file path) share these functions; no third-party imports.
"""

from __future__ import annotations

import math


def episode_successes(eval_info: dict) -> list[bool] | None:
    """Per-episode success of a single-task ``eval_info.json``, in episode order; None when not recorded."""
    tasks = eval_info.get("per_task") or []
    if len(tasks) > 1:
        raise ValueError(f"Expected one task per eval cell, got {len(tasks)}; pair the tasks separately.")
    successes = tasks[0].get("metrics", {}).get("successes") if tasks else None
    return None if successes is None else [bool(s) for s in successes]


def discordant(a: list[bool], b: list[bool]) -> tuple[int, int]:
    """Episodes solved only by `a` and only by `b`."""
    if len(a) != len(b):
        raise ValueError(f"Paired cells need the same episodes; got {len(a)} and {len(b)}.")
    return sum(x and not y for x, y in zip(a, b, strict=True)), sum(
        y and not x for x, y in zip(a, b, strict=True)
    )


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for `b` and `c` discordant episodes (binomial test at 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilson(wins: int, n: int) -> tuple[float, float]:
    """95% Wilson interval of a success rate, in percent."""
    p, z = wins / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def paired_difference(a: list[bool], b: list[bool]) -> dict:
    """Success of `a` minus `b` in points, the discordant counts and the McNemar p-value."""
    only_a, only_b = discordant(a, b)
    return {
        "difference": 100 * (only_a - only_b) / len(a),
        "only_a": only_a,
        "only_b": only_b,
        "p": mcnemar_exact(only_a, only_b),
    }


def format_p(p: float) -> str:
    """A p-value for a results table."""
    return "<0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.2f}"
