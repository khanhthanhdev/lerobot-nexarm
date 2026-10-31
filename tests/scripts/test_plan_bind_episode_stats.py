import importlib.util
import json
import runpy
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
D1_DIR = ROOT / "experiments" / "261027-plan-bind-two-step"
D1B_SUMMARIZE = ROOT / "experiments" / "261029-plan-bind-d1b-seeds" / "summarize.py"


def _load_stats():
    spec = importlib.util.spec_from_file_location("episode_stats", D1_DIR / "episode_stats.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats = _load_stats()


def _eval_info(successes: list[bool]) -> dict:
    n = len(successes)
    return {
        "per_task": [{"metrics": {"successes": successes}}],
        "overall": {"n_episodes": n, "pc_success": 100 * sum(successes) / n},
    }


def _write_cell(eval_dir: Path, name: str, info: dict) -> None:
    (eval_dir / name).mkdir(parents=True)
    (eval_dir / name / "eval_info.json").write_text(json.dumps(info), encoding="utf-8")


def _summarize(eval_dir: Path, capsys, monkeypatch) -> str:
    monkeypatch.setattr(sys, "argv", ["summarize.py", str(eval_dir)])
    runpy.run_path(str(D1B_SUMMARIZE), run_name="__main__")
    return capsys.readouterr().out


def test_mcnemar_exact_known_values():
    assert stats.mcnemar_exact(1, 16) == pytest.approx(0.000275, abs=1e-5)
    assert stats.mcnemar_exact(16, 1) == stats.mcnemar_exact(1, 16)
    assert stats.mcnemar_exact(0, 6) == 0.03125
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(5, 5) == 1.0


def test_episode_successes_and_paired_difference():
    assert stats.episode_successes(_eval_info([True, False, True])) == [True, False, True]
    assert stats.episode_successes({"per_task": [{"metrics": {}}], "overall": {}}) is None
    with pytest.raises(ValueError, match="one task"):
        stats.episode_successes({"per_task": [{}, {}]})
    result = stats.paired_difference([True, True, False, True], [False, True, False, False])
    assert result == {"difference": 50.0, "only_a": 2, "only_b": 0, "p": 0.5}
    with pytest.raises(ValueError, match="same episodes"):
        stats.discordant([True], [True, False])


def test_d1b_summary_pairs_seeds_against_both_controls(tmp_path, capsys, monkeypatch):
    eval_dir = tmp_path / "eval"
    for seed in (1000, 1001):
        _write_cell(eval_dir, f"planbind_L0_latency_s{seed}", _eval_info([True] * 18 + [False] * 2))
        _write_cell(eval_dir, f"onestep_L0_latency_s{seed}", _eval_info([False] * 10 + [True] * 10))
        _write_cell(eval_dir, f"onestep_pos_L0_latency_s{seed}", _eval_info([True] * 17 + [False] * 3))
    # An older cell without per-episode outcomes is reported unpaired, not dropped.
    _write_cell(eval_dir, "planbind_L6_latency_s1000", {"overall": {"n_episodes": 20, "pc_success": 90.0}})
    _write_cell(eval_dir, "onestep_L6_latency_s1000", _eval_info([True] * 10 + [False] * 10))

    out = _summarize(eval_dir, capsys, monkeypatch)
    assert "| onestep | 0 ms, latency | 1000 | +40 | 10 / 2 | 0.04 |" in out
    assert "| onestep_pos | 0 ms, latency | 1001 | +5 | 1 / 0 | 1.00 |" in out
    assert "| onestep | 0 ms, latency | mean | +40.0 | ahead on 2 of 2 seeds | - |" in out
    assert "+40 (unpaired)" in out
    assert "| onestep_pos | 2 | 85 (85 [64-95], 85 [64-95])" in out
    assert "pooled" not in out


@pytest.mark.skipif(not (D1_DIR / "outputs" / "full" / "eval").is_dir(), reason="D1 eval outputs not present")
def test_d1b_summary_on_d1_cells(tmp_path, capsys, monkeypatch):
    eval_dir = tmp_path / "eval"
    for cell in ("onestep_L0_none", "planbind_L0_none"):
        (eval_dir / f"{cell}_s1000").mkdir(parents=True)
        shutil.copy(
            D1_DIR / "outputs" / "full" / "eval" / cell / "eval_info.json", eval_dir / f"{cell}_s1000"
        )
    out = _summarize(eval_dir, capsys, monkeypatch)
    assert "| onestep | 0 ms, naive | 1000 | +56 | 29 / 1 | <0.0001 |" in out
