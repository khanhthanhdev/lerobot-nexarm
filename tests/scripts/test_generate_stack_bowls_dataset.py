# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.nexarm import generate_stack_bowls_dataset as generator, sim_dataset_utils as utils
from lerobot.motors.nexarm.nexarm import JOINT_NAMES
from lerobot.robots.nexarm_sim import NexArmSim, NexArmSimConfig, NexArmStackBowlsTask, SimCalibration

MODEL_PATH = Path(__file__).resolve().parents[2] / "sim" / "fusion_export" / "bowl_stack_scene.xml"


def _camera_free_config(**kwargs) -> NexArmSimConfig:
    return NexArmSimConfig(**{**kwargs, "camera_names": ()})


def test_recorded_actions_are_commands_not_delayed_actions() -> None:
    def first_actions(delay: int) -> np.ndarray:
        robot = NexArmSim(
            NexArmSimConfig(
                id="stack_bowls_test",
                model_path=MODEL_PATH,
                camera_names=(),
                settle_steps=0,
                action_delay_steps=delay,
            )
        )
        robot.connect()
        try:
            frames: list[dict[str, float]] = []
            generator.generate_episode(
                robot,
                NexArmStackBowlsTask(robot.backend),
                seed=0,
                record_frame=lambda _obs, action: frames.append(dict(action)),
            )
            return np.array([[a[f"{n}.pos"] for n in JOINT_NAMES] for a in frames[:10]])
        finally:
            robot.disconnect()

    # The first stage starts from the same home pose, so labels must not depend on the transport delay.
    np.testing.assert_allclose(first_actions(0), first_actions(2))


def test_domain_randomization_defaults_match_pick_place(tmp_path: Path) -> None:
    from examples.nexarm.generate_sim_dataset import parse_args as pick_place_parse_args

    root = ["--root", str(tmp_path / "ds")]
    for parse in (generator.parse_args, pick_place_parse_args):
        assert parse(root).domain_randomization
        assert not parse([*root, "--no-dr"]).domain_randomization
        assert parse([*root, "--dr"]).domain_randomization
        assert parse([*root, "--domain-randomization"]).domain_randomization


def test_gpus_select_one_worker_per_id_and_eval_split_seeds(tmp_path: Path) -> None:
    args = generator.parse_args(["--root", str(tmp_path / "ds"), "--gpus", "0,1", "--split", "eval"])
    assert args.gpus == ["0", "1"]
    assert args.workers == 2
    assert args.seed_start == utils.EVAL_SEED_OFFSET
    assert generator.parse_args(["--root", str(tmp_path / "ds")]).workers == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--gpus", "0,0"],
        ["--gpus", "a"],
        ["--workers", "0"],
        ["--split", "train", "--seed-start", str(utils.EVAL_SEED_OFFSET)],
    ],
)
def test_invalid_generation_args(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        generator.parse_args(["--root", str(tmp_path / "ds"), *extra])


@pytest.mark.parametrize("module_name", ["generate_stack_bowls_dataset", "generate_sim_dataset"])
def test_resume_requires_existing_logged_dataset(tmp_path: Path, module_name: str) -> None:
    parse = importlib.import_module(f"examples.nexarm.{module_name}").parse_args
    root = tmp_path / "ds"
    with pytest.raises(SystemExit):
        parse(["--root", str(root), "--resume"])
    root.mkdir()
    with pytest.raises(SystemExit):
        parse(["--root", str(root)])
    # A dataset without the per-attempt log cannot tell which seeds it already used.
    with pytest.raises(SystemExit):
        parse(["--root", str(root), "--resume"])
    utils.append_jsonl(root / utils.EPISODES_NAME, {"seed": 4, "accepted": False})
    utils.append_jsonl(root / utils.EPISODES_NAME, {"seed": 7, "accepted": True})
    # Appending to data encoded with another joint mapping would mix two angle conventions.
    with pytest.raises(SystemExit):
        parse(["--root", str(root), "--resume"])
    utils.write_report(root, {"joint_mapping": "per_joint_ctrlrange_v0"})
    with pytest.raises(SystemExit):
        parse(["--root", str(root), "--resume"])
    utils.write_report(root, {"joint_mapping": utils.JOINT_MAPPING_VERSION})

    args = parse(["--root", str(root), "--resume", "--gpus", "3"])
    assert (args.seed_start, args.workers) == (8, 1)
    # Sharded resumes plan seeds per shard block from the split start.
    args = parse(["--root", str(root), "--resume", "--workers", "2"])
    assert (args.seed_start, args.workers) == (0, 2)
    args = parse(["--root", str(root), "--resume", "--gpus", "0,1", "--split", "eval"])
    assert (args.seed_start, args.workers) == (utils.EVAL_SEED_OFFSET, 2)


def test_interrupted_sharded_resume_restores_the_original_dataset(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    backup = utils.resume_backup_path(root)
    backup.mkdir()
    (backup / "marker").write_text("original", encoding="utf-8")
    utils.recover_interrupted_resume(root)
    assert (root / "marker").read_text(encoding="utf-8") == "original"
    assert not backup.exists()
    # An intact root is never replaced by a stale backup.
    backup.mkdir()
    utils.recover_interrupted_resume(root)
    assert backup.exists()


def test_shard_seed_plan_resumes_inside_each_block() -> None:
    stride = utils.SHARD_SEED_STRIDE
    assert utils.plan_shard_seeds([], 0, [30, 30], "train") == [0, stride]
    # Seeds logged by an earlier single-process run and an earlier 3-shard run.
    logged = [0, 1, 2, 9, stride + 4, 2 * stride + 11]
    assert utils.plan_shard_seeds(logged, 0, [10, 10, 10, 10], "train") == [
        10,
        stride + 5,
        2 * stride + 12,
        3 * stride,
    ]
    offset = utils.EVAL_SEED_OFFSET
    assert utils.plan_shard_seeds([offset + 3], offset, [5, 5], "eval") == [offset + 4, offset + stride]
    with pytest.raises(ValueError, match="block ends"):
        utils.plan_shard_seeds([stride - 3], 0, [5], "train")
    with pytest.raises(ValueError, match="train seeds"):
        utils.plan_shard_seeds([], 0, [10] * 11, "train")


def test_launcher_flag_stripping() -> None:
    argv = ["--root", "out", "--gpus", "3", "--resume", "--workers=1", "--episodes", "2", "--no-dr"]
    # A single-GPU child runs the same command, resume included.
    assert utils._strip_flags(argv, utils._PARALLEL_FLAGS, ()) == [
        "--root",
        "out",
        "--resume",
        "--episodes",
        "2",
        "--no-dr",
    ]
    # Shards write fresh datasets with their own root, episodes and seeds.
    assert utils._strip_flags(argv) == ["--no-dr"]


def test_calibration_precedence_and_fps_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    SimCalibration(action_delay_steps=3, fps=30).save(path)
    root = tmp_path / "ds"

    args = generator.parse_args(["--root", str(root), "--calibration", str(path)])
    assert utils.delay_schedule(args) == (3, (2, 4))
    args = generator.parse_args(
        ["--root", str(root), "--calibration", str(path), "--action-delay-steps", "1"]
    )
    assert utils.delay_schedule(args) == (1, None)
    assert utils.delay_schedule(generator.parse_args(["--root", str(root)])) == (0, None)

    with pytest.raises(SystemExit):
        generator.parse_args(["--root", str(root), "--calibration", str(path), "--fps", "15"])
    assert not root.exists()


def test_combine_reports_and_episode_offsets() -> None:
    first = {
        "requested_episodes": 1,
        "attempts": 2,
        "accepted_seeds": [1],
        "rejected_attempts": [{"seed": 0}],
    }
    second = {"requested_episodes": 2, "attempts": 2, "accepted_seeds": [5, 6], "rejected_attempts": []}
    first["joint_mapping"] = second["joint_mapping"] = utils.JOINT_MAPPING_VERSION
    merged = utils.combine_reports([first, second])
    assert merged["joint_mapping"] == utils.JOINT_MAPPING_VERSION
    assert merged["attempts"] == 4
    assert merged["accepted_seeds"] == [1, 5, 6]
    assert merged["requested_episodes"] == 3
    assert merged["acceptance_rate"] == pytest.approx(0.75)


def test_generator_writes_follower_dataset_and_shared_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generator, "NexArmSimConfig", _camera_free_config)
    root = tmp_path / "dataset"
    base_args = [
        "--repo-id",
        "local/test_stack_bowls_gen",
        "--root",
        str(root),
        "--episodes",
        "1",
        "--max-attempts",
        "3",
        "--no-video",
        "--model",
        str(MODEL_PATH),
    ]
    assert generator.main(base_args) == 0

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["robot_type"] == "nexarm_follower"
    report = json.loads((root / utils.REPORT_NAME).read_text())
    assert report["domain_randomization"] is True
    assert len(report["accepted_seeds"]) == 1
    assert report["validation"]["episodes"] == 1
    assert report["provenance"]["model_sha256"]
    rows = utils.read_jsonl(root / utils.EPISODES_NAME)
    assert sum(row["accepted"] for row in rows) == 1
    assert all(row["dr_params"]["objects"] for row in rows)

    # Resuming appends episodes after the logged seeds and keeps one merged report.
    assert generator.main([*base_args, "--resume", "--no-dr"]) == 0
    report = json.loads((root / utils.REPORT_NAME).read_text())
    assert report["joint_mapping"] == utils.JOINT_MAPPING_VERSION
    assert report["validation"]["episodes"] == 2
    assert len(report["accepted_seeds"]) == 2
    assert len(set(report["accepted_seeds"])) == 2
    rows = utils.read_jsonl(root / utils.EPISODES_NAME)
    assert sorted(row["episode_index"] for row in rows if row["accepted"]) == [0, 1]


def test_combine_reports_refuses_mixed_joint_mappings() -> None:
    report = {"requested_episodes": 1, "attempts": 1, "accepted_seeds": [0], "rejected_attempts": []}
    with pytest.raises(ValueError, match="joint mappings"):
        utils.combine_reports([{**report, "joint_mapping": "a"}, {**report, "joint_mapping": "b"}])
