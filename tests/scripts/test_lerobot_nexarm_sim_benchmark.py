# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import argparse
import csv
import json

import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_nexarm_sim_benchmark import (
    EpisodeMetrics,
    _write_results,
    parse_args,
    parse_policy_spec,
    summarize_episodes,
    validate_policy_contract,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_parse_policy_spec_accepts_label_and_bare_path() -> None:
    labeled = parse_policy_spec("smolvla=org/nexarm-smolvla")
    bare = parse_policy_spec("outputs/train/act/pretrained_model")

    assert labeled.label == "smolvla"
    assert labeled.path == "org/nexarm-smolvla"
    assert bare.label == "pretrained_model"
    assert bare.path == "outputs/train/act/pretrained_model"


@pytest.mark.parametrize("value", ["=checkpoint", "act=", "="])
def test_parse_policy_spec_rejects_empty_parts(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_policy_spec(value)


def test_parse_args_rejects_duplicate_labels() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--policy", "act=one", "--policy", "act=two"])


def test_validate_policy_contract_requires_six_actions_and_matching_cameras() -> None:
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        device="cpu",
    )

    validate_policy_contract(
        config,
        camera_names=["front", "wrist"],
        camera_height=48,
        camera_width=64,
    )
    with pytest.raises(ValueError, match="benchmark cameras"):
        validate_policy_contract(
            config,
            camera_names=["front"],
            camera_height=48,
            camera_width=64,
        )


def test_episode_summary_and_artifacts(tmp_path) -> None:
    episodes = [
        EpisodeMetrics(
            seed=10,
            success=True,
            reason="success",
            steps=30,
            simulation_time_s=1.0,
            wall_time_s=2.0,
            control_hz=15.0,
            inference_mean_ms=10.0,
            inference_p50_ms=9.0,
            inference_p95_ms=15.0,
        ),
        EpisodeMetrics(
            seed=11,
            success=False,
            reason="timeout",
            steps=60,
            simulation_time_s=2.0,
            wall_time_s=3.0,
            control_hz=20.0,
            inference_mean_ms=12.0,
            inference_p50_ms=11.0,
            inference_p95_ms=18.0,
        ),
    ]
    summary = summarize_episodes(episodes)

    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["mean_control_hz"] == pytest.approx(17.5)
    assert summary["termination_reasons"] == {"success": 1, "timeout": 1}

    payload = {
        "environment": {},
        "config": {},
        "results": [
            {
                "label": "act",
                "path": "checkpoint",
                "policy_type": "act",
                "status": "completed",
                **summary,
            }
        ],
    }
    json_path, csv_path = _write_results(tmp_path, payload)

    assert json.loads(json_path.read_text())["results"][0]["success_rate"] == pytest.approx(0.5)
    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["label"] == "act"
    assert json.loads(rows[0]["termination_reasons"]) == {"success": 1, "timeout": 1}
