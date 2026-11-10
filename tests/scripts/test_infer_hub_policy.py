# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Commands that examples/nexarm/infer_hub_policy.py runs for the Hub presets, without network or hardware."""

import sys
from pathlib import Path

import pytest

from examples.nexarm import infer_hub_policy


def build(*argv):
    args, extra = infer_hub_policy.parse_args(list(argv))
    preset = infer_hub_policy.resolve(args)
    return infer_hub_policy.command(args, preset, Path("policy"), extra)


def test_sim_eval_uses_the_single_order_the_model_was_trained_on():
    cmd = build("planbind", "--policy.planbind_latency=6")
    assert cmd[1:3] == ["-m", "lerobot_env_nexarm.stage_eval"]
    assert "--env.stack_order=red,blue,black" in cmd
    # Unknown flags reach lerobot-nexarm-eval unchanged.
    assert cmd[-1] == "--policy.planbind_latency=6"
    # TurboVLA is prompt-conditioned with no fixed order, so the env keeps cycling orders and prompts.
    assert not any(arg.startswith("--env.stack_order") for arg in build("turbovla"))


def test_planbind_preset_points_inside_the_multi_run_repo():
    args, _ = infer_hub_policy.parse_args(
        ["planbind", "--subfolder", "/train_onestep/checkpoints/060000/pretrained_model/"]
    )
    preset = infer_hub_policy.resolve(args)
    assert (preset.repo_id, preset.subfolder) == (
        "thanhkt/plan-bind-d1",
        "train_onestep/checkpoints/060000/pretrained_model",
    )
    assert infer_hub_policy.resolve(infer_hub_policy.parse_args(["planbind"])[0]).subfolder.startswith(
        "train_planbind/"
    )


def test_real_target_runs_rollout_with_the_downloaded_policy():
    cmd = build("bspline", "--target", "real", "--follower-port", "COM19", "--no-top-cam")
    assert Path(cmd[1]).name == "rollout.py"
    assert cmd[2:] == ["--policy-path", "policy", "--follower-port", "COM19", "--no-top-cam"]


def test_turbovla_on_the_real_arm_requires_its_prompt(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["infer_hub_policy.py", "turbovla", "--target", "real"])
    # Fails before any download, so a missing prompt costs nothing.
    with pytest.raises(SystemExit, match="--task"):
        infer_hub_policy.main()
