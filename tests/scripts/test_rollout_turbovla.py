# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import json

import numpy as np
import pytest

from examples.nexarm import rollout_turbovla as rollout
from lerobot.motors.nexarm.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS


def write_config(directory, payload):
    (directory / "config.json").write_text(json.dumps(payload))
    return directory / "steps_10_ema_pytorch_model.pt"


def test_legacy_checkpoint_falls_back_to_view_count_and_224(tmp_path):
    checkpoint = write_config(tmp_path, {"action": {"horizon": 16}, "vision": {"num_views": 3}})
    config = rollout.load_checkpoint_config(checkpoint)
    assert config.camera_names == ["front", "wrist", "top"]
    assert (config.image_size, config.fps, config.task, config.task_type) == (224, None, None, None)
    missing = rollout.load_checkpoint_config(tmp_path / "missing" / "model.pt")
    assert missing == rollout.TurboVLACheckpointConfig()


def test_saved_contract_and_inconsistent_camera_keys(tmp_path):
    checkpoint = write_config(
        tmp_path,
        {
            "vision": {"num_views": 2, "image_size": 192},
            "camera_keys": ["observation.images.wrist", "observation.images.front"],
            "task_type": "pick_place",
            "fps": 15,
            "task": "Pick up the red cube, place it in the green target zone, and release it.",
        },
    )
    config = rollout.load_checkpoint_config(checkpoint)
    assert config.camera_names == ["wrist", "front"]
    assert (config.image_size, config.fps, config.task_type) == (192, 15, "pick_place")

    write_config(tmp_path, {"vision": {"num_views": 3}, "camera_keys": ["observation.images.front"]})
    with pytest.raises(ValueError, match="do not match num_views"):
        rollout.load_checkpoint_config(checkpoint)


def test_task_type_and_prompt_precedence():
    saved = rollout.TurboVLACheckpointConfig(task_type="pick_place", task="saved prompt")
    legacy_three_view = rollout.TurboVLACheckpointConfig(num_views=3)
    legacy = rollout.TurboVLACheckpointConfig()

    assert rollout.resolve_task_type("stack_bowls", saved, []) == "stack_bowls"
    assert rollout.resolve_task_type("auto", saved, ["outputs/bowl"]) == "pick_place"
    assert rollout.resolve_task_type("auto", legacy_three_view, []) == "stack_bowls"
    assert rollout.resolve_task_type("auto", legacy, ["outputs/train/nexarm_turbovla_cotrain"]) == (
        "pick_place"
    )

    assert rollout.resolve_prompt("explicit", saved, "pick_place") == "explicit"
    assert rollout.resolve_prompt(None, saved, "pick_place") == "saved prompt"
    assert rollout.resolve_prompt(None, legacy, "pick_place") == rollout.PICK_PLACE_TASK
    assert rollout.resolve_prompt(None, legacy, "stack_bowls") == rollout.REAL_STACK_BOWLS_TASK


def test_default_prompts_match_training_data():
    from lerobot.robots.nexarm_sim.stack_bowls_task import get_task_instruction

    assert get_task_instruction("red", "blue", "black") == rollout.REAL_STACK_BOWLS_TASK


def test_gripper_direction_follows_shared_constants():
    threshold = rollout.DEFAULT_GRIPPER_THRESHOLD
    assert rollout.gripper_is_closed(GRIPPER_CLOSED_POS, threshold)
    assert not rollout.gripper_is_closed(GRIPPER_OPEN_POS, threshold)


def make_runner(camera_names, stats=None):
    runner = object.__new__(rollout.TurboVLAPolicyRunner)
    runner.camera_names = camera_names
    runner.num_views = len(camera_names)
    runner.stats = stats
    runner.binarize_gripper = True
    runner.gripper_threshold = rollout.DEFAULT_GRIPPER_THRESHOLD
    runner.debug_actions = False
    return runner


def test_views_follow_trained_order_and_missing_views_fail():
    runner = make_runner(["top", "front"])
    frames = {"front": np.zeros(1), "wrist": np.ones(1), "top": np.full(1, 2)}
    assert [int(view[0]) for view in runner.ordered_views(frames)] == [2, 0]
    with pytest.raises(ValueError, match="missing camera"):
        runner.ordered_views({"front": np.zeros(1)})
    with pytest.raises(ValueError, match="Expected 2 camera views"):
        runner.ordered_views([np.zeros(1)])


def test_gripper_snaps_to_dataset_closed_and_open_extremes():
    stats = {"action": {"min": [0.0] * 5 + [GRIPPER_OPEN_POS], "max": [1.0] * 5 + [GRIPPER_CLOSED_POS]}}
    runner = make_runner(["front"], stats)
    # Normalized -1 is the open extreme and +1 the closed extreme.
    action = np.array([[0.0] * 5 + [-0.9], [0.0] * 5 + [0.9]], dtype=np.float32)
    assert runner.unnormalize_action(action)[:, 5].tolist() == [GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS]


def test_cli_defaults_come_from_checkpoint():
    args = rollout.parse_args(["--checkpoint", "model.pt"])
    assert args.task is None and args.fps is None
    assert (args.front_cam, args.wrist_cam, args.top_cam, args.no_top_cam) == (0, 1, "auto", False)
    assert args.gripper_threshold == rollout.DEFAULT_GRIPPER_THRESHOLD


def test_diagnose_uses_saved_camera_order():
    from examples.nexarm.diagnose_turbovla import diagnostic_cameras

    features = {
        "observation.images.front": {"dtype": "video"},
        "observation.images.wrist": {"dtype": "video"},
        "observation.images.top": {"dtype": "video"},
    }
    saved = ("observation.images.wrist", "observation.images.front")
    assert diagnostic_cameras(features, saved, None) == list(saved)
    assert diagnostic_cameras(features, None, None) == [
        "observation.images.front",
        "observation.images.wrist",
        "observation.images.top",
    ]
    with pytest.raises(ValueError, match="differs"):
        diagnostic_cameras(features, saved, "front,wrist")
    with pytest.raises(ValueError, match="missing checkpoint camera"):
        diagnostic_cameras({"observation.images.front": {"dtype": "video"}}, saved, None)
