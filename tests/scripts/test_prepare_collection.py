# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""Shared NexArm camera flags and collection defaults, without hardware."""

import argparse
import sys
from types import SimpleNamespace

import pytest

from examples.nexarm import rollout
from lerobot_robot_nexarm import camera_config
from tools.nexarm import prepare_collection


def parse(*argv):
    parser = argparse.ArgumentParser()
    camera_config.add_camera_args(parser)
    return parser.parse_args(argv)


def test_collection_defaults_follow_port_and_root_conventions():
    args = prepare_collection.parse_args([])
    assert (args.leader_port, args.follower_port) == ("/dev/ttyUSB0", "/dev/ttyUSB1")
    assert args.root_repo_id == "thanhkt/nexarm_stack_bowls_top_20fps"
    assert (args.top_cam, args.no_top_cam) == ("auto", False)
    assert args.fps == 20


def test_camera_sources_accept_index_or_device_path():
    args = parse("--front-cam", "2", "--wrist-cam", "/dev/v4l/by-id/wrist", "--wrist-fourcc", "yuyv")
    cameras = camera_config.build_camera_dicts(args, top_serial="123")
    # Cameras capture at 30 fps whatever the control rate: the D435i colour stream has no 20 fps mode.
    assert cameras["front"] == {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
    assert cameras["wrist"]["index_or_path"].replace("\\", "/").endswith("/dev/v4l/by-id/wrist")
    assert cameras["wrist"]["fourcc"] == "YUYV"
    assert cameras["top"] == {
        "type": "intelrealsense",
        "serial_number_or_name": "123",
        "width": 640,
        "height": 480,
        "fps": 30,
        "use_depth": False,
    }
    assert list(camera_config.build_camera_dicts(parse("--no-top-cam"))) == ["front", "wrist"]
    with pytest.raises(SystemExit):
        parse("--front-fourcc", "MJPEG")


def test_missing_realsense_driver_names_the_fix(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)
    with pytest.raises(camera_config.CameraSetupError, match="--no-top-cam") as error:
        camera_config.build_camera_dicts(parse())
    assert "intelrealsense" in str(error.value)


def fake_realsense(monkeypatch, serials):
    devices = [SimpleNamespace(get_info=lambda _, serial=serial: serial) for serial in serials]
    module = SimpleNamespace(
        camera_info=SimpleNamespace(serial_number="serial"),
        context=lambda: SimpleNamespace(query_devices=lambda: devices),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", module)


def test_realsense_auto_detect_requires_exactly_one(monkeypatch):
    fake_realsense(monkeypatch, [])
    with pytest.raises(camera_config.CameraSetupError, match="--no-top-cam"):
        camera_config.find_realsense_serial()
    fake_realsense(monkeypatch, ["111", "222"])
    with pytest.raises(camera_config.CameraSetupError, match="--top-cam <serial>"):
        camera_config.find_realsense_serial()
    assert camera_config.find_realsense_serial("222") == "222"
    with pytest.raises(camera_config.CameraSetupError, match="not found"):
        camera_config.find_realsense_serial("333")
    fake_realsense(monkeypatch, ["111"])
    assert camera_config.find_realsense_serial() == "111"


def test_policy_camera_mismatch_names_the_flag():
    camera_config.check_policy_cameras(["front", "wrist", "top"], ["top", "front", "wrist"], "policy")
    with pytest.raises(SystemExit, match="add --no-top-cam"):
        camera_config.check_policy_cameras(["front", "wrist", "top"], ["front", "wrist"], "policy")
    with pytest.raises(SystemExit, match="remove --no-top-cam"):
        camera_config.check_policy_cameras(["front", "wrist"], ["front", "wrist", "top"], "policy")


@pytest.mark.parametrize("policy_type", ["act", "bspline"])
def test_rollout_reads_policy_image_inputs(monkeypatch, tmp_path, policy_type):
    import subprocess
    from pathlib import Path

    # Other test modules may set a headless MuJoCo backend that this host cannot load.
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    from lerobot_policy_bspline import BsplineConfig

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act import ACTConfig

    # bspline is a plugin type, registered only through register_third_party_plugins().
    config_class = {"act": ACTConfig, "bspline": BsplineConfig}[policy_type]
    config_class(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        device="cpu",
    ).save_pretrained(tmp_path)
    # A fresh interpreter, as when the script runs, so policy config types are not pre-registered.
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import sys, rollout; print(rollout.policy_camera_names(sys.argv[1]))",
            str(tmp_path),
        ],
        cwd=Path(rollout.__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "['front', 'top']"
    args = rollout.parse_args(["--policy-path", "owner/policy"])
    assert args.follower_port == "/dev/ttyUSB1"
    assert camera_config.camera_names(args) == ["front", "wrist", "top"]


@pytest.mark.parametrize("strategy", ["base", "dagger"])
def test_rollout_command_is_a_valid_lerobot_rollout_config(monkeypatch, tmp_path, strategy):
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  # registers the "opencv" camera type
    from lerobot.configs import parser
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act import ACTConfig
    from lerobot.rollout.configs import RolloutConfig
    from lerobot.utils import register_third_party_plugins

    register_third_party_plugins()
    ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        device="cpu",
    ).save_pretrained(tmp_path)
    dataset_args = ["--repo-id", "local/nexarm_dagger"] if strategy == "dagger" else []
    args = rollout.parse_args(
        ["--policy-path", str(tmp_path), "--no-top-cam", "--leader-port", "COM18", "--follower-port", "COM19"]
        + ["--strategy", strategy, "--task", "Pick up the cube", *dataset_args]
    )

    @parser.wrap()
    def load(cfg: RolloutConfig) -> RolloutConfig:
        return cfg

    # The command runs `python -m <module> <flags>`; LeRobot's parser reads the flags from sys.argv.
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", *rollout.rollout_command(args)[3:]])
    cfg = load()
    assert (cfg.robot.type, cfg.robot.port) == ("nexarm_follower", "COM19")
    assert cfg.fps == 20
    assert {camera.fps for camera in cfg.robot.cameras.values()} == {30}
    if strategy == "dagger":
        # DAgger hands control to the leader arm for corrections; LeRobot refuses it without one.
        assert (cfg.teleop.type, cfg.teleop.port) == ("nexarm_leader", "COM18")
        assert cfg.dataset.single_task == "Pick up the cube"
    else:
        assert cfg.teleop is None
