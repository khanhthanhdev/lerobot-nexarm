"""The NexArm collection wrapper keeps lerobot-record's collection flags outside upstream code."""

import pytest

from examples.nexarm import record_collection as wrapper
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.scripts import lerobot_record
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


def make_config(tmp_path, **overrides):
    dataset = DatasetRecordConfig(
        repo_id="test/session",
        single_task="Dummy task",
        root=tmp_path / "session",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    return wrapper.CollectionRecordConfig(
        robot=MockRobotConfig(),
        dataset=dataset,
        teleop=MockTeleopConfig(),
        play_sounds=False,
        **overrides,
    )


def test_rerun_save_path_requires_display_data(tmp_path):
    with pytest.raises(ValueError, match="--rerun_save_path requires --display_data=true"):
        make_config(tmp_path, rerun_save_path=str(tmp_path / "session.rrd"))


def test_rerun_save_path_reaches_the_visualization(monkeypatch, tmp_path):
    calls = []

    def fake_init_visualization(display_mode, **kwargs):
        calls.append((display_mode, kwargs))

    monkeypatch.setattr(lerobot_record, "init_visualization", fake_init_visualization)
    monkeypatch.setattr(lerobot_record, "shutdown_visualization", lambda *_: None)
    monkeypatch.setattr(lerobot_record, "log_visualization_data", lambda *_, **__: None)
    save_path = str(tmp_path / "session.rrd")

    dataset = wrapper.record_collection(make_config(tmp_path, display_data=True, rerun_save_path=save_path))

    assert dataset.num_episodes == 1
    assert calls == [
        ("rerun", {"session_name": "recording", "ip": None, "port": None, "rerun_save_path": save_path})
    ]
    # The wrapper only patches lerobot-record while it runs.
    assert lerobot_record.init_visualization is fake_init_visualization


def test_without_collection_flags_the_wrapper_is_plain_lerobot_record(tmp_path):
    dataset = wrapper.record_collection(make_config(tmp_path))

    assert dataset.num_episodes == 1
    assert dataset.root == tmp_path / "session"
