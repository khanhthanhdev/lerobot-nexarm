"""The NexArm collection wrapper keeps lerobot-record's collection flags outside upstream code."""

import threading
import time

import pytest

from examples.nexarm import record_collection as wrapper
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset
from lerobot.scripts import lerobot_record
from lerobot_robot_nexarm import collection_merge
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


class _StoppedListener:
    def stop(self):
        pass


def _press_enter_when_enabled(events, presses):
    """Simulate the operator pressing Enter: each press ends the current wait or episode once controls are on."""

    def run():
        for _ in range(presses):
            time.sleep(0.3)
            while not events.get("controls_enabled"):
                time.sleep(0.01)
            events["exit_early"] = True

    threading.Thread(target=run, daemon=True).start()


@pytest.mark.parametrize("merge_into_root", [False, True])
def test_manual_control_records_and_merges(monkeypatch, tmp_path, merge_into_root):
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "quiet": True,
        "controls_enabled": False,
    }

    def fake_listener(**kwargs):
        assert kwargs == {"manual_control": True}
        # One press starts the episode, the second ends it.
        _press_enter_when_enabled(events, 2)
        return _StoppedListener(), events

    monkeypatch.setattr(wrapper, "init_keyboard_listener", fake_listener)
    monkeypatch.setattr(collection_merge, "hub_root_info", lambda _: None)
    merge_flags = (
        {"root_repo_id": "test/root", "merge_root": str(tmp_path / "root")} if merge_into_root else {}
    )
    recording_log = tmp_path / "log" / "recording.log"

    dataset = wrapper.record_collection(
        make_config(tmp_path, manual_control=True, recording_log=str(recording_log), **merge_flags)
    )

    assert dataset.num_episodes == 1
    assert recording_log.exists()
    if merge_into_root:
        assert LeRobotDataset("test/root", root=tmp_path / "root").num_episodes == 1
