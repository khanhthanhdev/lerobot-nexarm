"""Verify manual episode boundaries without moving hardware."""

from types import SimpleNamespace

import pytest

from lerobot.utils import keyboard_input as ki
from lerobot.utils.recording_session import run_manual_session


def fake_dataset():
    dataset = SimpleNamespace(num_episodes=0, root="test-dataset", episode_buffer={"size": 0})

    def save():
        dataset.num_episodes += 1
        dataset.episode_buffer["size"] = 0

    dataset.has_pending_frames = lambda: dataset.episode_buffer["size"] > 0
    dataset.save_episode = save
    dataset.clear_episode_buffer = lambda: dataset.episode_buffer.update(size=0)
    return dataset


def test_manual_wait_save_before_reset_and_no_reset_frames(monkeypatch):
    dataset = fake_dataset()
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    calls = []
    screens = []

    def initial_wait(_):
        assert not calls
        assert dataset.episode_buffer["size"] == 0
        events["exit_early"] = True

    monkeypatch.setattr("lerobot.utils.recording_session.time.sleep", initial_wait)

    def run_phase(target):
        calls.append(target)
        if target is None:
            assert dataset.num_episodes == 1  # saved before entering reset
            assert dataset.episode_buffer["size"] == 0
        else:
            target.episode_buffer["size"] = 10
        events["exit_early"] = True

    run_manual_session(
        dataset, events, 2, run_phase, show=lambda count, status, _: screens.append((count, status))
    )
    assert calls == [dataset, None, dataset]
    assert dataset.num_episodes == 2
    assert screens[0][1].startswith("READY")
    assert (1, "RESET THE ENVIRONMENT — Enter / → when ready") in screens


@pytest.mark.parametrize("discard", [False, True])
def test_quit_during_recording_saves_unless_discarded(monkeypatch, discard):
    dataset = fake_dataset()
    events = {"exit_early": True, "rerecord_episode": False, "stop_recording": False}
    monkeypatch.setattr(
        "lerobot.utils.recording_session.time.sleep", lambda _: events.update(exit_early=True)
    )

    def run_phase(target):
        assert target is dataset
        dataset.episode_buffer["size"] = 10
        events.update(stop_recording=True, rerecord_episode=discard)

    run_manual_session(dataset, events, 2, run_phase, show=lambda *_: None)
    assert dataset.num_episodes == (0 if discard else 1)


def test_quit_when_ready_never_records(monkeypatch):
    dataset = fake_dataset()
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    monkeypatch.setattr(
        "lerobot.utils.recording_session.time.sleep", lambda _: events.update(stop_recording=True)
    )
    run_manual_session(
        dataset, events, 2, lambda _: pytest.fail("unexpected recording"), show=lambda *_: None
    )
    assert dataset.num_episodes == 0


def test_discard_then_reset_then_record(monkeypatch):
    dataset = fake_dataset()
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    monkeypatch.setattr(
        "lerobot.utils.recording_session.time.sleep", lambda _: events.update(exit_early=True)
    )
    calls = []

    def run_phase(target):
        calls.append(target)
        if len(calls) == 1:
            dataset.episode_buffer["size"] = 10
            events["rerecord_episode"] = True
        elif target is None:
            assert dataset.num_episodes == 0
            assert dataset.episode_buffer["size"] == 0
        else:
            assert not events["rerecord_episode"]
            dataset.episode_buffer["size"] = 10

    run_manual_session(dataset, events, 1, run_phase, show=lambda *_: None)
    assert calls == [dataset, None, dataset]
    assert dataset.num_episodes == 1


@pytest.mark.parametrize("key", ["enter", "right", "n"])
def test_manual_keys_are_quiet_and_ignored_during_save(monkeypatch, capsys, key):
    monkeypatch.setattr(ki.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(ki.TerminalKeyListener, "start", lambda _: None)
    listener, events = ki.init_keyboard_listener(manual_control=True)
    listener._on_key(key)
    assert not events["exit_early"]
    events["controls_enabled"] = True
    listener._on_key(key)
    assert events["exit_early"]
    assert capsys.readouterr().out == ""
    events["exit_early"] = False
    listener._on_key(key)  # key repeat doesn't immediately advance another phase
    assert not events["exit_early"]
    events["controls_enabled"] = False
    listener._on_key("esc")
    assert events["stop_recording"]


def test_manual_session_writes_real_dataset(monkeypatch, tmp_path):
    import numpy as np

    from lerobot.datasets import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint"]},
    }
    dataset = LeRobotDataset.create(
        "local/manual_test", 30, root=tmp_path / "dataset", features=features, use_videos=False
    )
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    monkeypatch.setattr(
        "lerobot.utils.recording_session.time.sleep", lambda _: events.update(exit_early=True)
    )
    phases = []

    def run_phase(target):
        phases.append(target)
        if target is not None:
            for _ in range(3):
                target.add_frame(
                    {
                        "action": np.array([1], dtype=np.float32),
                        "observation.state": np.array([2], dtype=np.float32),
                        "task": "test collection",
                    }
                )
        else:
            assert dataset.num_episodes == 1
            assert not dataset.has_pending_frames()

    try:
        run_manual_session(dataset, events, 2, run_phase, show=lambda *_: None)
        assert dataset.num_episodes == 2
        assert dataset.num_frames == 6
        assert phases == [dataset, None, dataset]
    finally:
        dataset.finalize()
