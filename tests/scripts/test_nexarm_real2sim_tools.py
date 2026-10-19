import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from examples.nexarm import (
    audit_real_dataset as audit_module,
    probe_joint_mapping as probe,
    servo_step_test as step_test,
)
from examples.nexarm.calibrate_sim import Episode, parse_episode_list
from examples.nexarm.overlay_recorded_frames import pose_model, side_by_side
from examples.nexarm.realsense_fovy import fovy_deg, principal_point_offset_px
from lerobot.motors.nexarm.mujoco_mapping import MUJOCO_JOINTS
from lerobot.motors.nexarm.nexarm import JOINT_NAMES

SCENE = Path(__file__).resolve().parents[2] / "sim/description/mjcf/scene.xml"


class RateLimitedBus:
    """Fake follower: each read moves every joint toward its target at ``rate`` ticks per read."""

    def __init__(self, positions, rate=20.0):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.target = self.positions.copy()
        self.rate = rate
        self.motion_params = []

    def connect(self):
        pass

    def disconnect(self):
        pass

    def enter_lerobot_mode(self):
        pass

    def exit_lerobot_mode(self):
        pass

    def set_torque(self, enable):
        self.torque = enable

    def write_motion_params(self, acc, speed=0):
        self.motion_params.append((acc, speed))

    def write_positions(self, positions):
        self.target = np.asarray(positions, dtype=np.float64)

    def read_positions(self, retries=3):
        self.positions += np.clip(self.target - self.positions, -self.rate, self.rate)
        return self.positions.round().astype(int).tolist()


def test_analyze_step_recovers_latency_and_speed():
    times = np.arange(0, 1.0, 0.002)
    positions = 1000 + np.clip((times - 0.05) * 2000, 0, 800)  # 50 ms dead time, then 2000 ticks/s
    metrics = step_test.analyze_step(times, positions, 1000, 1800, speed_window_s=0.1)
    assert metrics["latency_s"] == pytest.approx(0.058, abs=0.003)  # dead time + 2% of 800 ticks at 2000/s
    assert metrics["mean_speed_ticks_s"] == pytest.approx(2000, rel=0.02)
    assert metrics["peak_speed_ticks_s"] == pytest.approx(2000, rel=0.02)
    assert metrics["final_error_ticks"] == 0


def test_analyze_step_reports_unreached_targets_as_none():
    times = np.linspace(0, 1, 50)
    metrics = step_test.analyze_step(times, np.full(50, 1000.0), 1000, 1800, speed_window_s=0.1)
    assert (
        metrics["latency_s"] is None and metrics["rise_s"] is None and metrics["mean_speed_ticks_s"] is None
    )


def test_run_step_holds_other_joints_and_uses_its_clock():
    bus = RateLimitedBus([2048] * 6)
    ticks = iter(np.arange(0, 10, 0.01))
    times, positions = step_test.run_step(bus, [2048] * 6, 2, 2448, 0.5, clock=lambda: next(ticks))
    assert bus.target.tolist() == [2048, 2048, 2448, 2048, 2048, 2048]
    assert positions[-1] == 2448 and times[0] == pytest.approx(0.01)


def test_step_targets_refuse_out_of_range_moves():
    assert step_test.step_targets("elbow_flex", 2000, 500) == (2000, 2500)
    with pytest.raises(ValueError, match="raw range"):
        step_test.step_targets("gripper", 2700, 500)


def test_servo_step_test_main_restores_motion_params(tmp_path):
    bus = RateLimitedBus([2048] * 6, rate=200.0)
    out = tmp_path / "step.json"
    with patch.object(step_test, "NexArmMotorsBus", return_value=bus):
        argv = ["--joint", "wrist_flex", "--delta", "300", "--motion-speed", "2000", "0", "--repeats", "1"]
        assert step_test.main([*argv, "--timeout", "0.05", "--yes", "--out", str(out)]) == 0
    assert bus.motion_params == [(100, 2000), (100, 0), (100, 2000)]  # last call restores the defaults
    payload = json.loads(out.read_text())
    assert set(payload["summary"]) == {"0", "2000"} and len(payload["steps"]) == 4


@pytest.fixture(scope="module")
def sim_arm():
    return probe.SimArm(SCENE)


def test_upright_offsets_stand_the_arm_straight_up(sim_arm):
    sim_arm.forward(sim_arm.upright_offsets())
    points = [sim_arm.anchor(i) for i in (1, 2, 3)] + [sim_arm.jaw()]
    for lower, upper in zip(points, points[1:], strict=False):
        link = upper - lower
        # Links may sit sideways along the flex axis (x), but must not lean front or back.
        assert link[2] > 0 and abs(link[1]) < 1e-4


def _probe_readings(sim_arm, signs, upright_raw=None):
    """Readings a user would produce if the true mapping had these per-joint signs (qpos = sign * raw)."""
    sim_arm.forward(sim_arm.upright_offsets())
    up = np.array([0.0, 0.0, 1.0])
    home = list(upright_raw) if upright_raw is not None else [2048.0] * 6
    moved, directions = {}, {}
    for index, name in enumerate(probe.ARM_JOINTS):
        raw = list(home)
        axis = sim_arm.axis(index)
        if name in probe.FLEX_JOINTS:
            raw[index] += 1024  # tilt either way: raw goes up, the link goes wherever the true sign sends it
            sim_direction = probe.direction_of(np.cross(axis, up))
            directions[name] = sim_direction if signs[name] > 0 else probe.rotate_direction(sim_direction, 2)
        else:
            raw[index] += 1024 * signs[name] * np.sign(axis @ up)  # the prompt asks for counter-clockwise
        moved[name] = raw
    return home, moved, directions


def test_solve_mapping_recovers_signs(sim_arm):
    signs = {"shoulder_pan": 1, "shoulder_lift": -1, "elbow_flex": 1, "wrist_flex": -1, "wrist_roll": 1}
    home, moved, directions = _probe_readings(sim_arm, signs)
    sim_arm.forward(sim_arm.upright_offsets())
    camera_side = probe.direction_of(sim_arm.data.cam_xpos[sim_arm.camera_id] - sim_arm.jaw())
    result = probe.solve_mapping(sim_arm, home, moved, directions, camera_side)
    for name in probe.ARM_JOINTS:
        assert result["joints"][name]["sign"] == signs[name], name
    assert result["joints"]["wrist_roll"]["offset_rad"] == pytest.approx(result["sim_upright_qpos"][4])


def test_solve_mapping_uses_the_measured_upright_reading(sim_arm):
    # Vendor Home (raw 2048) bends the arm, so posing it straight up reads elsewhere, e.g. elbow at 3072.
    signs = {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": -1, "wrist_flex": 1, "wrist_roll": 1}
    upright_raw = [2048.0, 2048.0, 3072.0, 1900.0, 2048.0, 2048.0]
    upright, moved, directions = _probe_readings(sim_arm, signs, upright_raw)
    sim_arm.forward(sim_arm.upright_offsets())
    camera_side = probe.direction_of(sim_arm.data.cam_xpos[sim_arm.camera_id] - sim_arm.jaw())
    result = probe.solve_mapping(sim_arm, upright, moved, directions, camera_side)
    # The mapping must put the upright reading on the sim's upright pose, not raw 2048.
    np.testing.assert_allclose(
        probe.mapped_qpos(upright_raw, result["joints"]), result["sim_upright_qpos"], atol=1e-9
    )
    elbow = result["joints"]["elbow_flex"]
    assert elbow["offset_rad"] == pytest.approx(result["sim_upright_qpos"][2] + np.pi / 2)


def test_solve_mapping_turns_the_roll_zero_to_the_reported_camera_side(sim_arm):
    signs = dict.fromkeys(probe.ARM_JOINTS, 1)
    home, moved, directions = _probe_readings(sim_arm, signs)
    base = probe.solve_mapping(sim_arm, home, moved, directions, "front")
    turned = probe.solve_mapping(sim_arm, home, moved, directions, "left")
    delta = turned["joints"]["wrist_roll"]["offset_rad"] - base["joints"]["wrist_roll"]["offset_rad"]
    assert np.cos(delta) == pytest.approx(0, abs=1e-9)  # a quarter turn, either way round


def test_solve_mapping_rejects_sideways_tilts_and_tiny_moves(sim_arm):
    signs = dict.fromkeys(probe.ARM_JOINTS, 1)
    home, moved, directions = _probe_readings(sim_arm, signs)
    with pytest.raises(ValueError, match="quarter turn"):
        probe.solve_mapping(sim_arm, home, moved, {**directions, "shoulder_lift": "left"}, "front")
    moved["elbow_flex"] = list(home)
    with pytest.raises(ValueError, match="redo"):
        probe.solve_mapping(sim_arm, home, moved, directions, "front")


def test_run_probe_reads_each_pose():
    bus = RateLimitedBus([2048] * 6, rate=0)
    answers = iter(["", "", "f"] + ["", ""] + ["", "f", ""] * 3 + ["", ""])
    upright, moved, directions, camera_side = probe.run_probe(bus, ask=lambda _: next(answers))
    assert upright == [2048.0] * 6 and camera_side == "front"
    assert bus.torque is False
    assert set(moved) == set(probe.ARM_JOINTS) and set(directions) == set(probe.FLEX_JOINTS)


def test_mapped_qpos_applies_sign_and_offset():
    raw = [3072, 2048, 1024, 2048, 2048, 2833]
    np.testing.assert_allclose(probe.mapped_qpos(raw, None), [np.pi / 2, 0, -np.pi / 2, 0, 0])
    mapping = {name: {"sign": -1, "offset_rad": 0.5} for name in probe.ARM_JOINTS}
    np.testing.assert_allclose(
        probe.mapped_qpos(raw, mapping), [0.5 - np.pi / 2, 0.5, 0.5 + np.pi / 2, 0.5, 0.5]
    )


def test_pose_model_and_side_by_side():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    pose_model(model, data, [3072, 2048, 2048, 2048, 2048, 1195], None)
    pan = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, MUJOCO_JOINTS["shoulder_pan"])
    assert data.qpos[model.jnt_qposadr[pan]] == pytest.approx(np.pi / 2)
    real, sim = np.full((4, 5, 3), 200, np.uint8), np.full((4, 5, 3), 100, np.uint8)
    frame = side_by_side(real, sim)
    assert frame.shape == (4, 15, 3) and frame[0, 10, 0] == 150


def _episode(index, frames=450, lag=2, gripper_cap=None):
    t = np.arange(frames)
    actions = np.full((frames, 6), 2048.0)
    actions[:, 2] = 2048 + 600 * np.sin(t / 8)  # fast elbow, up to 2250 ticks/s at 30 fps
    actions[:, 5] = np.where((t // 60) % 2 == 0, 2833.0, 1990.0)  # close, then a partial release
    states = np.empty_like(actions)
    states[lag:], states[:lag] = actions[: frames - lag], actions[0]
    if gripper_cap is not None:  # the gripper cannot follow faster than gripper_cap ticks per frame
        for k in range(1, frames):
            states[k, 5] = states[k - 1, 5] + np.clip(
                actions[k - lag, 5] - states[k - 1, 5], -gripper_cap, gripper_cap
            )
        states[:, 5] = np.minimum(states[:, 5], 2785)  # an empty gripper stops short of the command
    return Episode(states=states, actions=actions, episode_index=index)


def test_audit_finds_lag_saturation_timing_and_partial_releases():
    episodes = [_episode(i, gripper_cap=120) for i in range(4)]
    episodes.append(_episode(4, frames=300, lag=0, gripper_cap=120))  # short and differently logged
    report = audit_module.audit(episodes, fps=30, episode_time_s=15, speed_window_s=0.2)
    assert report["timing_trusted_episodes"] == [0, 1, 2, 3]
    assert report["lag_frames_median"] == 2 and report["lag_anomaly_episodes"] == [4]
    speeds = report["speeds_ticks_s"]
    assert not speeds["elbow_flex"]["saturated"] and speeds["elbow_flex"]["speed_cap_ticks_s"] is None
    assert speeds["gripper"]["saturated"] and speeds["gripper"]["speed_cap_ticks_s"] <= 120 * 30
    gripper = report["gripper"]
    assert gripper["closed_settled_p95"] == 2785
    assert gripper["releases"] > 0 and gripper["release_open_command_median"] == 1990
    assert gripper["releases_above_assist_threshold"] == gripper["releases"]


def test_audit_without_episode_time_uses_every_episode():
    report = audit_module.audit([_episode(0, frames=200), _episode(1)], 30, None, 0.2)
    assert report["timing_trusted_episodes"] == [0, 1]


def test_parse_episode_list():
    assert parse_episode_list("3, 10-12,") == {3, 10, 11, 12}
    assert parse_episode_list("") == set()
    with pytest.raises(ValueError):
        parse_episode_list("a-b")


def test_fovy_from_intrinsics():
    assert fovy_deg(480, 615.0) == pytest.approx(42.6, abs=0.05)
    assert principal_point_offset_px(640, 480, 330.0, 236.0) == (10.0, -4.0)
    with pytest.raises(ValueError):
        fovy_deg(480, 0)


def test_joint_names_match_tools():
    assert JOINT_NAMES[:-1] == probe.ARM_JOINTS
