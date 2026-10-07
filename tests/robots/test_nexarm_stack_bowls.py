# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from lerobot.robots.nexarm_sim.mujoco_backend import NexArmMujocoBackend
from lerobot.robots.nexarm_sim.stack_bowls_task import COLORS, NexArmStackBowlsTask

MODEL = Path(__file__).resolve().parents[2] / "sim/fusion_export/bowl_stack_scene.xml"


@pytest.fixture
def task():
    backend = NexArmMujocoBackend(
        model_path=MODEL,
        fps=30,
        camera_width=64,
        camera_height=64,
        camera_names=(),
    )
    task = NexArmStackBowlsTask(backend, position_jitter_m=0.04)
    task.reset(seed=3, settle_steps=0)
    yield task
    backend.close()


def test_layout_reproducible_nonoverlapping(task):
    for seed in range(20):
        task.reset(seed=seed, settle_steps=0)
        positions = np.array([task.bowl_position(c) for c in COLORS])
        for i in range(3):
            for j in range(i):
                assert np.linalg.norm(positions[i, :2] - positions[j, :2]) >= 0.14
        task.reset(seed=seed, settle_steps=0)
        np.testing.assert_array_equal(positions, [task.bowl_position(c) for c in COLORS])


def test_contact_assistance_requires_both_jaws_on_same_bowl(task):
    backend = task.backend
    site_pos = backend.data.site_xpos[task._gripper_frame_site_id].copy()
    qadr = backend.model.jnt_qposadr[task._joint_ids["red"]]
    backend.data.qpos[qadr : qadr + 3] = site_pos + [0, 0, 0.09]
    mujoco.mj_forward(backend.model, backend.data)
    backend.data.ctrl[backend._actuator_ids["gripper"]] = backend.raw_to_control("gripper", 2833)
    task.target_bowl = "red"
    backend.data.ncon = 0
    task._on_physics_step()
    assert task.held_bowl is None  # proximity alone used to attach

    red_geom = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_GEOM, "bowl_red_wall_0")
    blue_geom = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_GEOM, "bowl_blue_wall_0")

    def contact(jaw, geom, distance=-0.001):
        con = mujoco.MjContact()
        con.geom1, con.geom2, con.dist = jaw, geom, distance
        mujoco.mj_addContact(backend.model, backend.data, con)

    contact(task._left_jaw_geom_id, red_geom)
    contact(task._right_jaw_geom_id, blue_geom)
    task._on_physics_step()
    assert task.held_bowl is None
    contact(task._right_jaw_geom_id, red_geom, 0.001)
    task._on_physics_step()
    assert task.held_bowl is None  # margin contact is not touching
    contact(task._right_jaw_geom_id, red_geom)
    task._on_physics_step()
    assert task.held_bowl == "red"
    backend.data.ncon = 0
    assert not task._is_gripper_disengaged()
    backend.data.ctrl[backend._actuator_ids["gripper"]] = backend.raw_to_control("gripper", 1195)
    task._on_physics_step()
    assert task.held_bowl is None
    assert task.held_rel_pos is None
    assert task._is_gripper_disengaged()


def test_stack_requires_release_upright_and_angular_stability(task):
    backend = task.backend
    for i, color in enumerate(task.current_order):
        qadr = backend.model.jnt_qposadr[task._joint_ids[color]]
        backend.data.qpos[qadr : qadr + 7] = [0, -0.2, 0.02 + i * 0.03, 1, 0, 0, 0]
    mujoco.mj_forward(backend.model, backend.data)
    backend.data.ncon = 0
    backend.data.cvel[:] = 0
    assert not task.status().success
    backend.data.time += 0.6
    assert task.status().success
    task.held_bowl = "red"
    assert not task.status().success
    task.held_bowl = None
    backend.data.cvel[task._body_ids["red"], 0] = 1
    assert not task.status().success
    assert task._hold_start_time is None
    backend.data.cvel[:] = 0
    backend.data.xmat[task._body_ids["red"], 8] = 0
    assert not task.status().success
    assert task._hold_start_time is None


@pytest.mark.parametrize("jitter", [-0.1, 0.051, float("nan")])
def test_invalid_jitter(task, jitter):
    with pytest.raises(ValueError, match="position_jitter"):
        NexArmStackBowlsTask(task.backend, position_jitter_m=jitter)


def test_generation_log_resume_and_merge(tmp_path):
    import json

    from examples.nexarm.generate_stack_bowls_dataset import (
        _generation_log,
        _log_generation,
        _merge_generation_logs,
        _next_seed,
    )

    first, second, merged = (tmp_path / name for name in ("first", "second", "merged"))
    _log_generation(first, {"type": "run", "next_seed": 30})
    _log_generation(first, {"episode_index": 0, "seed": 2})
    _log_generation(second, {"type": "run", "next_seed": 60})
    _log_generation(second, {"episode_index": None, "seed": 30})
    _log_generation(second, {"episode_index": 0, "seed": 31})
    _merge_generation_logs(
        [
            SimpleNamespace(root=first, meta=SimpleNamespace(total_episodes=1)),
            SimpleNamespace(root=second, meta=SimpleNamespace(total_episodes=1)),
        ],
        merged,
    )
    assert _next_seed(merged, 2) == 60
    records = [json.loads(line) for line in _generation_log(merged).read_text().splitlines()]
    assert records[-1]["episode_index"] == 1
    assert records[-2]["episode_index"] is None
