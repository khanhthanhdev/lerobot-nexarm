# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from pathlib import Path

import mujoco
import numpy as np
import pytest

from lerobot.motors.nexarm.nexarm import GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS
from lerobot.robots.nexarm_sim.mujoco_backend import NexArmMujocoBackend
from lerobot.robots.nexarm_sim.stack_bowls_task import COLORS, NexArmStackBowlsTask

MODEL = Path(__file__).resolve().parents[2] / "sim/description/mjcf/bowl_stack_scene.xml"


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
    backend.data.ctrl[backend._actuator_ids["gripper"]] = backend.raw_to_control(
        "gripper", GRIPPER_CLOSED_POS
    )
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
    backend.data.ctrl[backend._actuator_ids["gripper"]] = backend.raw_to_control("gripper", GRIPPER_OPEN_POS)
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


def test_held_bowl_turns_with_the_gripper(task):
    """The assist carries the bowl rigidly, so turning the wrist cannot swing the gripper into the bowl."""
    backend = task.backend
    model, data = backend.model, backend.data
    qadr = model.jnt_qposadr[task._joint_ids["red"]]
    data.qpos[qadr : qadr + 3] = data.site_xpos[task._gripper_frame_site_id] + [0, 0, 0.09]
    mujoco.mj_forward(model, data)
    data.ncon = 0
    red_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bowl_red_wall_0")
    for jaw in (task._left_jaw_geom_id, task._right_jaw_geom_id):
        con = mujoco.MjContact()
        con.geom1, con.geom2, con.dist = jaw, red_geom, -0.001
        mujoco.mj_addContact(model, data, con)
    data.ctrl[backend._actuator_ids["gripper"]] = backend.raw_to_control("gripper", GRIPPER_CLOSED_POS)
    task.target_bowl = "red"
    task._on_physics_step()
    assert task.held_bowl == "red"
    relative = task.held_rel_mat.copy()

    roll = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_5_link_4_to_link_5")
    data.qpos[model.jnt_qposadr[roll]] += 0.8
    mujoco.mj_forward(model, data)
    task._on_physics_step()
    mujoco.mj_forward(model, data)

    site = data.site_xmat[task._gripper_frame_site_id].reshape(3, 3)
    bowl = data.xmat[task._body_ids["red"]].reshape(3, 3)
    np.testing.assert_allclose(site.T @ bowl, relative, atol=1e-9)


def test_assist_release_threshold_is_configurable(task):
    """Real teleop releases often open only to about 1990, which the default threshold never sees."""
    backend = task.backend
    gripper = backend._actuator_ids["gripper"]
    task.held_bowl, task.held_rel_pos = "red", np.zeros(3)
    backend.data.ctrl[gripper] = backend.raw_to_control("gripper", 1990)
    task._on_physics_step()
    assert task.held_bowl == "red"

    task.assist_release_pos = 2300
    task._on_physics_step()
    assert task.held_bowl is None


@pytest.mark.parametrize("release", [GRIPPER_OPEN_POS - 1, 2400])
def test_invalid_assist_release_threshold(task, release):
    with pytest.raises(ValueError, match="assist_release_pos"):
        NexArmStackBowlsTask(task.backend, assist_release_pos=release)


@pytest.mark.parametrize("engage", [GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS])
def test_invalid_assist_engage_threshold(task, engage):
    with pytest.raises(ValueError, match="assist_engage_pos"):
        NexArmStackBowlsTask(task.backend, assist_engage_pos=engage)


def test_stages_latch_and_name_the_first_unreached(task):
    from lerobot.robots.nexarm_sim.stack_bowls_task import STAGES

    backend = task.backend
    status = task.status()
    assert not any(status.stages.values()) and status.failed_stage == "picked_middle"
    assert status.is_arm_static

    # Put the stack together without picking anything: every stage after the picks is reached.
    qadrs = [backend.model.jnt_qposadr[task._joint_ids[color]] for color in task.current_order]
    for i, qadr in enumerate(qadrs):
        backend.data.qpos[qadr : qadr + 7] = [0, -0.2, 0.02 + i * 0.03, 1, 0, 0, 0]
    mujoco.mj_forward(backend.model, backend.data)
    backend.data.ncon = 0
    status = task.status()
    assert [stage for stage in STAGES if not status.stages[stage]] == ["picked_middle", "picked_top"]
    assert status.failed_stage == "picked_middle"

    # Knocking the stack apart again does not unlatch them.
    backend.data.qpos[qadrs[2] : qadrs[2] + 3] = [0.15, -0.25, 0.02]
    mujoco.mj_forward(backend.model, backend.data)
    assert task.status().stages["released_top"]

    backend.data.qvel[task._arm_dof_ids[0]] = 1.0
    assert not task.status().is_arm_static
    task.reset(seed=3, settle_steps=0)
    assert not any(task.status().stages.values())


@pytest.mark.parametrize(
    ("where", "bowl_local_point", "touches"),
    [
        ("floor", (0, 0, -0.0175), "bowl_red_floor"),
        ("lower wall", (0.0493, 0, -0.0155), "bowl_red_shell_0_0"),
        ("rim", (0.0615, 0, 0.010), "bowl_red_rim_0"),
        ("open interior", (0, 0, 0.012), None),
    ],
)
def test_jaws_cannot_pass_through_the_bowl(task, where, bowl_local_point, touches):
    model, data = task.backend.model, task.backend.data
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "link_6_left_jaw_collision_0")
    qadr = model.jnt_qposadr[task._joint_ids["red"]]
    data.qpos[qadr : qadr + 3] = data.geom_xpos[jaw] - np.array(bowl_local_point)
    data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    touched = set()
    for con in data.contact[: data.ncon]:
        if jaw in (con.geom1, con.geom2):
            other = con.geom2 if con.geom1 == jaw else con.geom1
            touched.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other))
    touched = {name for name in touched if name.startswith("bowl_red")}
    if touches is None:
        assert not touched, where  # the inner jaw still fits inside the bowl
    else:
        assert touches in touched, where


def test_pressed_bowls_do_not_wedge_lock():
    """A bowl pressed into another and lifted out must not drag (then fling) the lower bowl with it."""
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    blue, red = (
        model.jnt_qposadr[task_id]
        for task_id in (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"bowl_{c}_joint") for c in ("blue", "red")
        )
    )
    blue_body, red_body = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bowl_{c}") for c in ("blue", "red")
    )
    data.qpos[red : red + 3] = data.qpos[blue : blue + 3] + [0, 0, 0.03]
    for _ in range(500):
        mujoco.mj_step(model, data)
    rest_z = data.xpos[blue_body][2]

    data.xfrc_applied[red_body, 2] = -20.0  # pressed in about as hard as the arm can
    for _ in range(500):
        mujoco.mj_step(model, data)
    data.xfrc_applied[red_body, 2] = 3.0  # then lifted straight out
    rise = 0.0
    for _ in range(500):
        mujoco.mj_step(model, data)
        rise = max(rise, data.xpos[blue_body][2] - rest_z)
    assert rise < 0.005
