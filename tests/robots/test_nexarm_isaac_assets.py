# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
"""The Isaac Sim inputs (URDF, nexarm_scene.json, bowl USDs) carry the same design as the MuJoCo model.

Isaac itself is not needed: these check the files the Isaac Lab environment builds its scene from.
"""

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET  # nosec B405 - parses the repo's own URDF
from pathlib import Path

import mujoco
import numpy as np
import pytest

from lerobot_robot_nexarm.motors.nexarm import JOINT_NAMES
from lerobot_robot_nexarm.nexarm_sim.mujoco_backend import RAW_RANGES, NexArmMujocoBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM = REPO_ROOT / "sim"
ISAAC_LAB = SIM / "isaac_lab"
STACK_SCENE = SIM / "description" / "mjcf" / "bowl_stack_scene.xml"
ROBOT_MJCF = SIM / "description" / "mjcf" / "nexarm.xml"
URDF = SIM / "description" / "urdf" / "nexarm.urdf"

sys.path.insert(0, str(ISAAC_LAB))
from nexarm_isaac.scene_spec import SceneSpec, camera_apertures  # noqa: E402


def _builder():
    spec = importlib.util.spec_from_file_location("build_isaac_assets", ISAAC_LAB / "build_isaac_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scene():
    return SceneSpec.load()


def test_isaac_assets_are_in_sync_with_the_mjcf():
    assert _builder().main(["--check"]) == 0, "run: python sim/isaac_lab/build_isaac_assets.py"


def test_urdf_collisions_and_kinematics_match_the_mjcf(scene):
    """Every MJCF collision box sits at the same place in the URDF, over random joint positions."""
    mjcf, urdf = (mujoco.MjModel.from_xml_path(str(p)) for p in (ROBOT_MJCF, URDF))
    mjcf_data, urdf_data = mujoco.MjData(mjcf), mujoco.MjData(urdf)
    mjcf_boxes = [g for g in range(mjcf.ngeom) if mjcf.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    urdf_boxes = [g for g in range(urdf.ngeom) if urdf.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    assert len(urdf_boxes) == len(mjcf_boxes) == urdf.ngeom
    # The URDF base_link frame is the world origin there; in the MJCF it sits on the mounting plate.
    base = np.array(scene.robot["base_pos"])
    assert scene.robot["base_quat"] == [1.0, 0.0, 0.0, 0.0]
    rng = np.random.default_rng(0)
    for _ in range(20):
        for j in range(urdf.njnt):
            name = mujoco.mj_id2name(urdf, mujoco.mjtObj.mjOBJ_JOINT, j)
            jm = mujoco.mj_name2id(mjcf, mujoco.mjtObj.mjOBJ_JOINT, name)
            q = rng.uniform(*mjcf.jnt_range[jm])
            mjcf_data.qpos[mjcf.jnt_qposadr[jm]] = q
            urdf_data.qpos[urdf.jnt_qposadr[j]] = q
        mujoco.mj_kinematics(mjcf, mjcf_data)
        mujoco.mj_kinematics(urdf, urdf_data)
        for g in mjcf_boxes:
            same_size = [
                u for u in urdf_boxes if np.allclose(mjcf.geom_size[g], urdf.geom_size[u], atol=2e-6)
            ]
            errors = [
                np.linalg.norm(mjcf_data.geom_xpos[g] - base - urdf_data.geom_xpos[u]) for u in same_size
            ]
            u = same_size[int(np.argmin(errors))]
            assert min(errors) < 1e-5
            np.testing.assert_allclose(
                mjcf_data.geom_xmat[g].reshape(3, 3), urdf_data.geom_xmat[u].reshape(3, 3), atol=1e-5
            )


def _urdf_forward_kinematics(path: Path, q: dict[str, float]) -> dict[str, np.ndarray]:
    """World pose (4x4) of every URDF link, from the joint origins and axes alone."""

    def rpy_matrix(roll, pitch, yaw):
        cr, sr, cp, sp, cy, sy = (
            np.cos(roll),
            np.sin(roll),
            np.cos(pitch),
            np.sin(pitch),
            np.cos(yaw),
            np.sin(yaw),
        )
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

    def axis_motion(joint, value):
        motion = np.eye(4)
        axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
        if joint.get("type") in ("revolute", "continuous"):
            quat, mat = np.zeros(4), np.zeros(9)
            mujoco.mju_axisAngle2Quat(quat, axis, value)
            mujoco.mju_quat2Mat(mat, quat)
            motion[:3, :3] = mat.reshape(3, 3)
        elif joint.get("type") == "prismatic":
            motion[:3, 3] = axis * value
        return motion

    joints = {j.find("child").get("link"): j for j in ET.parse(path).iter("joint")}  # nosec B314
    poses: dict[str, np.ndarray] = {}

    def pose(link: str) -> np.ndarray:
        if link not in poses:
            joint = joints.get(link)
            if joint is None:
                poses[link] = np.eye(4)
            else:
                origin = joint.find("origin")
                local = np.eye(4)
                local[:3, :3] = rpy_matrix(*[float(v) for v in origin.get("rpy", "0 0 0").split()])
                local[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
                motion = (
                    np.eye(4)
                    if joint.get("type") == "fixed"
                    else axis_motion(joint, q.get(joint.get("name"), 0.0))
                )
                poses[link] = pose(joint.find("parent").get("link")) @ local @ motion
        return poses[link]

    for link in joints:
        pose(link)
    return poses


def test_urdf_gripper_frame_link_is_the_mjcf_gripper_frame_site(scene):
    """End-effector poses name the same point in MuJoCo, Isaac and a URDF IK solver."""
    model = mujoco.MjModel.from_xml_path(str(ROBOT_MJCF))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame")
    base = np.array(scene.robot["base_pos"])
    rng = np.random.default_rng(2)
    for _ in range(20):
        q = {}
        for j in range(model.njnt):
            if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                q[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = rng.uniform(*model.jnt_range[j])
                data.qpos[model.jnt_qposadr[j]] = q[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        mujoco.mj_kinematics(model, data)
        tcp = _urdf_forward_kinematics(URDF, q)["gripper_frame"]
        np.testing.assert_allclose(tcp[:3, 3] + base, data.site_xpos[site], atol=1e-5)
        np.testing.assert_allclose(tcp[:3, :3], data.site_xmat[site].reshape(3, 3), atol=1e-5)


def test_urdf_joint_dynamics_match_the_mjcf():
    mjcf = mujoco.MjModel.from_xml_path(str(ROBOT_MJCF))
    for joint in ET.parse(URDF).iter("joint"):  # nosec B314
        dynamics = joint.find("dynamics")
        if dynamics is None:
            continue
        dof = mjcf.jnt_dofadr[mujoco.mj_name2id(mjcf, mujoco.mjtObj.mjOBJ_JOINT, joint.get("name"))]
        assert float(dynamics.get("damping")) == pytest.approx(mjcf.dof_damping[dof]), joint.get("name")
        assert float(dynamics.get("friction")) == pytest.approx(mjcf.dof_frictionloss[dof]), joint.get("name")


def test_action_map_matches_the_mujoco_backend(scene):
    backend = NexArmMujocoBackend(
        model_path=STACK_SCENE, fps=30, camera_width=8, camera_height=8, camera_names=()
    )
    try:
        rng = np.random.default_rng(1)
        lows = np.array([min(RAW_RANGES[n]) for n in JOINT_NAMES]) - 100
        highs = np.array([max(RAW_RANGES[n]) for n in JOINT_NAMES]) + 100
        for raw in rng.uniform(lows, highs, size=(200, 6)):
            expected = [
                backend.raw_to_control(name, value) for name, value in zip(JOINT_NAMES, raw, strict=True)
            ]
            np.testing.assert_allclose(scene.raw_to_target(raw), expected, atol=1e-9)
        backend.reset()
        home = scene.home_joint_positions()
        for joint, value in home.items():
            jid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            assert backend.data.qpos[backend.model.jnt_qposadr[jid]] == pytest.approx(value, abs=1e-9), joint
    finally:
        backend.close()


def test_scene_cameras_robot_and_bowls_match_the_mjcf(scene):
    model = mujoco.MjModel.from_xml_path(str(STACK_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cameras = {c["name"]: c for c in scene.data["cameras"]}
    assert set(cameras) == {"front", "top", "wrist"}
    for name in ("front", "top"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        np.testing.assert_allclose(cameras[name]["pos"], data.cam_xpos[cid], atol=1e-6)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, np.array(cameras[name]["quat"]))
        np.testing.assert_allclose(mat, data.cam_xmat[cid], atol=1e-5)
    assert cameras["wrist"]["parent_link"] == "cam_mount"
    for cam in cameras.values():
        horizontal, vertical = camera_apertures(cam["fovy_deg"], cam["width"], cam["height"])
        assert np.degrees(2 * np.arctan(vertical / 2)) == pytest.approx(cam["fovy_deg"])
        assert horizontal / vertical == pytest.approx(cam["width"] / cam["height"])

    mount = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "nexarm_mount")]
    assert scene.robot["base_pos"][2] == pytest.approx(mount[2], abs=1e-6)
    bowl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl_red")
    proxies = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bowl and model.geom_conaffinity[g]]
    assert len(scene.bowls["colliders"]) == len(proxies)
    assert scene.bowls["mass"] == pytest.approx(0.06, abs=1e-4)
    workcell = [
        g for g in range(model.ngeom) if (mujoco.mj_id2name(model, 5, g) or "").startswith("workcell_")
    ]
    assert len(scene.data["fixtures"]) == len(workcell)
    assert scene.resolve(scene.robot["urdf"]) == URDF.resolve()


def test_bowl_usd_matches_the_mjcf_proxies(scene, tmp_path):
    pytest.importorskip("pxr", reason="needs usd-core (or Isaac Sim's Python)")
    from nexarm_isaac.bowl_usd import write_bowl_usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = write_bowl_usd(
        scene.bowls, "red", scene.resolve(scene.bowls["visual"]["mesh"]), tmp_path / "bowl_red.usd"
    )
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(scene.bowls["mass"])
    cache = UsdGeom.XformCache()
    ends = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            half = prim.GetAttribute("height").Get() / 2
            xf = cache.GetLocalToWorldTransform(prim)
            ends.append([*xf.Transform(Gf.Vec3d(0, 0, half)), *xf.Transform(Gf.Vec3d(0, 0, -half))])

    model = mujoco.MjModel.from_xml_path(str(STACK_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl_red")
    origin, frame = data.xpos[body], data.xmat[body].reshape(3, 3)
    expected = []
    for g in range(model.ngeom):
        if (
            model.geom_bodyid[g] == body and model.geom_conaffinity[g]
        ):  # the capsule proxy, not the jaw-only rim plates
            axis = data.geom_xmat[g].reshape(3, 3)[:, 2] * model.geom_size[g, 1]
            centre = data.geom_xpos[g]
            expected.append([*(frame.T @ (centre + axis - origin)), *(frame.T @ (centre - axis - origin))])
    np.testing.assert_allclose(ends, expected, atol=1e-7)


def test_scene_json_is_valid_json_with_schema():
    data = json.loads((ISAAC_LAB / "nexarm_scene.json").read_text(encoding="utf-8"))
    assert data["schema"] == "nexarm_isaac_scene_v1"
