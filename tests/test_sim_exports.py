"""Unit and integration tests for exported simulation models (MuJoCo, URDF, meshes)."""

from pathlib import Path

import mujoco

SIM_DIR = Path(__file__).parent.parent / "sim"
FUSION_EXPORT_DIR = SIM_DIR / "fusion_export"
MESHES_DIR = FUSION_EXPORT_DIR / "meshes"
DESCRIPTION_DIR = SIM_DIR / "description"
ASSETS_DIR = SIM_DIR / "assets" / "meshes" / "visual"

EXPECTED_MESHES = [
    "base_link.stl",
    "link_1.stl",
    "link_2.stl",
    "link_3.stl",
    "link_4.stl",
    "link_5.stl",
    "link_6_gripper_base.stl",
    "link_6_pinion_gear.stl",
    "link_6_left_jaw.stl",
    "link_6_right_jaw.stl",
    "cam_mount.stl",
]


def test_exported_meshes_exist_and_non_empty():
    assert MESHES_DIR.exists(), f"Mesh directory {MESHES_DIR} does not exist"
    for mesh_name in EXPECTED_MESHES:
        mesh_path = MESHES_DIR / mesh_name
        assert mesh_path.exists(), f"Missing mesh file: {mesh_name}"
        assert mesh_path.stat().st_size > 1000, f"Mesh file {mesh_name} is too small or empty"


def test_mujoco_mjcf_loads_and_steps():
    mjcf_path = FUSION_EXPORT_DIR / "NexArm-sim.xml"
    assert mjcf_path.exists(), f"Missing {mjcf_path}"

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    # Robot has 5 arm joints + 1 pinion + 2 jaw slider joints = 8 nq/nv
    assert model.nq == 8, f"Expected nq=8, got {model.nq}"
    assert model.nv == 8, f"Expected nv=8, got {model.nv}"
    # 5 arm actuators + 1 gripper actuator = 6 controls
    assert model.nu == 6, f"Expected nu=6, got {model.nu}"

    # Verify physical step executes without instability
    for _ in range(10):
        mujoco.mj_step(model, data)


def test_scene_xml_loads_and_steps():
    scene_path = FUSION_EXPORT_DIR / "scene.xml"
    assert scene_path.exists(), f"Missing {scene_path}"

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    assert model.nu == 6, f"Expected nu=6, got {model.nu}"
    for _ in range(10):
        mujoco.mj_step(model, data)


def test_urdf_loads_in_mujoco():
    urdf_path = FUSION_EXPORT_DIR / "nexarm.urdf"
    assert urdf_path.exists(), f"Missing {urdf_path}"

    # MuJoCo's built-in URDF parser validates syntax, kinematics, and mass properties
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)

    assert model.nq == 8, f"Expected nq=8, got {model.nq}"
    assert model.nv == 8, f"Expected nv=8, got {model.nv}"

    for _ in range(10):
        mujoco.mj_step(model, data)


def test_positive_definite_inertia():
    mjcf_path = FUSION_EXPORT_DIR / "NexArm-sim.xml"
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))

    for body_id in range(model.nbody):
        mass = model.body_mass[body_id]
        if mass > 0:
            inertia = model.body_inertia[body_id]
            assert all(i > 0 for i in inertia), f"Non-positive principal inertia for body {body_id}"


def test_description_assets_visual_meshes_exist():
    assert ASSETS_DIR.exists(), f"Asset directory {ASSETS_DIR} does not exist"
    for mesh_name in EXPECTED_MESHES:
        mesh_path = ASSETS_DIR / mesh_name
        assert mesh_path.exists(), f"Missing visual asset: {mesh_name}"
        assert mesh_path.stat().st_size > 1000, f"Asset file {mesh_name} is too small or empty"


def test_description_urdf_loads_in_mujoco():
    urdf_path = DESCRIPTION_DIR / "urdf" / "nexarm.urdf"
    assert urdf_path.exists(), f"Missing {urdf_path}"

    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)

    assert model.nq == 8, f"Expected nq=8, got {model.nq}"
    assert model.nv == 8, f"Expected nv=8, got {model.nv}"
    for _ in range(10):
        mujoco.mj_step(model, data)


def test_description_xacro_exists_and_valid():
    xacro_path = DESCRIPTION_DIR / "urdf" / "nexarm.urdf.xacro"
    assert xacro_path.exists(), f"Missing {xacro_path}"
    content = xacro_path.read_text(encoding="utf-8")
    assert '<xacro:macro name="nexarm"' in content
    assert "joint_1_base_to_link_1" in content


def test_description_mjcf_loads_and_steps():
    mjcf_path = DESCRIPTION_DIR / "mjcf" / "nexarm.xml"
    assert mjcf_path.exists(), f"Missing {mjcf_path}"

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    assert model.nq == 8, f"Expected nq=8, got {model.nq}"
    assert model.nv == 8, f"Expected nv=8, got {model.nv}"
    assert model.nu == 6, f"Expected nu=6, got {model.nu}"
    for _ in range(10):
        mujoco.mj_step(model, data)


def test_description_scene_xml_loads_and_steps():
    scene_path = DESCRIPTION_DIR / "mjcf" / "scene.xml"
    assert scene_path.exists(), f"Missing {scene_path}"

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    assert model.nu == 6, f"Expected nu=6, got {model.nu}"
    assert model.ncam >= 2, f"Expected at least 2 cameras, got {model.ncam}"
    for _ in range(10):
        mujoco.mj_step(model, data)
