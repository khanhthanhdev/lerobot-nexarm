"""Unit and integration tests for exported simulation models (MuJoCo, URDF, meshes)."""

import sys
from pathlib import Path

import mujoco

SIM_DIR = Path(__file__).parent.parent / "sim"
DESCRIPTION_DIR = SIM_DIR / "description"
MJCF_DIR = DESCRIPTION_DIR / "mjcf"
ASSETS_DIR = SIM_DIR / "assets" / "meshes" / "visual"
MESHES_DIR = ASSETS_DIR
# The xacro's mesh_dir for a standalone URDF next to the .xacro.
STANDALONE_MESH_DIR = "../../assets/meshes/visual"

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


def test_positive_definite_inertia():
    mjcf_path = MJCF_DIR / "nexarm.xml"
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


def test_xacro_compilation_modes():
    import xacro

    xacro_path = DESCRIPTION_DIR / "urdf" / "nexarm.urdf.xacro"
    doc_ros = xacro.process_file(str(xacro_path))
    assert "package://nexarm_description/meshes/visual/base_link.stl" in doc_ros.toxml()

    doc_standalone = xacro.process_file(str(xacro_path), mappings={"mesh_dir": STANDALONE_MESH_DIR})
    assert f"{STANDALONE_MESH_DIR}/base_link.stl" in doc_standalone.toxml()


def test_xacro_generated_urdf_loads_in_mujoco():
    import os

    import xacro

    xacro_path = DESCRIPTION_DIR / "urdf" / "nexarm.urdf.xacro"
    doc_standalone = xacro.process_file(str(xacro_path), mappings={"mesh_dir": STANDALONE_MESH_DIR})

    orig_cwd = os.getcwd()
    try:
        os.chdir(xacro_path.parent)
        model = mujoco.MjModel.from_xml_string(doc_standalone.toxml())
        data = mujoco.MjData(model)
        assert model.nq == 8
        assert model.nv == 8
        for _ in range(10):
            mujoco.mj_step(model, data)
    finally:
        os.chdir(orig_cwd)


def test_urdf_loads_in_rerun_tree():
    try:
        import rerun as rr
    except ImportError:
        return

    urdf_path = DESCRIPTION_DIR / "urdf" / "nexarm.urdf"
    tree = rr.urdf.UrdfTree.from_file_path(str(urdf_path))
    joints = tree.joints()
    assert len(joints) >= 8
    assert tree.get_joint_by_name("joint_1_base_to_link_1") is not None


def test_description_scene_camera_rendering():
    import numpy as np

    scene_path = DESCRIPTION_DIR / "mjcf" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=120, width=160)
    try:
        renderer.update_scene(data, camera="front")
        img_front = renderer.render()
        assert img_front.shape == (120, 160, 3)
        assert img_front.dtype == np.uint8

        renderer.update_scene(data, camera="wrist")
        img_wrist = renderer.render()
        assert img_wrist.shape == (120, 160, 3)
        assert img_wrist.dtype == np.uint8
    finally:
        renderer.close()


# --- Fusion extract merge (sim/apply_fusion_extract.py) ---------------------------------------------------


def _load_applier():
    import importlib.util

    spec = importlib.util.spec_from_file_location("apply_fusion_extract", SIM_DIR / "apply_fusion_extract.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _synthetic_extract(applier, table_top_z_mm=-33.0):
    """An extract that reproduces the committed MJCF inertials and kinematics."""
    import xml.etree.ElementTree as ET  # nosec B405

    import numpy as np

    mjcf = MJCF_DIR / "nexarm.xml"
    robot = applier.read_mjcf(mjcf)
    root = ET.parse(mjcf).getroot()  # nosec B314
    bodies = {}
    for name, (_suffix, filename) in {
        k: (None, f"{k}.stl") for k in [m[:-4] for m in EXPECTED_MESHES]
    }.items():
        inertial = root.find(f".//body[@name='{name}']/inertial")
        com = np.array([float(v) for v in inertial.get("pos").split()]) - robot.visual_offsets.get(name, 0)
        bodies[name] = {
            "mass_kg": float(inertial.get("mass")),
            "com_mm": (com * 1000).tolist(),
            "fullinertia": [float(v) for v in inertial.get("fullinertia").split()],
            "mesh": filename,
        }
    joints = {}
    for name in [
        "joint_1_base_to_link_1",
        "joint_2_link_1_to_link_2",
        "joint_3_link_2_to_link_3",
        "joint_4_link_3_to_link_4",
        "joint_5_link_4_to_link_5",
        "left_jaw_slide_joint",
        "right_jaw_slide_joint",
    ]:
        kind, pos, axis = robot.joints[name]
        joints[name] = {
            "type": "revolute" if kind == "hinge" else "prismatic",
            "origin_mm": (pos * 1000).tolist(),
            "axis": axis.tolist(),
        }
    return {
        "schema": applier.SCHEMA,
        "bodies": bodies,
        "joints": joints,
        "workcell": {
            "table_top_z_mm": table_top_z_mm,
            "boxes": [
                {
                    "occurrence": "workbench:1",
                    "body": "Tabletop",
                    "material": "Plywood",
                    "min_mm": [0, 0, -58],
                    "max_mm": [1, 1, -33],
                },
                {
                    "occurrence": "gantry_frame:1",
                    "body": "Plate Under Arm",
                    "material": "Aluminum 6061",
                    "min_mm": [455.4, -20.0, -13.0],
                    "max_mm": [623.4, 100.0, -5.0],
                },
            ],
            "skipped_non_box_bodies": [{"occurrence": "C270_2020_Mount:1", "body": "Bracket"}],
            "top_camera": {
                "lens_mm": [499.4, -286.0, 736.7],
                "view_dir": [0, 0, -1],
                "long_axis": [0, -1, 0],
            },
        },
    }


def _copy_sim_tree(tmp_path):
    import shutil

    sim = tmp_path / "sim"
    shutil.copytree(DESCRIPTION_DIR, sim / "description")
    shutil.copytree(ASSETS_DIR, sim / "assets" / "meshes" / "visual")
    # What the Fusion exporter stages for the merge.
    shutil.copytree(ASSETS_DIR, sim / "fusion_export" / "meshes")
    return sim


def test_apply_fusion_extract_merges_without_touching_tuning(tmp_path):
    import json

    import numpy as np

    applier = _load_applier()
    sim = _copy_sim_tree(tmp_path)
    extract = _synthetic_extract(applier)
    extract["bodies"]["link_2"]["mass_kg"] = 0.2
    path = tmp_path / "extract.json"
    path.write_text(json.dumps(extract), encoding="utf-8")

    applier.apply_extract(path, sim)

    robot_xml = (sim / "description" / "mjcf" / "nexarm.xml").read_text(encoding="utf-8")
    assert 'kp="80" kv="4.0" forcerange="-6.37 6.37"' in robot_xml  # hand tuning survives
    for scene in ("scene.xml", "bowl_stack_scene.xml"):
        model = mujoco.MjModel.from_xml_path(str(sim / "description" / "mjcf" / scene))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        link_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_2")
        assert model.body_mass[link_2] == 0.2
        mount = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "nexarm_mount")
        assert np.isclose(model.body_pos[mount][2], 0.033)
        top = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
        assert np.allclose(model.cam_pos[top], [-0.04, -0.35, 0.7697], atol=1e-4)
        # The plate fixture is in the scene, the tabletop is not (the floor is the tabletop).
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "workcell_plate_under_arm") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "workcell_tabletop") < 0

    urdf = mujoco.MjModel.from_xml_path(str(sim / "description" / "urdf" / "nexarm.urdf"))
    assert np.isclose(urdf.body_mass[mujoco.mj_name2id(urdf, mujoco.mjtObj.mjOBJ_BODY, "link_2")], 0.2)

    # Applying the same extract again changes nothing.
    before = robot_xml
    applier.apply_extract(path, sim)
    assert (sim / "description" / "mjcf" / "nexarm.xml").read_text(encoding="utf-8") == before


def test_apply_fusion_extract_rejects_moved_joint(tmp_path):
    import json

    import pytest

    applier = _load_applier()
    sim = _copy_sim_tree(tmp_path)
    extract = _synthetic_extract(applier)
    extract["joints"]["joint_3_link_2_to_link_3"]["origin_mm"][2] += 5.0  # off the hinge line
    path = tmp_path / "extract.json"
    path.write_text(json.dumps(extract), encoding="utf-8")
    original = (sim / "description" / "mjcf" / "nexarm.xml").read_text(encoding="utf-8")

    with pytest.raises(applier.ExtractMismatchError, match="joint_3_link_2_to_link_3"):
        applier.apply_extract(path, sim)
    assert (sim / "description" / "mjcf" / "nexarm.xml").read_text(encoding="utf-8") == original


def test_apply_fusion_extract_rejects_com_outside_mesh(tmp_path):
    import json

    import pytest

    applier = _load_applier()
    sim = _copy_sim_tree(tmp_path)
    extract = _synthetic_extract(applier)
    extract["bodies"]["link_5"]["com_mm"][1] += 150.0  # a world-vs-arm frame mix-up
    path = tmp_path / "extract.json"
    path.write_text(json.dumps(extract), encoding="utf-8")

    with pytest.raises(applier.ExtractMismatchError, match="link_5: centre of mass"):
        applier.apply_extract(path, sim)


def test_committed_mjcf_centres_of_mass_lie_inside_meshes():
    import numpy as np

    applier = _load_applier()
    model = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "nexarm.xml"))
    robot = applier.read_mjcf(MJCF_DIR / "nexarm.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for mesh_name in EXPECTED_MESHES:
        name = mesh_name[:-4]
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        mount = model.body_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "nexarm_mount")]
        com_arm = data.xipos[body] - mount - robot.visual_offsets.get(name, 0)
        lo, hi = applier.stl_bounds(MESHES_DIR / mesh_name)
        assert np.all(com_arm * 1000 >= lo - 2) and np.all(com_arm * 1000 <= hi + 2), name


# --- Collision geometry and scene sanity -------------------------------------------------------------------

COVERED_LINKS = [
    "base_link",
    "link_1",
    "link_2",
    "link_3",
    "link_4",
    "link_5",
    "link_6_gripper_base",
    "cam_mount",
]


def _collision_coverage(model, body_name, tol=0.001):
    """Share of the body's visual mesh vertices that lie inside (or within ``tol`` of) its collision geoms."""
    import numpy as np

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    visual = next(g for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    collision = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]

    def rot(quat):
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        return mat.reshape(3, 3)

    mesh = model.geom_dataid[visual]
    start = model.mesh_vertadr[mesh]
    points = model.mesh_vert[start : start + model.mesh_vertnum[mesh]] @ rot(model.geom_quat[visual]).T
    points = points + model.geom_pos[visual]
    covered = np.zeros(len(points), bool)
    for geom in collision:
        local = (points - model.geom_pos[geom]) @ rot(model.geom_quat[geom])
        size = model.geom_size[geom]
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            covered |= np.all(np.abs(local) <= size + tol, axis=1)
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axial = np.clip(local[:, 2], -size[1], size[1])
            covered |= np.linalg.norm(local - np.c_[0 * axial, 0 * axial, axial], axis=1) <= size[0] + tol
    return covered.mean()


def test_link_collisions_cover_the_link_meshes():
    model = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "nexarm.xml"))
    for name in COVERED_LINKS:
        assert _collision_coverage(model, name) >= 0.95, name


def test_robot_keyframes_have_no_contacts_in_the_workcell():
    model = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "scene.xml"))
    data = mujoco.MjData(model)
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    for key in range(model.nkey):
        mujoco.mj_resetData(model, data)
        data.qpos[:8] = model.key_qpos[key][:8]
        mujoco.mj_forward(model, data)
        contacts = [
            (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
            )
            for c in data.contact[: data.ncon]
            if cube not in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
        ]
        assert not contacts, (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, key), contacts)


def test_bowls_weigh_60_g_and_start_apart():
    model = mujoco.MjModel.from_xml_path(str(MJCF_DIR / "bowl_stack_scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bowls = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bowl_{c}") for c in ("red", "blue", "black")
    }
    for body in bowls:
        assert abs(model.body_mass[body] - 0.060) < 1e-4  # 33 proxies x 1.818 g
    touching = [
        c
        for c in data.contact[: data.ncon]
        if {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} <= bowls
    ]
    assert not touching
