"""Comprehensive exporter from Autodesk Fusion 360 to MuJoCo MJCF, URDF, and STL meshes.

Executed inside Fusion 360 via the Fusion MCP script runner.
"""

import json
import os
import xml.etree.ElementTree as ET  # nosec B405 - this exporter creates XML for robotics
from typing import Any

import adsk.core
import adsk.fusion

WSL_BASE = r"\\wsl.localhost\Ubuntu\home\thanh\code\vinuni\lerobot-nexarm\sim"
OUTPUT_DIR = os.path.join(WSL_BASE, "fusion_export")
DESCRIPTION_DIR = os.path.join(WSL_BASE, "description")
ASSETS_DIR = os.path.join(WSL_BASE, "assets", "meshes", "visual")

BODY_SPECS = {
    "base_link": ("base_link:1", "base_link.stl"),
    "link_1": ("link_1:1", "link_1.stl"),
    "link_2": ("link_2:1", "link_2.stl"),
    "link_3": ("link_3:1", "link_3.stl"),
    "link_4": ("link_4:1", "link_4.stl"),
    "link_5": ("link_5:1", "link_5.stl"),
    "link_6_gripper_base": (
        "gripper_base:1",
        "link_6_gripper_base.stl",
    ),
    "link_6_pinion_gear": (
        "pinion_gear:1",
        "link_6_pinion_gear.stl",
    ),
    "link_6_left_jaw": ("left_jaw:1", "link_6_left_jaw.stl"),
    "link_6_right_jaw": ("right_jaw:1", "link_6_right_jaw.stl"),
    "cam_mount": ("cam_mount:1", "cam_mount.stl"),
}

ARM_JOINTS = [
    "joint_1_base_to_link_1",
    "joint_2_link_1_to_link_2",
    "joint_3_link_2_to_link_3",
    "joint_4_link_3_to_link_4",
    "joint_5_link_4_to_link_5",
]

# Calibrated joint definitions matching NexArm kinematics and physical limits
JOINT_DEFS = {
    "joint_1_base_to_link_1": {
        "fusion_names": ["Revolute 6", "joint_1_base_to_link_1"],
        "parent": "base_link",
        "child": "link_1",
        "type": "revolute",
        "axis": [0.0, 0.0, -1.0],
        "range": [-2.35619449019, 2.35619449019],  # +-135 deg
        "actuated": True,
        "kp": 100.0,
        "kv": 0.2,
    },
    "joint_2_link_1_to_link_2": {
        "fusion_names": ["Revolute 1", "joint_2_link_1_to_link_2"],
        "parent": "link_1",
        "child": "link_2",
        "type": "revolute",
        "axis": [-1.0, 0.0, 0.0],
        "range": [-2.09439510239, 2.09439510239],
        "actuated": True,
        "kp": 100.0,
        "kv": 0.2,
    },
    "joint_3_link_2_to_link_3": {
        "fusion_names": ["Revolute 7", "joint_3_link_2_to_link_3"],
        "parent": "link_2",
        "child": "link_3",
        "type": "revolute",
        "axis": [1.0, 0.0, 0.0],
        "range": [-2.35619449019, 2.35619449019],
        "actuated": True,
        "kp": 100.0,
        "kv": 0.2,
    },
    "joint_4_link_3_to_link_4": {
        "fusion_names": ["Revolute 8", "joint_4_link_3_to_link_4"],
        "parent": "link_3",
        "child": "link_4",
        "type": "revolute",
        "axis": [1.0, 0.0, 0.0],
        "range": [-1.74532925199, 1.74532925199],
        "actuated": True,
        "kp": 100.0,
        "kv": 0.2,
    },
    "joint_5_link_4_to_link_5": {
        "fusion_names": ["Revolute 9", "joint_5_link_4_to_link_5"],
        "parent": "link_4",
        "child": "link_5",
        "type": "revolute",
        "axis": [0.0, -1.0, 0.0],
        "range": [-3.14159265359, 3.14159265359],
        "actuated": True,
        "kp": 100.0,
        "kv": 0.2,
    },
    "right_jaw_slide_joint": {
        "fusion_names": ["Slider 11", "right_jaw_slide_joint"],
        "parent": "link_6_gripper_base",
        "child": "link_6_right_jaw",
        "type": "prismatic",
        "axis": [-1.0, 0.0, 0.0],
        "range": [-0.0255, 0.0],
        "actuated": True,
        "kp": 1000.0,
        "kv": 5.0,
    },
    "left_jaw_slide_joint": {
        "fusion_names": ["Slider 10", "left_jaw_slide_joint"],
        "parent": "link_6_gripper_base",
        "child": "link_6_left_jaw",
        "type": "prismatic",
        "axis": [-1.0, 0.0, 0.0],
        "range": [0.0, 0.0255],
        "actuated": False,
        "kp": 1000.0,
        "kv": 5.0,
    },
    "gripper_pinion_joint": {
        "fusion_names": ["Revolute 3", "gripper_pinion_joint"],
        "parent": "link_6_gripper_base",
        "child": "link_6_pinion_gear",
        "type": "revolute",
        "axis": [0.0, 0.0, -1.0],
        "range": [-3.14159, 3.14159],
        "actuated": False,
        "kp": 100.0,
        "kv": 0.2,
    },
}

COLLISION_SPECS = {
    "base_link": [
        {"type": "box", "pos": "0.53937 0.06397 0.030", "size": "0.050 0.060 0.030"},
    ],
    "link_1": [
        {"type": "box", "pos": "0.53937 0.06397 0.076", "size": "0.044 0.036 0.040"},
    ],
    "link_2": [
        {
            "type": "capsule",
            "fromto": "0.57447 0.06397 0.10645 0.51944 0.19693 0.29168",
            "size": "0.032",
        },
    ],
    "link_3": [
        {
            "type": "capsule",
            "fromto": "0.51944 0.19693 0.29168 0.51912 0.06552 0.23041",
            "size": "0.024",
        },
    ],
    "link_4": [
        {
            "type": "capsule",
            "fromto": "0.51912 0.06552 0.23041 0.53937 0.01829 0.23042",
            "size": "0.024",
        },
    ],
    "link_5": [
        {
            "type": "capsule",
            "fromto": "0.53937 0.01829 0.23042 0.53937 -0.02102 0.23060",
            "size": "0.022",
        },
    ],
    "link_6_gripper_base": [
        {"type": "box", "pos": "0.53937 -0.02327 0.22519", "size": "0.052 0.008 0.021"},
    ],
    "link_6_left_jaw": [
        {
            "type": "box",
            "pos": "0.50559 -0.057 0.23044",
            "size": "0.027 0.010 0.015",
            "friction": "3 0.01 0.001",
            "condim": "4",
        },
    ],
    "link_6_right_jaw": [
        {
            "type": "box",
            "pos": "0.57314 -0.057 0.23044",
            "size": "0.027 0.010 0.015",
            "friction": "3 0.01 0.001",
            "condim": "4",
        },
    ],
}

EXCLUDED_CONTACTS = [
    ("base_link", "link_1"),
    ("link_1", "link_2"),
    ("link_2", "link_3"),
    ("link_3", "link_4"),
    ("link_4", "link_5"),
    ("link_4", "cam_mount"),
    ("link_5", "cam_mount"),
    ("link_5", "link_6_gripper_base"),
    ("link_6_gripper_base", "link_6_pinion_gear"),
    ("link_6_gripper_base", "link_6_left_jaw"),
    ("link_6_gripper_base", "link_6_right_jaw"),
    ("link_6_left_jaw", "link_6_right_jaw"),
]


def _format(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _point_m(point: adsk.core.Point3D) -> list[float]:
    return [point.x * 0.01, point.y * 0.01, point.z * 0.01]


def _inertial_attributes(occurrence: adsk.fusion.Occurrence) -> dict[str, Any]:
    properties = occurrence.physicalProperties
    mass = properties.mass
    center = properties.centerOfMass
    x, y, z = center.x, center.y, center.z

    (
        succeeded,
        ixx_origin,
        iyy_origin,
        izz_origin,
        ixy_origin,
        iyz_origin,
        ixz_origin,
    ) = properties.getXYZMomentsOfInertia()
    if not succeeded:
        raise RuntimeError(f"Unable to calculate inertia for {occurrence.name}")

    # Shift tensor from world origin to center of mass, convert to kg*m^2
    ixx = (ixx_origin - mass * (y * y + z * z)) * 1e-4
    iyy = (iyy_origin - mass * (x * x + z * z)) * 1e-4
    izz = (izz_origin - mass * (x * x + y * y)) * 1e-4
    ixy = (ixy_origin + mass * x * y) * 1e-4
    ixz = (ixz_origin + mass * x * z) * 1e-4
    iyz = (iyz_origin + mass * y * z) * 1e-4

    return {
        "mass": mass,
        "com": [x * 0.01, y * 0.01, z * 0.01],
        "inertia": (ixx, iyy, izz, ixy, ixz, iyz),
    }


def _write_scene(output_dir: str, model_origin_m: list[float], include_file: str = "NexArm-sim.xml") -> None:
    scene = ET.Element("mujoco", {"model": "NexArm Scene"})
    ET.SubElement(scene, "include", {"file": include_file})
    ET.SubElement(
        scene,
        "statistic",
        {
            "center": f"0 0.2 {model_origin_m[2] + 0.2:.6g}",
            "extent": "0.8",
        },
    )
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(
        visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"}
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

    asset = ET.SubElement(scene, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.15 0.2",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )

    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 3", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "1 1 0.05",
            "rgba": "0.8 0.8 0.8 1",
            "friction": "1 0.005 0.0001",
            "contype": "2",
            "conaffinity": "2",
        },
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "front",
            "mode": "fixed",
            "pos": "0.360630977001 -0.613972762847 0.55",
            "xyaxes": "0.857493 0.514496 0 -0.240504 0.400841 0.884016",
            "fovy": "55",
        },
    )
    # Interactive target cube for grasping
    cube = ET.SubElement(
        worldbody,
        "body",
        {"name": "cube", "pos": "-0.000369022999 -0.243972762847 0.01"},
    )
    ET.SubElement(cube, "freejoint", {"name": "cube_joint"})
    ET.SubElement(
        cube,
        "geom",
        {
            "name": "cube_collision",
            "type": "box",
            "size": "0.01 0.01 0.01",
            "mass": "0.02",
            "rgba": "0.85 0.15 0.1 1",
            "friction": "1.2 0.01 0.001",
            "contype": "2",
            "conaffinity": "3",
        },
    )
    # Target zone for placement
    target = ET.SubElement(
        worldbody,
        "body",
        {"name": "target_zone", "pos": "-0.089369022999 -0.243972762847 0.002"},
    )
    ET.SubElement(
        target,
        "geom",
        {
            "name": "target_zone_visual",
            "type": "cylinder",
            "size": "0.05 0.002",
            "rgba": "0.1 0.8 0.2 0.45",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    ET.indent(scene, space="  ")
    scene_path = os.path.join(output_dir, "scene.xml")
    ET.ElementTree(scene).write(scene_path, encoding="utf-8", xml_declaration=True)


def run(_context: str):
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise RuntimeError("No active Fusion design found.")

    root = design.rootComponent
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mesh_dir = os.path.join(OUTPUT_DIR, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # 1. Map occurrences
    all_occs = {occ.name: occ for occ in root.allOccurrences}

    found_occs = {}
    for body_name, (needle, _filename) in BODY_SPECS.items():
        matched = None
        for name, occ in all_occs.items():
            if needle in name:
                matched = occ
                break
        if not matched:
            raise RuntimeError(f"Could not find occurrence for {body_name} (needle: '{needle}')")
        found_occs[body_name] = matched

    # 2. Export STL meshes
    export_mgr = design.exportManager
    exported_meshes = []
    for body_name, (_needle, filename) in BODY_SPECS.items():
        occ = found_occs[body_name]
        mesh_path = os.path.join(mesh_dir, filename)
        opts = export_mgr.createSTLExportOptions(occ, mesh_path)
        opts.isBinaryFormat = True
        opts.isOneFilePerBody = False
        opts.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementLow
        opts.unitType = adsk.fusion.DistanceUnits.MillimeterDistanceUnits
        opts.sendToPrintUtility = False
        if not export_mgr.execute(opts):
            raise RuntimeError(f"Failed to export STL for {body_name} to {mesh_path}")

        # Also write to standard assets/meshes/visual directory
        asset_mesh_path = os.path.join(ASSETS_DIR, filename)
        opts_asset = export_mgr.createSTLExportOptions(occ, asset_mesh_path)
        opts_asset.isBinaryFormat = True
        opts_asset.isOneFilePerBody = False
        opts_asset.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementLow
        opts_asset.unitType = adsk.fusion.DistanceUnits.MillimeterDistanceUnits
        opts_asset.sendToPrintUtility = False
        export_mgr.execute(opts_asset)

        exported_meshes.append(filename)

    # 3. Resolve Joint Positions from Fusion 360 geometryTwoTransform
    all_joints = list(root.allJoints)
    resolved_joint_origins = {}
    for joint_id, jdef in JOINT_DEFS.items():
        matched_joint = None
        for j in all_joints:
            if j.name in jdef["fusion_names"]:
                matched_joint = j
                break
        if matched_joint:
            try:
                t = matched_joint.geometryTwoTransform.translation
                resolved_joint_origins[joint_id] = [t.x * 0.01, t.y * 0.01, t.z * 0.01]
            except Exception:  # nosec B110
                pass

    # Standard fallback coordinates if direct modeling joints lack transform handles
    default_origins = {
        "joint_1_base_to_link_1": [0.539369022999, 0.0639727628473, 0.049],
        "joint_2_link_1_to_link_2": [0.574469022999, 0.0639727628473, 0.106444648759],
        "joint_3_link_2_to_link_3": [0.519438641154, 0.196933979432, 0.291684061805],
        "joint_4_link_3_to_link_4": [0.519119139885, 0.0655199676701, 0.230405822363],
        "joint_5_link_4_to_link_5": [0.539369022999, 0.018286208573, 0.230422993883],
        "right_jaw_slide_joint": [0.539369022999, -0.027946, 0.230436],
        "left_jaw_slide_joint": [0.539369022999, -0.027946, 0.230436],
        "gripper_pinion_joint": [0.539369022999, -0.021654, 0.230436],
    }
    for k, v in default_origins.items():
        if k not in resolved_joint_origins:
            resolved_joint_origins[k] = v

    model_origin_m = list(resolved_joint_origins["joint_1_base_to_link_1"])
    model_origin_m[2] = 0.0

    # 4. Extract Physical Properties for all bodies
    body_physics = {}
    for body_name, occ in found_occs.items():
        body_physics[body_name] = _inertial_attributes(occ)

    # 5. Generate MuJoCo MJCF
    mujoco = ET.Element("mujoco", {"model": "NexArm"})
    ET.SubElement(
        mujoco,
        "compiler",
        {"angle": "radian", "meshdir": "meshes", "autolimits": "true"},
    )
    ET.SubElement(
        mujoco,
        "option",
        {
            "timestep": "0.002",
            "integrator": "implicitfast",
            "gravity": "0 0 -9.81",
            "cone": "elliptic",
            "impratio": "10",
        },
    )

    defaults = ET.SubElement(mujoco, "default")
    robot_defaults = ET.SubElement(defaults, "default", {"class": "nexarm"})
    ET.SubElement(
        robot_defaults,
        "joint",
        {"damping": "0.2", "frictionloss": "0.1", "armature": "0.005"},
    )
    ET.SubElement(robot_defaults, "position", {"kp": "100", "kv": "0.2"})

    gripper_defaults = ET.SubElement(defaults, "default", {"class": "gripper"})
    ET.SubElement(gripper_defaults, "joint", {"damping": "0.01", "frictionloss": "0", "armature": "0"})
    gripper_actuator_defaults = ET.SubElement(defaults, "default", {"class": "gripper_actuator"})
    ET.SubElement(gripper_actuator_defaults, "position", {"kp": "1000", "kv": "5"})

    visual_defaults = ET.SubElement(defaults, "default", {"class": "visual"})
    ET.SubElement(visual_defaults, "geom", {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2"})

    collision_defaults = ET.SubElement(defaults, "default", {"class": "collision"})
    ET.SubElement(
        collision_defaults,
        "geom",
        {
            "contype": "1",
            "conaffinity": "3",
            "group": "3",
            "rgba": "0 0 0 0",
            "friction": "1.1 0.01 0.001",
        },
    )

    assets = ET.SubElement(mujoco, "asset")
    for body_name, (_needle, filename) in BODY_SPECS.items():
        ET.SubElement(
            assets,
            "mesh",
            {"name": body_name, "file": filename, "scale": "0.001 0.001 0.001"},
        )

    worldbody = ET.SubElement(mujoco, "worldbody")
    mount = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "nexarm_mount",
            "pos": _format(tuple(-val for val in model_origin_m)),
            "childclass": "nexarm",
        },
    )

    def _add_mjcf_body(parent_elem, bname, jid=None):
        phys = body_physics[bname]
        body_elem = ET.SubElement(parent_elem, "body", {"name": bname, "pos": "0 0 0", "quat": "1 0 0 0"})
        if jid:
            jdef = JOINT_DEFS[jid]
            j_attrs = {
                "name": jid,
                "pos": _format(resolved_joint_origins[jid]),
                "type": "hinge" if jdef["type"] == "revolute" else "slide",
                "axis": _format(jdef["axis"]),
                "limited": "true",
                "range": _format(jdef["range"]),
            }
            if "jaw" in jid:
                j_attrs["class"] = "gripper"
            ET.SubElement(body_elem, "joint", j_attrs)

        ET.SubElement(body_elem, "geom", {"name": f"{bname}_visual", "class": "visual", "mesh": bname})

        # Inertial
        ixx, iyy, izz, ixy, ixz, iyz = phys["inertia"]
        ET.SubElement(
            body_elem,
            "inertial",
            {
                "mass": f"{phys['mass']:.12g}",
                "pos": _format(phys["com"]),
                "fullinertia": _format((ixx, iyy, izz, ixy, ixz, iyz)),
            },
        )

        # Collision geoms
        if bname in COLLISION_SPECS:
            for idx, col in enumerate(COLLISION_SPECS[bname]):
                col_attrs = dict(col)
                col_attrs["name"] = f"{bname}_collision_{idx}"
                col_attrs["class"] = "collision"
                ET.SubElement(body_elem, "geom", col_attrs)

        return body_elem

    # Build MuJoCo tree
    b_base = _add_mjcf_body(mount, "base_link")
    b_link1 = _add_mjcf_body(b_base, "link_1", "joint_1_base_to_link_1")
    b_link2 = _add_mjcf_body(b_link1, "link_2", "joint_2_link_1_to_link_2")
    b_link3 = _add_mjcf_body(b_link2, "link_3", "joint_3_link_2_to_link_3")
    b_link4 = _add_mjcf_body(b_link3, "link_4", "joint_4_link_3_to_link_4")
    b_link5 = _add_mjcf_body(b_link4, "link_5", "joint_5_link_4_to_link_5")
    b_grip = _add_mjcf_body(b_link5, "link_6_gripper_base")
    _add_mjcf_body(b_grip, "link_6_pinion_gear", "gripper_pinion_joint")
    _add_mjcf_body(b_grip, "link_6_left_jaw", "left_jaw_slide_joint")
    _add_mjcf_body(b_grip, "link_6_right_jaw", "right_jaw_slide_joint")

    # Camera mount on link 4
    b_cam = _add_mjcf_body(b_link4, "cam_mount")
    ET.SubElement(
        b_cam,
        "camera",
        {
            "name": "wrist",
            "mode": "fixed",
            "pos": "0.53937 -0.022 0.279",
            "xyaxes": "-1 0 0 0 -0.656059 0.754710",
            "fovy": "92.2",
        },
    )
    ET.SubElement(
        b_grip,
        "site",
        {
            "name": "gripper_frame",
            "pos": "0.539 -0.09 0.230",
            "size": "0.005",
            "rgba": "0.1 0.8 0.1 1",
        },
    )

    # Contact exclusions
    contact = ET.SubElement(mujoco, "contact")
    for b1, b2 in EXCLUDED_CONTACTS:
        ET.SubElement(contact, "exclude", {"name": f"{b1}_to_{b2}", "body1": b1, "body2": b2})

    # Equality constraint for jaws
    equality = ET.SubElement(mujoco, "equality")
    ET.SubElement(
        equality,
        "joint",
        {
            "name": "gripper_jaws_opposed_1_to_1",
            "joint1": "right_jaw_slide_joint",
            "joint2": "left_jaw_slide_joint",
            "polycoef": "0 -1 0 0 0",
            "solref": "0.002 1",
            "solimp": "0.99 0.999 0.0005 0.5 2",
        },
    )

    # Actuators
    actuators = ET.SubElement(mujoco, "actuator")
    for jid in ARM_JOINTS:
        jdef = JOINT_DEFS[jid]
        ET.SubElement(
            actuators,
            "position",
            {
                "name": f"{jid}_control",
                "class": "nexarm",
                "joint": jid,
                "ctrllimited": "true",
                "ctrlrange": _format(jdef["range"]),
            },
        )
    ET.SubElement(
        actuators,
        "position",
        {
            "name": "gripper_control",
            "class": "gripper_actuator",
            "joint": "right_jaw_slide_joint",
            "ctrllimited": "true",
            "ctrlrange": _format(JOINT_DEFS["right_jaw_slide_joint"]["range"]),
        },
    )

    ET.indent(mujoco, space="  ")
    mjcf_path = os.path.join(OUTPUT_DIR, "NexArm-sim.xml")
    ET.ElementTree(mujoco).write(mjcf_path, encoding="utf-8", xml_declaration=True)
    _write_scene(OUTPUT_DIR, model_origin_m)

    desc_mjcf_dir = os.path.join(DESCRIPTION_DIR, "mjcf")
    os.makedirs(desc_mjcf_dir, exist_ok=True)
    ET.ElementTree(mujoco).write(
        os.path.join(desc_mjcf_dir, "nexarm.xml"), encoding="utf-8", xml_declaration=True
    )
    _write_scene(desc_mjcf_dir, model_origin_m, "nexarm.xml")

    # 6. Generate Standard URDF (nexarm.urdf)
    urdf = ET.Element("robot", {"name": "nexarm"})

    # Material tags
    mat_grey = ET.SubElement(urdf, "material", {"name": "dark_grey"})
    ET.SubElement(mat_grey, "color", {"rgba": "0.2 0.2 0.2 1.0"})
    mat_black = ET.SubElement(urdf, "material", {"name": "black"})
    ET.SubElement(mat_black, "color", {"rgba": "0.1 0.1 0.1 1.0"})

    # Kinematic chain for URDF:
    # link_name -> (parent_link, joint_name_from_parent)
    urdf_chain = [
        ("base_link", None, None),
        ("link_1", "base_link", "joint_1_base_to_link_1"),
        ("link_2", "link_1", "joint_2_link_1_to_link_2"),
        ("link_3", "link_2", "joint_3_link_2_to_link_3"),
        ("link_4", "link_3", "joint_4_link_3_to_link_4"),
        ("link_5", "link_4", "joint_5_link_4_to_link_5"),
        ("link_6_gripper_base", "link_5", "joint_5_to_gripper_fixed"),
        ("link_6_pinion_gear", "link_6_gripper_base", "gripper_pinion_joint"),
        ("link_6_left_jaw", "link_6_gripper_base", "left_jaw_slide_joint"),
        ("link_6_right_jaw", "link_6_gripper_base", "right_jaw_slide_joint"),
        ("cam_mount", "link_4", "joint_4_to_cam_fixed"),
    ]

    link_frames = {
        "base_link": [model_origin_m[0], model_origin_m[1], 0.0],
        "link_1": resolved_joint_origins["joint_1_base_to_link_1"],
        "link_2": resolved_joint_origins["joint_2_link_1_to_link_2"],
        "link_3": resolved_joint_origins["joint_3_link_2_to_link_3"],
        "link_4": resolved_joint_origins["joint_4_link_3_to_link_4"],
        "link_5": resolved_joint_origins["joint_5_link_4_to_link_5"],
        "link_6_gripper_base": resolved_joint_origins["joint_5_link_4_to_link_5"],
        "link_6_pinion_gear": resolved_joint_origins["gripper_pinion_joint"],
        "link_6_left_jaw": resolved_joint_origins["left_jaw_slide_joint"],
        "link_6_right_jaw": resolved_joint_origins["right_jaw_slide_joint"],
        "cam_mount": resolved_joint_origins["joint_4_link_3_to_link_4"],
    }

    for lname, pname, jname in urdf_chain:
        link_elem = ET.SubElement(urdf, "link", {"name": lname})
        phys = body_physics[lname]
        com = phys["com"]
        mass = phys["mass"]
        ixx, iyy, izz, ixy, ixz, iyz = phys["inertia"]
        l_frame = link_frames[lname]

        # Inertial
        inertial = ET.SubElement(link_elem, "inertial")
        ET.SubElement(
            inertial,
            "origin",
            {
                "xyz": f"{com[0] - l_frame[0]:.6g} {com[1] - l_frame[1]:.6g} {com[2] - l_frame[2]:.6g}",
                "rpy": "0 0 0",
            },
        )
        ET.SubElement(inertial, "mass", {"value": f"{mass:.6g}"})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": f"{ixx:.6e}",
                "ixy": f"{ixy:.6e}",
                "ixz": f"{ixz:.6e}",
                "iyy": f"{iyy:.6e}",
                "iyz": f"{iyz:.6e}",
                "izz": f"{izz:.6e}",
            },
        )

        # Visual
        visual = ET.SubElement(link_elem, "visual")
        ET.SubElement(
            visual,
            "origin",
            {"xyz": f"{-l_frame[0]:.6g} {-l_frame[1]:.6g} {-l_frame[2]:.6g}", "rpy": "0 0 0"},
        )
        geom_v = ET.SubElement(visual, "geometry")
        ET.SubElement(
            geom_v, "mesh", {"filename": f"meshes/{BODY_SPECS[lname][1]}", "scale": "0.001 0.001 0.001"}
        )
        ET.SubElement(visual, "material", {"name": "dark_grey"})

        # Collision
        collision = ET.SubElement(link_elem, "collision")
        ET.SubElement(
            collision,
            "origin",
            {"xyz": f"{-l_frame[0]:.6g} {-l_frame[1]:.6g} {-l_frame[2]:.6g}", "rpy": "0 0 0"},
        )
        geom_c = ET.SubElement(collision, "geometry")
        ET.SubElement(
            geom_c, "mesh", {"filename": f"meshes/{BODY_SPECS[lname][1]}", "scale": "0.001 0.001 0.001"}
        )

        # Joint connecting to parent
        if pname is not None:
            joint_elem = ET.SubElement(urdf, "joint", {"name": jname})
            ET.SubElement(joint_elem, "parent", {"link": pname})
            ET.SubElement(joint_elem, "child", {"link": lname})

            p_frame = link_frames[pname]
            rel_xyz = [l_frame[i] - p_frame[i] for i in range(3)]
            rel_xyz_str = f"{rel_xyz[0]:.6g} {rel_xyz[1]:.6g} {rel_xyz[2]:.6g}"

            if jname.endswith("_fixed"):
                joint_elem.attrib["type"] = "fixed"
                ET.SubElement(joint_elem, "origin", {"xyz": rel_xyz_str, "rpy": "0 0 0"})
            else:
                jdef = JOINT_DEFS[jname]
                joint_elem.attrib["type"] = jdef["type"]
                ET.SubElement(joint_elem, "origin", {"xyz": rel_xyz_str, "rpy": "0 0 0"})
                ET.SubElement(joint_elem, "axis", {"xyz": _format(jdef["axis"])})
                rmin, rmax = jdef["range"]
                ET.SubElement(
                    joint_elem,
                    "limit",
                    {
                        "lower": f"{rmin:.6g}",
                        "upper": f"{rmax:.6g}",
                        "effort": "5.0",
                        "velocity": "3.14",
                    },
                )
                if jname == "left_jaw_slide_joint":
                    ET.SubElement(
                        joint_elem,
                        "mimic",
                        {"joint": "right_jaw_slide_joint", "multiplier": "-1", "offset": "0"},
                    )

    ET.indent(urdf, space="  ")
    urdf_path = os.path.join(OUTPUT_DIR, "nexarm.urdf")
    ET.ElementTree(urdf).write(urdf_path, encoding="utf-8", xml_declaration=True)

    desc_urdf_dir = os.path.join(DESCRIPTION_DIR, "urdf")
    os.makedirs(desc_urdf_dir, exist_ok=True)
    ET.ElementTree(urdf).write(
        os.path.join(desc_urdf_dir, "nexarm.urdf"), encoding="utf-8", xml_declaration=True
    )

    result = {
        "success": True,
        "document": app.activeDocument.name,
        "outputDirectory": OUTPUT_DIR,
        "descriptionDirectory": DESCRIPTION_DIR,
        "assetsDirectory": ASSETS_DIR,
        "mjcf": mjcf_path,
        "scene": os.path.join(OUTPUT_DIR, "scene.xml"),
        "urdf": urdf_path,
        "exportedMeshCount": len(exported_meshes),
        "exportedMeshes": exported_meshes,
    }
    print(json.dumps(result, indent=2))
