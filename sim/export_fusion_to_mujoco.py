"""Export the active Fusion 360 NexArm assembly as MuJoCo MJCF.

This script is executed inside Fusion 360 through the Fusion MCP script runner.
It writes through WSL's UNC share so the generated model lands in this checkout.
"""

import json
import math
import os
import xml.etree.ElementTree as ET

import adsk.core
import adsk.fusion

OUTPUT_DIR = (
    r"\\wsl.localhost\Ubuntu\home\thanh\code\vinuni\lerobot-nexarm"
    r"\sim\fusion_export"
)

BODY_SPECS = {
    "base_link": ("base_link:1", "base_link.stl"),
    "link_1": ("link_1:1", "link_1.stl"),
    "link_2": ("link_2:1", "link_2.stl"),
    "link_3": ("link_3:1", "link_3.stl"),
    "link_4": ("link_4:1", "link_4.stl"),
    "link_5": ("link_5:1", "link_5.stl"),
    "link_6_gripper_base": (
        "link_6:1+gripper_base:1",
        "link_6_gripper_base.stl",
    ),
    "link_6_pinion_gear": (
        "link_6:1+pinion_gear:1",
        "link_6_pinion_gear.stl",
    ),
    "link_6_left_jaw": ("link_6:1+left_jaw:1", "link_6_left_jaw.stl"),
    "link_6_right_jaw": ("link_6:1+right_jaw:1", "link_6_right_jaw.stl"),
    "cam_mount": ("cam_mount:1", "cam_mount.stl"),
}

ARM_JOINTS = [
    "joint_1_base_to_link_1",
    "joint_2_link_1_to_link_2",
    "joint_3_link_2_to_link_3",
    "joint_4_link_3_to_link_4",
    "joint_5_link_4_to_link_5",
]

# Stable primitive contact geometry. The detailed STL files stay visual-only:
# using the exported meshes directly for contact creates overlapping convex
# hulls and makes the arm oscillate. Coordinates use the world-aligned link
# frames emitted by this exporter.
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
        {"type": "box", "pos": "0.53937 -0.02192 0.23047", "size": "0.010 0.023 0.047"},
    ],
    "link_6_left_jaw": [
        {"type": "box", "pos": "0.50559 -0.03991 0.22931", "size": "0.017 0.029 0.020"},
    ],
    "link_6_right_jaw": [
        {"type": "box", "pos": "0.57314 -0.03991 0.23155", "size": "0.017 0.029 0.020"},
    ],
}


def _format(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _point_m(point: adsk.core.Point3D) -> list[float]:
    # Fusion's API length unit is centimeters; MuJoCo uses meters.
    return [point.x * 0.01, point.y * 0.01, point.z * 0.01]


def _vector(vector: adsk.core.Vector3D) -> list[float]:
    values = [vector.x, vector.y, vector.z]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _inertial_attributes(
    occurrence: adsk.fusion.Occurrence,
) -> dict[str, str]:
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
        raise RuntimeError(f"Unable to calculate inertia for {occurrence.fullPathName}")

    # Fusion reports kg*cm^2 at the world origin. Shift the tensor to the
    # center of mass, retain the world-aligned body frame, then convert to
    # kg*m^2. Fusion's off-diagonal values are already inertia-tensor terms.
    ixx = (ixx_origin - mass * (y * y + z * z)) * 1e-4
    iyy = (iyy_origin - mass * (x * x + z * z)) * 1e-4
    izz = (izz_origin - mass * (x * x + y * y)) * 1e-4
    ixy = (ixy_origin + mass * x * y) * 1e-4
    ixz = (ixz_origin + mass * x * z) * 1e-4
    iyz = (iyz_origin + mass * y * z) * 1e-4

    return {
        "mass": f"{mass:.12g}",
        "pos": _format(_point_m(center)),
        "fullinertia": _format((ixx, iyy, izz, ixy, ixz, iyz)),
    }


def _joint_origin_m(joint: object) -> list[float]:
    if joint.objectType == "adsk::fusion::Joint":
        return _point_m(joint.geometryOneTransform.translation)
    return _point_m(joint.geometry.origin)


def _joint_attributes(joint: object) -> dict[str, str]:
    motion = joint.jointMotion
    attributes = {
        "name": joint.name,
        "pos": _format(_joint_origin_m(joint)),
    }

    if motion.objectType == "adsk::fusion::RevoluteJointMotion":
        revolute = adsk.fusion.RevoluteJointMotion.cast(motion)
        limits = revolute.rotationLimits
        attributes.update(
            {
                "type": "hinge",
                "axis": _format(_vector(revolute.rotationAxisVector)),
            }
        )
        if limits.isMinimumValueEnabled and limits.isMaximumValueEnabled:
            attributes.update(
                {
                    "limited": "true",
                    "range": _format((limits.minimumValue, limits.maximumValue)),
                }
            )
    elif motion.objectType == "adsk::fusion::SliderJointMotion":
        slider = adsk.fusion.SliderJointMotion.cast(motion)
        limits = slider.slideLimits
        attributes.update(
            {
                "type": "slide",
                "axis": _format(_vector(slider.slideDirectionVector)),
            }
        )
        if limits.isMinimumValueEnabled and limits.isMaximumValueEnabled:
            attributes.update(
                {
                    "limited": "true",
                    "range": _format((limits.minimumValue * 0.01, limits.maximumValue * 0.01)),
                }
            )
    else:
        raise RuntimeError(f"Unsupported movable joint type for {joint.name}: {motion.objectType}")
    return attributes


def _native_joint_config(joint: object) -> dict[str, object]:
    native = joint.nativeObject if joint.nativeObject else joint
    attribute = native.attributes.itemByName("mujoco", "config")
    if not attribute:
        raise RuntimeError(f"Joint {joint.name} has no mujoco/config metadata")
    return json.loads(attribute.value)


def _add_body(
    parent: ET.Element,
    name: str,
    occurrence: adsk.fusion.Occurrence,
    joint: object | None = None,
) -> ET.Element:
    body = ET.SubElement(
        parent,
        "body",
        {"name": name, "pos": "0 0 0", "quat": "1 0 0 0"},
    )
    if joint is not None:
        ET.SubElement(body, "joint", _joint_attributes(joint))
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{name}_visual",
            "type": "mesh",
            "mesh": name,
            "class": "visual",
        },
    )
    for index, collision_attributes in enumerate(COLLISION_SPECS.get(name, [])):
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{name}_collision_{index}",
                "class": "collision",
                **collision_attributes,
            },
        )
    ET.SubElement(body, "inertial", _inertial_attributes(occurrence))
    return body


def _write_scene(output_dir: str) -> None:
    scene = ET.Element("mujoco", {"model": "NexArm-scene"})
    ET.SubElement(scene, "include", {"file": "NexArm-sim.xml"})
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", {"offwidth": "640", "offheight": "480"})
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"},
    )
    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"pos": "0 0 3", "dir": "0 0 -1", "directional": "true"},
    )
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
            "pos": "0.9 -0.55 0.55",
            "xyaxes": "0.857493 0.514496 0 -0.240504 0.400841 0.884016",
            "fovy": "55",
        },
    )
    cube = ET.SubElement(worldbody, "body", {"name": "cube", "pos": "0.539 -0.18 0.025"})
    ET.SubElement(cube, "freejoint", {"name": "cube_joint"})
    ET.SubElement(
        cube,
        "geom",
        {
            "name": "cube_collision",
            "type": "box",
            "size": "0.018 0.018 0.018",
            "mass": "0.05",
            "rgba": "0.85 0.15 0.1 1",
            "friction": "1.2 0.01 0.001",
            "contype": "2",
            "conaffinity": "3",
        },
    )
    ET.indent(scene, space="  ")
    ET.ElementTree(scene).write(
        os.path.join(output_dir, "scene.xml"),
        encoding="utf-8",
        xml_declaration=True,
    )


def run(_context: str):
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise RuntimeError("The active Fusion document is not a design")

    root = design.rootComponent
    root_attribute = root.attributes.itemByName("mujoco", "config")
    if not root_attribute:
        raise RuntimeError("The root component has no mujoco/config metadata")
    model_config = json.loads(root_attribute.value)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mesh_dir = os.path.join(OUTPUT_DIR, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    occurrences = {occ.fullPathName: occ for occ in root.allOccurrences}
    missing_occurrences = sorted(path for path, _filename in BODY_SPECS.values() if path not in occurrences)
    if missing_occurrences:
        raise RuntimeError(f"Missing required occurrences: {missing_occurrences}")

    joints = {joint.name: joint for joint in list(root.allJoints) + list(root.allAsBuiltJoints)}
    required_joints = set(ARM_JOINTS) | {
        "gripper_pinion_joint",
        "left_jaw_slide_joint",
        "right_jaw_slide_joint",
    }
    missing_joints = sorted(required_joints - joints.keys())
    if missing_joints:
        raise RuntimeError(f"Missing required joints: {missing_joints}")

    export_manager = design.exportManager
    exported_meshes = []
    for body_name, (path, filename) in BODY_SPECS.items():
        mesh_path = os.path.join(mesh_dir, filename)
        options = export_manager.createSTLExportOptions(occurrences[path], mesh_path)
        if not options:
            raise RuntimeError(f"Unable to create STL export options for {path}")
        options.isBinaryFormat = True
        options.isOneFilePerBody = False
        options.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementMedium
        options.unitType = adsk.fusion.DistanceUnits.MillimeterDistanceUnits
        options.sendToPrintUtility = False
        if not export_manager.execute(options):
            raise RuntimeError(f"STL export failed for {path}")
        exported_meshes.append({"name": body_name, "path": mesh_path})

    mujoco = ET.Element("mujoco", {"model": model_config.get("model", "NexArm")})
    ET.SubElement(
        mujoco,
        "compiler",
        {"angle": "radian", "meshdir": "meshes", "autolimits": "true"},
    )
    ET.SubElement(
        mujoco,
        "option",
        {"timestep": "0.002", "integrator": "implicitfast", "gravity": "0 0 -9.81"},
    )

    defaults = ET.SubElement(mujoco, "default")
    robot_defaults = ET.SubElement(defaults, "default", {"class": "nexarm"})
    ET.SubElement(
        robot_defaults,
        "joint",
        {"damping": "0.2", "frictionloss": "0.1", "armature": "0.005"},
    )
    actuator_defaults = model_config.get("default_actuator", {})
    ET.SubElement(
        robot_defaults,
        "position",
        {
            "kp": str(actuator_defaults.get("kp", 100.0)),
            "kv": str(actuator_defaults.get("kv", 0.2)),
        },
    )
    visual_defaults = ET.SubElement(defaults, "default", {"class": "visual"})
    ET.SubElement(
        visual_defaults,
        "geom",
        {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2"},
    )
    collision_defaults = ET.SubElement(defaults, "default", {"class": "collision"})
    ET.SubElement(
        collision_defaults,
        "geom",
        {
            "contype": "1",
            "conaffinity": "2",
            "group": "3",
            "rgba": "0 0 0 0",
            "friction": "1.1 0.01 0.001",
        },
    )

    assets = ET.SubElement(mujoco, "asset")
    for body_name, (_path, filename) in BODY_SPECS.items():
        ET.SubElement(
            assets,
            "mesh",
            {
                "name": body_name,
                "file": filename,
                "scale": "0.001 0.001 0.001",
            },
        )

    worldbody = ET.SubElement(mujoco, "worldbody")
    base = _add_body(
        worldbody,
        "base_link",
        occurrences[BODY_SPECS["base_link"][0]],
    )
    link_1 = _add_body(
        base,
        "link_1",
        occurrences[BODY_SPECS["link_1"][0]],
        joints[ARM_JOINTS[0]],
    )
    link_2 = _add_body(
        link_1,
        "link_2",
        occurrences[BODY_SPECS["link_2"][0]],
        joints[ARM_JOINTS[1]],
    )
    link_3 = _add_body(
        link_2,
        "link_3",
        occurrences[BODY_SPECS["link_3"][0]],
        joints[ARM_JOINTS[2]],
    )
    link_4 = _add_body(
        link_3,
        "link_4",
        occurrences[BODY_SPECS["link_4"][0]],
        joints[ARM_JOINTS[3]],
    )
    link_5 = _add_body(
        link_4,
        "link_5",
        occurrences[BODY_SPECS["link_5"][0]],
        joints[ARM_JOINTS[4]],
    )
    gripper_base = _add_body(
        link_5,
        "link_6_gripper_base",
        occurrences[BODY_SPECS["link_6_gripper_base"][0]],
    )
    _add_body(
        gripper_base,
        "link_6_pinion_gear",
        occurrences[BODY_SPECS["link_6_pinion_gear"][0]],
        joints["gripper_pinion_joint"],
    )
    _add_body(
        gripper_base,
        "link_6_left_jaw",
        occurrences[BODY_SPECS["link_6_left_jaw"][0]],
        joints["left_jaw_slide_joint"],
    )
    _add_body(
        gripper_base,
        "link_6_right_jaw",
        occurrences[BODY_SPECS["link_6_right_jaw"][0]],
        joints["right_jaw_slide_joint"],
    )
    cam_mount = _add_body(
        link_4,
        "cam_mount",
        occurrences[BODY_SPECS["cam_mount"][0]],
    )
    ET.SubElement(
        cam_mount,
        "camera",
        {
            "name": "wrist",
            "mode": "fixed",
            "pos": "0.539 0.015 0.295",
            "xyaxes": "0 -0.485643 0.874157 1 0 0",
            "fovy": "70",
        },
    )
    ET.SubElement(
        gripper_base,
        "site",
        {
            "name": "gripper_frame",
            "pos": "0.539 -0.09 0.230",
            "size": "0.005",
            "rgba": "0.1 0.8 0.1 1",
        },
    )

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
    right_limits = adsk.fusion.SliderJointMotion.cast(joints["right_jaw_slide_joint"].jointMotion).slideLimits
    right_travel_m = (right_limits.maximumValue - right_limits.minimumValue) * 0.01
    ET.SubElement(
        equality,
        "joint",
        {
            "name": "gripper_pinion_to_right_jaw",
            "joint1": "gripper_pinion_joint",
            "joint2": "right_jaw_slide_joint",
            "polycoef": f"0 {math.tau / right_travel_m:.12g} 0 0 0",
            "solref": "0.002 1",
            "solimp": "0.99 0.999 0.0005 0.5 2",
        },
    )

    actuators = ET.SubElement(mujoco, "actuator")
    actuator_count = 0
    for joint_name in ARM_JOINTS + ["right_jaw_slide_joint"]:
        joint = joints[joint_name]
        joint_config = _native_joint_config(joint)
        if not joint_config.get("actuated", False):
            continue
        joint_attributes = _joint_attributes(joint)
        actuator_name = (
            "gripper_control"
            if joint_name == "right_jaw_slide_joint"
            else joint_name.replace("joint_", "joint_", 1) + "_control"
        )
        attributes = {
            "name": actuator_name,
            "class": "nexarm",
            "joint": joint_name,
        }
        if "range" in joint_attributes:
            attributes.update(
                {
                    "ctrllimited": "true",
                    "ctrlrange": joint_attributes["range"],
                }
            )
        ET.SubElement(actuators, joint_config.get("actuator", "position"), attributes)
        actuator_count += 1

    ET.indent(mujoco, space="  ")
    model_path = os.path.join(OUTPUT_DIR, "NexArm-sim.xml")
    ET.ElementTree(mujoco).write(model_path, encoding="utf-8", xml_declaration=True)
    _write_scene(OUTPUT_DIR)

    print(
        json.dumps(
            {
                "document": app.activeDocument.name,
                "outputDirectory": OUTPUT_DIR,
                "model": model_path,
                "meshCount": len(exported_meshes),
                "actuatorCount": actuator_count,
                "jointCount": len(required_joints),
                "gripperOpeningMm": right_travel_m * 2000.0,
            },
            indent=2,
        )
    )
