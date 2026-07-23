#!/usr/bin/env python
"""Generate MuJoCo, URDF, and ASCII USD descriptions from spec.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_SPEC = ROOT / "spec.json"
GENERATED = ROOT / "generated"


def _values(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _rpy_to_quat(rpy: list[float]) -> str:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return _values(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _source_joint_contract(
    spec: dict[str, Any], spec_path: Path
) -> dict[str, tuple[list[float], list[float]]]:
    source = (spec_path.parent / spec["source"]["arm_description"]).resolve()
    root = ET.parse(source).getroot()
    result: dict[str, tuple[list[float], list[float]]] = {}
    for joint in root.findall(".//joint"):
        if "axis" in joint.attrib and "range" in joint.attrib:
            result[joint.attrib["name"]] = (
                [float(value) for value in joint.attrib["axis"].split()],
                [float(value) for value in joint.attrib["range"].split()],
            )
    return result


def validate_source(spec: dict[str, Any], spec_path: Path) -> None:
    source = _source_joint_contract(spec, spec_path)
    for joint in spec["joints"]:
        actual = source.get(joint["source_joint"])
        if actual is None:
            raise ValueError(f"Fusion asset is missing {joint['source_joint']}")
        axis, limits = actual
        if any(abs(a - b) > 1e-6 for a, b in zip(axis, joint["axis"], strict=True)):
            raise ValueError(f"axis drift for {joint['source_joint']}: {axis} != {joint['axis']}")
        if any(abs(a - b) > 1e-6 for a, b in zip(limits, joint["range"], strict=True)):
            raise ValueError(f"limit drift for {joint['source_joint']}: {limits} != {joint['range']}")


def _add_arm_mjcf(parent: ET.Element, side: str, spec: dict[str, Any]) -> None:
    mount = spec["arms"][side]
    arm = ET.SubElement(
        parent,
        "body",
        name=f"{side}_arm_mount",
        pos=_values(mount["mount_xyz_m"]),
        quat=_rpy_to_quat(mount["mount_rpy_rad"]),
    )
    lengths = tuple(spec["kinematics"]["link_lengths_m"])
    current = arm
    for index, (joint, length) in enumerate(zip(spec["joints"][:5], lengths, strict=True), 1):
        body = ET.SubElement(current, "body", name=f"{side}_link_{index}", pos="0 0 0", gravcomp="1")
        ET.SubElement(
            body,
            "joint",
            name=f"{side}_{joint['feature']}",
            type="hinge",
            axis=_values(joint["axis"]),
            range=_values(joint["range"]),
            limited="true",
            damping="0.3",
            armature="0.005",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{side}_link_{index}_collision",
            type="capsule",
            fromto=f"0 0 0 0 {length:.12g} 0",
            size="0.025",
            mass="0.25",
            rgba="0.72 0.75 0.78 1",
            contype="1",
            conaffinity="6",
        )
        current = ET.SubElement(body, "body", name=f"{side}_link_{index}_tip", pos=f"0 {length:.12g} 0")
    gripper = spec["joints"][5]
    gripper_body = ET.SubElement(current, "body", name=f"{side}_gripper", pos="0 0 0", gravcomp="1")
    ET.SubElement(gripper_body, "geom", type="box", size="0.025 0.045 0.025", mass="0.2")
    ET.SubElement(
        gripper_body,
        "site",
        name=f"{side}_tcp",
        pos="0 0.055 0",
        size="0.005",
        rgba="0.1 0.8 0.1 1",
    )
    right_jaw = ET.SubElement(gripper_body, "body", name=f"{side}_right_jaw")
    ET.SubElement(
        right_jaw,
        "joint",
        name=f"{side}_gripper",
        type="slide",
        axis=_values(gripper["axis"]),
        range=_values(gripper["range"]),
        limited="true",
        damping="0.2",
    )
    ET.SubElement(
        right_jaw,
        "geom",
        name=f"{side}_right_jaw_collision",
        type="box",
        pos="0 0.055 0",
        size="0.008 0.035 0.04",
        friction="3 0.01 0.001",
        condim="4",
    )
    left_jaw = ET.SubElement(gripper_body, "body", name=f"{side}_left_jaw")
    ET.SubElement(
        left_jaw,
        "joint",
        name=f"{side}_gripper_mimic",
        type="slide",
        axis=_values(gripper["axis"]),
        range="0 0.0255",
        limited="true",
    )
    ET.SubElement(
        left_jaw,
        "geom",
        name=f"{side}_left_jaw_collision",
        type="box",
        pos="0 0.055 0",
        size="0.008 0.035 0.04",
        friction="3 0.01 0.001",
        condim="4",
    )
    camera = spec["cameras"].get(f"{side}_wrist")
    if camera:
        ET.SubElement(
            gripper_body,
            "camera",
            name=f"{side}_wrist",
            pos=_values(camera["xyz_m"]),
            quat=_rpy_to_quat(camera["rpy_rad"]),
            fovy=str(camera["fovy_deg"]),
        )


def generate_mjcf(spec: dict[str, Any]) -> str:
    model = ET.Element("mujoco", model=spec["name"])
    ET.SubElement(model, "compiler", angle="radian", autolimits="true")
    ET.SubElement(model, "option", timestep="0.002", gravity="0 0 -9.81")
    default = ET.SubElement(model, "default")
    ET.SubElement(default, "position", kp="100", kv="0.3")
    world = ET.SubElement(model, "worldbody")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.05", contype="4", conaffinity="3")
    table_top = spec["task"]["table_top_m"]
    cube_half = spec["task"]["cube_half_size_m"]
    table = ET.SubElement(world, "body", name="table", pos=f"0.55 0 {table_top - 0.03:.12g}")
    ET.SubElement(
        table,
        "geom",
        name="table_collision",
        type="box",
        size="0.45 0.55 0.03",
        friction="0.2 0.005 0.0001",
        contype="4",
        conaffinity="3",
    )
    base = ET.SubElement(world, "body", name="mobile_base", pos="0 0 0.09")
    ET.SubElement(base, "joint", name="base_x", type="slide", axis="1 0 0", damping="1")
    ET.SubElement(base, "joint", name="base_y", type="slide", axis="0 1 0", damping="1")
    ET.SubElement(base, "joint", name="base_yaw", type="hinge", axis="0 0 1", damping="1")
    footprint = spec["base"]["footprint_m"]
    ET.SubElement(
        base,
        "geom",
        name="base_collision",
        type="box",
        size=_values([footprint[0] / 2, footprint[1] / 2, footprint[2] / 2]),
        mass=str(spec["base"]["mass_kg"]),
        contype="1",
        conaffinity="6",
    )
    lift = ET.SubElement(base, "body", name="lift_carriage", gravcomp="1")
    ET.SubElement(
        lift,
        "joint",
        name="lift_axis",
        type="slide",
        axis=_values(spec["lift"]["axis"]),
        range=_values([value / 1000 for value in spec["lift"]["range_mm"]]),
        limited="true",
        damping="4",
    )
    ET.SubElement(lift, "geom", type="box", size="0.06 0.28 0.08", mass=str(spec["lift"]["mass_kg"]))
    front = spec["cameras"]["front"]
    ET.SubElement(
        lift,
        "camera",
        name="front",
        pos=_values(front["xyz_m"]),
        quat=_rpy_to_quat(front["rpy_rad"]),
        fovy=str(front["fovy_deg"]),
    )
    _add_arm_mjcf(lift, "left", spec)
    _add_arm_mjcf(lift, "right", spec)
    cube = ET.SubElement(world, "body", name="cube", pos=f"0.55 0.15 {table_top + cube_half:.12g}")
    ET.SubElement(cube, "freejoint", name="cube_joint")
    ET.SubElement(
        cube,
        "geom",
        name="cube_collision",
        type="box",
        size=_values([cube_half] * 3),
        mass="0.05",
        rgba="0.85 0.15 0.1 1",
        contype="2",
        conaffinity="5",
    )
    target = ET.SubElement(world, "body", name="target_zone", pos=f"0.55 -0.15 {table_top + 0.002:.12g}")
    ET.SubElement(
        target,
        "geom",
        name="target_zone_visual",
        type="cylinder",
        size=f"{spec['task']['target_radius_m']:.12g} 0.002",
        rgba="0.1 0.8 0.2 0.45",
        contype="0",
        conaffinity="0",
    )
    equality = ET.SubElement(model, "equality")
    for side in ("left", "right"):
        ET.SubElement(
            equality,
            "joint",
            name=f"{side}_gripper_mimic_constraint",
            joint1=f"{side}_gripper_mimic",
            joint2=f"{side}_gripper",
            polycoef="0 -1 0 0 0",
        )
    contact = ET.SubElement(model, "contact")
    for side in ("left", "right"):
        ET.SubElement(
            contact,
            "exclude",
            name=f"{side}_jaw_pair_exclude",
            body1=f"{side}_left_jaw",
            body2=f"{side}_right_jaw",
        )
    actuator = ET.SubElement(model, "actuator")
    for side in ("left", "right"):
        for joint in spec["joints"]:
            ET.SubElement(
                actuator,
                "position",
                name=f"{side}_{joint['feature']}_control",
                joint=f"{side}_{joint['feature']}",
                ctrlrange=_values(joint["range"]),
                ctrllimited="true",
            )
    ET.SubElement(
        actuator,
        "position",
        name="lift_axis_control",
        joint="lift_axis",
        ctrlrange=_values([value / 1000 for value in spec["lift"]["range_mm"]]),
        ctrllimited="true",
        kp="300",
        kv="10",
    )
    ET.indent(model)
    return ET.tostring(model, encoding="unicode", xml_declaration=True)


def generate_urdf(spec: dict[str, Any]) -> str:
    robot = ET.Element("robot", name=spec["name"])
    ET.SubElement(robot, "link", name="world")
    ET.SubElement(robot, "link", name="base_x_link")
    x_joint = ET.SubElement(robot, "joint", name="base_x", type="prismatic")
    ET.SubElement(x_joint, "parent", link="world")
    ET.SubElement(x_joint, "child", link="base_x_link")
    ET.SubElement(x_joint, "axis", xyz="1 0 0")
    ET.SubElement(x_joint, "limit", lower="-10", upper="10", effort="1000", velocity="0.5")
    ET.SubElement(robot, "link", name="base_y_link")
    y_joint = ET.SubElement(robot, "joint", name="base_y", type="prismatic")
    ET.SubElement(y_joint, "parent", link="base_x_link")
    ET.SubElement(y_joint, "child", link="base_y_link")
    ET.SubElement(y_joint, "axis", xyz="0 1 0")
    ET.SubElement(y_joint, "limit", lower="-10", upper="10", effort="1000", velocity="0.5")
    ET.SubElement(robot, "link", name="mobile_base")
    base_joint = ET.SubElement(robot, "joint", name="base_yaw", type="continuous")
    ET.SubElement(base_joint, "parent", link="base_y_link")
    ET.SubElement(base_joint, "child", link="mobile_base")
    ET.SubElement(base_joint, "axis", xyz="0 0 1")
    ET.SubElement(base_joint, "limit", effort="1000", velocity="1.57079632679")
    ET.SubElement(robot, "link", name="lift_carriage")
    lift_joint = ET.SubElement(robot, "joint", name="lift_axis", type="prismatic")
    ET.SubElement(lift_joint, "parent", link="mobile_base")
    ET.SubElement(lift_joint, "child", link="lift_carriage")
    ET.SubElement(lift_joint, "axis", xyz=_values(spec["lift"]["axis"]))
    lift_range = spec["lift"]["range_mm"]
    ET.SubElement(
        lift_joint,
        "limit",
        lower=str(lift_range[0] / 1000),
        upper=str(lift_range[1] / 1000),
        effort="500",
        velocity="0.25",
    )
    for side in ("left", "right"):
        previous = "lift_carriage"
        mount = spec["arms"][side]
        mount_link = f"{side}_arm_mount"
        ET.SubElement(robot, "link", name=mount_link)
        fixed = ET.SubElement(robot, "joint", name=f"{side}_arm_mount_joint", type="fixed")
        ET.SubElement(fixed, "parent", link=previous)
        ET.SubElement(fixed, "child", link=mount_link)
        ET.SubElement(fixed, "origin", xyz=_values(mount["mount_xyz_m"]), rpy=_values(mount["mount_rpy_rad"]))
        previous = mount_link
        for index, joint in enumerate(spec["joints"], 1):
            link = f"{side}_link_{index}" if joint["feature"] != "gripper" else f"{side}_gripper"
            ET.SubElement(robot, "link", name=link)
            node = ET.SubElement(
                robot,
                "joint",
                name=f"{side}_{joint['feature']}",
                type="prismatic" if joint["feature"] == "gripper" else "revolute",
            )
            ET.SubElement(node, "parent", link=previous)
            ET.SubElement(node, "child", link=link)
            ET.SubElement(node, "origin", xyz="0 0.08 0", rpy="0 0 0")
            ET.SubElement(node, "axis", xyz=_values(joint["axis"]))
            ET.SubElement(
                node,
                "limit",
                lower=str(joint["range"][0]),
                upper=str(joint["range"][1]),
                effort="100",
                velocity="3",
            )
            previous = link
        mimic_link = f"{side}_left_jaw"
        ET.SubElement(robot, "link", name=mimic_link)
        mimic = ET.SubElement(robot, "joint", name=f"{side}_gripper_mimic", type="prismatic")
        ET.SubElement(mimic, "parent", link=previous)
        ET.SubElement(mimic, "child", link=mimic_link)
        ET.SubElement(mimic, "axis", xyz=_values(spec["joints"][5]["axis"]))
        ET.SubElement(mimic, "limit", lower="0", upper="0.0255", effort="100", velocity="0.1")
        ET.SubElement(mimic, "mimic", joint=f"{side}_gripper", multiplier="-1", offset="0")
    ET.indent(robot)
    return ET.tostring(robot, encoding="unicode", xml_declaration=True)


def generate_usda(spec: dict[str, Any]) -> str:
    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{spec["name"]}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        f'def Xform "{spec["name"]}" (',
        '    prepend apiSchemas = ["PhysicsArticulationRootAPI"]',
        ")",
        "{",
        '    def Xform "world_anchor" {}',
        '    def Xform "base_x_link" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}',
        '    def PhysicsPrismaticJoint "base_x"',
        "    {",
        f"        rel physics:body0 = </{spec['name']}/world_anchor>",
        f"        rel physics:body1 = </{spec['name']}/base_x_link>",
        '        uniform token physics:axis = "X"',
        "        float physics:lowerLimit = -10",
        "        float physics:upperLimit = 10",
        "    }",
        '    def Xform "base_y_link" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}',
        '    def PhysicsPrismaticJoint "base_y"',
        "    {",
        f"        rel physics:body0 = </{spec['name']}/base_x_link>",
        f"        rel physics:body1 = </{spec['name']}/base_y_link>",
        '        uniform token physics:axis = "Y"',
        "        float physics:lowerLimit = -10",
        "        float physics:upperLimit = 10",
        "    }",
        '    def Xform "mobile_base" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}',
        '    def PhysicsRevoluteJoint "base_yaw"',
        "    {",
        f"        rel physics:body0 = </{spec['name']}/base_y_link>",
        f"        rel physics:body1 = </{spec['name']}/mobile_base>",
        '        uniform token physics:axis = "Z"',
        "    }",
        '    def Xform "lift_carriage" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {}',
        '    def PhysicsPrismaticJoint "lift_axis"',
        "    {",
        f"        rel physics:body0 = </{spec['name']}/mobile_base>",
        f"        rel physics:body1 = </{spec['name']}/lift_carriage>",
        '        uniform token physics:axis = "Z"',
        f"        float physics:lowerLimit = {spec['lift']['range_mm'][0] / 1000}",
        f"        float physics:upperLimit = {spec['lift']['range_mm'][1] / 1000}",
        "    }",
    ]
    for side in ("left", "right"):
        xyz = spec["arms"][side]["mount_xyz_m"]
        lines.extend(
            [
                f'    def Xform "{side}_arm_mount" (prepend apiSchemas = ["PhysicsRigidBodyAPI"])',
                "    {",
                f"        double3 xformOp:translate = ({xyz[0]}, {xyz[1]}, {xyz[2]})",
                '        uniform token[] xformOpOrder = ["xformOp:translate"]',
                "    }",
                f'    def PhysicsFixedJoint "{side}_mount_joint"',
                "    {",
                f"        rel physics:body0 = </{spec['name']}/lift_carriage>",
                f"        rel physics:body1 = </{spec['name']}/{side}_arm_mount>",
                "    }",
            ]
        )
        parent = f"{side}_arm_mount"
        for joint in spec["joints"]:
            child = f"{side}_{joint['feature']}_link"
            joint_type = "PhysicsPrismaticJoint" if joint["feature"] == "gripper" else "PhysicsRevoluteJoint"
            axis_index = max(range(3), key=lambda index: abs(joint["axis"][index]))
            axis = ("X", "Y", "Z")[axis_index]
            lines.extend(
                [
                    f'    def Xform "{child}" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {{}}',
                    f'    def {joint_type} "{side}_{joint["feature"]}"',
                    "    {",
                    f"        rel physics:body0 = </{spec['name']}/{parent}>",
                    f"        rel physics:body1 = </{spec['name']}/{child}>",
                    f'        uniform token physics:axis = "{axis}"',
                    f"        float physics:lowerLimit = {joint['range'][0]}",
                    f"        float physics:upperLimit = {joint['range'][1]}",
                    "    }",
                ]
            )
            parent = child
    for camera in spec["cameras"]:
        lines.append(f'    def Camera "{camera}" {{}}')
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--check", action="store_true", help="Validate checked-in outputs without writing.")
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_source(spec, spec_path)
    outputs = {
        "mobile_bi_nexarm.xml": generate_mjcf(spec),
        "mobile_bi_nexarm.urdf": generate_urdf(spec),
        "mobile_bi_nexarm.usda": generate_usda(spec),
    }
    if args.check:
        stale = [
            name
            for name, content in outputs.items()
            if not (GENERATED / name).is_file() or (GENERATED / name).read_text(encoding="utf-8") != content
        ]
        if stale:
            raise SystemExit(f"generated assets are stale: {', '.join(stale)}")
    else:
        GENERATED.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            (GENERATED / name).write_text(content, encoding="utf-8")
    digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    print(f"validated {len(spec['joints'])} joints; spec sha256={digest}")


if __name__ == "__main__":
    main()
