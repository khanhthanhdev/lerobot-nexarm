"""Derive the Isaac Sim inputs from the MuJoCo model, so both simulators share one design.

The MJCF in ``sim/description/mjcf/`` is the reference: the Fusion export writes its masses, joints, cameras and
workcell, and its collision primitives are fitted to the meshes. This script copies that design to Isaac:

- ``sim/description/urdf/nexarm.urdf`` and ``nexarm.urdf.xacro`` get the MJCF collision boxes (in place of the
  raw CAD meshes, whose convex hulls overlap), the MJCF joint damping and friction, and a fixed
  ``gripper_frame`` link at the MJCF ``gripper_frame`` site, so an end-effector pose names the same point in
  MuJoCo, Isaac and a URDF IK solver.
- ``sim/isaac_lab/nexarm_scene.json`` describes everything the Isaac Lab environment builds around the robot:
  the robot pose, drive gains and raw-tick action mapping, the workcell fixtures, the three cameras, the bowls
  with their contact proxies, and the stacking task thresholds.

It needs only MuJoCo (no Isaac), and runs after every Fusion export (``sim/apply_fusion_extract.py`` calls it).

    python sim/isaac_lab/build_isaac_assets.py          # write
    python sim/isaac_lab/build_isaac_assets.py --check  # exit 1 if the outputs are stale
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import xml.etree.ElementTree as ET  # nosec B405 - parses the repo's own URDF
from pathlib import Path

import mujoco
import numpy as np

SIM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SIM_DIR.parent
SCHEMA = "nexarm_isaac_scene_v1"
SCENE_JSON = Path("isaac_lab/nexarm_scene.json")
URDF = Path("description/urdf/nexarm.urdf")
XACRO = Path("description/urdf/nexarm.urdf.xacro")
STACK_SCENE = Path("description/mjcf/bowl_stack_scene.xml")
CAMERA_RESOLUTION = (640, 480)  # The real follower records every camera at 640x480.

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# --- small rigid-transform helpers (4x4 homogeneous matrices) ---


def _pose(pos, mat) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = np.asarray(mat, dtype=float).reshape(3, 3)
    t[:3, 3] = pos
    return t


def _quat_to_mat(quat) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def _mat_to_quat(mat) -> list[float]:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=float).flatten())
    if quat[0] < 0:
        quat = -quat
    return _round(quat)


def _rpy_to_mat(rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _mat_to_rpy(mat) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    pitch = np.arcsin(-np.clip(mat[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-9:
        roll = np.arctan2(mat[2, 1], mat[2, 2])
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
    else:  # gimbal lock: put everything in yaw
        roll = 0.0
        yaw = np.arctan2(-mat[0, 1], mat[1, 1])
    return np.array([roll, pitch, yaw])


def _round(values, digits: int = 6) -> list[float]:
    return [round(float(v), digits) + 0.0 for v in np.asarray(values).flatten()]


def _fmt(values) -> str:
    return " ".join(f"{v:.6g}" for v in _round(values))


# --- URDF link frames in the MuJoCo world ---


def _urdf_links(urdf_text: str) -> dict[str, dict]:
    """URDF link name -> visual mesh file stem and visual origin."""
    links = {}
    for link in ET.fromstring(urdf_text).iter("link"):  # nosec B314
        visual = link.find("visual")
        if visual is None or visual.find("geometry/mesh") is None:
            continue
        origin = visual.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        stem = Path(visual.find("geometry/mesh").get("filename")).stem
        links[link.get("name")] = {"mesh": stem, "visual": _pose(xyz, _rpy_to_mat(rpy))}
    return links


def _mesh_frame(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """World pose of an STL's own frame. MuJoCo stores a mesh recentred on its inertia frame."""
    mesh = model.geom_dataid[geom]
    offset = _pose(model.mesh_pos[mesh], _quat_to_mat(model.mesh_quat[mesh]))
    return _pose(data.geom_xpos[geom], data.geom_xmat[geom]) @ np.linalg.inv(offset)


def link_frames(model: mujoco.MjModel, data: mujoco.MjData, urdf_text: str) -> dict[str, dict]:
    """For each URDF link: its MJCF body and the world pose of the link frame, matched through the shared mesh.

    A URDF link frame differs from the MJCF body frame (the MJCF keeps the Fusion assembly frame); placing the
    same STL in both fixes the offset between them.
    """
    mesh_geoms = {}
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_geoms.setdefault(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[geom]), geom
            )
    frames = {}
    for name, link in _urdf_links(urdf_text).items():
        geom = mesh_geoms.get(link["mesh"])
        if geom is None:
            raise ValueError(f"URDF link {name!r} uses mesh {link['mesh']!r}, which the MJCF does not")
        frames[name] = {
            "body": int(model.geom_bodyid[geom]),
            "world": _mesh_frame(model, data, geom) @ np.linalg.inv(link["visual"]),
        }
    return frames


# --- URDF and xacro sync ---


def _collision_boxes(
    model: mujoco.MjModel, data: mujoco.MjData, body: int, link_world: np.ndarray
) -> list[dict]:
    boxes = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or (
            model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0
        ):
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"Only box collisions are mirrored to the URDF, got geom {geom} on body {body}")
        local = np.linalg.inv(link_world) @ _pose(data.geom_xpos[geom], data.geom_xmat[geom])
        boxes.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                "xyz": local[:3, 3],
                "rpy": _mat_to_rpy(local[:3, :3]),
                "size": 2 * model.geom_size[geom],
            }
        )
    return boxes


def _sync_tcp(text: str, tcp: dict) -> str:
    """Write the ``gripper_frame`` link, fixed at the MJCF ``gripper_frame`` site, after its parent link."""
    nl = "\r\n" if "\r\n" in text else "\n"
    self_closing = " />" if '" />' in text else "/>"
    prefix = r"(?:\$\{prefix\})?"
    # Drop the block this function wrote last time (the blank line before it included).
    text = re.sub(
        rf"\r?\n(?:[ \t]*\r?\n)?[ \t]*<!-- Tool centre point[^\n]*\n[ \t]*<link name=\"{prefix}gripper_frame\"[^>]*>\r?\n"
        rf"[ \t]*<joint name=\"{prefix}gripper_frame_joint\".*?</joint>",
        "",
        text,
        flags=re.S,
    )
    parent = re.compile(
        rf'(?P<indent>[ \t]*)<link name="(?P<prefix>{prefix}){re.escape(tcp["parent"])}">.*?</link>', re.S
    )
    match = parent.search(text)
    if match is None:
        raise ValueError(f"Link {tcp['parent']!r} not found")
    indent, name_prefix = match.group("indent"), match.group("prefix")
    step = "\t" if "\t" in indent else "  "
    lines = [
        "",
        "",
        f"{indent}<!-- Tool centre point: the MJCF gripper_frame site (sim/isaac_lab/build_isaac_assets.py) -->",
        f'{indent}<link name="{name_prefix}gripper_frame"{self_closing}',
        f'{indent}<joint name="{name_prefix}gripper_frame_joint" type="fixed">',
        f'{indent}{step}<parent link="{name_prefix}{tcp["parent"]}"{self_closing}',
        f'{indent}{step}<child link="{name_prefix}gripper_frame"{self_closing}',
        f'{indent}{step}<origin xyz="{_fmt(tcp["xyz"])}" rpy="{_fmt(tcp["rpy"])}"{self_closing}',
        f"{indent}</joint>",
    ]
    return text[: match.end()] + nl.join(lines) + text[match.end() :]


def _sync_description(
    text: str, collisions: dict[str, list[dict]], dynamics: dict[str, tuple[float, float]]
) -> str:
    """Replace each link's collisions and each joint's dynamics; keeps the file's own formatting."""
    nl = "\r\n" if "\r\n" in text else "\n"
    self_closing = " />" if '" />' in text else "/>"
    prefix = r"(?:\$\{prefix\})?"
    for link, boxes in collisions.items():
        block = re.compile(rf'(?P<indent>[ \t]*)<link name="{prefix}{re.escape(link)}">.*?</link>', re.S)
        match = block.search(text)
        if match is None:
            raise ValueError(f"Link {link!r} not found")
        body = re.sub(r"[ \t]*<collision\b[^>]*>.*?</collision>[ \t]*\r?\n", "", match.group(0), flags=re.S)
        child = match.group("indent") + ("  " if "\t" not in match.group("indent") else "\t")
        step = child[len(match.group("indent")) :]
        lines = []
        for box in boxes:
            lines += [
                f'{child}<collision name="{box["name"]}">',
                f'{child}{step}<origin xyz="{_fmt(box["xyz"])}" rpy="{_fmt(box["rpy"])}"{self_closing}',
                f"{child}{step}<geometry>",
                f'{child}{step}{step}<box size="{_fmt(box["size"])}"{self_closing}',
                f"{child}{step}</geometry>",
                f"{child}</collision>",
            ]
        closing = match.group("indent") + "</link>"
        body = body[: body.rindex(closing)] + "".join(line + nl for line in lines) + closing
        text = text[: match.start()] + body + text[match.end() :]
    for joint, (damping, friction) in dynamics.items():
        block = re.compile(rf'<joint name="{prefix}{re.escape(joint)}".*?</joint>', re.S)
        match = block.search(text)
        if match is None:
            raise ValueError(f"Joint {joint!r} not found")
        new = re.sub(
            r'<dynamics damping="[^"]*" friction="[^"]*"',
            f'<dynamics damping="{damping:.6g}" friction="{friction:.6g}"',
            match.group(0),
        )
        text = text[: match.start()] + new + text[match.end() :]
    return text


# --- scene description ---


def _excluded_pairs(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """Body pairs named in the MJCF <exclude> elements (MuJoCo packs each as body1 << 16 | body2)."""
    return [(int(signature >> 16), int(signature & 0xFFFF)) for signature in model.exclude_signature]


def _rgba(model: mujoco.MjModel, geom: int) -> list[float]:
    mat = model.geom_matid[geom]
    return _round(model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[geom], 4)


def _actions(model: mujoco.MjModel) -> list[dict]:
    """Per follower feature: the joints it drives and the affine raw-tick -> joint-target map the MuJoCo backend uses."""
    from lerobot.motors.nexarm.mujoco_mapping import (
        HOME_POSITIONS,
        MUJOCO_JOINTS,
        RAW_RANGES,
        raw_to_joint_position,
    )
    from lerobot.motors.nexarm.nexarm import JOINT_NAMES

    actions = []
    for feature in JOINT_NAMES:
        joint = MUJOCO_JOINTS[feature]
        joint_range = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
        raw_low, raw_high = sorted(RAW_RANGES[feature])
        lo, hi = (raw_to_joint_position(feature, raw, joint_range) for raw in (raw_low, raw_high))
        # The map is affine between the raw limits (clamped to the joint range); measure it away from the clamps.
        a, b = raw_low + 0.25 * (raw_high - raw_low), raw_low + 0.75 * (raw_high - raw_low)
        qa, qb = (raw_to_joint_position(feature, raw, joint_range) for raw in (a, b))
        scale = (qb - qa) / (b - a)
        offset = qa - scale * a
        mid = 0.5 * (a + b)
        if not np.isclose(raw_to_joint_position(feature, mid, joint_range), scale * mid + offset, atol=1e-9):
            raise ValueError(f"The raw-tick map of {feature!r} is not affine")
        joints = [joint]
        if feature == "gripper":
            joints.append("left_jaw_slide_joint")  # follows the right jaw (MJCF equality, URDF mimic)
        actions.append(
            {
                "feature": feature,
                "joints": joints,
                "raw_range": [raw_low, raw_high],
                "home_raw": float(HOME_POSITIONS[feature]),
                "scale": float(scale),
                "offset": float(offset),
                "target_range": _round(sorted((lo, hi)), 9),
            }
        )
    return actions


def _drives(model: mujoco.MjModel) -> dict[str, dict]:
    """Joint drive parameters from the MJCF actuators and dof properties.

    A MuJoCo position actuator applies kp (target - q) - kv qdot plus the joint's passive damping; a PhysX drive
    has one damping term, so it takes kv + damping. MuJoCo frictionloss (a torque) has no PhysX equivalent and is
    reported for reference only.
    """
    actuated = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    jaw_actuator = actuated[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_jaw_slide_joint")]
    drives = {}
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        dof = model.jnt_dofadr[joint]
        act = actuated.get(joint, jaw_actuator if name == "left_jaw_slide_joint" else None)
        kp = float(model.actuator_gainprm[act, 0]) if act is not None else 0.0
        kv = float(-model.actuator_biasprm[act, 2]) if act is not None else 0.0
        drives[name] = {
            "type": "prismatic" if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE else "revolute",
            "range": _round(model.jnt_range[joint], 9),
            "stiffness": kp,
            "damping": round(kv + float(model.dof_damping[dof]), 6),
            "effort": float(model.actuator_forcerange[act, 1]) if act is not None else 0.0,
            "armature": float(model.dof_armature[dof]),
            "frictionloss": float(model.dof_frictionloss[dof]),
        }
    return drives


def _bowls(model: mujoco.MjModel) -> dict:
    from lerobot.robots.nexarm_sim.stack_bowls_task import COLORS

    first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bowl_{COLORS[0]}")
    colliders, visual = [], None
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != first:
            continue
        local = _pose(model.geom_pos[geom], _quat_to_mat(model.geom_quat[geom]))
        kind = model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            frame = local @ np.linalg.inv(_pose(model.mesh_pos[mesh], _quat_to_mat(model.mesh_quat[mesh])))
            visual = {
                "mesh": "../assets/meshes/visual/bowl.stl",
                "scale": _round(model.mesh_scale[mesh]),
                "pos": _round(frame[:3, 3]),
                "quat": _mat_to_quat(frame[:3, :3]),
            }
            continue
        size = model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            shape = {"type": "capsule", "radius": float(size[0]), "half_length": float(size[1])}
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape = {"type": "cylinder", "radius": float(size[0]), "half_length": float(size[1])}
        else:
            raise ValueError(f"Unsupported bowl collider type {kind}")
        shape.update(pos=_round(local[:3, 3]), quat=_mat_to_quat(local[:3, :3]))
        colliders.append(shape)
        friction = float(model.geom_friction[geom, 0])
    colors = {}
    for color in COLORS:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bowl_{color}")
        geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_type[g] == 7)
        colors[color] = _rgba(model, geom)
    return {
        "mass": round(float(model.body_mass[first]), 6),
        "com": _round(model.body_ipos[first]),
        "inertia": [float(v) for v in model.body_inertia[first]],
        "inertia_frame_quat": _round(model.body_iquat[first]),
        "friction": friction,
        "colors": colors,
        "visual": visual,
        "colliders": colliders,
    }


def build_scene(
    sim_dir: Path = SIM_DIR,
) -> tuple[dict, dict[str, list[dict]], dict[str, tuple[float, float]]]:
    """Return the scene description, the URDF collision boxes and the URDF joint dynamics."""
    from lerobot.robots.nexarm_sim import stack_bowls_task as task

    model = mujoco.MjModel.from_xml_path(str(sim_dir / STACK_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # qpos0: every arm joint at 0, bowls at their default slots
    urdf_text = (sim_dir / URDF).read_text(encoding="utf-8")
    frames = link_frames(model, data, urdf_text)

    collisions = {link: _collision_boxes(model, data, f["body"], f["world"]) for link, f in frames.items()}
    drives = _drives(model)
    # The URDF carries the passive joint damping only; the actuator's kv belongs to the controller.
    dynamics = {}
    for name in drives:
        dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        dynamics[name] = (float(model.dof_damping[dof]), float(model.dof_frictionloss[dof]))

    base = frames["base_link"]["world"]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame")
    site_link = next(n for n, f in frames.items() if f["body"] == model.site_bodyid[site])
    site_local = np.linalg.inv(frames[site_link]["world"]) @ _pose(data.site_xpos[site], data.site_xmat[site])

    fixtures = []
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        if name.startswith("workcell_"):
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError(f"Workcell geom {name!r} is not a box")
            fixtures.append(
                {
                    "name": name.removeprefix("workcell_"),
                    "pos": _round(data.geom_xpos[geom]),
                    "quat": _mat_to_quat(data.geom_xmat[geom]),
                    "size": _round(2 * model.geom_size[geom]),
                    "rgba": _rgba(model, geom),
                }
            )

    cameras = []
    for cam in range(model.ncam):
        world = _pose(data.cam_xpos[cam], data.cam_xmat[cam])
        body = model.cam_bodyid[cam]
        parent = None if body == 0 else next(n for n, f in frames.items() if f["body"] == body)
        local = world if parent is None else np.linalg.inv(frames[parent]["world"]) @ world
        cameras.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam),
                "parent_link": parent,
                # MuJoCo cameras look along -Z with +Y up: Isaac Lab's "opengl" convention.
                "pos": _round(local[:3, 3]),
                "quat": _mat_to_quat(local[:3, :3]),
                "fovy_deg": float(model.cam_fovy[cam]),
                "width": CAMERA_RESOLUTION[0],
                "height": CAMERA_RESOLUTION[1],
            }
        )

    defaults = {
        k: v.default
        for k, v in inspect.signature(task.NexArmStackBowlsTask.__init__).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    scene = {
        "schema": SCHEMA,
        "generated_by": "sim/isaac_lab/build_isaac_assets.py",
        "source": STACK_SCENE.as_posix(),
        "frame": "Positions are metres in the environment frame: the table top is z = 0, as in the MJCF world.",
        "physics": {"gravity": _round(model.opt.gravity), "control_hz": 30},
        "table": {
            "top_z": 0.0,
            "friction": float(model.geom_friction[floor, 0]),
            "rgba": _rgba(model, floor),
        },
        "robot": {
            "urdf": "../description/urdf/nexarm.urdf",
            "base_pos": _round(base[:3, 3]),
            "base_quat": _mat_to_quat(base[:3, :3]),
            "drives": drives,
            "actions": _actions(model),
            "gripper_frame": {
                "link": site_link,
                "pos": _round(site_local[:3, 3]),
                "quat": _mat_to_quat(site_local[:3, :3]),
            },
            "jaw_links": ["link_6_left_jaw", "link_6_right_jaw"],
            # MJCF <exclude> pairs. PhysX already skips pairs joined by a joint; the others need filtering.
            "contact_excludes": [
                [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) for body in pair]
                for pair in _excluded_pairs(model)
            ],
        },
        "fixtures": fixtures,
        "cameras": cameras,
        "bowls": _bowls(model),
        "stack_bowls": {
            "colors": list(task.COLORS),
            "slots": [list(s) for s in task.BOWL_SLOTS_M],
            "min_separation_m": task.MIN_BOWL_SEPARATION_M,
            "position_jitter_m": 0.04,
            "max_concentric_error_m": defaults["max_concentric_error_m"],
            "min_nesting_delta_m": defaults["min_nesting_delta_m"],
            "max_nesting_delta_m": defaults["max_nesting_delta_m"],
            "success_hold_s": defaults["success_hold_s"],
            "max_linear_speed_mps": 0.05,
            "max_angular_speed_rps": 0.5,
            "max_tilt_deg": 15.0,
            "assist_engage_raw": task.GRIPPER_ASSIST_ENGAGE_POS,
            "assist_release_raw": defaults["assist_release_pos"],
            "assist_max_distance_m": task.ASSIST_MAX_DISTANCE_M,
        },
    }
    return scene, collisions, dynamics


def render(sim_dir: Path = SIM_DIR) -> dict[Path, str]:
    """The text every output should have."""
    scene, collisions, dynamics = build_scene(sim_dir)
    outputs = {sim_dir / SCENE_JSON: json.dumps(scene, indent=2) + "\n"}
    frame = scene["robot"]["gripper_frame"]
    tcp = {
        "parent": frame["link"],
        "xyz": frame["pos"],
        "rpy": _round(_mat_to_rpy(_quat_to_mat(frame["quat"]))),
    }
    for path in (URDF, XACRO):
        text = (sim_dir / path).read_bytes().decode("utf-8")
        outputs[sim_dir / path] = _sync_tcp(_sync_description(text, collisions, dynamics), tcp)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sim-dir", type=Path, default=SIM_DIR)
    parser.add_argument(
        "--check", action="store_true", help="Fail if an output is out of date; write nothing."
    )
    args = parser.parse_args(argv)
    stale = []
    for path, text in render(args.sim_dir).items():
        current = path.read_bytes().decode("utf-8") if path.exists() else None
        if current == text:
            continue
        stale.append(path)
        if not args.check:
            path.write_bytes(text.encode("utf-8"))
    verb = "Stale" if args.check else "Updated"
    for path in stale:
        print(
            f"{verb}: {path.relative_to(args.sim_dir.parent) if args.sim_dir.parent in path.parents else path}"
        )
    if not stale:
        print("Isaac assets are up to date.")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
