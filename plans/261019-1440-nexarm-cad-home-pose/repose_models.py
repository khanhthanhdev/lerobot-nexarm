"""Move the committed NexArm models from the old CAD zero pose to the new one (the real Home pose).

One-off migration for plan 261019-1440. The Fusion zero pose changed (arm in an L at raw 2048, gripper
closed at raw 2833), so every value that lives in the arm frame and is kept by hand must follow its link:
joint positions and axes, the hand-tuned collision geoms, the sites, the wrist camera, the ``ready``
keyframe, and the URDF/xacro joint and visual origins. ``sim/apply_fusion_extract.py`` then merges the new
meshes, inertials and refitted collision boxes, and its axis-line check confirms this migration.

How: fit the old model's joint angles so that its joint axis lines land on the new Fusion extract, take each
body's rigid motion from MuJoCo forward kinematics at those angles, and apply it to the body's items.

    python plans/261019-1440-nexarm-cad-home-pose/repose_models.py [--check]
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET  # nosec B405 - parses committed model files
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SIM = REPO / "sim"
MJCF = SIM / "description/mjcf/nexarm.xml"
URDFS = (SIM / "description/urdf/nexarm.urdf", SIM / "description/urdf/nexarm.urdf.xacro")
EXTRACT = SIM / "fusion_export/fusion_extract.json"

ARM = ("joint_2_link_1_to_link_2", "joint_3_link_2_to_link_3", "joint_4_link_3_to_link_4")
# Joint -> the next joint whose axis line it must carry onto the extract (point for 2 and 3, axis for 4).
NEXT = {
    "joint_2_link_1_to_link_2": "joint_3_link_2_to_link_3",
    "joint_3_link_2_to_link_3": "joint_4_link_3_to_link_4",
    "joint_4_link_3_to_link_4": "joint_5_link_4_to_link_5",
}
# The Fusion gripper zero is now closed, which is the MJCF slide zero: the jaw meshes need no shift.
JAW_VISUALS = ("link_6_left_jaw_visual", "link_6_right_jaw_visual")


def fmt(values) -> str:
    return " ".join(f"{float(v) + 0.0:.9g}" for v in values)


def read(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def set_attrs(text: str, tag: str, name: str, attrs: dict[str, str | None]) -> str:
    """Set (or drop, for None) attributes on the single ``<tag name="name" .../>`` in ``text``."""
    pattern = re.compile(rf'<{tag}\b[^>]*\bname="{re.escape(name)}"[^>]*?/?>')
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"expected one <{tag} name={name!r}>, found {len(matches)}")
    old = matches[0].group(0)
    new = old
    for key, value in attrs.items():
        attr = re.compile(rf'\s{key}="[^"]*"')
        if value is None:
            new = attr.sub("", new)
        elif attr.search(new):
            new = attr.sub(f' {key}="{value}"', new, count=1)
        else:
            end = len(new) - (2 if new.endswith("/>") else 1)
            new = new[:end].rstrip() + f' {key}="{value}" ' + new[end:].lstrip()
    return text[: matches[0].start()] + new + text[matches[0].end() :]


def fit_angles(model: mujoco.MjModel, extract: dict) -> np.ndarray:
    """Old-model qpos that puts its joint axis lines on the extract's (arm joints only)."""
    data = mujoco.MjData(model)
    qpos = np.zeros(model.nq)

    def world_line(name: str) -> tuple[np.ndarray, np.ndarray]:
        jid = model.joint(name).id
        return data.xanchor[jid].copy(), data.xaxis[jid].copy()

    mount = model.body("nexarm_mount").id
    for _ in range(4):
        for name in ARM:
            mujoco.mj_kinematics(model, data)
            anchor, axis = world_line(name)
            offset = data.xpos[mount]
            nxt = NEXT[name]
            if nxt == "joint_5_link_4_to_link_5":
                current = world_line(nxt)[1]
                target = np.asarray(extract["joints"][nxt]["axis"], dtype=float)
                target *= np.sign(current @ target)
            else:
                current = world_line(nxt)[0] - anchor
                target = np.asarray(extract["joints"][nxt]["origin_mm"]) / 1000.0 + offset - anchor
            current = current - axis * (current @ axis)
            target = target - axis * (target @ axis)
            qpos[model.joint(name).qposadr[0]] += np.arctan2(
                axis @ np.cross(current, target), current @ target
            )
            data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return qpos


def body_motions(model: mujoco.MjModel, qpos: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Body name -> (R, t) taking a point in the old arm frame to the new arm frame."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mount = model.body("nexarm_mount").id
    motions = {}
    for b in range(model.nbody):
        name = model.body(b).name
        rot = data.xmat[b].reshape(3, 3) @ data.xmat[mount].reshape(3, 3).T
        motions[name] = (rot, data.xpos[b] - data.xpos[mount] - rot @ np.zeros(3))
    return motions


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def quat_to_mat(quat) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def repose_mjcf(text: str, motions: dict, qpos: np.ndarray, model: mujoco.MjModel) -> str:
    root = ET.fromstring(text)  # nosec B314 - committed file

    def owner_items(body: ET.Element):
        for child in body:
            if child.tag in ("joint", "geom", "site", "camera"):
                yield child

    for body in root.iter("body"):
        name = body.get("name")
        if name == "nexarm_mount":
            continue
        rot, shift = motions[name]

        def move(p: np.ndarray, rot=rot, shift=shift) -> np.ndarray:
            return rot @ p + shift

        for item in owner_items(body):
            item_name = item.get("name")
            pos = np.array([float(v) for v in (item.get("pos") or "0 0 0").split()])
            if item.tag == "geom" and item.get("class") == "visual":
                if item_name in JAW_VISUALS:
                    text = set_attrs(text, "geom", item_name, {"pos": None})
                continue
            attrs = {"pos": fmt(np.round(move(pos), 9 if item.tag == "joint" else 6))}
            if item.tag == "joint":
                axis = np.array([float(v) for v in item.get("axis", "0 0 1").split()])
                attrs["axis"] = fmt(np.round(rot @ axis, 6))
            elif item.tag == "geom":
                quat = rot @ quat_to_mat(item.get("quat", "1 0 0 0").split())
                if not np.allclose(quat, np.eye(3), atol=1e-6):
                    attrs["quat"] = fmt(np.round(mat_to_quat(quat), 6))
            elif item.tag == "camera":
                axes = np.array([float(v) for v in item.get("xyaxes").split()]).reshape(2, 3)
                attrs["xyaxes"] = fmt(np.round((rot @ axes.T).T.flatten(), 6))
            text = set_attrs(text, item.tag, item_name, attrs)

    # A keyframe names a physical pose: express it against the new zero. ``home`` and ``open`` are the Home
    # pose, which is now the zero itself.
    for key in root.iter("key"):
        if key.get("name") in ("home", "open"):
            continue
        qpos_key = np.array([float(v) for v in key.get("qpos").split()])
        ctrl = np.array([float(v) for v in key.get("ctrl").split()])
        for name in ARM:
            qpos_key[model.joint(name).qposadr[0]] -= qpos[model.joint(name).qposadr[0]]
            ctrl[model.actuator(f"{name}_control").id] -= qpos[model.joint(name).qposadr[0]]
        text = set_attrs(
            text, "key", key.get("name"), {"qpos": fmt(np.round(qpos_key, 4)), "ctrl": fmt(np.round(ctrl, 4))}
        )
    return text


def urdf_frames(text: str) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, str]]]:
    """URDF link name -> link frame origin in the arm frame (all frames are translations), and joint -> (parent, child)."""
    clean = text.replace("${prefix}", "")
    root = ET.fromstring(re.sub(r"<xacro:[^>]*>|</xacro:[^>]*>", "", clean).replace("xmlns:xacro", "data-x"))  # nosec B314
    joints = {}
    origins = {}
    for joint in root.iter("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        xyz = np.array([float(v) for v in origin.get("xyz").split()]) if origin is not None else np.zeros(3)
        if origin is not None and any(float(v) for v in origin.get("rpy", "0 0 0").split()):
            raise RuntimeError(f"{joint.get('name')}: rotated joint frames are not supported")
        joints[joint.get("name")] = (parent.get("link"), child.get("link"))
        origins[joint.get("name")] = xyz
    frames = {"base_link": np.zeros(3)}
    while len(frames) <= len(joints):
        progress = False
        for name, (parent, child) in joints.items():
            if parent in frames and child not in frames:
                frames[child] = frames[parent] + origins[name]
                progress = True
        if not progress:
            break
    return frames, joints


def repose_urdf(text: str, motions: dict, arm_offset: np.ndarray) -> str:
    """Move URDF link frames with their bodies; frames stay translations of the arm frame."""
    frames, joints = urdf_frames(text)
    # URDF frames are relative to base_link, whose frame sits at -visual origin in the arm frame.
    new = {}
    for link, origin in frames.items():
        body = link if link in motions else None
        if body is None:
            continue
        rot, shift = motions[body]
        moved = rot @ (origin + arm_offset) + shift - arm_offset
        if not np.allclose(moved, origin, atol=1e-7):
            new[link] = moved
    for name, (parent, child) in joints.items():
        if child not in new:
            continue
        xyz = new[child] - new.get(parent, frames[parent])
        block = re.compile(rf'(<joint name="(?:\$\{{prefix\}})?{name}".*?</joint>)', re.S).search(text)
        joint = block.group(1)
        joint = re.sub(
            r'(<origin xyz=")[^"]*(")',
            lambda m, v=xyz: m.group(1) + fmt(np.round(v, 6)) + m.group(2),
            joint,
            count=1,
        )
        axis = re.search(r'<axis xyz="([^"]+)"', joint)
        if axis:
            rot = motions[child][0]
            value = rot @ np.array([float(v) for v in axis.group(1).split()])
            joint = joint.replace(axis.group(0), f'<axis xyz="{fmt(np.round(value, 6))}"')
        text = text[: block.start(1)] + joint + text[block.end(1) :]
    for link, origin in new.items():
        block = re.compile(rf'(<link name="(?:\$\{{prefix\}})?{link}">.*?</link>)', re.S).search(text)
        if block is None:
            continue
        part = block.group(1)
        visual = re.search(r'(<visual>\s*<origin xyz=")([^"]*)(")', part)
        if visual is None:
            continue
        xyz = -(origin + arm_offset)
        part = part[: visual.start(2)] + fmt(np.round(xyz, 6)) + part[visual.end(2) :]
        text = text[: block.start(1)] + part + text[block.end(1) :]
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="Print the fit without writing files")
    args = parser.parse_args()

    extract = json.loads(EXTRACT.read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    qpos = fit_angles(model, extract)
    for name in ARM:
        print(
            f"{name}: old-zero angle of the new Home {np.degrees(qpos[model.joint(name).qposadr[0]]):+.2f} deg"
        )
    motions = body_motions(model, qpos)

    mjcf_text = repose_mjcf(read(MJCF), motions, qpos, model)
    # base_link's URDF frame in the arm frame is minus its visual origin.
    base_visual = re.search(
        r'<link name="(?:\$\{prefix\})?base_link">.*?<visual>\s*<origin xyz="([^"]+)"', read(URDFS[0]), re.S
    )
    arm_offset = -np.array([float(v) for v in base_visual.group(1).split()])
    urdf_texts = {path: repose_urdf(read(path), motions, arm_offset) for path in URDFS}
    if args.check:
        return
    write(MJCF, mjcf_text)
    for path, text in urdf_texts.items():
        write(path, text)
    print(f"Updated {MJCF.relative_to(REPO)} and {', '.join(str(p.relative_to(REPO)) for p in URDFS)}")


if __name__ == "__main__":
    main()
