"""Merge a Fusion extract (``fusion_extract.json``) into the committed NexArm MuJoCo and URDF models.

``sim/exporter/fusion_export_all.py`` writes the extract and meshes from Fusion into the staging folder
``sim/fusion_export/``. This script then:

1. checks the extract against the committed kinematics (joint axes and axis lines) and refuses to continue if
   they disagree, because the raw-to-joint mapping and calibrations depend on them;
2. checks that every centre of mass lies inside its mesh, which catches frame mistakes;
3. updates each body's mass, centre of mass and inertia in the MJCF, URDF and xacro, keeping everything that
   was tuned by hand (actuators, contacts, materials, sensors, keyframes);
4. raises the arm so the Fusion tabletop is the scene floor (z = 0);
5. writes ``workcell.xml`` (the box-shaped fixtures around the arm), includes it in every scene, and moves the
   ``top`` camera to the Fusion D435i and the ``front`` camera onto the Fusion C270 mount;
6. refits the arm-link collision boxes and copies the staged meshes into ``sim/assets/meshes/visual/``;
7. runs ``sim/isaac_lab/build_isaac_assets.py``, which mirrors the result to the URDF collisions and the Isaac
   Lab scene description.

Everything it edits lives in the one canonical tree: ``sim/description/`` and ``sim/assets/meshes/visual/``.

Usage:
    uv run python sim/apply_fusion_extract.py [--extract sim/fusion_export/fusion_extract.json] [--check]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import struct
import xml.etree.ElementTree as ET  # nosec B405 - parses committed model files
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

SIM_DIR = Path(__file__).resolve().parent
SCHEMA = "nexarm_fusion_extract_v1"

ROBOT_MJCF = "description/mjcf/nexarm.xml"
URDF_FILES = ("description/urdf/nexarm.urdf", "description/urdf/nexarm.urdf.xacro")
# workcell.xml is written next to these scenes.
SCENES = ("description/mjcf/scene.xml", "description/mjcf/bowl_stack_scene.xml")
WORKCELL_FILE = "workcell.xml"
STAGED_MESH_DIR = "fusion_export/meshes"  # written by the Fusion exporter
MESH_DIR = "assets/meshes/visual"  # the only mesh folder the models load

# Kinematic tolerances against the committed MJCF.
AXIS_MIN_DOT = 0.999
AXIS_LINE_TOL_M = 1e-3
# The pinion is visual only (fixed to the gripper body in practice); its axis is reported, not enforced.
UNCHECKED_AXIS_JOINTS = {"gripper_pinion_joint"}
COM_MESH_PAD_M = 2e-3
# Arm links whose collision boxes are refitted from the exported mesh: body -> (box count, oriented boxes).
# Counts and orientation give the least box volume at full mesh coverage. The finger pads
# (``finger_collision``) are tuned for grasping and stay hand-maintained; the pinion has no collision.
COLLISION_BOXES = {
    "base_link": (1, False),
    "link_1": (1, False),
    "link_2": (3, True),
    "link_3": (1, True),
    "link_4": (1, False),
    "link_5": (1, False),
    "cam_mount": (2, True),
}

# Fixture colours by Fusion material name (first matching keyword wins).
FIXTURE_MATERIALS = (
    ("extrusion", "workcell_extrusion", "0.08 0.08 0.09 1"),
    ("pvc", "workcell_panel", "0.92 0.92 0.90 1"),
    ("plywood", "workcell_wood", "0.62 0.47 0.30 1"),
    ("aluminum", "workcell_plate", "0.72 0.73 0.75 1"),
)
DEFAULT_FIXTURE_MATERIAL = ("workcell_default", "0.5 0.5 0.5 1")
TABLE_OCCURRENCE_PREFIX = "workbench"


class ExtractMismatchError(RuntimeError):
    """The Fusion extract disagrees with the committed kinematics."""


@dataclass
class Report:
    changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def print(self) -> None:
        for line in self.changes:
            print(line)
        for line in self.warnings:
            print(f"WARNING: {line}")


def _fmt(values) -> str:
    return " ".join(f"{float(v) + 0.0:.9g}" for v in values)  # + 0.0 turns -0.0 into 0


def _read(path: Path) -> str:
    """Read keeping the file's own line endings, so a merge only touches the values it changes."""
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def _vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([float(v) for v in text.split()]) if text else np.array(default, dtype=float)


def stl_vertices(path: Path) -> np.ndarray:
    """Return the unique vertices of a binary STL, in the file's units."""
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    records = np.frombuffer(
        data, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=count, offset=84
    )
    return np.unique(records["v"].reshape(-1, 3), axis=0).astype(float)


def stl_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the (min, max) vertex bounds of a binary STL, in the file's units."""
    vertices = stl_vertices(path)
    return vertices.min(axis=0), vertices.max(axis=0)


def _box(points: np.ndarray, oriented: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smallest box around ``points`` along the frame axes, or along their principal axes when ``oriented``.

    Returns (centre, axes as columns, half sizes). Principal axes within ~6 degrees of a frame axis snap to
    it, so boxes stay aligned with the CAD wherever the part is.
    """
    centre = points.mean(axis=0)
    axes = np.eye(3)
    if oriented:
        axes = np.linalg.svd(points - centre, full_matrices=False)[2].T
        for i in range(3):
            j = int(np.argmax(np.abs(axes[:, i])))
            if abs(axes[j, i]) > 0.995:
                axes[:, i] = np.eye(3)[j]
        q, _ = np.linalg.qr(axes)
        axes = q * np.sign(np.diag(q.T @ axes))
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
    local = (points - centre) @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    return centre + axes @ ((lo + hi) / 2), axes, (hi - lo) / 2


def fit_collision_boxes(points: np.ndarray, count: int, oriented: bool) -> list:
    """Cover ``points`` with ``count`` boxes, splitting them into slabs along their main axis (1-D k-means)."""
    if count == 1:
        return [_box(points, oriented)]
    centre = points.mean(axis=0)
    along = (points - centre) @ np.linalg.svd(points - centre, full_matrices=False)[2][0]
    edges = np.quantile(along, np.linspace(0, 1, count + 1))
    seeds = (edges[:-1] + edges[1:]) / 2
    for _ in range(50):
        label = np.argmin(np.abs(along[:, None] - seeds[None]), axis=1)
        seeds = np.array([along[label == i].mean() if np.any(label == i) else seeds[i] for i in range(count)])
    return [_box(points[label == i], oriented) for i in range(count) if np.sum(label == i) > 10]


def collision_geoms(name: str, boxes: list) -> list[str]:
    tags = []
    for index, (centre, axes, half) in enumerate(boxes):
        attrs = (
            f'name="{name}_collision_{index}" class="collision" type="box" '
            f'pos="{_fmt(np.round(centre, 6))}" size="{_fmt(np.round(half, 6))}"'
        )
        if not np.allclose(axes, np.eye(3)):
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, axes.flatten())
            attrs += f' quat="{_fmt(np.round(quat, 6))}"'
        tags.append(f"<geom {attrs} />")
    return tags


def replace_collisions(text: str, name: str, tags: list[str], label: str) -> str:
    """Swap a body's ``<name>_collision_N`` geoms for ``tags``, or add them after its inertial."""
    newline = "\r\n" if "\r\n" in text else "\n"
    old = list(re.finditer(rf'([ \t]*)<geom [^>]*name="{name}_collision_\d+"[^>]*/>\r?\n', text))
    if old:
        start, end, indent = old[0].start(), old[-1].end(), old[0].group(1)
        if text[start:end].count("<") != len(old):
            raise ExtractMismatchError(f"{label}: {name} collision geoms are not consecutive")
    else:
        body = text.find(f'<body name="{name}"')
        inertial = re.compile(r"([ \t]*)<inertial\b[^>]*/>\r?\n").search(text, max(body, 0))
        if body < 0 or inertial is None:
            raise ExtractMismatchError(f"{label}: cannot place collision geoms for {name}")
        start = end = inertial.end()
        indent = inertial.group(1)
    return text[:start] + "".join(indent + tag + newline for tag in tags) + text[end:]


@dataclass
class MjcfRobot:
    joints: dict[str, tuple[str, np.ndarray, np.ndarray]]  # name -> (type, pos, axis)
    visual_offsets: dict[str, np.ndarray]  # body -> visual geom pos
    mount_pos: np.ndarray


def fitted_collisions(mesh_dir: Path, extract: dict, robot: MjcfRobot) -> dict[str, list[str]]:
    """Collision boxes for the ``COLLISION_BOXES`` links, fitted to their exported meshes in the body frame."""
    collisions = {}
    for name, (count, oriented) in COLLISION_BOXES.items():
        points = stl_vertices(mesh_dir / extract["bodies"][name]["mesh"]) / 1000.0
        points = points + robot.visual_offsets.get(name, np.zeros(3))
        collisions[name] = collision_geoms(name, fit_collision_boxes(points, count, oriented))
    return collisions


def read_mjcf(path: Path) -> MjcfRobot:
    root = ET.parse(path).getroot()  # nosec B314 - committed file
    joints = {
        j.get("name"): (j.get("type", "hinge"), _vec(j.get("pos")), _vec(j.get("axis"), (0, 0, 1)))
        for j in root.iter("joint")
        if j.get("name")
    }
    offsets = {}
    for body in root.iter("body"):
        visual = body.find(f"geom[@name='{body.get('name')}_visual']")
        if visual is not None:
            offsets[body.get("name")] = _vec(visual.get("pos"))
    mount = root.find(".//body[@name='nexarm_mount']")
    if mount is None:
        raise ExtractMismatchError(f"{path} has no nexarm_mount body")
    return MjcfRobot(joints, offsets, _vec(mount.get("pos")))


def validate(extract: dict, robot: MjcfRobot, mesh_dir: Path, report: Report) -> None:
    if extract.get("schema") != SCHEMA:
        raise ExtractMismatchError(f"Unsupported extract schema {extract.get('schema')!r}, expected {SCHEMA}")
    errors = []
    for name, joint in extract["joints"].items():
        if name not in robot.joints:
            errors.append(f"{name}: not in the committed MJCF")
            continue
        _, pos, axis = robot.joints[name]
        axis = axis / np.linalg.norm(axis)
        fusion_axis = np.asarray(joint["axis"], dtype=float)
        fusion_axis /= np.linalg.norm(fusion_axis)
        dot = abs(float(axis @ fusion_axis))
        if dot < AXIS_MIN_DOT:
            message = f"{name}: Fusion axis {np.round(fusion_axis, 3)} vs MJCF {np.round(axis, 3)}"
            if name in UNCHECKED_AXIS_JOINTS:
                report.warnings.append(message + " (visual-only joint, not enforced)")
            else:
                errors.append(message)
            continue
        if joint["type"] == "revolute":
            offset = np.asarray(joint["origin_mm"], dtype=float) / 1000.0 - pos
            distance = float(np.linalg.norm(offset - (offset @ axis) * axis))
            if distance > AXIS_LINE_TOL_M:
                errors.append(f"{name}: Fusion axis line is {distance * 1000:.2f} mm from the MJCF axis line")
    missing = set(robot.joints) & {
        "joint_1_base_to_link_1",
        "joint_2_link_1_to_link_2",
        "joint_3_link_2_to_link_3",
        "joint_4_link_3_to_link_4",
        "joint_5_link_4_to_link_5",
        "left_jaw_slide_joint",
        "right_jaw_slide_joint",
    } - set(extract["joints"])
    errors.extend(f"{name}: missing from the extract" for name in sorted(missing))

    for name, body in extract["bodies"].items():
        lo, hi = stl_bounds(mesh_dir / body["mesh"])
        com = np.asarray(body["com_mm"], dtype=float)
        pad = COM_MESH_PAD_M * 1000.0
        if np.any(com < lo - pad) or np.any(com > hi + pad):
            errors.append(
                f"{name}: centre of mass {np.round(com, 1)} mm lies outside its mesh {lo.round(1)}..{hi.round(1)}"
            )
        if body["mass_kg"] <= 0:
            errors.append(f"{name}: non-positive mass {body['mass_kg']}")
        inertia = np.asarray(body["fullinertia"], dtype=float)
        tensor = np.array(
            [
                [inertia[0], inertia[3], inertia[4]],
                [inertia[3], inertia[1], inertia[5]],
                [inertia[4], inertia[5], inertia[2]],
            ]
        )
        if np.linalg.eigvalsh(tensor).min() <= 0:
            errors.append(f"{name}: inertia tensor is not positive definite")
    if errors:
        raise ExtractMismatchError("Fusion extract rejected:\n  " + "\n  ".join(errors))


def _replace_attrs(tag: str, attrs: dict[str, str]) -> str:
    for key, value in attrs.items():
        pattern = re.compile(rf'(\s{key}=")[^"]*(")')
        if not pattern.search(tag):
            raise ExtractMismatchError(f"Attribute {key} not found in {tag}")
        tag = pattern.sub(lambda m, v=value: m.group(1) + v + m.group(2), tag, count=1)
    return tag


def update_mjcf(
    text: str,
    extract: dict,
    robot: MjcfRobot,
    mount_z: float,
    report: Report,
    label: str,
    collisions: dict[str, list[str]] | None = None,
) -> str:
    for name, body in extract["bodies"].items():
        start = text.find(f'<body name="{name}"')
        if start < 0:
            raise ExtractMismatchError(f"{label}: body {name} not found")
        match = re.compile(r"<inertial\b[^>]*/>").search(text, start)
        if match is None:
            raise ExtractMismatchError(f"{label}: no inertial after body {name}")
        com = np.asarray(body["com_mm"], dtype=float) / 1000.0 + robot.visual_offsets.get(name, np.zeros(3))
        new_tag = _replace_attrs(
            match.group(0),
            {
                "mass": f"{body['mass_kg']:.6g}",
                "pos": _fmt(com),
                "fullinertia": " ".join(f"{v:.6e}" for v in body["fullinertia"]),
            },
        )
        old_mass = float(re.search(r'mass="([^"]+)"', match.group(0)).group(1))
        report.changes.append(f"{name}: mass {old_mass * 1000:.1f} g -> {body['mass_kg'] * 1000:.1f} g")
        text = text[: match.start()] + new_tag + text[match.end() :]

    for name, tags in (collisions or {}).items():
        text = replace_collisions(text, name, tags, label)

    mount = re.compile(r'<body name="nexarm_mount"[^>]*>').search(text)
    pos = robot.mount_pos.copy()
    pos[2] = mount_z
    text = text[: mount.start()] + _replace_attrs(mount.group(0), {"pos": _fmt(pos)}) + text[mount.end() :]
    return text


def update_urdf(text: str, extract: dict, label: str) -> str:
    for name, body in extract["bodies"].items():
        link = re.compile(rf'<link name="(?:\$\{{prefix\}})?{name}">.*?</link>', re.S).search(text)
        if link is None:
            raise ExtractMismatchError(f"{label}: link {name} not found")
        block = link.group(0)
        visual = re.search(r'<visual>\s*<origin xyz="([^"]+)"', block)
        inertial = re.search(r"<inertial>.*?</inertial>", block, re.S)
        if visual is None or inertial is None:
            raise ExtractMismatchError(f"{label}: link {name} lacks a visual origin or inertial")
        # The visual origin places the mesh (arm frame) in the link frame, so it also places the CoM.
        com = np.asarray(body["com_mm"], dtype=float) / 1000.0 + _vec(visual.group(1))
        ixx, iyy, izz, ixy, ixz, iyz = body["fullinertia"]
        new = inertial.group(0)
        com_text, mass_text = _fmt(com), f"{body['mass_kg']:.6g}"
        new = re.sub(r'(<origin xyz=")[^"]*(")', lambda m: m.group(1) + com_text + m.group(2), new, count=1)  # noqa: B023
        new = re.sub(r'(<mass value=")[^"]*(")', lambda m: m.group(1) + mass_text + m.group(2), new)  # noqa: B023
        for key, value in (
            ("ixx", ixx),
            ("ixy", ixy),
            ("ixz", ixz),
            ("iyy", iyy),
            ("iyz", iyz),
            ("izz", izz),
        ):
            new = re.sub(rf'({key}=")[^"]*(")', lambda m, v=value: m.group(1) + f"{v:.6e}" + m.group(2), new)
        block = block[: inertial.start()] + new + block[inertial.end() :]
        text = text[: link.start()] + block + text[link.end() :]
    return text


def _fixture_material(material: str) -> tuple[str, str]:
    lowered = material.lower()
    for keyword, name, rgba in FIXTURE_MATERIALS:
        if keyword in lowered:
            return name, rgba
    return DEFAULT_FIXTURE_MATERIAL


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_").lower()


def workcell_xml(extract: dict, mount: np.ndarray) -> str:
    """Static fixtures in MJCF world coordinates (arm frame + ``mount``)."""
    lines = [
        "<!-- Generated by sim/apply_fusion_extract.py from the Fusion workcell. Do not edit by hand. -->",
        '<mujoco model="NexArm workcell">',
        "  <asset>",
    ]
    used = {}
    geoms = []
    for box in extract["workcell"]["boxes"]:
        if box["occurrence"].lower().startswith(TABLE_OCCURRENCE_PREFIX):
            continue  # the scene floor is the tabletop
        lo = np.asarray(box["min_mm"], dtype=float) / 1000.0 + mount
        hi = np.asarray(box["max_mm"], dtype=float) / 1000.0 + mount
        material, rgba = _fixture_material(box["material"])
        used[material] = rgba
        geoms.append(
            f'    <geom name="workcell_{_safe_name(box["body"])}" type="box" pos="{_fmt((lo + hi) / 2)}" '
            f'size="{_fmt((hi - lo) / 2)}" material="{material}" />'
        )
    lines += [f'    <material name="{name}" rgba="{rgba}" />' for name, rgba in sorted(used.items())]
    lines += [
        "  </asset>",
        "  <default>",
        '    <default class="workcell">',
        '      <geom contype="2" conaffinity="2" friction="1 0.005 0.0001" />',
        "    </default>",
        "  </default>",
        "  <worldbody>",
        '    <body name="workcell" childclass="workcell">',
        *geoms,
        "    </body>",
        "  </worldbody>",
        "</mujoco>",
        "",
    ]
    return "\n".join(lines)


# Scene camera name -> extract key.
CAMERAS = {"top": "top_camera", "front": "front_camera"}


def camera_pose(camera: dict, mount: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MJCF ``pos`` and ``xyaxes`` for an extracted camera.

    The extract gives the viewing direction plus either the image's right direction (``long_axis``, the sensor
    bar of a depth camera) or its up direction (``up``).
    """
    pos = np.asarray(camera["lens_mm"], dtype=float) / 1000.0 + mount
    z_cam = -np.asarray(camera["view_dir"], dtype=float)
    z_cam /= np.linalg.norm(z_cam)
    if "long_axis" in camera:
        x_cam = np.asarray(camera["long_axis"], dtype=float)
        x_cam -= (x_cam @ z_cam) * z_cam
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
    else:
        y_cam = np.asarray(camera["up"], dtype=float)
        y_cam -= (y_cam @ z_cam) * z_cam
        y_cam /= np.linalg.norm(y_cam)
        x_cam = np.cross(y_cam, z_cam)
    return pos, np.concatenate([x_cam, y_cam])


def update_scene(text: str, cameras: dict[str, tuple[np.ndarray, np.ndarray]], label: str) -> str:
    if WORKCELL_FILE not in text:
        robot_include = re.compile(r'(\s*)<include file="[^"]+" />').search(text)
        if robot_include is None:
            raise ExtractMismatchError(f"{label}: no robot include")
        indent = robot_include.group(1)
        insert = f'{indent}<include file="{WORKCELL_FILE}" />'
        text = text[: robot_include.end()] + insert + text[robot_include.end() :]
    for name, (pos, xyaxes) in cameras.items():
        tag = re.compile(rf'<camera name="{name}"[^>]*/>').search(text)
        if tag is None:
            raise ExtractMismatchError(f"{label}: no {name} camera")
        new = _replace_attrs(
            tag.group(0), {"pos": _fmt(np.round(pos, 6)), "xyaxes": _fmt(np.round(xyaxes, 6))}
        )
        text = text[: tag.start()] + new + text[tag.end() :]
    return text


def apply_extract(extract_path: Path, sim_dir: Path = SIM_DIR, check_only: bool = False) -> Report:
    extract = json.loads(Path(extract_path).read_text(encoding="utf-8"))
    report = Report()
    robot = read_mjcf(sim_dir / ROBOT_MJCF)
    staged = sim_dir / STAGED_MESH_DIR
    validate(extract, robot, staged, report)

    mount_z = -extract["workcell"]["table_top_z_mm"] / 1000.0
    mount = robot.mount_pos.copy()
    mount[2] = mount_z
    report.changes.append(f"nexarm_mount z {robot.mount_pos[2] * 1000:.1f} mm -> {mount_z * 1000:.1f} mm")
    cameras = {}
    for name, key in CAMERAS.items():
        if not extract["workcell"].get(key):
            report.warnings.append(f"no {key} in the extract; {name} camera left unchanged")
            continue
        cameras[name] = camera_pose(extract["workcell"][key], mount)
        pos, xyaxes = cameras[name]
        report.changes.append(f"{name} camera -> pos {np.round(pos, 4)} xyaxes {np.round(xyaxes, 3)}")
    for skipped in extract["workcell"].get("skipped_non_box_bodies", []):
        report.warnings.append(f"not exported (not box-shaped): {skipped['occurrence']} / {skipped['body']}")

    collisions = fitted_collisions(staged, extract, robot)
    report.changes.append(f"collision boxes refitted for {', '.join(collisions)}")
    outputs: dict[Path, str | bytes] = {}
    path = sim_dir / ROBOT_MJCF
    outputs[path] = update_mjcf(_read(path), extract, robot, mount_z, report, ROBOT_MJCF, collisions)
    for rel in URDF_FILES:
        path = sim_dir / rel
        outputs[path] = update_urdf(_read(path), extract, rel)
    workcell = workcell_xml(extract, mount)
    for rel in SCENES:
        path = sim_dir / rel
        outputs[path] = update_scene(_read(path), cameras, rel)
        outputs[path.parent / WORKCELL_FILE] = workcell
    for body in extract["bodies"].values():
        outputs[sim_dir / MESH_DIR / body["mesh"]] = (staged / body["mesh"]).read_bytes()

    if not check_only:
        for path, content in outputs.items():
            if isinstance(content, bytes):
                path.write_bytes(content)
                continue
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(content)
        for path in sync_isaac_assets(sim_dir):
            report.changes.append(f"Isaac: updated {path.relative_to(sim_dir).as_posix()}")
    report.print()
    return report


def sync_isaac_assets(sim_dir: Path) -> list[Path]:
    """Mirror the merged MJCF to the URDF collisions and the Isaac Lab scene; returns the files changed."""
    spec = importlib.util.spec_from_file_location(
        "build_isaac_assets", SIM_DIR / "isaac_lab" / "build_isaac_assets.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    changed = []
    for path, text in builder.render(sim_dir).items():
        if path.exists() and path.read_bytes().decode("utf-8") == text:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        changed.append(path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--extract", type=Path, default=SIM_DIR / "fusion_export" / "fusion_extract.json")
    parser.add_argument("--sim-dir", type=Path, default=SIM_DIR)
    parser.add_argument("--check", action="store_true", help="Validate and report without writing files")
    args = parser.parse_args()
    apply_extract(args.extract, args.sim_dir, check_only=args.check)


if __name__ == "__main__":
    main()
