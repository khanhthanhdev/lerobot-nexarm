"""Extract the NexArm robot and its workcell from Autodesk Fusion for MuJoCo.

Executed inside Fusion through the Fusion MCP script runner (see ``sim/run_export.py``). It writes:

- one binary STL per robot body, in the arm frame (the ``base_link`` component frame that every committed
  mesh, joint position and collision primitive uses);
- ``fusion_extract.json`` with per-body mass, centre of mass and inertia in that frame, the Fusion joint
  origins and axes, and the workcell: tabletop height, box-shaped fixtures, and the overhead camera pose.

It does not write MJCF or URDF. The committed models carry hand-tuned actuators, materials, contacts and
sensors, so ``sim/apply_fusion_extract.py`` merges the extracted values into them instead.
"""

import json
import math
import os
import struct

import adsk.core
import adsk.fusion

# Replaced by sim/run_export.py with the repository's sim/ directory, as Fusion sees it.
EXPORT_SIM_DIR = ""

# MJCF body name -> (occurrence path suffix, mesh file name).
BODY_SPECS = {
    "base_link": ("base_link:1", "base_link.stl"),
    "link_1": ("link_1:1", "link_1.stl"),
    "link_2": ("link_2:1", "link_2.stl"),
    "link_3": ("link_3:1", "link_3.stl"),
    "link_4": ("link_4:1", "link_4.stl"),
    "link_5": ("link_5:1", "link_5.stl"),
    "link_6_gripper_base": ("link_6:1+gripper_base:1", "link_6_gripper_base.stl"),
    "link_6_pinion_gear": ("link_6:1+pinion_gear:1", "link_6_pinion_gear.stl"),
    "link_6_left_jaw": ("link_6:1+left_jaw:1", "link_6_left_jaw.stl"),
    "link_6_right_jaw": ("link_6:1+right_jaw:1", "link_6_right_jaw.stl"),
    "cam_mount": ("cam_mount:1", "cam_mount.stl"),
}

# MJCF joint name -> accepted Fusion joint names (current design first, older exports after).
JOINT_NAMES = {
    "joint_1_base_to_link_1": ("joint1", "joint_1_base_to_link_1", "Revolute 6"),
    "joint_2_link_1_to_link_2": ("joint2", "joint_2_link_1_to_link_2", "Revolute 1"),
    "joint_3_link_2_to_link_3": ("joint3", "joint_3_link_2_to_link_3", "Revolute 7"),
    "joint_4_link_3_to_link_4": ("joint4", "joint_4_link_3_to_link_4", "Revolute 8"),
    "joint_5_link_4_to_link_5": ("joint5", "joint_5_link_4_to_link_5", "Revolute 9"),
    "gripper_pinion_joint": ("gripper_base_joint", "gripper_pinion_joint", "Revolute 3"),
    "left_jaw_slide_joint": ("left_jaw_joint", "left_jaw_slide_joint", "Slider 10"),
    "right_jaw_slide_joint": ("right_jaw_joint", "right_jaw_slide_joint", "Slider 11"),
}

TABLE_OCCURRENCE_PREFIX = "workbench"
TOP_CAMERA_OCCURRENCE_PREFIX = "d435i"
# Bodies of the top-camera occurrence that are its clamp, not the camera module.
TOP_CAMERA_CLAMP_MARKER = "clamp"
FRONT_CAMERA_OCCURRENCE_PREFIX = "c270"
# The C270 mount holds no camera model. Its lens sits about this far above the shelf face and ahead of the
# shelf centre (half the webcam's height and depth); verify with tools/nexarm/overlay_recorded_frames.py.
C270_LENS_ABOVE_SHELF_MM = 16.0
C270_LENS_FORWARD_MM = 15.0
# A fixture body becomes a MuJoCo box only if it fills its bounding box this well.
MIN_BOX_FILL = 0.95


def _occurrence(root, suffix):
    matches = [occ for occ in root.allOccurrences if occ.fullPathName.endswith(suffix)]
    if len(matches) != 1:
        found = [occ.fullPathName for occ in matches]
        raise RuntimeError(f"Expected one occurrence ending in '{suffix}', found {found}")
    return matches[0]


def _top_level(occ):
    while occ.assemblyContext:
        occ = occ.assemblyContext
    return occ


def _mm(point):
    return [point.x * 10.0, point.y * 10.0, point.z * 10.0]


def _to_arm(point, world_to_arm):
    p = point.copy()
    p.transformBy(world_to_arm)
    return _mm(p)


def _rotate(vector, matrix):
    v = vector.copy()
    v.transformBy(matrix)
    return [v.x, v.y, v.z]


def _rotation(matrix):
    return [[matrix.getCell(i, j) for j in range(3)] for i in range(3)]


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _transform_stl(path, matrix):
    """Rewrite a binary STL (mm) with ``matrix`` applied to every vertex and normal."""
    with open(path, "rb") as handle:
        data = handle.read()
    count = struct.unpack_from("<I", data, 80)[0]
    rot = _rotation(matrix)
    tx, ty, tz = (matrix.getCell(i, 3) * 10.0 for i in range(3))
    out = bytearray(data[:84])
    for index in range(count):
        values = struct.unpack_from("<12fH", data, 84 + index * 50)
        floats = []
        for offset in range(0, 12, 3):
            x, y, z = values[offset : offset + 3]
            rx = rot[0][0] * x + rot[0][1] * y + rot[0][2] * z
            ry = rot[1][0] * x + rot[1][1] * y + rot[1][2] * z
            rz = rot[2][0] * x + rot[2][1] * y + rot[2][2] * z
            if offset:  # vertices translate, the normal does not
                rx, ry, rz = rx + tx, ry + ty, rz + tz
            floats.extend((rx, ry, rz))
        out += struct.pack("<12fH", *floats, values[12])
    with open(path, "wb") as handle:
        handle.write(bytes(out))


def _is_identity(matrix, tol=1e-9):
    return all(
        abs(matrix.getCell(i, j) - (1.0 if i == j else 0.0)) <= tol for i in range(4) for j in range(4)
    )


def _export_mesh(design, occ, path, native_to_arm):
    opts = design.exportManager.createSTLExportOptions(occ, path)
    opts.isBinaryFormat = True
    opts.isOneFilePerBody = False
    opts.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementLow
    opts.unitType = adsk.fusion.DistanceUnits.MillimeterDistanceUnits
    opts.sendToPrintUtility = False
    if not design.exportManager.execute(opts):
        raise RuntimeError(f"Failed to export STL to {path}")
    # Fusion writes an occurrence's STL in its own component frame and ignores where the assembly places
    # it, so a link offset inside the assembly would otherwise be lost.
    if not _is_identity(native_to_arm):
        _transform_stl(path, native_to_arm)


def _inertial(occ, world_to_arm, arm_rot_t, world_rot):
    props = occ.getPhysicalProperties(adsk.fusion.CalculationAccuracy.HighCalculationAccuracy)
    mass = props.mass
    com_world = props.centerOfMass
    x, y, z = com_world.x, com_world.y, com_world.z
    ok, ixx_o, iyy_o, izz_o, ixy_o, iyz_o, ixz_o = props.getXYZMomentsOfInertia()
    if not ok:
        raise RuntimeError(f"Unable to calculate inertia for {occ.fullPathName}")
    # Moments about the world origin in kg*cm^2 -> about the centre of mass, world axes, kg*m^2.
    ixx = (ixx_o - mass * (y * y + z * z)) * 1e-4
    iyy = (iyy_o - mass * (x * x + z * z)) * 1e-4
    izz = (izz_o - mass * (x * x + y * y)) * 1e-4
    ixy = (ixy_o + mass * x * y) * 1e-4
    ixz = (ixz_o + mass * x * z) * 1e-4
    iyz = (iyz_o + mass * y * z) * 1e-4
    tensor_world = [[ixx, -ixy, -ixz], [-ixy, iyy, -iyz], [-ixz, -iyz, izz]]
    # Arm axes: I_arm = R^T I_world R, where R maps arm axes to world axes.
    tensor = _matmul(_matmul(arm_rot_t, tensor_world), world_rot)
    return {
        "mass_kg": mass,
        "com_mm": _to_arm(com_world, world_to_arm),
        # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz (tensor elements).
        "fullinertia": [
            tensor[0][0],
            tensor[1][1],
            tensor[2][2],
            tensor[0][1],
            tensor[0][2],
            tensor[1][2],
        ],
    }


def _joints(root, world_to_arm, arm_rot_t):
    by_name = {}
    for joint in list(root.allJoints) + list(root.allAsBuiltJoints):
        by_name.setdefault(joint.name, joint)
    result = {}
    for mjcf_name, fusion_names in JOINT_NAMES.items():
        joint = next((by_name[n] for n in fusion_names if n in by_name), None)
        if joint is None:
            raise RuntimeError(
                f"No Fusion joint for {mjcf_name}; tried {list(fusion_names)}, found {sorted(by_name)}"
            )
        motion = joint.jointMotion
        if motion.jointType == adsk.fusion.JointTypes.RevoluteJointType:
            axis, value, kind = motion.rotationAxisVector, motion.rotationValue, "revolute"
        elif motion.jointType == adsk.fusion.JointTypes.SliderJointType:
            axis, value, kind = motion.slideDirectionVector, motion.slideValue, "prismatic"
        else:
            raise RuntimeError(f"Fusion joint {joint.name} is neither revolute nor slider")
        if abs(value) > 1e-6:
            raise RuntimeError(
                f"Fusion joint {joint.name} is at {value}, not 0; meshes must be exported at the zero pose"
            )
        # Joint geometry is reported in the top-level arm assembly's frame; joint axes in world axes.
        origin = joint.geometry.origin.copy()
        origin.transformBy(_top_level(joint.assemblyContext).transform2)
        result[mjcf_name] = {
            "fusion_name": joint.name,
            "type": kind,
            "origin_mm": _to_arm(origin, world_to_arm),
            "axis": [
                sum(arm_rot_t[i][k] * c for k, c in enumerate((axis.x, axis.y, axis.z))) for i in range(3)
            ],
        }
    return result


def _bbox_in_arm(bbox, world_to_arm):
    corners = []
    for x in (bbox.minPoint.x, bbox.maxPoint.x):
        for y in (bbox.minPoint.y, bbox.maxPoint.y):
            for z in (bbox.minPoint.z, bbox.maxPoint.z):
                corners.append(_to_arm(adsk.core.Point3D.create(x, y, z), world_to_arm))
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    return lo, hi


def _fixtures(root, arm_top, world_to_arm):
    boxes, skipped = [], []
    for occ in root.occurrences:
        if occ == arm_top or occ.name.lower().startswith(TOP_CAMERA_OCCURRENCE_PREFIX):
            continue
        for body in occ.bRepBodies:
            if not body.isVisible or not body.isSolid:
                continue
            lo, hi = _bbox_in_arm(body.boundingBox, world_to_arm)
            box_volume_cm3 = math.prod((hi[i] - lo[i]) / 10.0 for i in range(3))
            fill = body.volume / box_volume_cm3 if box_volume_cm3 > 0 else 0.0
            entry = {
                "occurrence": occ.name,
                "body": body.name,
                "material": body.material.name if body.material else "",
                "min_mm": lo,
                "max_mm": hi,
            }
            (boxes if fill >= MIN_BOX_FILL else skipped).append(entry)
    return boxes, skipped


def _top_camera(root, world_to_arm, arm_rot_t):
    matches = [o for o in root.occurrences if o.name.lower().startswith(TOP_CAMERA_OCCURRENCE_PREFIX)]
    if not matches:
        return None
    occ = matches[0]
    bodies = [b for b in occ.bRepBodies if TOP_CAMERA_CLAMP_MARKER not in b.name.lower()]
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for body in bodies:
        b_lo, b_hi = _bbox_in_arm(body.boundingBox, world_to_arm)
        lo = [min(lo[i], b_lo[i]) for i in range(3)]
        hi = [max(hi[i], b_hi[i]) for i in range(3)]
    transform = occ.transform2
    # The camera component is modelled with its lens facing -Z and its long (sensor) axis along X.
    long_axis = [transform.getCell(i, 0) for i in range(3)]
    view_dir = [-transform.getCell(i, 2) for i in range(3)]
    to_arm = lambda v: [sum(arm_rot_t[i][k] * v[k] for k in range(3)) for i in range(3)]  # noqa: E731
    long_axis, view_dir = to_arm(long_axis), to_arm(view_dir)
    # Lens position: the module's face centre on the viewing side.
    lens = [(lo[i] + hi[i]) / 2.0 + view_dir[i] * (hi[i] - lo[i]) / 2.0 for i in range(3)]
    return {
        "occurrence": occ.name,
        "lens_mm": lens,
        "view_dir": view_dir,
        "long_axis": long_axis,
        "module_min_mm": lo,
        "module_max_mm": hi,
    }


def _front_camera(root, world_to_arm, arm_rot_t):
    """Pose of the webcam resting on the tilted shelf of the C270 mount (the camera itself is not modelled)."""
    matches = [o for o in root.occurrences if o.name.lower().startswith(FRONT_CAMERA_OCCURRENCE_PREFIX)]
    if not matches:
        return None
    to_arm = lambda v: [sum(arm_rot_t[i][k] * v[k] for k in range(3)) for i in range(3)]  # noqa: E731
    shelves = []
    for body in matches[0].bRepBodies:
        for face in body.faces:
            if face.geometry.surfaceType != adsk.core.SurfaceTypes.PlaneSurfaceType:
                continue
            ok, normal = face.evaluator.getNormalAtPoint(face.pointOnFace)
            # A shelf faces upward but is tilted; flat tops and vertical walls are not shelves.
            if ok and 0.2 < normal.z < 0.95:
                shelves.append((face.area, face, to_arm([normal.x, normal.y, normal.z])))
    if not shelves:
        raise RuntimeError(f"{matches[0].name} has no tilted upward face to rest a camera on")
    _area, face, up = max(shelves, key=lambda s: s[0])
    # The camera sits on the shelf and looks down its slope: world down projected onto the shelf plane.
    down = to_arm([0.0, 0.0, -1.0])
    dot = sum(d * u for d, u in zip(down, up))  # noqa: B905 - Fusion's Python may predate zip(strict=)
    view = [d - dot * u for d, u in zip(down, up)]  # noqa: B905
    norm = math.sqrt(sum(v * v for v in view))
    view = [v / norm for v in view]
    centre = _to_arm(face.centroid, world_to_arm)
    lens = [centre[i] + up[i] * C270_LENS_ABOVE_SHELF_MM + view[i] * C270_LENS_FORWARD_MM for i in range(3)]
    return {"occurrence": matches[0].name, "lens_mm": lens, "view_dir": view, "up": up, "shelf_mm": centre}


def run(_context: str):
    if not EXPORT_SIM_DIR:
        raise RuntimeError("EXPORT_SIM_DIR is empty; run this through sim/run_export.py")
    app = adsk.core.Application.get()
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise RuntimeError("No active Fusion design found.")
    root = design.rootComponent

    occs = {name: _occurrence(root, suffix) for name, (suffix, _) in BODY_SPECS.items()}
    base = occs["base_link"]
    arm_top = _top_level(base)
    world_to_arm = base.transform2.copy()
    world_to_arm.invert()
    world_rot = _rotation(base.transform2)
    arm_rot_t = _transpose(world_rot)

    # Staging only: sim/apply_fusion_extract.py validates the meshes before copying them into sim/assets/.
    mesh_dirs = [os.path.join(EXPORT_SIM_DIR, "fusion_export", "meshes")]
    for directory in mesh_dirs:
        os.makedirs(directory, exist_ok=True)

    bodies = {}
    for name, (_suffix, filename) in BODY_SPECS.items():
        occ = occs[name]
        native_to_arm = occ.transform2.copy()
        native_to_arm.transformBy(world_to_arm)
        for directory in mesh_dirs:
            _export_mesh(design, occ, os.path.join(directory, filename), native_to_arm)
        bodies[name] = _inertial(occ, world_to_arm, arm_rot_t, world_rot)
        bodies[name]["mesh"] = filename
        bodies[name]["mesh_offset_mm"] = [native_to_arm.getCell(i, 3) * 10.0 for i in range(3)]

    table = [o for o in root.occurrences if o.name.lower().startswith(TABLE_OCCURRENCE_PREFIX)]
    if len(table) != 1:
        raise RuntimeError(f"Expected one '{TABLE_OCCURRENCE_PREFIX}' occurrence, found {len(table)}")
    table_lo, table_hi = _bbox_in_arm(table[0].boundingBox, world_to_arm)
    boxes, skipped = _fixtures(root, arm_top, world_to_arm)

    extract = {
        "schema": "nexarm_fusion_extract_v1",
        "document": app.activeDocument.name,
        "frame": "arm (base_link component) frame, millimetres",
        "bodies": bodies,
        "joints": _joints(root, world_to_arm, arm_rot_t),
        "workcell": {
            "table_top_z_mm": table_hi[2],
            "table_min_mm": table_lo,
            "table_max_mm": table_hi,
            "boxes": boxes,
            "skipped_non_box_bodies": skipped,
            "top_camera": _top_camera(root, world_to_arm, arm_rot_t),
            "front_camera": _front_camera(root, world_to_arm, arm_rot_t),
        },
    }
    out_path = os.path.join(EXPORT_SIM_DIR, "fusion_export", "fusion_extract.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(extract, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"success": True, "extract": out_path, "meshes": len(bodies)}))
