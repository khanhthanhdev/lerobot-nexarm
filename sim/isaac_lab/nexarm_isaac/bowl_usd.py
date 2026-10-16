"""Write a bowl as a USD rigid body with the same contact proxies, mass and inertia as the MJCF bowl.

The MJCF bowl collides through a ring of capsules and a base disc, not through its mesh, because a convex hull
of a bowl fills it and bowls could not nest. The USD copies those proxies one for one, and shows the STL.
Needs only ``pxr`` (Isaac Sim ships it; ``pip install usd-core`` elsewhere).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (n, 3) and triangle indices (m, 3) of a binary or ASCII STL, duplicate vertices merged."""
    raw = Path(path).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0] if len(raw) >= 84 else 0
    if len(raw) == 84 + 50 * count:
        records = np.frombuffer(raw, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), offset=84)
        corners = records["v"].reshape(-1, 3).astype(float)
    else:
        corners = np.array(
            [
                [float(v) for v in line.split()[1:4]]
                for line in raw.decode("ascii").splitlines()
                if "vertex" in line
            ]
        )
    vertices, indices = np.unique(corners, axis=0, return_inverse=True)
    return vertices, indices.reshape(-1, 3)


def _set_pose(xformable, pos, quat_wxyz) -> None:
    from pxr import Gf, UsdGeom

    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
    w, x, y, z = map(float, quat_wxyz)
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))


def write_bowl_usd(bowls: dict, color: str, stl_path: Path, out_path: Path) -> Path:
    """Write ``bowls`` (the ``bowls`` entry of nexarm_scene.json) in ``color`` to ``out_path``."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Bowl")
    stage.SetDefaultPrim(root.GetPrim())

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(float(bowls["mass"]))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, bowls["com"])))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, bowls["inertia"])))
    w, x, y, z = map(float, bowls["inertia_frame_quat"])
    mass.CreatePrincipalAxesAttr(Gf.Quatf(w, x, y, z))

    # Visual: the STL in its own frame, coloured.
    vertices, triangles = read_stl(stl_path)
    visual = bowls["visual"]
    mesh = UsdGeom.Mesh.Define(stage, "/Bowl/visual")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray.FromNumpy((vertices * np.asarray(visual["scale"])).astype(np.float32))
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(triangles)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(triangles.flatten().tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    _set_pose(mesh, visual["pos"], visual["quat"])
    rgba = bowls["colors"][color]
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgba[:3]))])
    look = UsdShade.Material.Define(stage, "/Bowl/Looks/paint")
    shader = UsdShade.Shader.Define(stage, "/Bowl/Looks/paint/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgba[:3])))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(look)

    # Contact: the MJCF proxies, invisible, with the MJCF sliding friction.
    physics = UsdShade.Material.Define(stage, "/Bowl/Looks/contact")
    material = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    material.CreateStaticFrictionAttr(float(bowls["friction"]))
    material.CreateDynamicFrictionAttr(float(bowls["friction"]))
    material.CreateRestitutionAttr(0.0)
    for i, shape in enumerate(bowls["colliders"]):
        path = f"/Bowl/collider_{i:02d}"
        if shape["type"] == "capsule":
            geom = UsdGeom.Capsule.Define(stage, path)
        elif shape["type"] == "cylinder":
            geom = UsdGeom.Cylinder.Define(stage, path)
        else:
            raise ValueError(f"Unsupported bowl collider {shape['type']!r}")
        # MuJoCo capsules and cylinders run along their local Z, with a half length.
        geom.CreateAxisAttr(UsdGeom.Tokens.z)
        geom.CreateRadiusAttr(float(shape["radius"]))
        geom.CreateHeightAttr(2.0 * float(shape["half_length"]))
        geom.CreatePurposeAttr(UsdGeom.Tokens.guide)
        _set_pose(geom, shape["pos"], shape["quat"])
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
            physics, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    stage.GetRootLayer().Export(str(out_path))
    return out_path
