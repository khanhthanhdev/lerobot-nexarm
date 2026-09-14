# Collision Geometries

In accordance with the design patterns of **MuJoCo Menagerie** and **OpenArm**, collision dynamics in this description package are primarily handled by optimized geometric primitives (`capsule` and `box` tags) defined directly inside the URDF and MJCF models.

Using primitives rather than raw CAD meshes provides:

1. **Zero oscillation / micro-chatter** during high-speed contact and grasping.
2. **Deterministic, analytical contact normals** for physics solvers.
3. **Low computational overhead** for massively parallel simulation (Isaac Lab / MuJoCo Warp / MJX).

If convex decomposition meshes (`*_collision.stl`) are required for specific geometry, they should be placed in this directory.
