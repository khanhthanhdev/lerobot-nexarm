# ManiSkill data environment

This directory is a separate `uv` project because ManiSkill/SAPIEN GPU wheels
must not constrain LeRobot's base environment. From this directory run
`uv sync`, then set `PYTHONPATH` to the LeRobot checkout's `src` directory.

`agent.py` registers the generated URDF as `mobile_bi_nexarm`, using joint
names for every controller and qpos lookup. `skills.py` implements the staged
pick/place trace and bounded repair policy. `generate.py` launches parallel
ManiSkill environments and writes accepted episodes through the neutral
artifact schema.
