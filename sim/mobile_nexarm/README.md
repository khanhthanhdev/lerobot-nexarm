# Mobile bimanual NexArm simulation

`spec.json` is the canonical mechanical and learning contract. Run
`uv run python sim/mobile_nexarm/generate_assets.py` after changing it; the
generator writes `generated/mobile_bi_nexarm.{xml,urdf,usda}` and validates the
source Fusion joint axes and limits before replacing any output.

The first policy profile is deliberately fixed to 16 scalar features and the
`front` plus `left_wrist` RGB cameras. MuJoCo is the CPU reference backend.
ManiSkill and Isaac live under this directory and import their SDKs lazily, so
the base LeRobot environment stays usable without either NVIDIA stack.
