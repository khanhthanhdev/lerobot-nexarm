# NexArm video2sim environment

Copy `environment.example.json` outside the repository, fill every absolute
external path, and run `uv run python preflight.py <manifest> --check-imports`.
No stage assumes a `/home/...` layout.

`recipe.json` preserves the validated AlohaMini baseline: landscape capture,
8 FPS extraction, LingBot poses/depth, 4 mm TSDF, 350k dense initialization,
10k packed iterations, spatial-only pruning, and a 300k-triangle collider.
Reconstruction, NuRec export, and Isaac remain in the interpreters declared by
the manifest; they are intentionally not dependencies of LeRobot's base lock.
