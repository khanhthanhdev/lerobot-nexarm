# Cross-Project Consistency Review — last 15 commits on main

Range: `31689507~1..HEAD` (63 files, +6249/−527). Three parallel read-only reviews (sim/data-gen, training/rollout/recording, docs), merged and de-duplicated. Key claims spot-checked by the lead against source. No tests were run.

Shared conventions that **are** consistent: joint names/order (`JOINT_NAMES`), `{joint}.pos` keys in raw servo ticks, gripper raw range 1195–2833, 480×640 @ 30 fps, camera names front/wrist/top, stats keys `action`/`observation.state`.

## Critical — wrong behavior or broken pipeline by default

1. **Leader/follower serial ports swapped between scripts and docs.** `prepare_collection.py:23-24` and `DATA_COLLECTION_GUIDE.md:97-99` use follower=USB0/leader=USB1; `calibrate_arms.py:27-28`, `record.yaml:4,31`, `teleoperate.yaml`, `inference.yaml:5`, `rollout_turbovla.py:95`, `teleoperate_cartesian.py:39`, `vision_grasp.py:40`, `run.md:236` and `README.md:270` use the reverse. `run.md` contradicts itself (`:236` vs `:588-589`). Mixing commands drives the passive leader as the follower and records state from the wrong arm.
2. **Sim raw-tick → joint-angle scale differs from the servo.** `mujoco_backend.py:169-186` stretches 0..4095 over each actuator `ctrlrange` (±135°, ±120°, ±100°…), while HX-30HM is 4096 ticks per 360° (`docs/2_ESP32_Development_Basics.md:166`). Only `wrist_roll` matches. Sim and real datasets encode different angles for the same number; mixed training and sim→real replay are inconsistent, and `calibrate_sim.py` absorbs the error into kp. Verified: arm `RAW_RANGES` is the full 0..4095 (`nexarm.py:65-66`), so pan maps 270° over 4096 ticks (0.066°/tick vs 0.088°). Arm `jnt_range` equals `ctrlrange`, so the kinematics converter agrees with the backend on arm joints. (Needs hardware confirmation that joints are direct-drive.)
3. **Gripper direction disagrees between the two raw↔MuJoCo converters.** `mujoco_backend.py:173-174,184-185` inverts the gripper (2833 → closed); `motors/nexarm/kinematics_dynamics.py:114-127` does not (2833 → open) and uses `jnt_range` instead of `ctrlrange`. `sim/README.md` documents a third version. `teleoperate_cartesian.py:98` vs `:120` also disagree on which raw value is "open".
4. **Top camera on by default in collection, off by default elsewhere.** `prepare_collection.py:33-36` (`--top-cam default="auto"`) requires a RealSense and `pyrealsense2` unless `--no-top-cam`; no doc mentions `--no-top-cam`, `intelrealsense` is in no `run.md` profile (only `pipeline.md` installs it), and `collect.sh` uses `--no-sync`. Without a RealSense: crash at startup. With one: 3-camera sessions fail to merge into the 2-camera Hub root `thanhkt/nexarm_stack_bowls` after recording (`collection_merge.py:57` → `aggregate.py` feature check). ACT trained on that data needs `observation.images.top`, which `rollout.py:41` and `inference.yaml` omit by default. Sim defaults to 3 cameras (`config_nexarm_sim.py:30`, `envs/nexarm.py:88`, benchmark `:563`, `generate_sim_dataset.py` has no option), and `tests/robots/test_nexarm_sim.py:176-181` asserts 3 cameras in a test named "matches physical feature contract".
5. **Sim and real datasets cannot be merged with standard tools.** Sim writes `robot_type="nexarm_sim"` (`generate_sim_dataset.py:344`, `generate_stack_bowls_dataset.py:460`), real writes `"nexarm_follower"`; `validate_all_metadata` requires equality (plus the camera set from item 4). Only `train_turbovla.py:731` works around it by intersecting cameras.
6. **Timestamped repo ids break documented follow-up commands.** `lerobot_record.py:385` → `stamp_repo_id()` appends `_YYYYMMDD_HHMMSS`, but `run.md:389/400/410` (viz/replay/train), `sim_calibration.md:51,67`, and `DATA_COLLECTION_GUIDE.md:198,208` reuse the plain id. `pipeline.md:47` collects into the default root `thanhkt/...` (`prepare_collection.py:44`) but trains from `<user>/nexarm_stack_bowls` via Hub (`pipeline.md:74`) → 404.

## High — TurboVLA train/rollout contract not persisted

7. **Rollout guesses the sim task from file names.** `rollout_turbovla.py:364-370` picks stack_bowls only if `num_views==3` or the path contains "bowl"/"paper_finetuned"/"turbovla_ddp". The documented co-train run (`turbovla_guide.md:113-115`, `--cameras front,wrist`, output `nexarm_turbovla_cotrain`) is rolled out in the pick-place scene with the pick-cube prompt.
8. **Rollout ignores the trained image size.** `train_turbovla.py:755-756` forces the size and saves `vision.image_size`; `diagnose_turbovla.py:86-90` restores it; `rollout_turbovla.py:200` hard-codes 224 and uses default processor preprocessing.
9. **Camera keys/order not saved.** Training allows any subset/order (`train_turbovla.py:293-305`) but `config.json` saves only `num_views`; rollout always feeds `front,wrist,top`[:n] and pads missing views silently (`rollout_turbovla.py:317-325,391-393`). `diagnose_turbovla.py:83` detects cameras from one dataset while training intersects all datasets.
10. **Prompts at rollout differ from training.** Default `--task` is the pick-cube prompt (`rollout_turbovla.py:67`), used verbatim in `run_real`; real data uses "Stack the bowls with red on bottom…" (`prepare_collection.py:47-48`). `run.md:443` and `turbovla_guide.md:154,168` use off-distribution prompts; `pipeline.md:100` is correct.
11. **Camera source spec differs between collection and rollout/record.** Collection takes `/dev/v4l/by-id` paths and per-camera fourcc (MJPG/YUYV); `record.py`, `teleoperate.py`, `rollout.py`, `rollout_turbovla.py` take `type=int` and no fourcc, although `pipeline.md:56` says they "take the same camera flags".

## Medium — sim fidelity and generator divergence

12. **Stack-bowls labels the delayed action, pick-place labels the commanded action.** `generate_stack_bowls_dataset.py:216-228,408-410` records `send_action`'s return (delayed when `action_delay_steps>0`); `generate_sim_dataset.py:189-197` records the command (enforced by a test); real recording labels the teleop action. `NexArmSim.send_action` and `NexArmFollower.send_action` return different things.
13. **Physics step ≠ control period.** `mujoco_backend.py:161-162`: `round((1/30)/0.002)=17` → 34.0 ms per frame vs 33.3 ms stamped (~2% fast).
14. **Calibration fps never checked; delay precedence differs.** `calibrate_sim.py:280` saves fps; loaders ignore it. `NexArmSim.connect` lets calibration override config delay; the generator lets the CLI override calibration.
15. **Calibration/DR only in the pick-place generator.** Not available in the benchmark (`lerobot_nexarm_sim_benchmark.py:372-379`), `NexArmPickPlaceEnv`, or the stack-bowls generator, contradicting `sim_calibration.md:126`.
16. **Opposite DR defaults; `--dr` means different things.** On by default (no-op flag) in `generate_sim_dataset.py:375-418`, off by default in `generate_stack_bowls_dataset.py:678-683`. Object DR is hard-coded to `cube` (`mujoco_backend.py:245-262`), so bowls never randomize. `apply_calibration` also replaces the asymmetric default DR ranges for all joints, including the unfitted gripper.
17. **Real speed cap not modelled in sim.** `NexArmFollowerConfig.motion_speed=2000` vs uncapped sim; `nexarm.py:48` and `:275-278` contradict each other on whether the cap applies to CMD 97.
18. **"Shared" helpers used by only one generator.** `sim_dataset_utils.py` is unused by the stack-bowls generator, which has different provenance files, no train/eval seed split, and no `--workers`. Gripper open/closed thresholds are scattered (2400/1600, 1800, 2000/2400).
19. **Collection sessions can vary fps/resolution/cameras but merge requires identical features**; the failure only appears after recording. `train_stack_bowls_act.py:44,55` hard-requires 30 fps and 480×640. `collection_merge.py:40` fails on the first session for a new root repo.

## Docs — commands that error as written

20. Non-existent flags: `--num-episodes` → `--episodes` (`run.md:187,192`, `README.md:243`); `--camera-index` → `--cam-index` (`run.md:323`, `README.md:356`); camera alignment `--camera top --real-camera-index --blend-alpha` and `[`/`]` keys (`run.md:335-341`, `README.md:436-438`) vs `--camera-name {front,wrist}`, `--cam-index`, `+`/`-` (`calibrate_camera_alignment.py:30-31,186-188`).
21. `pipeline.md:51`: bare `--top-cam` is an argparse error (no `nargs="?"`). `sim_calibration.md:148` asks to align `top`, which the tool cannot select.
22. `run.md:587-592` recommends a fixed `--dataset.root` that hits `FileExistsError` on the second run (`dataset_metadata.py:792`) and collides with the sim generator's default output.
23. `pipeline.md:122-139` benchmarks stack-bowls checkpoints with the pick-and-place benchmark. `pipeline.md:37` treats `--gpus N` as a count; it is a comma-separated ID list.
24. Install profiles conflict (`pipeline.md:20-23` vs `stack_bowls_act.md:6` vs `run.md:73-88`); a narrower `uv sync` removes `nexarm`/`intelrealsense`/test extras.
25. Minor drift: `pipeline.md:78` `--cameras` default; `AGENT_GUIDE.md:216` vs `:135` reset time; `AGENT_GUIDE.md:460` top camera height/RGB-D; `sim_calibration.md:25` "cameras optional"; `stack_bowls_act.md:3` episode count; `run.md` links an absolute `file:///home/marinelab/...` path.

## Low — help text vs code

26. `merge_hub_datasets.py:152` documents a cache default but uses a temp dir, and deletes source Hub repos unless `--keep-sources` (not in docstring). `upload_dataset.py:51-55` `--large-folder` is `store_true` with `default=True` (no-op). Rollout fps is not tied to the saved dataset fps (harmless while everything is 30 fps).

## Suggested fix order

1. One leader/follower port convention everywhere (item 1).
2. Decide the sim tick→angle mapping and gripper direction (items 2–3), then re-run `calibrate_sim`.
3. Make the top camera consistently opt-in or opt-out across collection, sim defaults, ACT/rollout and docs; add a `robot_type` override for sim datasets (items 4–5).
4. Persist task type, camera keys/order, image size and fps in the TurboVLA `config.json` and apply them in `TurboVLAPolicyRunner` (items 7–10).
5. Fix doc commands and dataset-id flow (items 6, 20–24).

## Unresolved questions

- Is real gripper raw 2833 physically closed?
- Are the arm joints direct-drive (so 4096 ticks = 360° at the joint)? Does shoulder_lift sign/zero match between sim and the real follower?
- Does the CMD 56 speed cap apply to CMD 97 writes?
- Which port convention is canonical (USB0 = leader or follower)?
