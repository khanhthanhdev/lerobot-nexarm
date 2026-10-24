# so101-nexus simulation best practices, mapped to NexArm

Date: 2026-10-16. This is a read-only study: no NexArm code was changed. It was re-verified after pulling the latest `main`.

## What this report is

**Source:** `C:\Users\Administrator\code\so101-nexus` at commit `7bdb671`. Fetching origin brought no new commits. It is a MuJoCo, Warp, and MJBatch environment suite for the SO101 arm, which has 5 hinge joints plus a gripper.

**Target:** the NexArm MuJoCo twin in this repo at `88b1df68`, which merges PR #2, "one sim asset tree, stack-bowls gym env and an Isaac Lab twin". The twin is made of:

- `src/lerobot/robots/nexarm_sim/`
- `src/lerobot/envs/nexarm.py`
- the model tree `sim/description/mjcf/` (`nexarm.xml`, `scene.xml`, `bowl_stack_scene.xml`, `workcell.xml`)

`sim/fusion_export/` is now untracked export staging. NexArm, like SO101, has 5 arm hinges plus a gripper.

**Method:** so101-nexus is an ordinary source repository, so the work was source analysis, not binary reverse engineering. The REA MCP server also failed to connect. Each claim cites `file:line`. so101 references point into so101-nexus, and NexArm statuses were rechecked against `88b1df68`.

**Port the methods, not the numbers.** SO101 uses Feetech STS3215 servos. NexArm uses Hiwonder HX-30HM and HX-65HM servos (`motors/nexarm/nexarm.py:20,69`). so101's gains, force ranges, and timestep are therefore not targets for NexArm.

**Status legend:**

- **HAVE**: NexArm already does this.
- **PARTIAL**: NexArm does part of it.
- **GAP**: NexArm is missing it.
- **N/A**: optional, or does not apply.
- **FIXED**: closed by PR #2.

## 0. Post-pull verification

I ran the NexArm sim test suites after the pull with `uv run --frozen pytest`:

- `tests/envs/test_nexarm_env.py`
- `tests/robots/test_nexarm_sim*.py`
- `tests/robots/test_nexarm_isaac_assets.py`
- `tests/robots/test_nexarm_stack_bowls.py`
- `tests/test_sim_exports.py`

The result was 95 passed, 8 failed, and 1 skipped. Both causes are specific to this machine, not regressions in the merge.

- **Seven failures are rendering tests.** This Windows Server has no OpenGL driver (`GLFWError 65542: WGL: The driver does not appear to support OpenGL`). The affected tests are `test_nexarm_env_gym_checker`, `obs_types[pixels*]`, `render_rgb`, `test_backend_steps_and_renders`, `test_sim_robot_matches_physical_feature_contract`, and `test_description_scene_camera_rendering`. Rerun them on a machine with a GPU or an EGL/OSMesa setup.
- **One failure is a real portability bug in the stale check:** `test_isaac_assets_are_in_sync_with_the_mjcf` fails with `Stale: sim\isaac_lab\nexarm_scene.json`.
  - With `core.autocrlf=true`, the checkout has CRLF line endings, while `build_isaac_assets.py:502` renders the file with `"\n"`. The `--check` step then compares raw bytes (`:526-528`).
  - With line endings normalized, the content matches, so the JSON is actually in sync.
  - The URDF and xacro pass because their expected text is derived from the file read from disk.
  - Fix: normalize `\r\n` before comparing, or add `sim/isaac_lab/*.json text eol=lf` to `.gitattributes`.

---

## 1. Already in place in NexArm

These are already present, so no rebuild is needed.

- **Physics-step and frame alignment.** The timestep is shrunk so an integer number of substeps spans exactly one frame (`mujoco_backend.py:167-170`). so101 enforces the same rule with a divisibility check (`config.py:875-883`).
- **Solver basics.** NexArm uses `implicitfast`, `cone="elliptic"`, and `impratio="10"` (`sim/description/mjcf/nexarm.xml:3`), which match so101 (`scene.py:29-33`).
- **Collision filtering.** NexArm has adjacent-body `<exclude>` pairs and separate visual and collision classes. PR #2 adds tests that the collisions cover the meshes and that the keyframes are contact-free (`tests/test_sim_exports.py:392,398`).
- **Domain randomization.** NexArm randomizes camera pose and FOV, light, floor and object color, object friction and mass, and joint damping and frictionloss. It caches nominal values and resets to them (`mujoco_backend.py:148-325`). This is broader than so101, which mostly randomizes visuals.
- **Real-to-sim calibration.** NexArm fits kp, damping, and frictionloss scales plus action latency, with an fps guard and a mapping-version guard (`calibration.py`). so101 has no equivalent.
- **Other reset and step machinery:**
  - an action-latency queue (`mujoco_backend.py:373-387`)
  - separate RNG streams for domain randomization and placement (`pick_place_task.py:93-95`)
  - success gated on a hold time plus a slow object (`pick_place_task.py:144`)
  - overlap rejection in stack-bowls (`stack_bowls_task.py:148`)
- **Interface contract.** `robot_type="nexarm_follower"` is shared by sim and real (`nexarm_sim.py:41`). The raw-unit contract at the LeRobot boundary is clamped to `RAW_RANGES`. `send_action` returns the clamped command (`nexarm_sim.py:121`).
- **Twin consistency, new in PR #2.** The URDF collisions, joint dynamics, and `gripper_frame` match MuJoCo to within 10 µm over random poses (`tests/robots/test_nexarm_isaac_assets.py:47,139,158`). The models also compile from any working directory (`tests/robots/test_nexarm_sim_assets.py:28-43`). This is stronger than so101's single TCP-parity test.

---

## 2. Gap table, ordered by priority

### P1: affects sim-to-real transfer or training correctness

| # | Practice (so101 evidence) | NexArm status at `88b1df68` | Action |
|---|---|---|---|
| 1 | **Rank-5 IK.** A 5-DoF arm cannot reach every 6-D pose. so101 uses damped least squares (DLS) over position and orientation, with the orientation rows down-weighted (`ee_orientation_weight=0.01`). The weight is applied to both the error and the Jacobian (`kinematics.py:25-29,227-232`), with damping 0.05 and q clamped on each iteration (`base_env.py:630-690`). | **PARTIAL.** `solve_ik` handles position only (`mujoco_backend.py:397-466`), so the wrist orientation comes out arbitrary. | Add an optional weighted orientation term. NexArm has the same rank-5 structure. Use it for scripted data generation and for `teleoperate_cartesian.py`, so grasps approach at a consistent pitch. |
| 2 | **Physical grasp predicate.** so101 counts only object-to-finger contacts, requires a normal force of at least 0.5 N on each side, and requires the force-weighted inward normals to oppose (`dot <= -0.3`) (`grasp.py:25-58`, `base_env.py:707-779`). The opposition test rejects both fingers pressing the same face. so101 also aggregates normals over every collision geom of the object (`CHANGELOG.md:302`). | **PARTIAL.** Any contact by both jaws counts as a grasp (`pick_place_task.py:183-191`). There is no force floor and no opposition test. | Add `mj_contactForce` force floors and the opposing-normal test. This matters most where grasp state gates the reward or labels datasets. |
| 3 | **Determinism tests.** so101 checks that same-seed rollouts are bit-identical (`tests/mujoco/test_determinism_goldens.py:179`). It also keeps golden values (`RTOL=1e-5`, regenerated with `REGEN_GOLDENS=1`, `:38,66-71`) and replays a seed across fresh processes (`test_env_invariants.py:79`). | **PARTIAL (mostly fixed by PR #2).** A bit-identical same-seed rollout test now runs with domain randomization and pose noise on, and it checks that a different seed diverges (`tests/envs/test_nexarm_env.py:319`). Golden values and the cross-process check are still missing. | Add a golden-value test, which catches silent drift from MuJoCo upgrades or Fusion re-merges, and a cross-process replay test. |
| 4 | **Reusable invariant suite.** so101 checks that observations stay in the observation space, rewards are finite, seeded resets are deterministic, and random actions never crash (`testing/invariants.py:34,52,73,94`). | **FIXED.** `test_nexarm_env_invariants_under_random_actions` covers both envs and both reward types (`tests/envs/test_nexarm_env.py:330`). | None. |
| 5 | **TCP-frame parity between URDF and MJCF** (`CHANGELOG.md:488`). | **FIXED.** The URDF `gripper_frame` link matches the MJCF site (`sim/description/urdf/nexarm.urdf:233-237`), and a test locks the two together (`test_nexarm_isaac_assets.py:139`). | None. |
| 6 | **Joint velocities in the state.** so101 made `JointVelocities` a default observation (`CHANGELOG.md:461`, a breaking change). | **GAP.** `agent_pos` contains positions only (`envs/nexarm.py:244`). | Velocity transfers to the real arm only if the real follower produces an identical signal. so101 uses a finite difference at the control dt (`lerobot_dataset.py:131-229`). Either compute it the same way in sim and on hardware, or keep it as a privileged sim-only RL observation. Make it opt-in so existing checkpoints keep working. |
| 7 | **Initial-pose noise.** so101 adds about 0.02 rad of uniform noise to the rest pose, then re-clamps (`base_env.py:529-547`). | **FIXED, opt-in.** `init_qpos_noise_rad` exists. It is drawn after domain randomization and applied to the 5 arm joints only (`mujoco_backend.py:327-347`, test `test_nexarm_env.py:364`). The default is `0.0`. | Turn it on (about 0.02) in the RL and scripted-dataset configs. Otherwise every episode still starts at exact HOME. |

### P2: training quality and data quality

| # | Practice (so101 evidence) | NexArm status at `88b1df68` | Action |
|---|---|---|---|
| 8 | **Potential-based shaping with a fixed budget.** so101 rewards `Phi(s') - Phi(s)`, which keeps the optimal policy unchanged (`rewards.py:107-136`). Terms use tanh shaping (`rewards.py:18-36`), facet weights sum to 1 (`config.py:569-582`), and potentials are monotone along the ideal trajectory (`CHANGELOG.md:516`). | **PARTIAL.** Both dense rewards are absolute sums. Pick-place sums reach, grasp, lift, target, a place bonus of 2, and a success bonus of 5 (`envs/nexarm.py:371-407`). Stack-bowls sums reach, grasp, carry, stage (2 per stage), and success (5) (`envs/nexarm.py:460-487`). The scale is unbounded per step, so loitering near the goal pays. | Convert the terms to potential differences, normalize the budget to 1, and keep sparse success separate. Then walk Phi along a scripted IK trajectory from `generate_sim_dataset.py` and assert that every step is non-negative (`skills/reward-engineering/SKILL.md`). |
| 9 | **Reward breakdown in `info`** (`config.py:633-682`). | **FIXED.** `info["reward_components"]` names every term and sums exactly to the reward (`envs/nexarm.py:230-239,324`, tests `:330,350`). | Also record the components in teleop and scripted datasets, as so101 does (`teleop/dataset.py:139-252`). |
| 10 | **Stricter "settled" success.** so101 requires object linear speed of at most 0.01 m/s and angular speed of at most 0.5 rad/s. It also requires a static arm (all `qvel` below 0.2) and no grasp (`pick_and_place.py:227-247`, `base_env.py:781-787`). Its v2 adds a table-support check: support of at least 0.9 times the weight (`placement.py:379-411`). | **PARTIAL.** Pick-place checks linear speed only, against 0.05 (`pick_place_task.py:144`). | Add angular-speed and arm-static checks. Consider the support-force test for bowl stacking. |
| 11 | **Rejection-sampled spawning.** so101 samples in polar coordinates (radius 0.10–0.30 m, ±90°) and rejects samples whose bounding radii plus clearance overlap. It tries 100 times, then falls back (`spawn_utils.py:22-81`). Objects get a random yaw (`:84-99`), and the spawn height is analytic (`:128-173`). | **PARTIAL.** Pick-place applies a single nudge when the cube lands too close to the target (`pick_place_task.py:103-104`), and the cube yaw is always 0 (quaternion `1 0 0 0`). Stack-bowls already rejects overlaps. | Reuse the stack-bowls rejection loop in pick-place and add a random yaw. |
| 12 | **Delta and normalized action modes.** so101 offers 5 modes, absolute or delta, in joint space or end-effector space (`config.py:48-72`). Deltas are scaled to [-1, 1]: 0.05 rad for arm joints and 0.2 rad for the gripper (`base_env.py:80,1084-1115`). Recording always uses absolute joints (`teleop/session.py:198`). | **PARTIAL.** NexArm has only the absolute `raw` and `normalized` modes (`envs/nexarm.py:123,138`). | Add a joint-target delta mode for RL only, and keep datasets absolute. so101 notes that a mismatch between record and train modes fails silently (`control-modes.mdx:165-168`). |
| 13 | **Clamp targets to ctrlrange ∩ jnt_range** (`base_env.py:287-295`, `test_menagerie_model.py:69-82`). | **FIXED.** A test asserts that ctrlrange lies inside the joint range (`tests/robots/test_nexarm_sim_assets.py:63`). | None. |

### P3: physics fidelity and rendering

| # | Practice (so101 evidence) | NexArm status at `88b1df68` | Action |
|---|---|---|---|
| 14 | **NoSlip.** so101 sets `noslip_iterations=3` and also cuts the solver to `iterations=10`, `ls_iterations=20` (`scene.py:29-33`). MuJoCo's defaults are 100, 50, and 0, with the Newton solver (checked with `mujoco.MjOption()`), so so101's 10/20 is a speed cut, not a fidelity gain. | **PARTIAL.** `nexarm.xml:3` uses the defaults: Newton, with NoSlip off. | Benchmark `noslip_iterations=3`, which reduces slip of held objects, with `examples/nexarm/benchmark_sim.py`. Don't copy 10/20 unless profiling shows the solver is the bottleneck. |
| 15 | **Finger contact tuning.** so101 gives finger geoms `condim=6`, `priority=1`, `friction="1 5e-3 5e-4"`, and `solref="0.01 1"` (`SO101_menagerie/so101.xml:31-38`). It also has an optional per-finger `gripper_solimp` override (`physics.py:23-36`). | **PARTIAL.** Finger boxes use `condim=4` and `solimp="0.9 0.95 0.001"` (`nexarm.xml:22`). | Try `condim=6` with `priority=1` on the fingers. This is the prerequisite for #16. |
| 16 | **Grasping with physics only, no assist.** so101 has no weld or attach. | **RISK, larger after PR #2.** Stack-bowls still defaults to `contact_assisted` (`stack_bowls_task.py:72`). The assist now carries the bowl rigidly, including its rotation (`:227`). Policies trained on assisted grasps can learn closures that slip on the real arm. | This is an option, not a mandatory fix. After #15, measure how often unassisted bowl grasps succeed. Keep the assist only where physics cannot hold the bowl, and flag assisted datasets in their metadata. |
| 17 | **Equality-constraint time constant versus MuJoCo's floor.** MuJoCo enforces a minimum time constant of 2 × timestep (refsafe). | **FIXED.** The jaw coupling now uses `solref="0.004 1"` (`nexarm.xml:142`), and a test guards it at every supported fps (`test_nexarm_sim_assets.py:76`). | None. |
| 18 | **Mass and rest-pose provenance.** so101 keeps total mass constant across convex parts (`object_slots.py:218-253`). Masses come from hull volume times density (`GSO_PROVENANCE.md:31-36`), and rest poses are checked with a settle test (`scripts/validate_object_rest_poses.py:148`). | **PARTIAL.** Bowls are 60 g, matching the real ones, and start apart (`tests/test_sim_exports.py:417`). Centres of mass are checked to lie inside the meshes (`:331`). | If mesh objects are added, port the settle-test validator. |
| 19 | **Hide unused objects instead of recompiling.** so101 zeroes their collisions and parks them at z = -10 (`spawn_utils.py:187-195`). | **N/A for now.** | Use this when tasks need variable object counts. |
| 20 | **Bounded shadow-casting key light.** A directional light shadow-maps the infinite floor, which causes speckle moiré in wrist views. so101 uses a spotlight key light (cutoff 25°), a directional fill with `castshadow=false` (`scene.py:51-67`), and `offsamples=8`, `shadowsize=8192` (`scene.py:45-49`). | **GAP.** There is one directional light with default `<quality>` (`sim/description/mjcf/scene.xml:7-17`, `bowl_stack_scene.xml:7-33`). | Adopt the spotlight key light, a non-shadowing fill, and higher `offsamples`/`shadowsize`. The wrist camera feeds the policy. |
| 21 | **Reproducible rendering profile.** so101 has a default profile (8× MSAA, shadows) and a reproducible one (4× MSAA, no shadows). It fixes the profile at construction and records the GL vendor and the MSAA actually granted (`rendering.py:9-58`, `base_env.py:350-434`). | **GAP.** | Add a profile option and record the renderer in dataset metadata. The same headless problem that broke the rendering tests here argues for recording which renderer produced each dataset. |
| 22 | **Camera-intrinsics provenance.** so101 derives its wrist FOV from the sensor geometry (`SO101_menagerie/PROVENANCE.md:13-25`). | **HAVE.** `realsense_fovy.py` and `calibrate_camera_alignment.py` exist. | Record the derivation in `sim/README.md`. |

### P4: LeRobot integration and data contracts

| # | Practice (so101 evidence) | NexArm status at `88b1df68` | Action |
|---|---|---|---|
| 23 | **Observe, then act.** so101 reads the observation before sending the action, and records the action after clipping (`teleop/recorder.py:443-465`, `sim_follower.py:431`). | **HAVE.** | Keep this order in any new recorder. |
| 24 | **Seed the follower from the leader pose at reset** (`teleop/recorder.py:232-247`). | **PARTIAL.** `set_joint_positions` exists (`mujoco_backend.py:349`), and reset now goes through it. | Call it with the leader pose at episode start in sim teleop, so the arm doesn't snap. |
| 25 | **Per-episode reproducibility sidecars** (`teleop/dataset.py:46-82`). | **PARTIAL.** `last_domain_params` exists, but only in memory. | Persist per episode: the seed, the domain-randomization parameters, the MJCF hash, the MuJoCo version, the calibration file, and the grasp-assist mode. |
| 26 | **Explicit public API surface** (`docs/.../api/stability.mdx:10-46`). | **N/A** (this is a fork). | Document the `nexarm_sim` keys and env IDs (`NexArmPickPlace`, `NexArmStackBowls-v0`) as the stable contract. |
| 27 | **EnvHub packaging** (`envhub.py:59-150,229,352`). | **N/A.** | Optional. |
| 28 | **GPU and batched backends.** so101 has a Warp backend with a float64 box-box patch for thin finger boxes, and MJBatch. | **N/A for MuJoCo.** PR #2 adds an Isaac Lab twin (`sim/isaac_lab/nexarm_isaac/`) that has not been run yet. | NexArm's finger collision geoms are also thin boxes. Watch for the same float32 box-box problem in GPU physics. |

---

## 3. so101 anti-patterns to avoid

1. **Mixing units in one vector.** so101 records arm joints in degrees and the gripper as `RANGE_0_100`. Its `RolloutRecorder` then applies `deg2rad` to the whole vector (`policy_adapters/recorder.py:138-141`), which silently corrupts the gripper (`docs/.../concepts/lerobot.mdx:139-145`). NexArm's single raw-unit contract avoids this, so keep it.
2. **Labelling actions inconsistently.** so101's rollout path records the action before clipping, while teleop records it after (`recorder.py:137` vs `sim_follower.py:431`).
3. **Duplicating constants inconsistently.** `forcerange` is ±3.35 in one place and ±2.94 in another (`so101_new_calib.xml:20-27,171-176`).
4. **Config fields that do nothing.** `disable_torque_on_disconnect` is never read (`sim_follower_config.py:26`).
5. **Hard-coding robot names throughout env code** (`base_env.py:243-257`). NexArm centralizes them in `mujoco_mapping.py`.

## 4. Not a NexArm gap

so101 uses `(4096 - 1) / 2π` ticks per radian (`normalization.py:27`). NexArm uses `4096 / 2π`, which is versioned (`JOINT_MAPPING_VERSION`) and comes from the servo constant. Keep NexArm's value.

## Unresolved questions

1. Is the stack-bowls grasp assist (#16) a deliberate long-term choice, or a stopgap until the finger contacts are tuned? PR #2 made it stronger, carrying the bowl rigidly with rotation.
2. Should `init_qpos_noise_rad` (#7) default to non-zero for RL and scripted datasets? The default is currently `0.0`.
3. The Isaac Lab twin has not been run on a GPU yet. Until it has, the claims about it are untested.
