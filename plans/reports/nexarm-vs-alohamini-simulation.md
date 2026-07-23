# Feature Comparison: NexArm and AlohaMini Simulation

## Source manifest

- Source: `/home/thanh/code/vinuni/lerobot_alohamini`
- Source ref: `main` at `7d9022dc71baa811740983c43b1feb01db958630`
- Local project: `/home/thanh/code/vinuni/lerobot-nexarm`
- Local ref: `main` at `1cf6b33a12c18145e7f33e599368f5d3aab1f967`
- Mode: compare only
- Scope: robot physics, task environments, data generation, sim-to-real contracts, and photoreal scenes

## Verdict

AlohaMini has more simulation infrastructure, but NexArm has a better starting
contract for incremental development. Extend `NexArmSim` into a mobile bimanual
MuJoCo robot first. Port AlohaMini's task/reset/reward and dataset-generation
ideas after the real 16-dimensional robot contract is stable. Treat the
phone-video-to-Isaac pipeline as a later visual-domain project, not a prerequisite
for controlling two arms, a base, and a lift.

## Head-to-head

| Aspect | AlohaMini | NexArm | Recommendation |
| --- | --- | --- | --- |
| Primary physics | ManiSkill/SAPIEN/PhysX | MuJoCo | Stay MuJoCo-first |
| Robot model | Dual arms, virtual planar base, lift, parallel grippers | One fixed arm and gripper | Create one composite mobile-bimanual MJCF |
| LeRobot contract | Converted after episode generation by a bridge | `NexArmSim` directly implements `Robot` | Preserve the direct `Robot` interface |
| Physical feature compatibility | Bridge currently targets only the 16-D 5-DoF profile | Same six feature names and raw ranges as physical NexArm | Expand the existing contract to 16-D |
| Task environments | Reset, randomization, reward, success, YCB objects | Red cube scene without task environment | Port the task semantics, not the engine |
| Automated data | Scripted skills, planner/repair loop, batch episode writers | Manual leader recording and policy rollout | Add deterministic scripted tasks after physics validation |
| Parallel simulation | GPU-vectorized ManiSkill | Single MuJoCo instance | Start single-instance; vectorize only when data throughput is limiting |
| Photoreal scenes | Phone video to splat/collider to Isaac USD | Simple rendered scene | Defer until control and task behavior work |
| Tests | Bridge and import tests; heavy engines need external environments | Raw mapping, rendering, contact, and physical feature-contract tests | Keep NexArm's focused tests and add whole-robot/task tests |

## AlohaMini simulation anatomy

AlohaMini contains two largely separate products:

1. `data_engine` vendors ManiSkill robot agents, task environments, scripted
   skills, planners, success checks, and episode writers. The base is represented
   by virtual root X/Y/yaw joints rather than simulated Kiwi wheels.
2. `video2sim` reconstructs a room from phone video, trains a Gaussian splat,
   creates a collision mesh, and assembles an Isaac Sim scene. It needs several
   external environments and substantial NVIDIA GPU resources.

The LeRobot bridge converts simulated root positions into body velocity,
converts lift metres into millimetres, and exports a 16-D dataset. It explicitly
leaves arm-unit alignment to the caller, and it currently uses the 5-DoF arm
profile even though a 6-joint Pro V3 simulator exists.

## NexArm simulation anatomy

`NexArmSim` already:

- registers as a LeRobot `Robot`;
- accepts the same raw six-position action dictionary as `NexArmFollower`;
- converts raw positions to MuJoCo radians/metres;
- renders front and wrist cameras;
- supports physical-leader teleoperation, dataset recording, replay, and policy rollout;
- tests raw conversion, camera rendering, jaw contact, clamping, and feature compatibility.

It does not yet provide a Gymnasium task with automatic reset, reward, success,
termination, or randomized object placement. Its actuator gains and contact
model are stable approximations rather than identified physical motor dynamics.

## Recommended simulated embodiment

```text
MobileBiNexArmSim (same 16-D LeRobot contract as real robot)
├── planar chassis
│   ├── x
│   ├── y
│   └── yaw
├── lift slide
├── left NexArm
├── right NexArm
├── front/chest camera
└── left/right wrist cameras
```

Initially, use virtual planar X/Y/yaw joints like AlohaMini instead of modelling
wheel-ground contact. This tests policies, reachability, collisions, cameras,
and task sequencing without tyre-friction tuning. Add explicit Kiwi wheels only
when wheel slip, odometry, or low-level chassis control is itself part of the
research question.

## Dependency matrix

| Component | Status | Local equivalent or gap |
| --- | --- | --- |
| One-arm MuJoCo model | EXISTS | `sim/fusion_export/scene.xml` |
| One-arm LeRobot simulator | EXISTS | `NexArmSim` |
| Physical raw-action mapping | EXISTS | `NexArmMujocoBackend` |
| Two-arm model | NEW | Duplicate/mirror arm bodies under a shared lift carriage |
| Planar base | NEW | Virtual X/Y/yaw joints first |
| Lift | NEW | One limited slide joint |
| Full 16-D feature contract | NEW | Must match `MobileBiNexArm` exactly |
| Automatic task reset | NEW | Reset robot and objects deterministically |
| Reward/success/termination | NEW | Gymnasium or LeRobot environment wrapper |
| Object randomization | NEW | Seeded pose and object variants |
| Scripted demonstrations | NEW | Deterministic state-machine or Cartesian skills |
| Photoreal room reconstruction | OPTIONAL | AlohaMini `video2sim` idea |
| Explicit Kiwi wheel physics | OPTIONAL | Add only for locomotion-system research |

## Challenge decisions

| # | Decision | AlohaMini answer | NexArm answer | Risk if wrong | Choice |
| --- | --- | --- | --- | --- | --- |
| 1 | Which simulator first? | ManiSkill plus Isaac tools | MuJoCo already works in-repo | A new engine delays usable results and doubles maintenance | Extend MuJoCo |
| 2 | Wheels or virtual base? | Virtual planar base in ManiSkill | No base yet | Wheel contact tuning can consume the project before manipulation works | Use X/Y/yaw joints first |
| 3 | What is the canonical action schema? | Bridge-generated 16-D profile with known profile drift | Direct physical-compatible six-D schema | Sim data becomes unusable if names, order, or units diverge | Define the real 16-D schema once and import it in both |
| 4 | Is photoreal reconstruction necessary? | Supported through a complex video-to-Isaac pipeline | Simple cameras and scene | Heavy dependencies distract from control and task correctness | Defer photoreal scenes |
| 5 | How should tasks be evaluated? | ManiSkill reset/reward/success methods | No automatic task environment | Policies cannot be compared reproducibly | Port task semantics early |
| 6 | How should demonstrations be generated? | Scripted skills and planner/repair loops | Physical leader or learned rollout | Complex planning may hide model/contact errors | Start with deterministic state machines |
| 7 | Is sim-to-real ready? | Arm units and newer Pro schema are not aligned | Raw interface matches, dynamics are approximate | A policy can exploit inaccurate dynamics or zero offsets | Calibrate kinematics, cameras, latency, and noise before transfer |

## Risk

Overall risk is **medium** for a virtual-base mobile bimanual MuJoCo model and
**high** for immediate wheel-contact plus photoreal Isaac adoption. The critical
unknowns are the final mechanical assembly, joint frames, lift geometry, camera
extrinsics, collision geometry, and real chassis feedback.

## Recommendation

The most valuable AlohaMini feature to adopt next is its task layer:
deterministic reset, seeded object randomization, reward, success, and scripted
episode generation. Build those on top of an expanded `NexArmSim`, while keeping
the action names and physical raw ranges identical to the eventual real
`MobileBiNexArm`.

This comparison intentionally stops before an implementation plan or code port.
