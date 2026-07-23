# Feature Comparison: Mobile Bimanual NexArm

## Source manifest

- Source: `/home/thanh/code/vinuni/lerobot_alohamini`
- Source ref: `main` at `7d9022dc71baa811740983c43b1feb01db958630`
- Local project: `/home/thanh/code/vinuni/lerobot-nexarm`
- Local ref: `main` at `1cf6b33a12c18145e7f33e599368f5d3aab1f967`
- Mode: compare only
- Scope: dual follower arms, dual leader arms, mobile base, lift, transport, action schema, safety, recording, and simulation

## Verdict

Use AlohaMini's whole-robot contract as the product model, but build it from the
existing NexArm components instead of copying AlohaMini's hardware topology.
The target should be one composite LeRobot robot containing two independent
`NexArmFollower` instances, one chassis controller, one lift controller, and
top-level cameras. Each physical subsystem should keep its own connection,
watchdog, calibration, and tests.

For two current NexArms on a holonomic base, the natural action/state contract
has 16 values:

1. Six left-arm values: five positioning joints plus gripper.
2. Six right-arm values: five positioning joints plus gripper.
3. Three body-frame velocities: `x.vel`, `y.vel`, and `theta.vel`.
4. One lift position: `lift_axis.height_mm`.

## Head-to-head

| Aspect | `lerobot_alohamini` | Current `lerobot-nexarm` | Recommendation |
| --- | --- | --- | --- |
| Embodiment | Two arms, three-wheel Kiwi base, lift, cameras | One leader and one follower arm, optional cameras | Add a composite embodiment; retain the existing single-arm classes |
| Arm transport | Two Feetech buses; left bus also owns wheels and lift | One USB serial/ESP32/AT32 path per arm | Use one independent USB serial path per arm |
| Bimanual structure | AlohaMini implements both buses directly in one large class | The repo already contains `BimanualMixin` and compositional bimanual examples | Follow `BiSOFollower`/`BiSOLeader`, not AlohaMini's monolith |
| Action names | `arm_left_*`, `arm_right_*`, base velocity, lift height | Unprefixed six-joint action | Use `left_*`, `right_*`, `x.vel`, `y.vel`, `theta.vel`, `lift_axis.height_mm` |
| Base control | Kiwi inverse kinematics on the robot host, raw wheel velocity on the motor bus | No LeRobot chassis component; vendor firmware contains chassis commands | Add a typed chassis component and keep kinematics above the firmware driver |
| Lift control | Velocity-mode motor with multi-turn position tracking, homing, soft limits | No lift component | Add a lift interface with absolute height observation, homing, limits, and timeout |
| Teleoperation | Two leaders plus keyboard merged in custom scripts | Generic LeRobot teleoperate/record with one teleoperator | Add `BiNexArmLeader`; add base/lift input through a composite teleoperator or action processor |
| PC/robot split | ZMQ client on PC, host process on Raspberry Pi | Arm is directly connected to the LeRobot process | Add a remote boundary only when the robot carries a Pi/Jetson; keep the physical composite behind it |
| Dataset | One stable schema for arms, body velocity, lift, and cameras | Six arm actions and observations | Define the final schema before recording any production dataset |
| Safety | Base/lift watchdog, current protection, lift guards; arm watchdog is incomplete | Arm clamping, corrupt-read filtering, controlled torque-off; no command watchdog | Combine both approaches and stop every subsystem on stale commands |
| Tests | Simulation bridge tests; little physical-controller isolation | Focused protocol and follower lifecycle tests | Preserve NexArm's test style and add hardware-free chassis, lift, composite, and timeout tests |
| Simulation | ManiSkill/Isaac stack; bridge currently fixed to a 5-DoF arm profile | One-arm MuJoCo model | Build the real 16-D contract first, then make simulation emit that exact contract |

## Source anatomy worth learning

### Stable whole-robot schema

AlohaMini exposes both arm states, body-frame velocity, and lift height through
the same `observation_features` and `action_features`. This makes recording,
training, replay, and policy evaluation use one contract. The important idea is
the stable named schema, not AlohaMini's exact class layout.

### Body-space base commands

Policies and teleoperators use `x.vel`, `y.vel`, and `theta.vel`; only the robot
driver converts these into wheel commands. This prevents wheel order, wheel
radius, and chassis geometry from leaking into datasets and policies.

Copy the Kiwi matrix only if the physical base has three independently driven
omni wheels at the assumed mounting angles. A differential or mecanum chassis
needs its own kinematics while preserving the body-space API.

### Separate command and observation transport

AlohaMini conflates queued commands so the newest command wins and bounds
observation requests so stale camera frames cannot grow without limit. This is
useful when the robot computer is remote, but the transport should wrap the
whole composite robot rather than be embedded separately in both arms.

### Lift as an observable actuator

The lift is represented by absolute millimetres, even though its motor is driven
in velocity mode. That is the correct policy-facing abstraction. The missing
piece to improve is persistent or absolute position feedback: a software
multi-turn counter loses its reference after restart and must home safely.

### One recording loop

AlohaMini merges leader-arm action, keyboard base action, and keyboard lift
action before calling `robot.send_action()`. The dataset therefore records the
command actually associated with each observation. Keep this single timestamped
loop; do not record arms and chassis in independent processes and merge them
later.

## Local anatomy to preserve

### NexArm protocol isolation

`NexArmMotorsBus` already owns framing, retries, serialization, and the ESP32
bridge commands. Keep it arm-only. A chassis protocol should be another driver
with an equally narrow API rather than additional conditionals in
`NexArmMotorsBus`.

### Existing arm safety and tests

`NexArmFollower` already clamps outgoing positions, filters corrupt endpoint
reads, warns on implausible jumps, configures motion limits, and holds position
before torque-off. Reusing two instances preserves this behavior and lets the
existing tests remain authoritative.

### Existing bimanual composition

`BiSOFollower` and `BiSOLeader` already demonstrate the local convention:
instantiate two normal devices, prefix their features, split actions by prefix,
and delegate lifecycle operations. A `BiNexArmFollower` and `BiNexArmLeader`
should follow this pattern.

## Suggested component boundary

```text
MobileBiNexArm (LeRobot Robot)
├── left_arm: NexArmFollower
├── right_arm: NexArmFollower
├── chassis: KiwiChassis
├── lift: LiftAxis
└── cameras: top-level camera map

BiNexArmTeleoperator
├── left_leader: NexArmLeader
├── right_leader: NexArmLeader
└── base/lift input: gamepad or keyboard
```

`MobileBiNexArm.get_observation()` should gather all subsystem observations into
one named dictionary. `send_action()` should validate the complete action,
route each prefix to its owner, and return the values actually sent. On any
partial connection failure, it should stop and disconnect every subsystem that
was already connected.

## Dependency matrix

| Component | Status | Local equivalent or gap |
| --- | --- | --- |
| Single follower arm | EXISTS | `NexArmFollower` |
| Single leader arm | EXISTS | `NexArmLeader` |
| Dual-arm lifecycle | EXISTS | `BimanualMixin` |
| Dual-arm routing example | EXISTS | `BiSOFollower` and `BiSOLeader` |
| Arm protocol | EXISTS | `NexArmMotorsBus` |
| Composite NexArm robot | NEW | Should own both arms, chassis, lift, and cameras |
| Composite NexArm teleoperator | NEW | Should own both leaders and base/lift input |
| Chassis protocol | NEW | Separate serial driver and configuration |
| Kiwi kinematics | NEW | Adapt equations only after geometry is confirmed |
| Lift protocol/control | NEW | Absolute height, homing, limits, and stop behavior |
| Whole-robot watchdog | NEW | Must stop arms, base, and lift |
| Remote robot transport | OPTIONAL | Add after the local composite contract works |
| Mobile-bimanual tests | NEW | Unit tests plus hardware smoke tests |
| Mobile-bimanual simulation | NEW | Must match the real action schema and units |

## Challenge decisions

| # | Decision | Source answer | Local answer | Risk if wrong | Choice |
| --- | --- | --- | --- | --- | --- |
| 1 | Must the robot be remote? | AlohaMini assumes PC plus Raspberry Pi over ZMQ | NexArm currently runs directly over USB | Added latency and failure modes before the hardware works | Make remote transport a wrapper added after local bring-up |
| 2 | Is the chassis really Kiwi? | Three independently controlled omni wheels | Prior chassis work mentioned an ESP32, L298N, and IMU | Copying the matrix onto different geometry produces unsafe motion | Confirm wheel count, geometry, drivers, and encoders before choosing kinematics |
| 3 | Where should base traffic go? | Through the left arm's Feetech bus | NexArm arm UART GPIO16/17 is already occupied by the AT32 path | Coupling can interrupt arm control and complicate emergency stops | Give the chassis ESP32 its own host connection |
| 4 | How is lift position recovered? | Home on startup and track multiple turns in software | No lift hardware contract exists yet | A lost zero can drive into a hard stop | Require limit switch or absolute reference plus guarded homing |
| 5 | What happens on stale commands? | AlohaMini stops base and lift after one second | NexArm has no command-age watchdog | Arms can retain the last policy target while the platform stops | Add a whole-robot supervisor and firmware-local base timeout |
| 6 | Are reads synchronized enough? | AlohaMini reads buses sequentially at roughly 30 Hz | Two NexArm reads can each retry up to 150 ms | Dataset state may mix different physical times and miss FPS | Timestamp reads, measure latency, and consider concurrent arm polling |
| 7 | Should base/lift be learned immediately? | AlohaMini records every dimension together | Current NexArm tasks and datasets are arm-only | Exploration complexity grows sharply and hides basic arm failures | Validate dual-arm fixed-base operation before learning locomotion |

## Risk

Overall risk is **medium**. The software composition is straightforward because
the repository already has bimanual patterns, but chassis geometry, motor-driver
capacity, lift feedback, power distribution, emergency-stop behavior, and
cross-device timing remain hardware-critical assumptions.

## Recommendation

Build in four boundaries:

1. Prove two followers and two leaders with prefixed actions while the base is
   fixed and the lift is disabled.
2. Add the chassis as a separately tested component with odometry/body-velocity
   feedback and a firmware-local watchdog.
3. Add lift homing, absolute height, soft limits, and mechanical end-stop
   protection.
4. Wrap the completed composite robot in remote transport and only then record
   full mobile-manipulation datasets.

This comparison intentionally stops before an implementation plan or code port.
