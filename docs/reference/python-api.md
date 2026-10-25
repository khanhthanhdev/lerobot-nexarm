# Python API: kinematics, dynamics and the motor bus

## Kinematics and dynamics

Module: [`packages/lerobot_robot_nexarm/src/lerobot_robot_nexarm/motors/kinematics_dynamics.py`](../../packages/lerobot_robot_nexarm/src/lerobot_robot_nexarm/motors/kinematics_dynamics.py). The solver is backed by MuJoCo:

```python
import numpy as np
from lerobot_robot_nexarm.motors import NexArmKinematicsDynamics

kd = NexArmKinematicsDynamics()

# 5 arm joint positions in radians (an array), or a {joint_name: raw servo value} mapping
q = np.array([0.0, 0.4, 0.8, -0.4, 0.0])

# 1. Forward Kinematics: position, rotation_matrix, quaternion_wxyz, rpy_rad, rpy_deg
fk = kd.forward_kinematics(q)
print("EE Position [x, y, z]:", fk["position"])
print("EE RPY (degrees):", fk["rpy_deg"])

# 2. Geometric Jacobian (6 x 5)
J = kd.compute_jacobian(q)
print("Jacobian shape:", J.shape)

# 3. Damped Least-Squares Inverse Kinematics: raw servo positions per joint, or None
target_pos = np.array([0.22, 0.0, 0.15])
q_sol = kd.inverse_kinematics(target_pos)  # optional: target_rpy=..., current_joint_positions={name: raw}
if q_sol is not None:
    print("IK Solution (raw):", q_sol)

# 4. Mass / Inertia Matrix M(q)
M = kd.mass_matrix(q)

# 5. Gravity Compensation Torques, per joint
tau_g = kd.gravity_compensation_torques(q)
print("Gravity torques (Nm):", tau_g)
```

## Direct motor bus access

Directly query and command the 6 HX-30HM servos via `NexArmMotorsBus`:

```python
from lerobot_robot_nexarm.motors import NexArmMotorsBus

bus = NexArmMotorsBus(port="/dev/ttyUSB1")
bus.connect()

# CMD 56: acceleration ramp (0-254) and speed limit (0-3400 raw units/s, 0 = no limit)
bus.write_motion_params(acc=50, speed=0)

# Read 6 servo positions (0-4095 range)
positions = bus.read_positions()
print("Current positions:", positions)

# Enable / disable torque
bus.set_torque(False)  # Free movement
bus.set_torque(True)   # Active hold

bus.disconnect()
```

## Idle torque timeout

To prevent servo overheating during pauses in teleoperation or data collection, `NexArmFollowerConfig` has an automatic idle torque timeout:

- `idle_timeout_s`: Defaults to `20.0` seconds.
- When no new command is received for $> 20$ seconds, motor torque automatically turns off.
- The next `send_action()` seamlessly re-engages torque.
