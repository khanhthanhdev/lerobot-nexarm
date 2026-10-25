# Python API: kinematics, dynamics and the motor bus

## Kinematics and dynamics

Module: [`packages/lerobot_robot_nexarm/src/lerobot_robot_nexarm/motors/kinematics_dynamics.py`](../../packages/lerobot_robot_nexarm/src/lerobot_robot_nexarm/motors/kinematics_dynamics.py). The solver is backed by MuJoCo:

```python
import numpy as np
from lerobot_robot_nexarm.motors import NexArmKinematicsDynamics

kd = NexArmKinematicsDynamics()

# 5 arm joint positions in radians
q = np.array([0.0, 0.4, 0.8, -0.4, 0.0])

# 1. Forward Kinematics
fk = kd.forward_kinematics(q)
print("EE Position [x, y, z]:", fk["position"])
print("EE RPY (degrees):", np.rad2deg(fk["rpy"]))

# 2. Geometric Jacobian (6 x 5)
J = kd.jacobian(q)
print("Jacobian shape:", J.shape)

# 3. Damped Least-Squares Inverse Kinematics
target_pos = np.array([0.22, 0.0, 0.15])
success, q_sol = kd.inverse_kinematics(target_pos, q_init=q)
if success:
    print("IK Solution (rad):", q_sol)

# 4. Mass / Inertia Matrix M(q)
M = kd.mass_matrix(q)

# 5. Gravity Compensation Torques
tau_g = kd.gravity_torques(q)
print("Gravity torques (Nm):", tau_g)
```

## Direct motor bus access

Directly query and command the 6 HX-30HM servos via `NexArmMotorsBus`:

```python
from lerobot_robot_nexarm.motors import NexArmMotorsBus

bus = NexArmMotorsBus(port="/dev/ttyUSB1")
bus.connect()

# Set motion profile: speed (0-3400 raw units/s), acceleration ramp (0-254)
bus.set_motion_profile(speed=2000, acc=100)

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
