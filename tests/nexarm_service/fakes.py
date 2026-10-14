# Copyright 2026 The NexArm authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); see the LICENSE file at the repo root.

"""Hardware fakes shared by the engine tests: an in-memory arm that records every write."""

import numpy as np

from lerobot.motors.nexarm.nexarm import JOINT_NAMES


class FakeBus:
    def __init__(self, positions=None):
        self.positions = list(positions or [2048] * 6)
        self.writes: list[list[int]] = []
        self.torque_calls: list[bool] = []
        self.bridge_calls: list[bool] = []
        self.connected = False

    @property
    def is_connected(self):
        return self.connected

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def read_positions(self):
        return list(self.positions)

    def write_positions(self, positions):
        self.writes.append(list(positions))
        self.positions = list(positions)  # the arm follows instantly

    def set_torque(self, enable):
        self.torque_calls.append(enable)

    def enter_lerobot_mode(self):
        self.bridge_calls.append(True)

    def exit_lerobot_mode(self):
        self.bridge_calls.append(False)


class FakeFollower:
    robot_type = "nexarm_follower"
    name = "nexarm_follower"

    def __init__(self, bus, with_camera=False):
        self.bus = bus
        self.sent: list[dict] = []
        self.cameras = {"front": object()} if with_camera else {}
        joints = {f"{n}.pos": float for n in JOINT_NAMES}
        self.action_features = dict(joints)
        self.observation_features = {**joints, **dict.fromkeys(self.cameras, (48, 64, 3))}

    @property
    def is_connected(self):
        return self.bus.connected

    def connect(self, calibrate=True):
        self.bus.connect()
        self.bus.set_torque(True)

    def get_observation(self):
        obs = {f"{n}.pos": float(p) for n, p in zip(JOINT_NAMES, self.bus.read_positions(), strict=True)}
        for key in self.cameras:
            obs[key] = np.zeros((48, 64, 3), dtype=np.uint8)
        return obs

    def send_action(self, action):
        self.sent.append(action)
        self.bus.write_positions([int(action[f"{n}.pos"]) for n in JOINT_NAMES])
        return action

    def disconnect(self):
        self.bus.exit_lerobot_mode()
        self.bus.disconnect()


class FakeLeader:
    def __init__(self, action):
        self.action = action
        self.connected = False

    @property
    def is_connected(self):
        return self.connected

    def connect(self, calibrate=True):
        self.connected = True

    def get_action(self):
        return dict(self.action)

    def disconnect(self):
        self.connected = False


class FakeHardware:
    """Injectable replacement for Hardware; follower and leader positions are set by the test."""

    def __init__(self, follower_pos=None, leader_action=None, leader_raw=None):
        self.follower_bus = FakeBus(follower_pos)
        self.follower = FakeFollower(self.follower_bus)
        self.camera_follower = FakeFollower(self.follower_bus, with_camera=True)
        self.leader = FakeLeader(leader_action or {f"{n}.pos": 2048.0 for n in JOINT_NAMES})
        self.buses = {"LEADER": FakeBus(leader_raw), "FOLLOWER": self.follower_bus}

    def make_follower(self, profile, with_cameras=False):
        return self.camera_follower if with_cameras else self.follower

    def make_leader(self, profile):
        return self.leader

    def make_bus(self, port, baudrate):
        return self.buses[port]
