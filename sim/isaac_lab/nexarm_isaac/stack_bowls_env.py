"""Isaac Lab twin of ``NexArmStackBowls-v0``: stack the red, blue and black bowls in the commanded order.

The scene, robot, cameras and task rules come from ``nexarm_scene.json``, which is derived from the MJCF, so this
environment matches the MuJoCo one:

- the same robot (masses, joint limits, collision boxes, servo gains) on the same 33 mm mounting plate;
- the same workcell fixtures, the three cameras at their MJCF poses and fields of view, and the 60 g bowls with
  the MJCF contact proxies;
- the same action units: the follower's raw servo ticks, as ``raw`` or normalized to [-1, 1];
- the same grasp assist: while the gripper is commanded closed and both jaws touch one bowl, the bowl is carried
  rigidly with the gripper; commanding it open releases it;
- the same success test (concentric, nested, released, still and upright for ``success_hold_s``).

Each environment draws its own stacking order; the policy observation lists the bowls bottom, middle, top.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .robot_cfg import make_robot_cfg
from .scene_spec import USD_DIR, SceneSpec, camera_apertures

PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
SPEC = SceneSpec.load()
CONTROL_HZ = SPEC.data["physics"]["control_hz"]
# Physics steps per control step. MuJoCo runs about 2 ms steps; 17 per 30 Hz frame gives 1.96 ms, as there.
DECIMATION = 17


@configclass
class NexArmStackBowlsEnvCfg(DirectRLEnvCfg):
    decimation: int = DECIMATION
    episode_length_s: float = 30.0
    # Follower features: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
    action_space: int = 6
    # joint positions (6, normalized ticks), joint velocities (6), gripper frame (3), bottom/middle/top bowl
    # positions (9), which of them is held (3), previous action (6).
    observation_space: int = 33
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / (CONTROL_HZ * DECIMATION),
        render_interval=DECIMATION,
        gravity=tuple(SPEC.data["physics"]["gravity"]),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=SPEC.data["table"]["friction"],
            dynamic_friction=SPEC.data["table"]["friction"],
            friction_combine_mode="max",  # MuJoCo uses the larger friction of the two geoms
            restitution=0.0,
        ),
    )
    # Each cell holds the workcell (about 1.2 m deep with the front camera post), so keep cells 2 m apart.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=2.0, replicate_physics=True)
    # Filled from nexarm_scene.json when left as None.
    robot_cfg: ArticulationCfg | None = None
    bowl_usd_dir: str = str(USD_DIR)

    # "normalized": actions in [-1, 1] over each servo's raw range (for RL). "raw": servo ticks, as recorded.
    action_mode: str = "normalized"
    position_jitter_m: float = SPEC.task["position_jitter_m"]
    # Filter the MJCF <exclude> pairs and let the rest of the arm collide with itself, as in MuJoCo.
    self_collisions: bool = True

    # Off by default: RL from state does not need them. ``--enable_cameras`` is required when on.
    enable_cameras: bool = False
    camera_width: int = 640
    camera_height: int = 480
    # Where the wrist camera hangs, relative to the robot prim (the URDF importer keeps link names).
    wrist_camera_parent: str = "cam_mount"

    reward_type: str = "dense"
    jaw_contact_threshold_n: float = 0.01


class NexArmStackBowlsEnv(DirectRLEnv):
    cfg: NexArmStackBowlsEnvCfg

    def __init__(self, cfg: NexArmStackBowlsEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.robot_cfg is None:
            cfg.robot_cfg = make_robot_cfg(SPEC)
        if cfg.action_mode not in ("normalized", "raw"):
            raise ValueError(f"action_mode must be 'normalized' or 'raw', got {cfg.action_mode!r}")
        super().__init__(cfg, render_mode, **kwargs)

        device = self.device
        amap = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in SPEC.action_map.items()}
        self._raw_low, self._raw_high = amap["raw_low"], amap["raw_high"]
        self._scale, self._offset = amap["scale"], amap["offset"]
        self._target_low, self._target_high = amap["target_low"], amap["target_high"]
        self._home_raw = amap["home_raw"]
        # One column per driven joint; the gripper feature drives both jaws.
        self._feature_of_joint = torch.tensor(
            [i for i, a in enumerate(SPEC.robot["actions"]) for _ in a["joints"]], device=device
        )
        self._joint_ids, _ = self.robot.find_joints(SPEC.action_joints, preserve_order=True)
        self._feature_joint_ids = [
            self._joint_ids[SPEC.action_joints.index(a["joints"][0])] for a in SPEC.robot["actions"]
        ]
        frame = SPEC.robot["gripper_frame"]
        self._frame_body = self.robot.find_bodies(frame["link"])[0][0]
        self._frame_pos = torch.tensor(frame["pos"], device=device).repeat(self.num_envs, 1)
        self._frame_quat = torch.tensor(frame["quat"], device=device).repeat(self.num_envs, 1)

        task = SPEC.task
        self._slots = torch.tensor(task["slots"], device=device)
        self._permutations = torch.tensor(PERMUTATIONS, device=device)
        self._order = torch.zeros(self.num_envs, 3, dtype=torch.long, device=device)  # colour index per role
        self._raw_action = self._home_raw.repeat(self.num_envs, 1)
        self._prev_action = torch.zeros(self.num_envs, 6, device=device)
        self._held = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        self._held_rel_pos = torch.zeros(self.num_envs, 3, device=device)
        self._held_rel_quat = torch.zeros(self.num_envs, 4, device=device)
        self._held_rel_quat[:, 0] = 1.0
        self._hold_time = torch.zeros(self.num_envs, device=device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

    # --- scene ---

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.bowls: list[RigidObject] = []
        for i, color in enumerate(SPEC.task["colors"]):
            self.bowls.append(
                RigidObject(
                    RigidObjectCfg(
                        prim_path=f"/World/envs/env_.*/Bowl_{color}",
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=f"{self.cfg.bowl_usd_dir}/bowl_{color}.usd",
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                solver_position_iteration_count=16, max_depenetration_velocity=1.0
                            ),
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(SPEC.task["slots"][i])),
                    )
                )
            )
        colors = SPEC.task["colors"]
        self.jaw_sensors = [
            ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{link}",
                    filter_prim_paths_expr=[f"/World/envs/env_.*/Bowl_{c}" for c in colors],
                )
            )
            for link in SPEC.robot["jaw_links"]
        ]

        # The table top is the ground plane (z = 0), as the MJCF floor.
        table = SPEC.data["table"]
        ground = sim_utils.GroundPlaneCfg(
            color=tuple(table["rgba"][:3]),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=table["friction"],
                dynamic_friction=table["friction"],
                friction_combine_mode="max",
            ),
        )
        ground.func("/World/ground", ground)
        for fixture in SPEC.data["fixtures"]:
            cfg = sim_utils.CuboidCfg(
                size=tuple(fixture["size"]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(fixture["rgba"][:3])),
            )
            cfg.func(
                f"/World/envs/env_0/Workcell/{fixture['name']}",
                cfg,
                translation=tuple(fixture["pos"]),
                orientation=tuple(fixture["quat"]),
            )
        if self.cfg.self_collisions:
            _filter_excluded_pairs("/World/envs/env_0/Robot", SPEC.robot["contact_excludes"])

        self.cameras: dict[str, TiledCamera] = {}
        if self.cfg.enable_cameras:
            for cam in SPEC.data["cameras"]:
                parent = "" if cam["parent_link"] is None else f"Robot/{self.cfg.wrist_camera_parent}/"
                horizontal, vertical = camera_apertures(
                    cam["fovy_deg"], self.cfg.camera_width, self.cfg.camera_height
                )
                self.cameras[cam["name"]] = TiledCamera(
                    TiledCameraCfg(
                        prim_path=f"/World/envs/env_.*/{parent}Camera_{cam['name']}",
                        offset=TiledCameraCfg.OffsetCfg(
                            pos=tuple(cam["pos"]), rot=tuple(cam["quat"]), convention="opengl"
                        ),
                        data_types=["rgb"],
                        spawn=sim_utils.PinholeCameraCfg(
                            focal_length=1.0,
                            horizontal_aperture=horizontal,
                            vertical_aperture=vertical,
                            clipping_range=(0.01, 5.0),
                        ),
                        width=self.cfg.camera_width,
                        height=self.cfg.camera_height,
                    )
                )

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        for color, bowl in zip(SPEC.task["colors"], self.bowls, strict=True):
            self.scene.rigid_objects[f"bowl_{color}"] = bowl
        for link, sensor in zip(SPEC.robot["jaw_links"], self.jaw_sensors, strict=True):
            self.scene.sensors[link] = sensor
        for name, camera in self.cameras.items():
            self.scene.sensors[f"camera_{name}"] = camera
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
        light.func("/World/Light", light)

    # --- actions ---

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.clone()
        if self.cfg.action_mode == "normalized":
            actions = actions.clamp(-1.0, 1.0)
            raw = self._raw_low + 0.5 * (actions + 1.0) * (self._raw_high - self._raw_low)
        else:
            raw = actions
        self._raw_action = torch.max(torch.min(raw, self._raw_high), self._raw_low)
        self._prev_action = actions
        target = (self._scale * self._raw_action + self._offset).clamp(self._target_low, self._target_high)
        self._joint_targets = target[:, self._feature_of_joint]

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)
        self._grasp_assist()

    def _gripper_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_pos = self.robot.data.body_pos_w[:, self._frame_body]
        body_quat = self.robot.data.body_quat_w[:, self._frame_body]
        return body_pos + quat_apply(body_quat, self._frame_pos), quat_mul(body_quat, self._frame_quat)

    def _bowl_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World positions, quaternions, linear and angular velocities, stacked by colour: (N, 3, ...)."""
        return (
            torch.stack([b.data.root_pos_w for b in self.bowls], dim=1),
            torch.stack([b.data.root_quat_w for b in self.bowls], dim=1),
            torch.stack([b.data.root_lin_vel_w for b in self.bowls], dim=1),
            torch.stack([b.data.root_ang_vel_w for b in self.bowls], dim=1),
        )

    def _jaw_contacts(self) -> torch.Tensor:
        """(N, 2 jaws, 3 bowls) bool: the jaw touches that bowl."""
        forces = [s.data.force_matrix_w[:, 0].norm(dim=-1) for s in self.jaw_sensors]  # (N, 3) each
        return torch.stack(forces, dim=1) > self.cfg.jaw_contact_threshold_n

    def _grasp_assist(self) -> None:
        """The MuJoCo task's contact assist, per environment."""
        task = SPEC.task
        gripper_raw = self._raw_action[:, -1]
        closing = gripper_raw > task["assist_engage_raw"]
        opening = gripper_raw < task["assist_release_raw"]
        site_pos, site_quat = self._gripper_frame()
        pos, quat, _, _ = self._bowl_states()

        # Attach: both jaws on one bowl, close enough, gripper commanded closed.
        both = self._jaw_contacts().all(dim=1)  # (N, 3)
        dist = (pos - site_pos[:, None]).norm(dim=-1).masked_fill(~both, math.inf)
        nearest_dist, nearest = dist.min(dim=1)
        attach = closing & (self._held < 0) & (nearest_dist < task["assist_max_distance_m"])
        if attach.any():
            ids = attach.nonzero(as_tuple=False).squeeze(-1)
            colour = nearest[ids]
            inverse = quat_conjugate(site_quat[ids])
            self._held[ids] = colour
            self._held_rel_pos[ids] = quat_apply(inverse, pos[ids, colour] - site_pos[ids])
            self._held_rel_quat[ids] = quat_mul(inverse, quat[ids, colour])
        self._held[opening] = -1

        # Carry, only while commanded closed (as in MuJoCo): place each held bowl rigidly on the gripper frame.
        for colour, bowl in enumerate(self.bowls):
            ids = ((self._held == colour) & closing).nonzero(as_tuple=False).squeeze(-1)
            if ids.numel() == 0:
                continue
            new_pos = site_pos[ids] + quat_apply(site_quat[ids], self._held_rel_pos[ids])
            new_quat = quat_mul(site_quat[ids], self._held_rel_quat[ids])
            bowl.write_root_pose_to_sim(torch.cat([new_pos, new_quat], dim=-1), env_ids=ids)
            bowl.write_root_velocity_to_sim(torch.zeros(ids.numel(), 6, device=self.device), env_ids=ids)

    # --- observations, rewards, terminations ---

    def _role_positions(self) -> torch.Tensor:
        """Bowl positions relative to the environment origin, ordered bottom, middle, top: (N, 3, 3)."""
        pos = self._bowl_states()[0] - self.scene.env_origins[:, None]
        return torch.gather(pos, 1, self._order[:, :, None].expand(-1, -1, 3))

    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self._feature_joint_ids]
        raw = ((joint_pos - self._offset) / self._scale).clamp(self._raw_low, self._raw_high)
        normalized = 2.0 * (raw - self._raw_low) / (self._raw_high - self._raw_low) - 1.0
        joint_vel = self.robot.data.joint_vel[:, self._feature_joint_ids]
        site_pos = self._gripper_frame()[0] - self.scene.env_origins
        held_role = (self._held[:, None] == self._order) & (self._held[:, None] >= 0)
        policy = torch.cat(
            [
                normalized,
                0.1 * joint_vel,
                site_pos,
                self._role_positions().flatten(1),
                held_role.float(),
                self._prev_action,
            ],
            dim=-1,
        )
        obs = {"policy": policy}
        if self.cameras:
            obs["images"] = {name: cam.data.output["rgb"][..., :3] for name, cam in self.cameras.items()}
            obs["agent_pos"] = raw  # the follower's raw ticks, like observation.state in the real dataset
        return obs

    def _stack_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per environment: middle nested in bottom, top nested in middle, and the whole stack valid and still."""
        task = SPEC.task
        roles = self._role_positions()
        _, quat, lin_vel, ang_vel = self._bowl_states()

        def nested(lower, upper):
            dz = upper[:, 2] - lower[:, 2]
            concentric = (upper[:, :2] - lower[:, :2]).norm(dim=-1) < task["max_concentric_error_m"]
            return concentric & (dz >= task["min_nesting_delta_m"]) & (dz <= task["max_nesting_delta_m"])

        middle_in_bottom = nested(roles[:, 0], roles[:, 1])
        top_in_middle = nested(roles[:, 1], roles[:, 2])
        released = (self._held < 0) & ~self._jaw_contacts().any(dim=(1, 2))
        # Bowl z axis in world: the last column of its rotation matrix; its z component is 1 - 2 (x^2 + y^2).
        upright = (1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)) > math.cos(
            math.radians(task["max_tilt_deg"])
        )
        still = (lin_vel.norm(dim=-1) < task["max_linear_speed_mps"]) & (
            ang_vel.norm(dim=-1) < task["max_angular_speed_rps"]
        )
        valid = middle_in_bottom & top_in_middle & released & (upright & still).all(dim=1)
        return middle_in_bottom, top_in_middle, valid

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # DirectRLEnv calls this before _get_rewards, so the success hold is tracked here.
        self._middle_in_bottom, self._top_in_middle, valid = self._stack_state()
        self._hold_time = torch.where(
            valid, self._hold_time + self.step_dt, torch.zeros_like(self._hold_time)
        )
        self._success = self._hold_time >= SPEC.task["success_hold_s"]
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["success"] = self._success.clone()
        return self._success, time_out

    def _get_rewards(self) -> torch.Tensor:
        middle_in_bottom = self._middle_in_bottom
        if self.cfg.reward_type == "sparse":
            return self._success.float()

        roles = self._role_positions()
        mover_role = torch.where(middle_in_bottom, 2, 1)
        index = torch.arange(self.num_envs, device=self.device)
        mover, base = roles[index, mover_role], roles[index, mover_role - 1]
        site_pos = self._gripper_frame()[0] - self.scene.env_origins
        reach = 1.0 - torch.tanh(10.0 * (site_pos - mover).norm(dim=-1))
        held = (self._held >= 0) & (self._held == self._order[index, mover_role])
        carry = torch.where(held, 1.0 - torch.tanh(5.0 * (mover[:, :2] - base[:, :2]).norm(dim=-1)), 0.0)
        stage = 2.0 * (middle_in_bottom.float() + (middle_in_bottom & self._top_in_middle).float())
        return reach + held.float() + carry + stage + 5.0 * self._success.float()

    # --- reset ---

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)

        # Layout: jitter the three slots, reject overlaps, then give each colour a random slot and yaw.
        task = SPEC.task
        jitter = self.cfg.position_jitter_m
        positions = self._slots.repeat(n, 1, 1)
        pending = torch.ones(n, dtype=torch.bool, device=self.device)
        for _ in range(100):
            k = int(pending.sum())
            if k == 0:
                break
            sample = self._slots.repeat(k, 1, 1)
            sample[..., :2] += (torch.rand(k, 3, 2, device=self.device) * 2 - 1) * jitter
            positions[pending] = sample
            gaps = torch.cdist(positions[..., :2], positions[..., :2])
            gaps = gaps + torch.eye(3, device=self.device) * 1e3
            pending = gaps.amin(dim=(1, 2)) < task["min_separation_m"]
        slot_of_colour = torch.argsort(torch.rand(n, 3, device=self.device), dim=1)
        yaw = (torch.rand(n, 3, device=self.device) * 2 - 1) * math.pi
        for colour, bowl in enumerate(self.bowls):
            pos = positions[torch.arange(n, device=self.device), slot_of_colour[:, colour]]
            pos = pos + self.scene.env_origins[env_ids]
            quat = torch.zeros(n, 4, device=self.device)
            quat[:, 0], quat[:, 3] = torch.cos(yaw[:, colour] / 2), torch.sin(yaw[:, colour] / 2)
            bowl.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
            bowl.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)

        self._order[env_ids] = self._permutations[
            torch.randint(0, len(PERMUTATIONS), (n,), device=self.device)
        ]
        self._held[env_ids] = -1
        self._hold_time[env_ids] = 0.0
        self._success[env_ids] = False
        self._raw_action[env_ids] = self._home_raw
        self._prev_action[env_ids] = 0.0

    def task_descriptions(self) -> list[str]:
        """The instruction for each environment, worded as in the MuJoCo task and the real dataset."""
        colors = SPEC.task["colors"]
        return [
            f"Stack the bowls with {colors[b]} on bottom, {colors[m]} in middle, and {colors[t]} on top."
            for b, m, t in self._order.tolist()
        ]


def _filter_excluded_pairs(robot_path: str, pairs: list[list[str]]) -> None:
    """Stop PhysX colliding the link pairs the MJCF excludes (pairs joined by a joint are skipped anyway)."""
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    links = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            links.setdefault(prim.GetName(), prim)
    for first, second in pairs:
        if first not in links or second not in links:
            continue  # merged away by the importer, or absent
        api = UsdPhysics.FilteredPairsAPI.Apply(links[first])
        api.CreateFilteredPairsRel().AddTarget(links[second].GetPath())
