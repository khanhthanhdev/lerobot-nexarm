"""Check the Isaac Lab NexArm stack-bowls environment against the MuJoCo behaviour it should match.

Run it once after ``build_usd.py`` on a machine with Isaac Lab:

    ./isaaclab.sh -p sim/isaac_lab/check_env.py --headless
    ./isaaclab.sh -p sim/isaac_lab/check_env.py --headless --enable_cameras --save_images out/

It checks that the bowls settle on the table, that the arm holds home and tracks a raw-tick command, that a
hand-placed stack counts as success only once released, and (with cameras) saves one frame per camera to compare
with the MuJoCo renders. It exits non-zero on the first failed check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--save_images", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from nexarm_isaac.stack_bowls_env import SPEC, NexArmStackBowlsEnv, NexArmStackBowlsEnvCfg  # noqa: E402


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        raise SystemExit(1)


def main() -> None:
    cfg = NexArmStackBowlsEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.action_mode = "raw"
    cfg.enable_cameras = bool(args.enable_cameras)
    if args.device is not None:
        cfg.sim.device = args.device
    env = NexArmStackBowlsEnv(cfg)
    device = env.device
    home = torch.tensor(SPEC.action_map["home_raw"], dtype=torch.float32, device=device)
    hold = home.repeat(env.num_envs, 1)

    env.reset()
    for _ in range(30):  # 1 s
        env.step(hold)
    roles = env._role_positions()
    check(bool((roles[..., 2] - SPEC.task["slots"][0][2]).abs().max() < 0.01), "bowls rest on the table")
    joint_pos = env.robot.data.joint_pos[:, env._feature_joint_ids]
    home_target = (env._scale * home + env._offset).clamp(env._target_low, env._target_high)
    error = (joint_pos - home_target).abs().max().item()
    check(error < 0.02, f"arm holds home (max error {error:.4f})")

    command = hold.clone()
    command[:, 0] = 2400  # shoulder_pan: +352 ticks
    for _ in range(30):
        env.step(command)
    target = env._scale[0] * 2400 + env._offset[0]
    error = (env.robot.data.joint_pos[:, env._feature_joint_ids[0]] - target).abs().max().item()
    check(error < 0.02, f"shoulder_pan tracks a raw command (error {error:.4f} rad)")

    # Place the stack by hand, gripper open and away: success after the hold time.
    for _ in range(15):
        env.step(hold)
    origins = env.scene.env_origins
    for role in range(3):
        for env_id in range(env.num_envs):
            colour = int(env._order[env_id, role])
            pose = torch.tensor([[0.0, -0.2, 0.02 + 0.03 * role, 1.0, 0.0, 0.0, 0.0]], device=device)
            pose[:, :3] += origins[env_id]
            env.bowls[colour].write_root_pose_to_sim(pose, env_ids=torch.tensor([env_id], device=device))
    opened = hold.clone()
    opened[:, -1] = SPEC.robot["actions"][-1]["raw_range"][0]
    succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    for _ in range(45):
        _, _, terminated, _, _ = env.step(opened)
        succeeded |= terminated
    check(bool(succeeded.all()), "a released, nested stack counts as success")

    if args.save_images is not None and env.cameras:
        from PIL import Image

        args.save_images.mkdir(parents=True, exist_ok=True)
        for name, camera in env.cameras.items():
            path = args.save_images / f"isaac_{name}.png"
            Image.fromarray(camera.data.output["rgb"][0, ..., :3].cpu().numpy()).save(path)
            print(f"Saved {path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
