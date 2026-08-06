"""Open one higherjump Quadhopper and repeatedly drop it for visual inspection."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect the higherjump HopperAsset model without a policy")
parser.add_argument("--powered", action="store_true", help="Use neutral collective instead of motors-off")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import Quadhopper_Gate  # noqa: F401 - registers the environment
from Quadhopper_Gate.gate_env import QuadhopperGateEnvCfg


def main():
    env_cfg = QuadhopperGateEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 4.0
    env_cfg.curriculum_stage = 0
    env_cfg.episode_length_s = 20.0
    env = None
    try:
        env = gym.make("Quadhopper-Gate-Direct-v0", cfg=env_cfg)
        env.reset()

        # Action[0] = -1 maps to zero collective command. With --powered, action 0
        # maps to the higherjump neutral command u=0.5. Desired body rates stay zero.
        actions = torch.zeros(1, 4, device=env.unwrapped.device)
        if not args_cli.powered:
            actions[:, 0] = -1.0

        print("[MODEL] Asset: Quadhopper_Isaac/model/HopperAsset.usd")
        print("[MODEL] Body: Body | leg: SpringLeg | joint: center_spring_joint")
        print("[MODEL] Motors: " + ("neutral u=0.5" if args_cli.powered else "off"))
        print("[MODEL] Inspect the long rod, round foot, four springs, sliding joint, and landing contact.")

        step = 0
        while simulation_app.is_running():
            with torch.inference_mode():
                env.step(actions)
            step += 1
            if step % 100 == 0:
                raw = env.unwrapped
                root_z = raw._robot.data.root_pos_w[0, 2].item()
                if raw._spring_joint_id is None:
                    joint_q = float("nan")
                else:
                    joint_q = raw._robot.data.joint_pos[0, raw._spring_joint_id].item()
                print(f"[MODEL] root_z={root_z:.4f} m, spring_q={joint_q:.4f} m")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
