"""Train one stage of the Quadhopper gate curriculum."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train the Quadhopper gate curriculum with RSL-RL PPO")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--stage", type=int, choices=range(7), default=0)
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Resume/transfer from a compatible stage checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys
from datetime import datetime

import gymnasium as gym

import isaaclab as _il

_ISAACLAB_SOURCE = os.path.join(os.path.dirname(_il.__file__), "source")
_ISAACLAB_RL = os.path.join(_ISAACLAB_SOURCE, "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)

import Quadhopper_Gate  # noqa: F401 - registers the Gym task
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Gate.gate_env import QuadhopperGateEnvCfg
from Quadhopper_Gate.ppo_cfg import QuadhopperGatePPORunnerCfg


def main():
    env_cfg = QuadhopperGateEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.curriculum_stage = args_cli.stage
    env = RslRlVecEnvWrapper(gym.make("Quadhopper-Gate-Direct-v0", cfg=env_cfg))

    runner_cfg = QuadhopperGatePPORunnerCfg()
    iterations = args_cli.iterations or runner_cfg.max_iterations
    log_dir = os.path.join(
        os.path.dirname(__file__), "logs", "rsl_rl", runner_cfg.experiment_name,
        f"stage_{args_cli.stage}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
