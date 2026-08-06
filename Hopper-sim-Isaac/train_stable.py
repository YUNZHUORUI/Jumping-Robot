"""Train the frozen 37-D stable-jump baseline supplied by the user."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train the canonical Quadhopper stable-jump baseline")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Resume an exact-compatible stable checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import isaaclab as _il

_ISAACLAB_SOURCE = os.path.join(os.path.dirname(_il.__file__), "source")
_ISAACLAB_RL = os.path.join(_ISAACLAB_SOURCE, "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)

import Quadhopper_Stable  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Stable.quadhopper_env import QuadhopperEnvCfg
from Quadhopper_Stable.rsl_rl_ppo_cfg import QuadhopperPPORunnerCfg


PROJECT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PROJECT_DIR / "Quadhopper_Stable"
EXPERIMENT_NAME = "quadhopper_stable_baseline"


def main():
    env_cfg = QuadhopperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.power_model_path = str(PACKAGE_DIR / "model" / "quadhopper_memory_power.pt")
    env_cfg.csv_log_path = str(PROJECT_DIR / "outputs" / "stable" / "on_quadhopper_sim.csv")
    env = RslRlVecEnvWrapper(gym.make("Quadhopper-Stable-Direct-v0", cfg=env_cfg))

    runner_cfg = QuadhopperPPORunnerCfg()
    runner_cfg.experiment_name = EXPERIMENT_NAME
    iterations = args_cli.iterations or runner_cfg.max_iterations
    log_dir = PROJECT_DIR / "logs" / "rsl_rl" / EXPERIMENT_NAME / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(log_dir), device="cuda:0")
    if args_cli.checkpoint:
        runner.load(str(Path(args_cli.checkpoint).expanduser().resolve()))
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
