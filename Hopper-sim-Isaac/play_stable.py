"""Play the newest stable-jump baseline checkpoint with no required arguments."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play the canonical stable-jump baseline")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

PROJECT_DIR = Path(__file__).resolve().parent
LOG_ROOT = PROJECT_DIR / "logs" / "rsl_rl" / "quadhopper_stable_baseline"


def latest_checkpoint() -> Path | None:
    candidates = list(LOG_ROOT.glob("*/model_*.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


checkpoint = Path(args_cli.checkpoint).expanduser() if args_cli.checkpoint else latest_checkpoint()
if checkpoint is None:
    parser.error(f"No stable checkpoint found under {LOG_ROOT}; train it first.")
checkpoint = checkpoint.resolve()
if not checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import gymnasium as gym
import isaaclab as _il
import torch

_ISAACLAB_SOURCE = os.path.join(os.path.dirname(_il.__file__), "source")
_ISAACLAB_RL = os.path.join(_ISAACLAB_SOURCE, "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)

import Quadhopper_Stable  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Stable.quadhopper_env import QuadhopperEnvCfg
from Quadhopper_Stable.rsl_rl_ppo_cfg import QuadhopperPPORunnerCfg


def main():
    env_cfg = QuadhopperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable" / "model" / "quadhopper_memory_power.pt")
    env_cfg.csv_log_path = str(PROJECT_DIR / "outputs" / "stable" / "on_quadhopper_sim.csv")
    env = RslRlVecEnvWrapper(gym.make("Quadhopper-Stable-Direct-v0", cfg=env_cfg))

    runner_cfg = QuadhopperPPORunnerCfg()
    runner_cfg.experiment_name = "quadhopper_stable_baseline"
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    print(f"[PLAY] Loading canonical stable checkpoint: {checkpoint}")
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device="cuda:0")
    # Gymnasium requires an explicit reset before the first step.  Merely
    # calling get_observations() does not transition the OrderEnforcing
    # wrapper into its running state.
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
