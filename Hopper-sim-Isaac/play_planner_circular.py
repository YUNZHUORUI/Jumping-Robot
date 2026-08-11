"""Play the newest planner circular checkpoint with no required arguments."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play planner-conditioned circular hopping")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--height_stage", choices=("low", "high", "alternate"), default="alternate")
parser.add_argument("--height_high", type=float, default=1.0)
parser.add_argument("--height_low", type=float, default=0.7)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = {
    "low": "quadhopper_planner_circular_v15_descend_to_070",
    "high": "quadhopper_planner_circular_v15_high_100",
    "alternate": "quadhopper_planner_circular_v15_alternate_070_100",
}
LOG_ROOT = PROJECT_DIR / "logs/rsl_rl" / EXPERIMENTS[args_cli.height_stage]


def latest_checkpoint() -> Path | None:
    candidates = list(LOG_ROOT.glob("*/model_*.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


checkpoint = Path(args_cli.checkpoint).expanduser() if args_cli.checkpoint else latest_checkpoint()
if checkpoint is None:
    parser.error(f"No planner circular checkpoint found under {LOG_ROOT}")
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

_ISAACLAB_RL = os.path.join(os.path.dirname(_il.__file__), "source", "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)

import Quadhopper_Planner_Circular  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Planner_Circular.planner_circular_env import PlannerCircularEnvCfg
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg


def main():
    env_cfg = PlannerCircularEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.debug_vis = True
    env_cfg.force_full_planner = True
    # Evaluation must expose policy quality rather than observation RNG.
    env_cfg.observation_noise_std = 0.0
    env_cfg.alternate_target_heights = args_cli.height_stage == "alternate"
    env_cfg.alternate_height_high = args_cli.height_high
    env_cfg.alternate_height_low = args_cli.height_low
    env_cfg.target_height = (
        args_cli.height_low if args_cli.height_stage == "low" else args_cli.height_high
    )
    # Evaluation is always the final fixed command, never the training schedule.
    env_cfg.fixed_height_curriculum = False
    env_cfg.symmetric_height_tracking = True
    env_cfg.require_apex_tolerance_for_hit = True
    env_cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    env_cfg.csv_log_path = str(PROJECT_DIR / "outputs/planner_circular/on_quadhopper_sim.csv")
    env = RslRlVecEnvWrapper(gym.make("Quadhopper-Planner-Circular-Direct-v0", cfg=env_cfg))
    runner = OnPolicyRunner(env, PlannerCircularPPORunnerCfg().to_dict(), log_dir=None, device=args_cli.device)
    print(f"[PLAY] Loading planner circular checkpoint: {checkpoint}")
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args_cli.device)
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
