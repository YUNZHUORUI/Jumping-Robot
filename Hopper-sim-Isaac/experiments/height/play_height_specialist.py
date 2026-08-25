"""Play and print per-hop metrics for a fixed-height full-policy specialist."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a fixed-height Quadhopper specialist")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--target_height", type=float, default=1.0)
parser.add_argument("--apex_tolerance", type=float, default=0.15)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab as _il
import torch

_ISAACLAB_RL = os.path.join(os.path.dirname(_il.__file__), "source", "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import Quadhopper_Planner_Circular  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Planner_Circular.planner_circular_env import PlannerCircularEnvCfg
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg

def main():
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    env_cfg = PlannerCircularEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env_cfg.debug_vis = True
    env_cfg.force_full_planner = True
    env_cfg.target_height = args_cli.target_height
    env_cfg.alternate_target_heights = False
    env_cfg.apex_tolerance = args_cli.apex_tolerance
    env_cfg.require_apex_tolerance_for_hit = True
    env_cfg.minimum_valid_apex = max(
        env_cfg.landing_root_height + 0.05,
        args_cli.target_height - env_cfg.apex_tolerance,
    )
    env_cfg.observation_noise_std = 0.0
    env_cfg.power_model_path = str(
        PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt"
    )
    env_cfg.csv_log_path = str(PROJECT_DIR / "outputs/height_specialist/play.csv")
    env = RslRlVecEnvWrapper(
        gym.make("Quadhopper-Planner-Circular-Direct-v0", cfg=env_cfg)
    )
    runner = OnPolicyRunner(
        env, PlannerCircularPPORunnerCfg().to_dict(), log_dir=None, device=args_cli.device
    )
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args_cli.device)
    obs, _ = env.reset()
    core = env.unwrapped
    hops = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if bool(core._touchdown_event[0].item()):
            hops += 1
            apex = float(core._cycle_max_z[0].item())
            landing = float(core._landing_error[0].item())
            hit = bool(core._target_hit_event[0].item())
            streak = int(core._consecutive_hits[0].item())
            maximum = int(core._max_consecutive_hits[0].item())
            print(
                f"[HOP {hops:03d}] target_apex={args_cli.target_height:.3f}m "
                f"apex={apex:.3f}m apex_err={abs(apex-args_cli.target_height):.3f}m "
                f"landing_err={landing:.3f}m hit={hit} streak={streak}/58 "
                f"max_streak={maximum}/58"
            )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
