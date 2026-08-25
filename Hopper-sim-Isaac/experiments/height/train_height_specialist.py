"""Fine-tune the full four-motor planner policy for one fixed apex height."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train a fixed-height Quadhopper specialist")
parser.add_argument("--source_checkpoint", type=str, required=True)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume source checkpoint including optimizer instead of policy-only transfer.",
)
parser.add_argument("--target_height", type=float, default=1.0)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--iterations", type=int, default=60)
parser.add_argument("--init_noise_std", type=float, default=0.08)
parser.add_argument("--learning_rate", type=float, default=1.0e-4)
parser.add_argument("--save_interval", type=int, default=5)
parser.add_argument("--apex_tolerance", type=float, default=0.15)
parser.add_argument("--apex_error_scale", type=float, default=-300.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.rendering_mode = "performance"
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

height_name = f"{args_cli.target_height:.2f}".replace(".", "p")
tolerance_name = int(round(args_cli.apex_tolerance * 100.0))
EXPERIMENT = f"quadhopper_height_specialist_{height_name}_v3_tol{tolerance_name:02d}"


def main():
    source = Path(args_cli.source_checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    env_cfg = PlannerCircularEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.debug_vis = False
    env_cfg.force_full_planner = True
    env_cfg.target_height = args_cli.target_height
    env_cfg.alternate_target_heights = False
    env_cfg.apex_tolerance = args_cli.apex_tolerance
    env_cfg.require_apex_tolerance_for_hit = True
    env_cfg.minimum_valid_apex = max(
        env_cfg.landing_root_height + 0.05,
        args_cli.target_height - env_cfg.apex_tolerance,
    )
    env_cfg.symmetric_height_tracking = True
    env_cfg.apex_error_penalty_scale = args_cli.apex_error_scale
    env_cfg.apex_event_reward_scale = 150.0
    env_cfg.apex_shortfall_penalty_scale = 0.0
    env_cfg.observation_noise_std = 0.0
    env_cfg.power_model_path = str(
        PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt"
    )
    env_cfg.csv_log_path = str(PROJECT_DIR / "outputs/height_specialist/training.csv")
    env = RslRlVecEnvWrapper(
        gym.make("Quadhopper-Planner-Circular-Direct-v0", cfg=env_cfg)
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = PROJECT_DIR / "logs/rsl_rl" / EXPERIMENT / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    runner_cfg = PlannerCircularPPORunnerCfg()
    runner_cfg.experiment_name = EXPERIMENT
    runner_cfg.save_interval = args_cli.save_interval
    runner_cfg.policy.init_noise_std = args_cli.init_noise_std
    runner_cfg.algorithm.entropy_coef = 0.0002
    runner_cfg.algorithm.learning_rate = args_cli.learning_rate
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(log_dir), device=args_cli.device)

    if args_cli.resume:
        runner.load(str(source), load_optimizer=True)
    else:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"].copy()
        state["std"] = torch.full_like(state["std"], args_cli.init_noise_std)
        initialization = log_dir / "initial_policy.pt"
        torch.save(
            {
                "model_state_dict": state,
                "iter": 0,
                "infos": {
                    "source_checkpoint": str(source),
                    "transfer": "fixed-height full-policy specialist; optimizer reset",
                    "target_height_m": args_cli.target_height,
                },
            },
            initialization,
        )
        runner.load(str(initialization), load_optimizer=False)
    print(
        f"[HEIGHT SPECIALIST] target={args_cli.target_height:.3f}m "
        f"valid_apex_min={env_cfg.minimum_valid_apex:.3f}m resume={args_cli.resume} "
        f"source={source} log={log_dir}"
    )
    runner.learn(num_learning_iterations=args_cli.iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
