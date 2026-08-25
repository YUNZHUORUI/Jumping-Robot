"""Train a one-dimensional collective residual on top of frozen circular v10."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train frozen-teacher height residual")
parser.add_argument("--teacher_checkpoint", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None, help="Exact residual checkpoint to resume")
parser.add_argument(
    "--policy_init_checkpoint",
    type=str,
    default=None,
    help="Initialize residual policy weights but reset optimizer and action std.",
)
parser.add_argument("--target_height", type=float, default=1.0)
parser.add_argument("--alternate_heights", action="store_true")
parser.add_argument("--height_high", type=float, default=1.15)
parser.add_argument("--height_low", type=float, default=1.0)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--residual_scale", type=float, default=2.0)
parser.add_argument("--init_noise_std", type=float, default=0.08)
parser.add_argument(
    "--reward_version",
    choices=("v2", "v3"),
    default="v3",
    help="v3 uses symmetric apex tracking and penalizes both over/undershoot.",
)
parser.add_argument("--save_interval", type=int, default=20)
parser.add_argument("--height_bias_high", type=float, default=0.0)
parser.add_argument("--height_bias_low", type=float, default=0.0)
parser.add_argument(
    "--normalize_height_command",
    action="store_true",
    help="Map low/high height commands to -1/+1 for the residual only.",
)
parser.add_argument("--allow_flight_residual", action="store_true")
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

from Quadhopper_Planner_Circular.height_residual_wrapper import TeacherCollectiveResidualVecEnv
from Quadhopper_Planner_Circular.planner_circular_env import PlannerCircularEnvCfg
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg

if args_cli.allow_flight_residual and args_cli.normalize_height_command:
    EXPERIMENT = "quadhopper_height_residual_v6_flight_collective"
elif args_cli.height_bias_high != 0.0 or args_cli.height_bias_low != 0.0:
    EXPERIMENT = "quadhopper_height_residual_v5_feedforward"
elif args_cli.reward_version == "v3" and args_cli.normalize_height_command:
    EXPERIMENT = "quadhopper_height_residual_v4_conditioned"
elif args_cli.reward_version == "v3":
    EXPERIMENT = "quadhopper_height_residual_v3_symmetric"
elif args_cli.alternate_heights:
    EXPERIMENT = "quadhopper_height_residual_v2_alternating"
else:
    EXPERIMENT = "quadhopper_height_residual_v1"


def main():
    teacher_checkpoint = Path(args_cli.teacher_checkpoint).expanduser().resolve()
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(teacher_checkpoint)

    env_cfg = PlannerCircularEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.debug_vis = False
    env_cfg.force_full_planner = True
    env_cfg.target_height = args_cli.target_height
    env_cfg.alternate_target_heights = args_cli.alternate_heights
    env_cfg.alternate_height_high = args_cli.height_high
    env_cfg.alternate_height_low = args_cli.height_low
    if args_cli.reward_version == "v3":
        env_cfg.symmetric_height_tracking = True
        env_cfg.apex_error_penalty_scale = -200.0
        env_cfg.apex_shortfall_penalty_scale = 0.0
    env_cfg.observation_noise_std = 0.0
    env_cfg.power_model_path = str(
        PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt"
    )
    env_cfg.csv_log_path = str(PROJECT_DIR / "outputs/height_residual/training.csv")

    gym_env = gym.make("Quadhopper-Planner-Circular-Direct-v0", cfg=env_cfg)
    base_env = RslRlVecEnvWrapper(gym_env)

    teacher_cfg = PlannerCircularPPORunnerCfg()
    teacher_runner = OnPolicyRunner(
        base_env, teacher_cfg.to_dict(), log_dir=None, device=args_cli.device
    )
    teacher_runner.load(str(teacher_checkpoint), load_optimizer=False)
    teacher_model = teacher_runner.alg.policy
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    residual_env = TeacherCollectiveResidualVecEnv(
        base_env,
        teacher_model,
        residual_scale=args_cli.residual_scale,
        stance_only=not args_cli.allow_flight_residual,
        normalize_height_command=args_cli.normalize_height_command,
        height_command_center=0.5 * (args_cli.height_high + args_cli.height_low),
        height_command_half_range=0.5 * (args_cli.height_high - args_cli.height_low),
        height_bias_high=args_cli.height_bias_high,
        height_bias_low=args_cli.height_bias_low,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = PROJECT_DIR / "logs/rsl_rl" / EXPERIMENT / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    runner_cfg = PlannerCircularPPORunnerCfg()
    runner_cfg.experiment_name = EXPERIMENT
    runner_cfg.save_interval = args_cli.save_interval
    runner_cfg.policy.init_noise_std = args_cli.init_noise_std
    runner_cfg.algorithm.entropy_coef = 0.0005
    residual_runner = OnPolicyRunner(
        residual_env, runner_cfg.to_dict(), log_dir=str(log_dir), device=args_cli.device
    )
    if args_cli.checkpoint:
        residual_runner.load(str(Path(args_cli.checkpoint).expanduser().resolve()), load_optimizer=True)
    elif args_cli.policy_init_checkpoint:
        source = Path(args_cli.policy_init_checkpoint).expanduser().resolve()
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"].copy()
        state["std"] = torch.full_like(state["std"], args_cli.init_noise_std)
        initialization = log_dir / "initial_residual_policy.pt"
        torch.save(
            {
                "model_state_dict": state,
                "iter": 0,
                "infos": {
                    "source_checkpoint": str(source),
                    "transfer": "variable-height policy initialization; optimizer reset",
                },
            },
            initialization,
        )
        residual_runner.load(str(initialization), load_optimizer=False)

    print(f"[HEIGHT RESIDUAL] frozen teacher: {teacher_checkpoint}")
    print(
        f"[HEIGHT RESIDUAL] target={args_cli.target_height:.2f} m "
        f"alternating={args_cli.alternate_heights} "
        f"high={args_cli.height_high:.2f} low={args_cli.height_low:.2f} "
        f"scale={args_cli.residual_scale:.2f} "
        f"reward={args_cli.reward_version} save_interval={args_cli.save_interval} "
        f"normalized_height_command={args_cli.normalize_height_command} "
        f"height_bias=({args_cli.height_bias_high:.2f},{args_cli.height_bias_low:.2f}) "
        f"stance_only={not args_cli.allow_flight_residual} log={log_dir}"
    )
    residual_runner.learn(num_learning_iterations=args_cli.iterations, init_at_random_ep_len=True)
    residual_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
