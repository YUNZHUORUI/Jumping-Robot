"""Visualize a frozen circular teacher plus learned collective residual."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play Quadhopper height residual")
parser.add_argument("--teacher_checkpoint", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--target_height", type=float, default=1.15)
parser.add_argument("--alternate_heights", action="store_true")
parser.add_argument("--height_high", type=float, default=1.15)
parser.add_argument("--height_low", type=float, default=1.0)
parser.add_argument("--residual_scale", type=float, default=1.25)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--height_bias_high", type=float, default=0.0)
parser.add_argument("--height_bias_low", type=float, default=0.0)
parser.add_argument("--normalize_height_command", action="store_true")
parser.add_argument("--allow_flight_residual", action="store_true")
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

from Quadhopper_Planner_Circular.height_residual_wrapper import TeacherCollectiveResidualVecEnv
from Quadhopper_Planner_Circular.planner_circular_env import PlannerCircularEnvCfg
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg

def main():
    teacher_checkpoint = Path(args_cli.teacher_checkpoint).expanduser().resolve()
    residual_checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    for checkpoint in (teacher_checkpoint, residual_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    env_cfg = PlannerCircularEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.debug_vis = True
    env_cfg.force_full_planner = True
    env_cfg.target_height = args_cli.target_height
    env_cfg.alternate_target_heights = args_cli.alternate_heights
    env_cfg.alternate_height_high = args_cli.height_high
    env_cfg.alternate_height_low = args_cli.height_low
    env_cfg.observation_noise_std = 0.0
    env_cfg.power_model_path = str(
        PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt"
    )
    env_cfg.csv_log_path = str(PROJECT_DIR / "outputs/height_residual/play.csv")

    base_env = RslRlVecEnvWrapper(
        gym.make("Quadhopper-Planner-Circular-Direct-v0", cfg=env_cfg)
    )
    teacher_runner = OnPolicyRunner(
        base_env,
        PlannerCircularPPORunnerCfg().to_dict(),
        log_dir=None,
        device=args_cli.device,
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
    residual_runner = OnPolicyRunner(
        residual_env,
        PlannerCircularPPORunnerCfg().to_dict(),
        log_dir=None,
        device=args_cli.device,
    )
    residual_runner.load(str(residual_checkpoint), load_optimizer=False)
    residual_policy = residual_runner.get_inference_policy(device=args_cli.device)

    print(f"[PLAY HEIGHT] teacher={teacher_checkpoint}")
    print(f"[PLAY HEIGHT] residual={residual_checkpoint}")
    print(
        f"[PLAY HEIGHT] target={args_cli.target_height:.2f} m "
        f"alternating={args_cli.alternate_heights} "
        f"high={args_cli.height_high:.2f} low={args_cli.height_low:.2f} "
        f"scale={args_cli.residual_scale:.2f} "
        f"normalized_height_command={args_cli.normalize_height_command} "
        f"height_bias=({args_cli.height_bias_high:.2f},{args_cli.height_bias_low:.2f}) "
        f"stance_only={not args_cli.allow_flight_residual}"
    )
    obs, _ = residual_env.reset()
    core = base_env.unwrapped
    touchdown_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            residual_actions = residual_policy(obs)
            obs, _, _, _ = residual_env.step(residual_actions)
        if args_cli.num_envs == 1 and bool(core._touchdown_event[0].item()):
            touchdown_count += 1
            apex = float(core._cycle_max_z[0].item())
            commanded_apex = float(core._apex_target_height[0].item())
            landing_error = float(core._landing_error[0].item())
            hit = bool(core._target_hit_event[0].item())
            streak = int(core._consecutive_hits[0].item())
            max_streak = int(core._max_consecutive_hits[0].item())
            print(
                f"[HOP {touchdown_count:03d}] "
                f"target_apex={commanded_apex:.3f}m apex={apex:.3f}m "
                f"apex_err={abs(apex - commanded_apex):.3f}m "
                f"landing_err={landing_error:.3f}m hit={hit} "
                f"streak={streak}/58 max_streak={max_streak}/58"
            )
            if bool(core._circle_complete_event[0].item()):
                print("[CIRCLE COMPLETE] 58 consecutive first-attempt landings.")

    residual_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
