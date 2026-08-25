"""Play or evaluate the hierarchical two-hop teacher plus residual policy."""

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play two-hop residual policy")
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--checkpoint", required=True, help="Residual policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--target_tolerance", type=float, default=0.10)
parser.add_argument("--collective_scale", type=float, default=0.03)
parser.add_argument("--attitude_scale", type=float, default=0.05)
parser.add_argument("--residual_slew_rate", type=float, default=0.02)
parser.add_argument("--next_tilt_deg", type=float, default=6.0)
parser.add_argument("--no_debug_vis", action="store_true")
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

import Quadhopper_Planner_Random  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg
from Quadhopper_Planner_Random.two_hop_residual_wrapper import (
    TeacherTwoHopResidualVecEnv,
)

def environment_cfg() -> PlannerRandomTwoHopEnvCfg:
    cfg = PlannerRandomTwoHopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    cfg.debug_vis = not args_cli.no_debug_vis
    cfg.force_full_planner = True
    cfg.observation_noise_std = 0.0
    cfg.randomize_dynamics = False
    cfg.randomize_action_delay = False
    cfg.target_height = 1.0
    cfg.alternate_target_heights = False
    cfg.fixed_height_curriculum = False
    cfg.symmetric_height_tracking = True
    cfg.require_apex_tolerance_for_hit = True
    cfg.relative_next_hop_observation = True
    cfg.short_hop_radius_min, cfg.short_hop_radius_max = 0.50, 0.80
    cfg.long_hop_radius_min, cfg.long_hop_radius_max = 0.80, 1.00
    cfg.max_turn_angle_deg = 180.0
    cfg.target_tolerance = args_cli.target_tolerance
    cfg.planner_landing_xy_velocity_scale = 0.0
    cfg.anticipatory_velocity_blend = 0.0
    cfg.anticipatory_tilt_rad = math.radians(args_cli.next_tilt_deg)
    cfg.anticipation_start_phase = 0.55
    cfg.prepared_attitude_tolerance_rad = math.radians(10.0)
    cfg.prepared_velocity_tolerance = 0.55
    cfg.power_model_path = str(
        PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt"
    )
    cfg.csv_log_path = str(
        PROJECT_DIR / "outputs/planner_random_two_hop_residual/play.csv"
    )
    return cfg


def main():
    teacher_checkpoint = Path(args_cli.teacher_checkpoint).expanduser().resolve()
    residual_checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    for checkpoint in (teacher_checkpoint, residual_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    base_env = RslRlVecEnvWrapper(
        gym.make("Quadhopper-Planner-Random-Two-Hop-Direct-v0", cfg=environment_cfg())
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

    env = TeacherTwoHopResidualVecEnv(
        base_env,
        teacher_model,
        collective_scale=args_cli.collective_scale,
        attitude_scale=args_cli.attitude_scale,
        residual_slew_rate=args_cli.residual_slew_rate,
        next_tilt_rad=math.radians(args_cli.next_tilt_deg),
    )
    residual_runner = OnPolicyRunner(
        env,
        PlannerCircularPPORunnerCfg().to_dict(),
        log_dir=None,
        device=args_cli.device,
    )
    residual_runner.load(str(residual_checkpoint), load_optimizer=False)
    policy = residual_runner.get_inference_policy(device=args_cli.device)
    obs, _ = env.reset()
    core = base_env.unwrapped
    p_t, p_t1 = core.commands.lookahead()
    first = torch.linalg.norm(p_t - core.commands.anchor_w, dim=1)
    second = torch.linalg.norm(p_t1 - p_t, dim=1)
    print(f"[PLAY] teacher={teacher_checkpoint}")
    print(f"[PLAY] residual={residual_checkpoint}")
    print(
        f"[PLAY] sampled first={first.min().item():.3f}--{first.max().item():.3f} m "
        f"second={second.min().item():.3f}--{second.max().item():.3f} m"
    )

    steps = 0
    latest_log = {}
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, _, _, extras = env.step(policy(obs))
        steps += 1
        latest_log = extras.get("log", latest_log)
        if args_cli.max_steps is not None and steps >= args_cli.max_steps:
            break

    if args_cli.max_steps is not None:
        touchdowns = core._touchdown_count.sum().float()
        hits = core._target_hit_count.sum().float()
        error = core._touchdown_error_sum.sum() / touchdowns.clamp_min(1.0)
        print(f"[EVAL-DIRECT] steps={steps}")
        print(f"[EVAL-DIRECT] touchdown_count={touchdowns.item():.0f}")
        print(f"[EVAL-DIRECT] target_hit_rate={(hits / touchdowns.clamp_min(1.0)).item():.6f}")
        print(f"[EVAL-DIRECT] touchdown_error_m={error.item():.6f}")
        print(
            "[EVAL-DIRECT] max_consecutive_hits="
            f"{core._max_consecutive_hits.float().mean().item():.6f}"
        )
        for key in sorted(latest_log):
            if key.startswith("Metrics/residual") or key == "Metrics/combined_action_clip_fraction":
                value = latest_log[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().mean().item()
                print(f"[EVAL] {key}={value}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
