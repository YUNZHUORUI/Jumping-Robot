"""Train anticipatory bounded residual control on a frozen two-hop teacher."""

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train two-hop high-level residual")
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--checkpoint", default=None, help="Exact residual checkpoint to resume")
parser.add_argument("--eval_checkpoint", default=None, help="Evaluate a residual checkpoint without learning")
parser.add_argument("--eval_steps", type=int, default=3000)
parser.add_argument("--state_planner", action="store_true", help="Train v42 causal stance touchdown-state planner instead of v38 motor residual")
parser.add_argument("--phase_residual", action="store_true", help="Train v54 phase-conditioned whole-hop motor residual")
parser.add_argument("--curriculum_iterations", type=float, default=160.0)
parser.add_argument("--curriculum_iteration_offset", type=float, default=0.0)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=200)
parser.add_argument("--target_height", type=float, default=1.0)
parser.add_argument("--target_tolerance", type=float, default=0.10)
parser.add_argument("--collective_scale", type=float, default=0.03)
parser.add_argument("--attitude_scale", type=float, default=0.05)
parser.add_argument("--residual_slew_rate", type=float, default=0.02)
parser.add_argument("--residual_penalty_scale", type=float, default=50.0)
parser.add_argument("--slew_penalty_scale", type=float, default=10.0)
parser.add_argument("--safety_gate_error_m", type=float, default=0.08)
parser.add_argument("--next_tilt_deg", type=float, default=6.0)
parser.add_argument("--init_noise_std", type=float, default=0.02)
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--continuous_queue",
    action="store_true",
    help="Keep P_(t+1) as the next P_t after every touchdown; this is now the default.",
)
parser.add_argument(
    "--pair_restart_queue",
    action="store_true",
    help="Legacy mode: after a long hop, restart a fresh short/long pair around the measured touchdown.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.continuous_queue and args_cli.pair_restart_queue:
    parser.error("--continuous_queue and --pair_restart_queue are mutually exclusive")
args_cli.continuous_queue = not args_cli.pair_restart_queue
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

import Quadhopper_Planner_Random  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg
from Quadhopper_Planner_Random.two_hop_residual_wrapper import (
    TeacherTwoHopResidualVecEnv,
)
from Quadhopper_Planner_Random.phase_residual_wrapper import (
    TeacherTwoHopPhaseResidualVecEnv,
)
from Quadhopper_Planner_Random.two_hop_state_planner_wrapper import (
    TeacherTwoHopStatePlannerVecEnv,
)

if args_cli.state_planner and args_cli.phase_residual:
    raise ValueError("--state_planner and --phase_residual are mutually exclusive")

if args_cli.phase_residual:
    EXPERIMENT = (
        "quadhopper_planner_random_two_hop_v55_phase_residual_continuous_queue_ppo"
        if args_cli.continuous_queue
        else "quadhopper_planner_random_two_hop_v54_phase_residual_ppo"
    )
elif args_cli.state_planner:
    EXPERIMENT = (
        "quadhopper_planner_random_two_hop_v55_state_planner_continuous_queue_ppo"
        if args_cli.continuous_queue
        else "quadhopper_planner_random_two_hop_v42_causal_stance_bc_ppo_curriculum"
    )
else:
    EXPERIMENT = (
        "quadhopper_planner_random_two_hop_v55_residual_continuous_queue_ppo"
        if args_cli.continuous_queue
        else "quadhopper_planner_random_two_hop_v38_conservative_residual"
    )


def configure_env() -> PlannerRandomTwoHopEnvCfg:
    cfg = PlannerRandomTwoHopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    cfg.debug_vis = False
    cfg.force_full_planner = True
    cfg.observation_noise_std = 0.0 if args_cli.eval_checkpoint else 0.002
    if args_cli.eval_checkpoint:
        cfg.randomize_dynamics = False
        cfg.randomize_action_delay = False
    cfg.target_height = args_cli.target_height
    cfg.alternate_target_heights = False
    cfg.fixed_height_curriculum = False
    cfg.symmetric_height_tracking = True
    cfg.require_apex_tolerance_for_hit = True
    cfg.relative_next_hop_observation = True
    cfg.restart_two_hop_pair = not args_cli.continuous_queue
    cfg.short_hop_radius_min = 0.50
    cfg.short_hop_radius_max = 0.80
    cfg.long_hop_radius_min = 0.80
    cfg.long_hop_radius_max = 1.00
    cfg.max_turn_angle_deg = 180.0
    if args_cli.state_planner or args_cli.phase_residual:
        # Distances remain continuously randomized throughout the curriculum.
        cfg.distance_curriculum_iterations = args_cli.curriculum_iterations
        cfg.distance_curriculum_iteration_offset = args_cli.curriculum_iteration_offset
        cfg.curriculum_short_radius_min = 0.50
        cfg.curriculum_short_radius_max = 0.65
        cfg.curriculum_long_radius_min = 0.80
        cfg.curriculum_long_radius_max = 0.90
        cfg.curriculum_max_turn_angle_deg = 30.0
        cfg.curriculum_by_hops = args_cli.phase_residual
    cfg.target_tolerance = args_cli.target_tolerance

    # The current touchdown remains settled, while its attitude is optimized
    # for the direction of the already-observed following hop.
    cfg.planner_landing_xy_velocity_scale = 0.0
    cfg.anticipatory_velocity_blend = 0.0
    cfg.anticipatory_tilt_rad = math.radians(args_cli.next_tilt_deg)
    cfg.anticipation_start_phase = 0.55
    cfg.anticipatory_attitude_reward_scale = 20.0
    cfg.anticipatory_attitude_penalty_scale = -30.0
    cfg.prepared_landing_reward_scale = 100.0
    cfg.pair_hit_reward_scale = 500.0
    cfg.prepared_attitude_tolerance_rad = math.radians(10.0)
    cfg.prepared_velocity_tolerance = 0.55

    cfg.terminate_on_target_miss = False
    cfg.target_hit_reward_scale = 280.0
    cfg.target_miss_penalty_scale = -200.0
    cfg.landing_error_penalty_scale = -280.0
    cfg.landing_precision_reward_scale = 170.0
    cfg.landing_precision_width = 0.05
    cfg.projected_landing_penalty_scale = -100.0
    cfg.streak_progress_reward_scale = 800.0
    cfg.circle_complete_reward_scale = 5000.0
    cfg.apex_event_reward_scale = 50.0
    cfg.apex_error_penalty_scale = -120.0
    cfg.height_progress_reward_scale = 30.0
    cfg.power_model_path = str(
        PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt"
    )
    cfg.csv_log_path = str(
        PROJECT_DIR / "outputs/planner_random_two_hop_residual/training.csv"
    )
    return cfg


def main():
    teacher_checkpoint = Path(args_cli.teacher_checkpoint).expanduser().resolve()
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(teacher_checkpoint)
    teacher_data = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher_width = teacher_data["model_state_dict"]["memory_a.rnn.weight_ih_l0"].shape[1]
    if teacher_width != 43:
        raise ValueError(f"Teacher must use the 43-D planner contract, got {teacher_width}")

    base_env = RslRlVecEnvWrapper(
        gym.make("Quadhopper-Planner-Random-Two-Hop-Direct-v0", cfg=configure_env())
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

    if args_cli.phase_residual:
        env = TeacherTwoHopPhaseResidualVecEnv(
            base_env,
            teacher_model,
            residual_slew_rate=args_cli.residual_slew_rate,
            residual_penalty_scale=args_cli.residual_penalty_scale,
            slew_penalty_scale=args_cli.slew_penalty_scale,
            safety_gate_error_m=args_cli.safety_gate_error_m,
        )
    elif args_cli.state_planner:
        env = TeacherTwoHopStatePlannerVecEnv(base_env, teacher_model)
    else:
        env = TeacherTwoHopResidualVecEnv(
            base_env,
            teacher_model,
            collective_scale=args_cli.collective_scale,
            attitude_scale=args_cli.attitude_scale,
            residual_slew_rate=args_cli.residual_slew_rate,
            residual_penalty_scale=args_cli.residual_penalty_scale,
            slew_penalty_scale=args_cli.slew_penalty_scale,
            next_tilt_rad=math.radians(args_cli.next_tilt_deg),
        )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = PROJECT_DIR / "logs/rsl_rl" / EXPERIMENT / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    runner_cfg = PlannerCircularPPORunnerCfg()
    runner_cfg.experiment_name = EXPERIMENT
    runner_cfg.save_interval = args_cli.save_interval
    if args_cli.phase_residual:
        runner_cfg.policy.init_noise_std = args_cli.init_noise_std
        runner_cfg.algorithm.learning_rate = 1.0e-4
        runner_cfg.algorithm.entropy_coef = 2.0e-4
    else:
        runner_cfg.policy.init_noise_std = 0.02 if args_cli.state_planner else args_cli.init_noise_std
        runner_cfg.algorithm.learning_rate = 1.0e-5 if args_cli.state_planner else 2.0e-5
        runner_cfg.algorithm.entropy_coef = 0.0 if args_cli.state_planner else 2.0e-5
    runner = OnPolicyRunner(
        env, runner_cfg.to_dict(), log_dir=str(log_dir), device=args_cli.device
    )
    if args_cli.eval_checkpoint:
        runner.load(
            str(Path(args_cli.eval_checkpoint).expanduser().resolve()),
            load_optimizer=False,
        )
    elif args_cli.checkpoint:
        runner.load(
            str(Path(args_cli.checkpoint).expanduser().resolve()),
            load_optimizer=True,
        )
    else:
        # A residual policy must initially reproduce the accepted teacher.
        # Zeroing only the final actor layer preserves the recurrent feature
        # initialization while making the deterministic residual exactly zero.
        actor_linears = [
            module
            for module in runner.alg.policy.actor.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        if not actor_linears:
            raise RuntimeError("Could not find the residual actor output layer")
        torch.nn.init.zeros_(actor_linears[-1].weight)
        torch.nn.init.zeros_(actor_linears[-1].bias)
        initialization = log_dir / "model_0_zero_residual.pt"
        torch.save(
            {
                "model_state_dict": runner.alg.policy.state_dict(),
                "iter": 0,
                "infos": {
                "teacher_checkpoint": str(teacher_checkpoint),
                "interface": (
                    "67-D observations, phase-conditioned 4-D motor residual"
                    if args_cli.phase_residual
                    else "66-D observations, causal stance-refined 4-D touchdown-state plan"
                    if args_cli.state_planner
                    else "56-D observations, 3-D collective/roll/pitch residual"
                ),
                "zero_residual_initialization": True,
                },
            },
            initialization,
        )

    print(f"[TWO-HOP RESIDUAL] frozen 43-D teacher: {teacher_checkpoint}")
    print(
        f"[TWO-HOP RESIDUAL] obs={67 if args_cli.phase_residual else 66 if args_cli.state_planner else 56} "
        f"actions={4 if (args_cli.phase_residual or args_cli.state_planner) else 3} target={args_cli.target_height:.2f} m "
        f"scales=({args_cli.collective_scale:.3f}, {args_cli.attitude_scale:.3f}) "
        f"slew={args_cli.residual_slew_rate:.3f} next_tilt={args_cli.next_tilt_deg:.1f} deg "
        f"phase_residual={args_cli.phase_residual} "
        f"log={log_dir}"
    )
    if args_cli.eval_checkpoint:
        policy = runner.get_inference_policy(device=args_cli.device)
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(args_cli.eval_steps):
                obs, _, _, _ = env.step(policy(obs))
        core = base_env.unwrapped
        touchdowns = core._touchdown_count.sum().float()
        hits = core._target_hit_count.sum().float()
        print(f"[EVAL-DIRECT] steps={args_cli.eval_steps}", flush=True)
        print(f"[EVAL-DIRECT] touchdown_count={touchdowns.item():.0f}", flush=True)
        print(f"[EVAL-DIRECT] target_hit_rate={(hits / touchdowns.clamp_min(1.0)).item():.6f}", flush=True)
        print(
            "[EVAL-DIRECT] touchdown_error_m="
            f"{(core._touchdown_error_sum.sum() / touchdowns.clamp_min(1.0)).item():.6f}",
            flush=True,
        )
        print(
            "[EVAL-DIRECT] max_consecutive_hits="
            f"{core._max_consecutive_hits.float().mean().item():.6f}",
            flush=True,
        )
        conditional_attempts = core._conditional_second_attempts.sum().float()
        pair_attempts = core._pair_attempts.sum().float()
        print(
            "[EVAL-DIRECT] conditional_second_hit_rate="
            f"{(core._conditional_second_hits.sum().float() / conditional_attempts.clamp_min(1.0)).item():.6f}",
            flush=True,
        )
        print(
            "[EVAL-DIRECT] two_hop_pair_success_rate="
            f"{(core._pair_hits.sum().float() / pair_attempts.clamp_min(1.0)).item():.6f}",
            flush=True,
        )
    else:
        runner.learn(num_learning_iterations=args_cli.iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
