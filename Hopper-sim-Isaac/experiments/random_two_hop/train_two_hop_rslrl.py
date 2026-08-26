"""Standard RSL-RL PPO baseline for the continuous two-hop planner.

This is the control experiment for ``train_two_hop_semimdp.py``.  It keeps the
same random P_t/P_(t+1) task and frozen 43-D teacher, but uses the stock
``OnPolicyRunner.learn()`` loop and logging.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train continuous two-hop planner with standard RSL-RL PPO")
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--checkpoint", default=None, help="RSL-RL checkpoint to resume")
parser.add_argument("--eval_checkpoint", default=None, help="Evaluate an RSL-RL checkpoint without learning")
parser.add_argument("--eval_steps", type=int, default=3000)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=200)
parser.add_argument("--num_steps_per_env", type=int, default=256)
parser.add_argument("--save_interval", type=int, default=20)
parser.add_argument("--target_tolerance", type=float, default=0.10)
parser.add_argument("--short_radius_min", type=float, default=0.20)
parser.add_argument("--short_radius_max", type=float, default=0.50)
parser.add_argument("--long_radius_min", type=float, default=0.50)
parser.add_argument("--long_radius_max", type=float, default=0.80)
parser.add_argument("--curriculum_iterations", type=float, default=0.0)
parser.add_argument("--curriculum_iteration_offset", type=float, default=0.0)
parser.add_argument("--curriculum_max_turn_angle_deg", type=float, default=30.0)
parser.add_argument("--disable_train_randomization", action="store_true")
parser.add_argument("--correction_mode", choices=("none", "descent", "landing"), default="landing")
parser.add_argument("--landing_correction_height", type=float, default=0.20)
parser.add_argument("--motor_correction_limit", type=float, default=0.014)
parser.add_argument("--velocity_feedback_gain", type=float, default=0.020)
parser.add_argument("--attitude_feedback_gain", type=float, default=0.100)
parser.add_argument("--state_reward_scale", type=float, default=80.0)
parser.add_argument("--expert_anchor_scale", type=float, default=20.0)
parser.add_argument("--stance_plan_rate", type=float, default=0.10)
parser.add_argument("--tail_error_threshold", type=float, default=0.10)
parser.add_argument("--short_tail_error_penalty", type=float, default=0.0)
parser.add_argument("--long_tail_error_penalty", type=float, default=0.0)
parser.add_argument("--init_noise_std", type=float, default=0.02)
parser.add_argument("--lr", type=float, default=1.0e-5)
parser.add_argument("--entropy_coef", type=float, default=0.0)
parser.add_argument("--desired_kl", type=float, default=0.015)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--pair_restart_queue", action="store_true",
                    help="Legacy mode: restart a new pair after long hops instead of continuous P_t/P_(t+1)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.target_tolerance <= 0.0:
    parser.error("--target_tolerance must be positive")
if args.tail_error_threshold <= 0.0:
    parser.error("--tail_error_threshold must be positive")
if not (0.0 < args.short_radius_min <= args.short_radius_max):
    parser.error("--short_radius_min/max must satisfy 0 < min <= max")
if not (0.0 < args.long_radius_min <= args.long_radius_max):
    parser.error("--long_radius_min/max must satisfy 0 < min <= max")

args.headless = True
args.rendering_mode = "performance"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab as _il
import torch

_ISAACLAB_RL = os.path.join(os.path.dirname(_il.__file__), "source", "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Quadhopper_Planner_Random  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg
from Quadhopper_Planner_Random.two_hop_state_planner_wrapper import TeacherTwoHopStatePlannerVecEnv


EXPERIMENT = "quadhopper_planner_random_two_hop_v58_rslrl_state_planner_continuous_queue_ppo"


def configure_env(evaluation: bool) -> PlannerRandomTwoHopEnvCfg:
    cfg = PlannerRandomTwoHopEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.debug_vis = False
    cfg.force_full_planner = True
    randomize_training = (not evaluation) and (not args.disable_train_randomization)
    cfg.observation_noise_std = 0.0 if not randomize_training else 0.002
    cfg.randomize_dynamics = randomize_training
    cfg.randomize_action_delay = randomize_training
    cfg.target_height = 1.0
    cfg.alternate_target_heights = False
    cfg.fixed_height_curriculum = False
    cfg.symmetric_height_tracking = True
    cfg.require_apex_tolerance_for_hit = True
    cfg.relative_next_hop_observation = True
    cfg.restart_two_hop_pair = args.pair_restart_queue
    cfg.short_hop_radius_min = args.short_radius_min
    cfg.short_hop_radius_max = args.short_radius_max
    cfg.long_hop_radius_min = args.long_radius_min
    cfg.long_hop_radius_max = args.long_radius_max
    cfg.max_turn_angle_deg = 180.0
    cfg.distance_curriculum_iterations = args.curriculum_iterations
    cfg.distance_curriculum_iteration_offset = args.curriculum_iteration_offset
    cfg.curriculum_short_radius_min = args.short_radius_min
    cfg.curriculum_short_radius_max = min(
        args.short_radius_max,
        args.short_radius_min + 0.5 * (args.short_radius_max - args.short_radius_min),
    )
    cfg.curriculum_long_radius_min = args.long_radius_min
    cfg.curriculum_long_radius_max = min(
        args.long_radius_max,
        args.long_radius_min + 0.5 * (args.long_radius_max - args.long_radius_min),
    )
    cfg.curriculum_max_turn_angle_deg = args.curriculum_max_turn_angle_deg
    cfg.curriculum_by_hops = False
    cfg.target_tolerance = args.target_tolerance
    cfg.planner_landing_xy_velocity_scale = 0.0
    cfg.anticipatory_velocity_blend = 0.0
    cfg.anticipatory_tilt_rad = math.radians(6.0)
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
    cfg.power_model_path = str(ROOT / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    cfg.csv_log_path = str(ROOT / "outputs/planner_random_two_hop_rslrl/training.csv")
    return cfg


def make_env(evaluation: bool):
    teacher_checkpoint = Path(args.teacher_checkpoint).expanduser().resolve()
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(teacher_checkpoint)
    teacher_data = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher_width = teacher_data["model_state_dict"]["memory_a.rnn.weight_ih_l0"].shape[1]
    if teacher_width != 43:
        raise ValueError(f"Teacher must use the 43-D planner contract, got {teacher_width}")

    base_env = RslRlVecEnvWrapper(
        gym.make("Quadhopper-Planner-Random-Two-Hop-Direct-v0", cfg=configure_env(evaluation))
    )
    teacher_runner = OnPolicyRunner(
        base_env, PlannerCircularPPORunnerCfg().to_dict(), log_dir=None, device=args.device
    )
    teacher_runner.load(str(teacher_checkpoint), load_optimizer=False)
    teacher_model = teacher_runner.alg.policy
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    env = TeacherTwoHopStatePlannerVecEnv(
        base_env,
        teacher_model,
        velocity_feedback_gain=args.velocity_feedback_gain,
        attitude_feedback_gain=args.attitude_feedback_gain,
        motor_correction_limit=args.motor_correction_limit,
        correction_mode=args.correction_mode,
        landing_correction_height=args.landing_correction_height,
        state_reward_scale=args.state_reward_scale,
        expert_anchor_scale=args.expert_anchor_scale,
        stance_plan_rate=args.stance_plan_rate,
        tail_error_threshold=args.tail_error_threshold,
        short_tail_error_penalty=args.short_tail_error_penalty,
        long_tail_error_penalty=args.long_tail_error_penalty,
    )
    return base_env, env


def zero_actor_output(policy) -> None:
    actor_linears = [
        module for module in policy.actor.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if not actor_linears:
        raise RuntimeError("Could not find actor output layer")
    torch.nn.init.zeros_(actor_linears[-1].weight)
    torch.nn.init.zeros_(actor_linears[-1].bias)


def print_eval(base_env, split_count, split_hit, split_error, split_tail10, split_tail15):
    core = base_env.unwrapped
    td = core._touchdown_count.sum().float().clamp_min(1.0)
    cond = core._conditional_second_attempts.sum().float().clamp_min(1.0)
    pairs = core._pair_attempts.sum().float().clamp_min(1.0)
    print(f"[EVAL-DIRECT] touchdown_count={td.item():.0f}", flush=True)
    print(f"[EVAL-DIRECT] target_hit_rate={(core._target_hit_count.sum() / td).item():.6f}", flush=True)
    print(f"[EVAL-DIRECT] touchdown_error_m={(core._touchdown_error_sum.sum() / td).item():.6f}", flush=True)
    print(f"[EVAL-DIRECT] max_consecutive_hits={core._max_consecutive_hits.float().mean().item():.6f}", flush=True)
    print(f"[EVAL-DIRECT] conditional_second_hit_rate={(core._conditional_second_hits.sum() / cond).item():.6f}", flush=True)
    print(f"[EVAL-DIRECT] two_hop_pair_success_rate={(core._pair_hits.sum() / pairs).item():.6f}", flush=True)
    print(f"[EVAL-DIRECT] prepared_landing_rate={(core._prepared_landing_count.sum().float() / td).item():.6f}", flush=True)
    for split_id, name in enumerate(("short", "long")):
        count = split_count[split_id].clamp_min(1.0)
        print(
            f"[EVAL-SPLIT] {name}_count={split_count[split_id].item():.0f} "
            f"{name}_hit_rate={(split_hit[split_id] / count).item():.6f} "
            f"{name}_touchdown_error_m={(split_error[split_id] / count).item():.6f} "
            f"{name}_tail10_rate={(split_tail10[split_id] / count).item():.6f} "
            f"{name}_tail15_rate={(split_tail15[split_id] / count).item():.6f}",
            flush=True,
        )


def main():
    evaluation = args.eval_checkpoint is not None
    base_env, env = make_env(evaluation)
    runner_cfg = PlannerCircularPPORunnerCfg()
    runner_cfg.experiment_name = EXPERIMENT
    runner_cfg.max_iterations = args.iterations
    runner_cfg.num_steps_per_env = args.num_steps_per_env
    runner_cfg.save_interval = args.save_interval
    runner_cfg.policy.init_noise_std = args.init_noise_std
    runner_cfg.algorithm.learning_rate = args.lr
    runner_cfg.algorithm.entropy_coef = args.entropy_coef
    runner_cfg.algorithm.desired_kl = args.desired_kl

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = ROOT / "logs/rsl_rl" / EXPERIMENT / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(log_dir), device=args.device)

    if args.eval_checkpoint:
        runner.load(str(Path(args.eval_checkpoint).expanduser().resolve()), load_optimizer=False)
        policy = runner.get_inference_policy(device=args.device)
        obs = env.get_observations()
        device = torch.device(args.device)
        split_count = torch.zeros(2, device=device)
        split_hit = torch.zeros(2, device=device)
        split_error = torch.zeros(2, device=device)
        split_tail10 = torch.zeros(2, device=device)
        split_tail15 = torch.zeros(2, device=device)
        with torch.inference_mode():
            for _ in range(args.eval_steps):
                core = base_env.unwrapped
                target_before, _ = core.commands.lookahead()
                target_before = target_before.clone()
                phase_was_short = ((core.commands.route_index % 2) == 0).clone()
                obs, _, _, _ = env.step(policy(obs))
                touchdown = core._touchdown_event
                if torch.any(touchdown):
                    landing_error = torch.linalg.norm(
                        core._robot.data.root_pos_w[:, :2] - target_before, dim=1
                    )
                    for split_id, mask in (
                        (0, touchdown & phase_was_short),
                        (1, touchdown & (~phase_was_short)),
                    ):
                        if torch.any(mask):
                            split_count[split_id] += mask.float().sum()
                            split_hit[split_id] += core._target_hit_event[mask].float().sum()
                            split_error[split_id] += landing_error[mask].sum()
                            split_tail10[split_id] += (landing_error[mask] > 0.10).float().sum()
                            split_tail15[split_id] += (landing_error[mask] > 0.15).float().sum()
        print_eval(base_env, split_count, split_hit, split_error, split_tail10, split_tail15)
    else:
        if args.checkpoint:
            runner.load(str(Path(args.checkpoint).expanduser().resolve()), load_optimizer=True)
        else:
            zero_actor_output(runner.alg.policy)
            torch.save(
                {
                    "model_state_dict": runner.alg.policy.state_dict(),
                    "iter": 0,
                    "infos": {
                        "teacher_checkpoint": str(Path(args.teacher_checkpoint).expanduser().resolve()),
                        "interface": "69-D standard RSL-RL state planner, 4-D touchdown-state plan",
                        "zero_residual_initialization": True,
                        "continuous_queue": not args.pair_restart_queue,
                    },
                },
                log_dir / "model_0_zero_state_planner.pt",
            )
        print(
            f"[RSLRL-TWO-HOP] log={log_dir} obs={env.get_observations()['policy'].shape[-1]} "
            f"actions={env.num_actions} "
            f"steps/env={args.num_steps_per_env} iterations={args.iterations}",
            flush=True,
        )
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
