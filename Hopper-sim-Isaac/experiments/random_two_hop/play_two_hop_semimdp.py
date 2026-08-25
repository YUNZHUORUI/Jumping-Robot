"""Play the event-driven two-hop Semi-MDP policy with GUI visualization."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play two-hop Semi-MDP policy")
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--checkpoint", required=True, help="Semi-MDP policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--target_tolerance", type=float, default=0.10)
parser.add_argument("--action_offset", type=str, default=None,
                    help="Comma-separated 4 values added to the selected policy action")
parser.add_argument("--short_action_offset", type=str, default=None,
                    help="Comma-separated 4 values added only to short-hop policy actions")
parser.add_argument("--long_action_offset", type=str, default=None,
                    help="Comma-separated 4 values added only to long-hop policy actions")
parser.add_argument("--action_scorer", default=None,
                    help="Optional learned scorer used to select long-hop candidate actions")
parser.add_argument("--first_action_scorer", default=None,
                    help="Optional learned scorer used to select short-hop candidate actions")
parser.add_argument("--scorer_quality_weight", type=float, default=0.0,
                    help="Weight of dense quality prediction in candidate ranking")
parser.add_argument("--scorer_threshold_override", type=float, default=None,
                    help="Override the second-hop scorer's saved selection threshold")
parser.add_argument("--first_scorer_threshold_override", type=float, default=None,
                    help="Override the first-hop scorer's saved selection threshold")
parser.add_argument("--short_radius_min", type=float, default=0.50)
parser.add_argument("--short_radius_max", type=float, default=0.80)
parser.add_argument("--long_radius_min", type=float, default=0.80)
parser.add_argument("--long_radius_max", type=float, default=1.00)
parser.add_argument("--correction_mode", choices=("none", "descent", "landing"), default="landing")
parser.add_argument("--landing_correction_height", type=float, default=0.20)
parser.add_argument("--motor_correction_limit", type=float, default=0.014)
parser.add_argument("--velocity_feedback_gain", type=float, default=0.020)
parser.add_argument("--attitude_feedback_gain", type=float, default=0.100)
parser.add_argument("--state_reward_scale", type=float, default=180.0)
parser.add_argument("--no_debug_vis", action="store_true")
parser.add_argument("--pair_restart_queue", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not (0.0 < args.short_radius_min <= args.short_radius_max):
    parser.error("--short_radius_min/max must satisfy 0 < min <= max")
if not (0.0 < args.long_radius_min <= args.long_radius_max):
    parser.error("--long_radius_min/max must satisfy 0 < min <= max")
def parse_action_offset(value: str | None, name: str) -> list[float] | None:
    if value is None:
        return None
    parsed = [float(x) for x in value.split(",")]
    if len(parsed) != 4:
        parser.error(f"{name} expects 4 comma-separated values")
    return parsed


action_offset_arg = parse_action_offset(args.action_offset, "--action_offset")
short_action_offset_arg = parse_action_offset(args.short_action_offset, "--short_action_offset")
long_action_offset_arg = parse_action_offset(args.long_action_offset, "--long_action_offset")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab as _il
import torch
from torch import nn
from torch.distributions import Normal

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
from Quadhopper_Planner_Random.second_hop_scorer import SecondHopScorer
from Quadhopper_Planner_Random.two_hop_state_planner_wrapper import TeacherTwoHopStatePlannerVecEnv

class HopActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 69, action_dim: int = 4):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )
        self.second_hop_adapter = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ELU(), nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(0.20)))

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        long_hop = obs[:, -2:-1]
        return self.actor(obs) + long_hop * self.second_hop_adapter(obs)

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mean(obs), self.log_std.exp().expand(obs.shape[0], -1))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


def load_model_state_compat(model: nn.Module, state: dict[str, torch.Tensor]):
    current = model.state_dict()
    for key in ("actor.0.weight", "critic.0.weight"):
        if key in state and state[key].shape != current[key].shape:
            if state[key].shape[0] != current[key].shape[0] or state[key].shape[1] > current[key].shape[1]:
                raise ValueError(f"Cannot migrate {key}: {state[key].shape} -> {current[key].shape}")
            expanded = torch.zeros_like(current[key])
            expanded[:, : state[key].shape[1]] = state[key]
            state[key] = expanded
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {key for key in missing if key.startswith("second_hop_adapter.")}
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")


def environment_cfg() -> PlannerRandomTwoHopEnvCfg:
    cfg = PlannerRandomTwoHopEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.debug_vis = not args.no_debug_vis
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
    cfg.restart_two_hop_pair = args.pair_restart_queue
    cfg.short_hop_radius_min = args.short_radius_min
    cfg.short_hop_radius_max = args.short_radius_max
    cfg.long_hop_radius_min = args.long_radius_min
    cfg.long_hop_radius_max = args.long_radius_max
    cfg.max_turn_angle_deg = 180.0
    cfg.distance_curriculum_iterations = 0.0
    cfg.target_tolerance = args.target_tolerance
    cfg.curriculum_by_hops = True
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
    cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    cfg.csv_log_path = str(PROJECT_DIR / "outputs/planner_random_two_hop_semimdp/play.csv")
    return cfg


def main():
    teacher_checkpoint = Path(args.teacher_checkpoint).expanduser().resolve()
    policy_checkpoint = Path(args.checkpoint).expanduser().resolve()
    for checkpoint in (teacher_checkpoint, policy_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    base_env = RslRlVecEnvWrapper(
        gym.make("Quadhopper-Planner-Random-Two-Hop-Direct-v0", cfg=environment_cfg())
    )
    teacher_runner = OnPolicyRunner(
        base_env,
        PlannerCircularPPORunnerCfg().to_dict(),
        log_dir=None,
        device=args.device,
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
        expert_anchor_scale=0.0,
    )
    model = HopActorCritic().to(args.device)
    data = torch.load(policy_checkpoint, map_location=args.device, weights_only=False)
    load_model_state_compat(model, data["model_state_dict"])
    model.eval()
    scorer = None
    scorer_offsets = None
    scorer_threshold = 0.0
    if args.action_scorer is not None:
        scorer_data = torch.load(Path(args.action_scorer).expanduser().resolve(), map_location=args.device, weights_only=False)
        scorer = SecondHopScorer().to(args.device)
        scorer.load_state_dict(scorer_data["model_state_dict"])
        scorer.eval()
        scorer_offsets = scorer_data["candidate_offsets"].to(args.device)
        scorer_threshold = float(scorer_data.get("selection_threshold", 0.0))
        if args.scorer_threshold_override is not None:
            scorer_threshold = args.scorer_threshold_override
    first_scorer = None
    first_scorer_offsets = None
    first_scorer_threshold = 0.0
    if args.first_action_scorer is not None:
        first_data = torch.load(Path(args.first_action_scorer).expanduser().resolve(), map_location=args.device, weights_only=False)
        first_scorer = SecondHopScorer().to(args.device)
        first_scorer.load_state_dict(first_data["model_state_dict"])
        first_scorer.eval()
        first_scorer_offsets = first_data["candidate_offsets"].to(args.device)
        first_scorer_threshold = float(first_data.get("selection_threshold", 0.0))
        if args.first_scorer_threshold_override is not None:
            first_scorer_threshold = args.first_scorer_threshold_override
    action_offset = (
        torch.tensor(action_offset_arg, device=args.device)
        if action_offset_arg is not None
        else None
    )
    short_action_offset = (
        torch.tensor(short_action_offset_arg, device=args.device)
        if short_action_offset_arg is not None
        else None
    )
    long_action_offset = (
        torch.tensor(long_action_offset_arg, device=args.device)
        if long_action_offset_arg is not None
        else None
    )

    def apply_selector(observations, base_action, mask, selector, offsets, threshold):
        if selector is None or not torch.any(mask):
            return base_action
        selected_obs = observations[mask]
        candidates = (base_action[mask, None, :] + offsets[None, :, :]).clamp(-1.0, 1.0)
        expanded_obs = selected_obs[:, None, :].expand(-1, len(offsets), -1)
        logits, quality = selector(
            expanded_obs.reshape(-1, selected_obs.shape[-1]),
            candidates.reshape(-1, 4),
        )
        ranking = (logits + args.scorer_quality_weight * quality).reshape(
            len(selected_obs), len(offsets)
        )
        best = ranking.argmax(dim=1)
        base_score = ranking[:, 0]
        use_candidate = ranking[torch.arange(len(selected_obs), device=args.device), best] - base_score > threshold
        output = base_action.clone()
        chosen = candidates[torch.arange(len(selected_obs), device=args.device), best]
        output[mask] = torch.where(use_candidate[:, None], chosen, base_action[mask])
        return output

    def selected_mean(observations: torch.Tensor) -> torch.Tensor:
        action = model.mean(observations)
        short_hop = observations[:, -3] > 0.5
        action = apply_selector(
            observations, action, short_hop, first_scorer, first_scorer_offsets,
            first_scorer_threshold,
        )
        long_hop = observations[:, -2] > 0.5
        action = apply_selector(
            observations, action, long_hop, scorer, scorer_offsets, scorer_threshold
        )
        if action_offset is not None:
            action = action + action_offset
        short_hop = observations[:, -3:-2] > 0.5
        long_hop = observations[:, -2:-1] > 0.5
        if short_action_offset is not None:
            action = torch.where(short_hop, action + short_action_offset, action)
        if long_action_offset is not None:
            action = torch.where(long_hop, action + long_action_offset, action)
        return action.clamp(-1.0, 1.0)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    with torch.no_grad():
        held = selected_mean(obs)

    core = base_env.unwrapped
    p_t, p_t1 = core.commands.lookahead()
    first = torch.linalg.norm(p_t - core.commands.anchor_w, dim=1)
    second = torch.linalg.norm(p_t1 - p_t, dim=1)
    print(f"[PLAY] teacher={teacher_checkpoint}", flush=True)
    print(f"[PLAY] policy={policy_checkpoint}", flush=True)
    print(
        f"[PLAY] queue={'pair-restart' if args.pair_restart_queue else 'continuous'} "
        f"correction={args.correction_mode} motor_limit={args.motor_correction_limit:.3f} "
        f"action_offset={args.action_offset or 'none'} "
        f"short_action_offset={args.short_action_offset or 'none'} "
        f"long_action_offset={args.long_action_offset or 'none'} "
        f"second_scorer={args.action_scorer or 'none'} first_scorer={args.first_action_scorer or 'none'}",
        flush=True,
    )
    print(
        f"[PLAY] sampled first={first.min().item():.3f}--{first.max().item():.3f} m "
        f"second={second.min().item():.3f}--{second.max().item():.3f} m",
        flush=True,
    )

    steps = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            obs_dict, _, dones, _ = env.step(held)
            dones = dones.reshape(-1) > 0
            obs = obs_dict["policy"]
            boundary = core._touchdown_event | dones
            if torch.any(boundary):
                held[boundary] = selected_mean(obs[boundary])
        steps += 1
        if args.max_steps is not None and steps >= args.max_steps:
            break

    if args.max_steps is not None:
        touchdowns = core._touchdown_count.sum().float().clamp_min(1.0)
        print(f"[EVAL-DIRECT] steps={steps}", flush=True)
        print(f"[EVAL-DIRECT] touchdown_count={touchdowns.item():.0f}", flush=True)
        print(
            f"[EVAL-DIRECT] target_hit_rate={(core._target_hit_count.sum().float() / touchdowns).item():.6f}",
            flush=True,
        )
        print(
            f"[EVAL-DIRECT] touchdown_error_m={(core._touchdown_error_sum.sum() / touchdowns).item():.6f}",
            flush=True,
        )
        print(
            f"[EVAL-DIRECT] max_consecutive_hits={core._max_consecutive_hits.float().mean().item():.6f}",
            flush=True,
        )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
