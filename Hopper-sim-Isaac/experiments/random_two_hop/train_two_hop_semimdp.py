"""Event-driven Semi-MDP PPO for the random two-hop touchdown planner.

One policy action is sampled at reset/touchdown and held for the complete hop.
Only completed hops enter PPO storage, so every stored log-probability owns the
entire discounted hop return.  Robot physics and the frozen 43-D teacher are
unchanged.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train event-driven two-hop Semi-MDP PPO")
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--reset_optimizer", action="store_true", help="Load checkpoint weights but start Adam from a fresh state")
parser.add_argument("--eval_checkpoint", default=None)
parser.add_argument("--grid_search", action="store_true", help="Evaluate a 5x5 constant forward/tilt residual grid")
parser.add_argument("--offset_grid_search", action="store_true",
                    help="Evaluate a 5x5 forward/tilt offset grid around the loaded policy")
parser.add_argument("--offset_grid_span", type=float, default=0.12,
                    help="Half-width of --offset_grid_search in normalized action units")
parser.add_argument("--offset_grid_phase", choices=("all", "short", "long"), default="all",
                    help="Apply --offset_grid_search offsets to all, short-hop, or long-hop actions")
parser.add_argument("--action_offset", type=str, default=None,
                    help="Comma-separated 4 values added to the selected policy action at eval time")
parser.add_argument("--short_action_offset", type=str, default=None,
                    help="Comma-separated 4 values added only to short-hop policy actions")
parser.add_argument("--long_action_offset", type=str, default=None,
                    help="Comma-separated 4 values added only to long-hop policy actions")
parser.add_argument("--constant_forward", type=float, default=None)
parser.add_argument("--constant_tilt", type=float, default=None)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--updates", type=int, default=40)
parser.add_argument("--transitions_per_update", type=int, default=256)
parser.add_argument("--eval_steps", type=int, default=3000)
parser.add_argument("--target_tolerance", type=float, default=0.10)
parser.add_argument("--short_radius_min", type=float, default=0.50)
parser.add_argument("--short_radius_max", type=float, default=0.80)
parser.add_argument("--long_radius_min", type=float, default=0.80)
parser.add_argument("--long_radius_max", type=float, default=1.00)
parser.add_argument("--curriculum_iterations", type=float, default=30.0,
                    help="With --curriculum_by_hops (always on here): completed hops per env "
                         "to reach the full 0.50-0.80 / 0.80-1.00 m, 180 deg ranges.  Pass 0 "
                         "to evaluate at the full ranges from the first step.")
parser.add_argument("--curriculum_iteration_offset", type=float, default=0.0)
parser.add_argument("--lr", type=float, default=2.0e-5, help="Initial Adam learning rate (cosine decay to lr/10)")
parser.add_argument("--std_start", type=float, default=0.03, help="Initial action std (annealed linearly to --std_end)")
parser.add_argument("--std_end", type=float, default=0.01, help="Final action std")
parser.add_argument("--gamma_frame", type=float, default=0.999, help="Per-frame discount inside a hop")
parser.add_argument("--anchor_scale", type=float, default=0.05, help="Weight of the zero-action anchor in the PPO loss")
parser.add_argument("--progress_log_every", type=int, default=300, help="Print rollout progress every N frames")
parser.add_argument("--disable_train_randomization", action="store_true",
                    help="Disable observation/dynamics/action-delay randomization during training for nominal precision fine-tuning")
parser.add_argument("--action_bias", type=str, default=None,
                    help="Comma-separated 4 values added to the actor's final bias on a fresh model "
                         "(warm-start near a known-good constant action, e.g. \"-0.5,0,0.5,0\").  "
                         "When set, --anchor_scale regularizes the squared distance from this bias "
                         "instead of from zero, keeping the policy near the warm-start region.")
parser.add_argument("--hit_reward", type=float, default=3.0, help="Event reward for a target hit")
parser.add_argument("--miss_penalty", type=float, default=-2.0, help="Event penalty for a target miss")
parser.add_argument("--precision_reward_scale", type=float, default=2.0,
                    help="Dense touchdown reward scale for landing close to the current target")
parser.add_argument("--precision_sigma", type=float, default=0.10,
                    help="Std-dev of the touchdown precision reward in meters")
parser.add_argument("--landing_error_penalty", type=float, default=0.0,
                    help="Linear touchdown penalty weight in reward per meter of landing error")
parser.add_argument("--short_precision_reward_scale", type=float, default=None,
                    help="Override --precision_reward_scale on short hops")
parser.add_argument("--long_precision_reward_scale", type=float, default=None,
                    help="Override --precision_reward_scale on long hops")
parser.add_argument("--short_precision_sigma", type=float, default=None,
                    help="Override --precision_sigma on short hops")
parser.add_argument("--long_precision_sigma", type=float, default=None,
                    help="Override --precision_sigma on long hops")
parser.add_argument("--short_landing_error_penalty", type=float, default=None,
                    help="Override --landing_error_penalty on short hops")
parser.add_argument("--long_landing_error_penalty", type=float, default=None,
                    help="Override --landing_error_penalty on long hops")
parser.add_argument("--pair_success_reward", type=float, default=12.0,
                    help="Delayed reward assigned to both eligible actions when the pair succeeds")
parser.add_argument("--prepared_reward", type=float, default=0.0,
                    help="Event reward for landing with next-hop velocity and attitude prepared")
parser.add_argument("--touchdown_attitude_penalty", type=float, default=0.0,
                    help="Event penalty weight for touchdown body-axis error toward the next hop")
parser.add_argument("--touchdown_next_velocity_penalty", type=float, default=0.0,
                    help="Event penalty weight for touchdown XY velocity error toward the next hop")
parser.add_argument("--pair_first_miss_penalty", type=float, default=-4.0,
                    help="Delayed penalty assigned only to the first action when hop one misses")
parser.add_argument("--pair_second_miss_penalty", type=float, default=-8.0,
                    help="Delayed penalty assigned to both actions when eligible hop two misses")
parser.add_argument("--eligible_second_weight", type=float, default=2.0,
                    help="PPO policy-loss weight for second hops whose first hop hit")
parser.add_argument("--second_hop_adapter_only", action="store_true",
                    help="Freeze the migrated shared actor and train only a long-hop residual adapter")
parser.add_argument("--action_scorer", default=None,
                    help="Optional learned second-hop state-action scorer checkpoint")
parser.add_argument("--first_action_scorer", default=None,
                    help="Optional first-hop state-action scorer trained on final pair success")
parser.add_argument("--scorer_quality_weight", type=float, default=0.0,
                    help="Weight of dense quality prediction in candidate ranking; zero uses hit probability only")
parser.add_argument("--scorer_threshold_override", type=float, default=None,
                    help="Override the second-hop scorer's saved selection threshold")
parser.add_argument("--first_scorer_threshold_override", type=float, default=None,
                    help="Override the first-hop scorer's saved selection threshold")
parser.add_argument("--passive_airborne", action="store_true",
                    help="Disable high-level descent motor correction; the hop must be prepared through the latched touchdown-state plan.")
parser.add_argument("--correction_mode", choices=("none", "descent", "landing"), default="descent",
                    help="Where high-level touchdown-state motor correction is allowed.")
parser.add_argument("--landing_correction_height", type=float, default=0.18,
                    help="Extra height above landing root height where landing-only correction may begin.")
parser.add_argument("--motor_correction_limit", type=float, default=0.03)
parser.add_argument("--velocity_feedback_gain", type=float, default=0.04)
parser.add_argument("--attitude_feedback_gain", type=float, default=0.20)
parser.add_argument("--state_reward_scale", type=float, default=80.0)
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
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.continuous_queue and args.pair_restart_queue:
    parser.error("--continuous_queue and --pair_restart_queue are mutually exclusive")
if args.grid_search and args.offset_grid_search:
    parser.error("--grid_search and --offset_grid_search are mutually exclusive")
if not (0.0 < args.short_radius_min <= args.short_radius_max):
    parser.error("--short_radius_min/max must satisfy 0 < min <= max")
if not (0.0 < args.long_radius_min <= args.long_radius_max):
    parser.error("--long_radius_min/max must satisfy 0 < min <= max")
if args.target_tolerance <= 0.0:
    parser.error("--target_tolerance must be positive")
if args.precision_sigma <= 0.0:
    parser.error("--precision_sigma must be positive")
for name in ("short_precision_sigma", "long_precision_sigma"):
    value = getattr(args, name)
    if value is not None and value <= 0.0:
        parser.error(f"--{name} must be positive")
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
short_precision_reward_scale = args.precision_reward_scale if args.short_precision_reward_scale is None else args.short_precision_reward_scale
long_precision_reward_scale = args.precision_reward_scale if args.long_precision_reward_scale is None else args.long_precision_reward_scale
short_precision_sigma = args.precision_sigma if args.short_precision_sigma is None else args.short_precision_sigma
long_precision_sigma = args.precision_sigma if args.long_precision_sigma is None else args.long_precision_sigma
short_landing_error_penalty = args.landing_error_penalty if args.short_landing_error_penalty is None else args.short_landing_error_penalty
long_landing_error_penalty = args.landing_error_penalty if args.long_landing_error_penalty is None else args.long_landing_error_penalty
args.continuous_queue = not args.pair_restart_queue
args.headless = True
args.rendering_mode = "performance"
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
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Quadhopper_Planner_Random  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg
from Quadhopper_Planner_Random.two_hop_state_planner_wrapper import TeacherTwoHopStatePlannerVecEnv
from Quadhopper_Planner_Random.second_hop_scorer import SecondHopScorer

passive_tuned = (
    args.passive_airborne
    or args.motor_correction_limit < 0.03
    or args.correction_mode != "descent"
    or args.prepared_reward > 0.0
    or args.touchdown_attitude_penalty > 0.0
    or args.touchdown_next_velocity_penalty > 0.0
)
if args.correction_mode == "landing":
    EXPERIMENT = "quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo"
elif passive_tuned:
    EXPERIMENT = "quadhopper_planner_random_two_hop_v56_passive_continuous_queue_semimdp_ppo"
elif args.continuous_queue:
    EXPERIMENT = "quadhopper_planner_random_two_hop_v55_continuous_queue_semimdp_ppo"
else:
    EXPERIMENT = "quadhopper_planner_random_two_hop_v49_second_hop_adapter_semimdp_ppo"


class HopActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 69, action_dim: int = 4, init_log_std: float = math.log(0.20)):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, action_dim))
        self.second_hop_adapter = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ELU(), nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.second_hop_adapter[-1].weight)
        nn.init.zeros_(self.second_hop_adapter[-1].bias)

    def mean(self, obs):
        # Pair context is appended as [short, long, first_hit].  Gating the
        # adapter makes the migrated policy exactly unchanged on hop one.
        long_hop = obs[:, -2:-1]
        return self.actor(obs) + long_hop * self.second_hop_adapter(obs)

    def distribution(self, obs):
        return Normal(self.mean(obs), self.log_std.exp().expand(obs.shape[0], -1))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def load_model_state_compat(model: nn.Module, state: dict[str, torch.Tensor]) -> bool:
    """Load old 66-D planners into the 69-D Markov pair-state model.

    Pair context is appended to the observation, so copying the old first
    layer columns and zero-initializing the final three columns preserves the
    old policy exactly until learning starts using the new information.
    """
    current = model.state_dict()
    migrated = False
    for key in ("actor.0.weight", "critic.0.weight"):
        if key in state and state[key].shape != current[key].shape:
            if state[key].shape[0] != current[key].shape[0] or state[key].shape[1] > current[key].shape[1]:
                raise ValueError(f"Cannot migrate {key}: {state[key].shape} -> {current[key].shape}")
            expanded = torch.zeros_like(current[key])
            expanded[:, :state[key].shape[1]] = state[key]
            state[key] = expanded
            migrated = True
            print(f"[MIGRATE] expanded {key} from 66-D to 69-D pair state", flush=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {key for key in missing if key.startswith("second_hop_adapter.")}
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    if allowed_missing:
        print("[MIGRATE] initialized zero-output second-hop adapter", flush=True)
    return migrated


def env_cfg(evaluation: bool) -> PlannerRandomTwoHopEnvCfg:
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
    cfg.restart_two_hop_pair = not args.continuous_queue
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
    cfg.curriculum_max_turn_angle_deg = 30.0
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
    cfg.power_model_path = str(ROOT / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    cfg.csv_log_path = str(ROOT / "outputs/planner_random_two_hop_semimdp/training.csv")
    return cfg


def make_env(evaluation: bool):
    teacher_path = Path(args.teacher_checkpoint).expanduser().resolve()
    base = RslRlVecEnvWrapper(gym.make("Quadhopper-Planner-Random-Two-Hop-Direct-v0", cfg=env_cfg(evaluation)))
    runner = OnPolicyRunner(base, PlannerCircularPPORunnerCfg().to_dict(), log_dir=None, device=args.device)
    runner.load(str(teacher_path), load_optimizer=False)
    teacher = runner.alg.policy
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    correction_mode = "none" if args.passive_airborne else args.correction_mode
    motor_limit = 0.0 if args.passive_airborne else args.motor_correction_limit
    velocity_gain = 0.0 if args.passive_airborne else args.velocity_feedback_gain
    attitude_gain = 0.0 if args.passive_airborne else args.attitude_feedback_gain
    return base, TeacherTwoHopStatePlannerVecEnv(
        base,
        teacher,
        velocity_feedback_gain=velocity_gain,
        attitude_feedback_gain=attitude_gain,
        motor_correction_limit=motor_limit,
        correction_mode=correction_mode,
        landing_correction_height=args.landing_correction_height,
        state_reward_scale=args.state_reward_scale,
        expert_anchor_scale=0.0,
    )


def save(path, model, optimizer, update):
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "update": update,
                "infos": {"interface": "66-D observation, 4-D held action, one transition per completed hop",
                          "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
                          "continuous_queue": args.continuous_queue,
                          "passive_airborne": args.passive_airborne,
                          "correction_mode": "none" if args.passive_airborne else args.correction_mode,
                          "landing_correction_height": args.landing_correction_height,
                          "motor_correction_limit": 0.0 if args.passive_airborne else args.motor_correction_limit,
                          "target_tolerance": args.target_tolerance,
                          "short_radius_min": args.short_radius_min,
                          "short_radius_max": args.short_radius_max,
                          "long_radius_min": args.long_radius_min,
                          "long_radius_max": args.long_radius_max,
                          "precision_reward_scale": args.precision_reward_scale,
                          "precision_sigma": args.precision_sigma,
                          "landing_error_penalty": args.landing_error_penalty,
                          "short_precision_reward_scale": short_precision_reward_scale,
                          "long_precision_reward_scale": long_precision_reward_scale,
                          "short_precision_sigma": short_precision_sigma,
                          "long_precision_sigma": long_precision_sigma,
                          "short_landing_error_penalty": short_landing_error_penalty,
                          "long_landing_error_penalty": long_landing_error_penalty,
                          "prepared_reward": args.prepared_reward,
                          "touchdown_attitude_penalty": args.touchdown_attitude_penalty,
                          "touchdown_next_velocity_penalty": args.touchdown_next_velocity_penalty,
                          "curriculum_by_hops": True,
                          "curriculum_iterations": args.curriculum_iterations,
                          "lr": args.lr, "std_start": args.std_start, "std_end": args.std_end,
                          "disable_train_randomization": args.disable_train_randomization,
                          "gamma_frame": args.gamma_frame, "anchor_scale": args.anchor_scale}}, path)


def metrics(core, prefix="[EVAL-DIRECT]"):
    td = core._touchdown_count.sum().float().clamp_min(1.0)
    cond = core._conditional_second_attempts.sum().float().clamp_min(1.0)
    pairs = core._pair_attempts.sum().float().clamp_min(1.0)
    print(f"{prefix} touchdown_count={td.item():.0f}", flush=True)
    print(f"{prefix} target_hit_rate={(core._target_hit_count.sum() / td).item():.6f}", flush=True)
    print(f"{prefix} touchdown_error_m={(core._touchdown_error_sum.sum() / td).item():.6f}", flush=True)
    print(f"{prefix} max_consecutive_hits={core._max_consecutive_hits.float().mean().item():.6f}", flush=True)
    print(f"{prefix} conditional_second_hit_rate={(core._conditional_second_hits.sum() / cond).item():.6f}", flush=True)
    print(f"{prefix} two_hop_pair_success_rate={(core._pair_hits.sum() / pairs).item():.6f}", flush=True)
    print(f"{prefix} prepared_landing_rate={(core._prepared_landing_count.sum().float() / td).item():.6f}", flush=True)
    print(f"{prefix} touchdown_attitude_error_rad={(core._touchdown_attitude_error_sum.sum() / td).item():.6f}", flush=True)
    print(f"{prefix} touchdown_next_velocity_error_mps={(core._touchdown_next_velocity_error_sum.sum() / td).item():.6f}", flush=True)


def main():
    torch.manual_seed(args.seed)
    constant_eval = args.constant_forward is not None or args.constant_tilt is not None
    evaluation = args.eval_checkpoint is not None or args.grid_search or args.offset_grid_search or constant_eval
    base, env = make_env(evaluation)
    device = torch.device(args.device)
    model = HopActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_update = 0
    checkpoint = args.eval_checkpoint or args.checkpoint
    if checkpoint:
        data = torch.load(Path(checkpoint).expanduser().resolve(), map_location=device, weights_only=False)
        migrated = load_model_state_compat(model, data["model_state_dict"])
        if args.checkpoint and "optimizer_state_dict" in data and not migrated and not args.reset_optimizer:
            optimizer.load_state_dict(data["optimizer_state_dict"])
        elif args.checkpoint and migrated:
            print("[MIGRATE] reset Adam state because first-layer shapes changed", flush=True)
        elif args.checkpoint and args.reset_optimizer:
            print("[MIGRATE] reset Adam state by request", flush=True)
        start_update = int(data.get("update", 0))
    elif args.action_bias is not None:
        # Warm-start a fresh policy near a known-good constant action.  The
        # actor's final layer starts zero-weighted, so the mean action equals
        # this bias until state-dependent corrections are learned.
        bias = [float(x) for x in args.action_bias.split(",")]
        if len(bias) != 4:
            raise ValueError("--action_bias expects 4 comma-separated values")
        with torch.no_grad():
            model.actor[-1].bias.copy_(torch.tensor(bias, device=device))
        print(f"[WARMSTART] actor final bias = {bias}", flush=True)
    if args.second_hop_adapter_only:
        for parameter in model.actor.parameters():
            parameter.requires_grad_(False)
        print("[TRAIN] shared actor frozen; optimizing second-hop adapter only", flush=True)
    scorer = None
    candidate_offsets = None
    scorer_threshold = 0.0
    if args.action_scorer is not None:
        scorer_data = torch.load(Path(args.action_scorer).expanduser().resolve(), map_location=device, weights_only=False)
        scorer = SecondHopScorer().to(device)
        scorer.load_state_dict(scorer_data["model_state_dict"])
        scorer.eval()
        candidate_offsets = scorer_data["candidate_offsets"].to(device)
        scorer_threshold = float(scorer_data.get("selection_threshold", 0.0))
        if args.scorer_threshold_override is not None:
            scorer_threshold = args.scorer_threshold_override
        print(f"[SELECTOR] loaded {len(candidate_offsets)} second-hop candidates", flush=True)
    first_scorer = None
    first_candidate_offsets = None
    first_scorer_threshold = 0.0
    if args.first_action_scorer is not None:
        first_data = torch.load(Path(args.first_action_scorer).expanduser().resolve(), map_location=device, weights_only=False)
        first_scorer = SecondHopScorer().to(device)
        first_scorer.load_state_dict(first_data["model_state_dict"])
        first_scorer.eval()
        first_candidate_offsets = first_data["candidate_offsets"].to(device)
        first_scorer_threshold = float(first_data.get("selection_threshold", 0.0))
        if args.first_scorer_threshold_override is not None:
            first_scorer_threshold = args.first_scorer_threshold_override
        print(f"[SELECTOR] loaded {len(first_candidate_offsets)} first-hop pair candidates", flush=True)

    def apply_selector(observations, base_action, mask, selector, offsets, threshold):
        if selector is None or not torch.any(mask):
            return base_action
        selected_obs = observations[mask]
        candidates = (base_action[mask, None, :] + offsets[None, :, :]).clamp(-1.0, 1.0)
        expanded_obs = selected_obs[:, None, :].expand(-1, len(offsets), -1)
        hit_logit, quality = selector(
            expanded_obs.reshape(-1, selected_obs.shape[-1]), candidates.reshape(-1, 4)
        )
        ranking = (hit_logit + args.scorer_quality_weight * quality).reshape(
            len(selected_obs), len(offsets)
        )
        best = ranking.argmax(dim=1)
        base_score = ranking[:, 0]
        use_candidate = ranking[torch.arange(len(selected_obs), device=device), best] - base_score > threshold
        output = base_action.clone()
        chosen = candidates[torch.arange(len(selected_obs), device=device), best]
        output[mask] = torch.where(use_candidate[:, None], chosen, base_action[mask])
        return output

    def selected_mean(observations):
        base_action = model.mean(observations)
        short_hop = observations[:, -3] > 0.5
        base_action = apply_selector(
            observations, base_action, short_hop, first_scorer, first_candidate_offsets,
            first_scorer_threshold,
        )
        long_hop = observations[:, -2] > 0.5
        base_action = apply_selector(
            observations, base_action, long_hop, scorer, candidate_offsets, scorer_threshold
        )
        if action_offset_tensor is not None:
            base_action = base_action + action_offset_tensor
        short_hop = observations[:, -3:-2] > 0.5
        long_hop = observations[:, -2:-1] > 0.5
        if short_action_offset_tensor is not None:
            base_action = torch.where(short_hop, base_action + short_action_offset_tensor, base_action)
        if long_action_offset_tensor is not None:
            base_action = torch.where(long_hop, base_action + long_action_offset_tensor, base_action)
        return base_action.clamp(-1.0, 1.0)
    bias_tensor = (
        torch.tensor([float(x) for x in args.action_bias.split(",")], device=device)
        if args.action_bias is not None
        else None
    )
    action_offset_tensor = (
        torch.tensor(action_offset_arg, device=device)
        if action_offset_arg is not None
        else None
    )
    short_action_offset_tensor = (
        torch.tensor(short_action_offset_arg, device=device)
        if short_action_offset_arg is not None
        else None
    )
    long_action_offset_tensor = (
        torch.tensor(long_action_offset_arg, device=device)
        if long_action_offset_arg is not None
        else None
    )

    def apply_phase_grid_offset(observations: torch.Tensor, action: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        if args.offset_grid_phase == "all":
            return (action + offset).clamp(-1.0, 1.0)
        if args.offset_grid_phase == "short":
            mask = observations[:, -3:-2] > 0.5
        else:
            mask = observations[:, -2:-1] > 0.5
        return torch.where(mask, action + offset, action).clamp(-1.0, 1.0)

    obs, _ = env.reset()
    obs = obs["policy"]
    with torch.no_grad():
        held = selected_mean(obs) if evaluation else model.distribution(obs).sample()

    if evaluation:
        group = None
        grid_actions = None
        grid_offsets = None
        split_count = torch.zeros(2, device=device)
        split_hit = torch.zeros(2, device=device)
        split_error = torch.zeros(2, device=device)
        if args.grid_search:
            values = torch.linspace(-0.5, 0.5, 5, device=device)
            forward, tilt = torch.meshgrid(values, values, indexing="ij")
            grid_actions = torch.zeros(25, 4, device=device)
            grid_actions[:, 0] = forward.flatten()
            grid_actions[:, 2] = tilt.flatten()
            group = torch.arange(args.num_envs, device=device) % 25
            held = grid_actions[group].clone()
        elif args.offset_grid_search:
            values = torch.linspace(-args.offset_grid_span, args.offset_grid_span, 5, device=device)
            forward, tilt = torch.meshgrid(values, values, indexing="ij")
            grid_offsets = torch.zeros(25, 4, device=device)
            grid_offsets[:, 0] = forward.flatten()
            grid_offsets[:, 2] = tilt.flatten()
            group = torch.arange(args.num_envs, device=device) % 25
            held = apply_phase_grid_offset(obs, selected_mean(obs), grid_offsets[group])
        elif constant_eval:
            held.zero_()
            held[:, 0] = args.constant_forward or 0.0
            held[:, 2] = args.constant_tilt or 0.0
        for _ in range(args.eval_steps):
            core = base.unwrapped
            target_before, _ = core.commands.lookahead()
            target_before = target_before.clone()
            phase_was_short = ((core.commands.route_index % 2) == 0).clone()
            obs_dict, _, dones, _ = env.step(held)
            dones = dones.reshape(-1) > 0  # bool mask, see rollout branch
            obs = obs_dict["policy"]
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
                        split_count[split_id] += mask.sum()
                        split_hit[split_id] += core._target_hit_event[mask].float().sum()
                        split_error[split_id] += landing_error[mask].sum()
            boundary = touchdown | dones
            if torch.any(boundary):
                with torch.no_grad():
                    if args.grid_search:
                        held[boundary] = grid_actions[group[boundary]]
                    elif args.offset_grid_search:
                        held[boundary] = apply_phase_grid_offset(
                            obs[boundary], selected_mean(obs[boundary]), grid_offsets[group[boundary]]
                        )
                    elif not constant_eval:
                        held[boundary] = selected_mean(obs[boundary])
        metrics(base.unwrapped)
        split_names = ("short", "long")
        for split_id, name in enumerate(split_names):
            count = split_count[split_id].clamp_min(1.0)
            print(
                f"[EVAL-SPLIT] {name}_count={split_count[split_id].item():.0f} "
                f"{name}_hit_rate={(split_hit[split_id] / count).item():.6f} "
                f"{name}_touchdown_error_m={(split_error[split_id] / count).item():.6f}",
                flush=True,
            )
        if args.grid_search or args.offset_grid_search:
            core = base.unwrapped
            for grid_id in range(25):
                mask = group == grid_id
                pair_attempts = core._pair_attempts[mask].sum().float().clamp_min(1.0)
                touchdowns = core._touchdown_count[mask].sum().float().clamp_min(1.0)
                values_to_print = grid_actions if args.grid_search else grid_offsets
                prefix = "[GRID]" if args.grid_search else "[OFFSET-GRID]"
                print(
                    f"{prefix} id={grid_id:02d} forward={values_to_print[grid_id,0].item():+.3f} "
                    f"tilt={values_to_print[grid_id,2].item():+.3f} "
                    f"pair={(core._pair_hits[mask].sum()/pair_attempts).item():.4f} "
                    f"hit={(core._target_hit_count[mask].sum()/touchdowns).item():.4f} "
                    f"error={(core._touchdown_error_sum[mask].sum()/touchdowns).item():.4f}",
                    flush=True,
                )
        env.close()
        simulation_app.close()
        return

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = ROOT / "logs/rsl_rl" / EXPERIMENT / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    save(log_dir / "model_0_expert.pt", model, optimizer, start_update)
    gamma_frame = args.gamma_frame
    hop_obs = obs.clone()
    dist = model.distribution(hop_obs)
    held = dist.sample().detach()
    hop_action = held.clone()
    hop_logp = dist.log_prob(hop_action).sum(-1).detach()
    hop_return = torch.zeros(args.num_envs, device=device)
    hop_discount = torch.ones(args.num_envs, device=device)
    hop_duration = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    storage = {key: [] for key in ("obs", "action", "logp", "return", "next_obs", "discount", "done", "weight")}
    pending = {
        "obs": torch.zeros_like(hop_obs), "action": torch.zeros_like(hop_action),
        "logp": torch.zeros_like(hop_logp), "return": torch.zeros_like(hop_return),
        "next_obs": torch.zeros_like(hop_obs), "discount": torch.ones_like(hop_discount),
        "done": torch.zeros_like(hop_return),
        "weight": torch.ones_like(hop_return),
    }
    pending_valid = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    train_split_count = torch.zeros(2, device=device)
    train_split_hit = torch.zeros(2, device=device)
    train_split_error = torch.zeros(2, device=device)

    wall_t0 = datetime.now()
    frame_count = 0
    collected = 0
    print(f"[ROLLOUT] collecting {args.transitions_per_update} hop transitions "
          f"({args.transitions_per_update // max(args.num_envs, 1)} hops/env) for update "
          f"{start_update + 1}/{start_update + args.updates}", flush=True)
    update = start_update
    while update < start_update + args.updates:
        core = base.unwrapped
        target_before, _ = core.commands.lookahead()
        target_before = target_before.clone()
        phase_was_short = ((core.commands.route_index % 2) == 0).clone()
        first_hit_before = core._first_hop_hit_for_pair.clone()
        obs_dict, reward, dones, _ = env.step(held)
        # RslRlVecEnvWrapper returns dones as int64 (N, 1).  Flatten AND cast
        # to bool: the boundary masks below must stay boolean, otherwise
        # `ids[mask]` silently becomes fancy indexing over values 0/1 and
        # asserts when a single env reaches a boundary (index 1 out of range).
        dones = dones.reshape(-1) > 0
        next_obs = obs_dict["policy"]
        # Keep only a tiny amount of low-level shaping.  The high-level return
        # is defined at touchdown so pair success cannot be drowned out by
        # hundreds of per-frame reward terms.
        hop_return += hop_discount * (0.0002 * reward)
        hop_discount *= gamma_frame
        hop_duration += 1
        boundary = base.unwrapped._touchdown_event | dones
        if torch.any(boundary):
            touchdown = core._touchdown_event
            hit = core._target_hit_event
            landing_error = torch.linalg.norm(
                core._robot.data.root_pos_w[:, :2] - target_before, dim=1
            )
            for split_id, split_mask in (
                (0, touchdown & phase_was_short),
                (1, touchdown & (~phase_was_short)),
            ):
                if torch.any(split_mask):
                    train_split_count[split_id] += split_mask.float().sum()
                    train_split_hit[split_id] += hit[split_mask].float().sum()
                    train_split_error[split_id] += landing_error[split_mask].sum()
            event_reward = torch.zeros(args.num_envs, device=device)
            event_reward[touchdown] += torch.where(
                hit[touchdown], torch.full_like(landing_error[touchdown], args.hit_reward),
                torch.full_like(landing_error[touchdown], args.miss_penalty),
            )
            precision_scale = torch.where(
                phase_was_short,
                torch.full_like(landing_error, short_precision_reward_scale),
                torch.full_like(landing_error, long_precision_reward_scale),
            )
            precision_sigma = torch.where(
                phase_was_short,
                torch.full_like(landing_error, short_precision_sigma),
                torch.full_like(landing_error, long_precision_sigma),
            )
            error_penalty = torch.where(
                phase_was_short,
                torch.full_like(landing_error, short_landing_error_penalty),
                torch.full_like(landing_error, long_landing_error_penalty),
            )
            event_reward[touchdown] += precision_scale[touchdown] * torch.exp(
                -landing_error[touchdown].square() / (2.0 * precision_sigma[touchdown].square())
            )
            event_reward[touchdown] -= error_penalty[touchdown] * landing_error[touchdown]
            event_reward[touchdown] += (
                args.prepared_reward
                * core._prepared_landing_event[touchdown].float()
            )
            event_reward[touchdown] -= (
                args.touchdown_attitude_penalty
                * core._touchdown_attitude_error[touchdown]
            )
            event_reward[touchdown] -= (
                args.touchdown_next_velocity_penalty
                * core._touchdown_next_velocity_error[touchdown]
            )
            eligible_second = touchdown & (~phase_was_short) & first_hit_before
            event_reward[eligible_second] += torch.where(
                hit[eligible_second], torch.full_like(landing_error[eligible_second], 8.0),
                torch.full_like(landing_error[eligible_second], -5.0),
            )
            event_reward[dones & (~touchdown)] -= 3.0
            hop_return += hop_discount * event_reward
            ids = torch.where(boundary)[0]
            defer_short = touchdown[ids] & phase_was_short[ids] & (~dones[ids])
            short_ids = ids[defer_short]
            finish_ids = ids[~defer_short]
            if len(short_ids) > 0:
                pending["obs"][short_ids] = hop_obs[short_ids]
                pending["action"][short_ids] = hop_action[short_ids]
                pending["logp"][short_ids] = hop_logp[short_ids]
                pending["return"][short_ids] = hop_return[short_ids]
                pending["next_obs"][short_ids] = next_obs[short_ids]
                pending["discount"][short_ids] = hop_discount[short_ids]
                pending["done"][short_ids] = dones[short_ids].float()
                pending["weight"][short_ids] = 1.0
                pending_valid[short_ids] = True
            if len(finish_ids) > 0:
                paired = finish_ids[pending_valid[finish_ids]]
                if len(paired) > 0:
                    # Credit the first-hop action with the outcome that only
                    # becomes known after the second touchdown.
                    first_hit = first_hit_before[paired]
                    pair_success = first_hit & hit[paired]
                    first_credit = torch.where(
                        pair_success,
                        torch.full_like(hop_return[paired], args.pair_success_reward),
                        torch.where(
                            first_hit,
                            torch.full_like(hop_return[paired], args.pair_second_miss_penalty),
                            torch.full_like(hop_return[paired], args.pair_first_miss_penalty),
                        ),
                    )
                    # First-hop action owns both its own eligibility and the
                    # eventual pair result.  The second-hop action owns pair
                    # success/failure only when hop one actually hit.
                    pending["return"][paired] += pending["discount"][paired] * first_credit
                    second_credit = torch.where(
                        pair_success,
                        torch.full_like(hop_return[paired], args.pair_success_reward),
                        torch.full_like(hop_return[paired], args.pair_second_miss_penalty),
                    )
                    hop_return[paired] += first_hit.float() * hop_discount[paired] * second_credit
                    pending["weight"][paired] = torch.where(
                        first_hit, torch.full_like(hop_return[paired], 1.5),
                        torch.ones_like(hop_return[paired]),
                    )
                    for key in storage:
                        storage[key].append(pending[key][paired].clone())
                    pending_valid[paired] = False
                for key, value in (
                    ("obs", hop_obs), ("action", hop_action), ("logp", hop_logp),
                    ("return", hop_return), ("next_obs", next_obs),
                    ("discount", hop_discount), ("done", dones.float()),
                    ("weight", torch.where(
                        eligible_second,
                        torch.full_like(hop_return, args.eligible_second_weight),
                        torch.ones_like(hop_return),
                    )),
                ):
                    storage[key].append(value[finish_ids].clone())
            with torch.no_grad():
                new_dist = model.distribution(next_obs[ids])
                new_action = new_dist.sample()
                held[ids] = new_action
                hop_action[ids] = new_action
                hop_logp[ids] = new_dist.log_prob(new_action).sum(-1)
            hop_obs[ids] = next_obs[ids]
            hop_return[ids] = 0.0
            hop_discount[ids] = 1.0
            hop_duration[ids] = 0
        obs = next_obs
        collected = sum(chunk.shape[0] for chunk in storage["obs"])
        frame_count += 1
        if frame_count % args.progress_log_every == 0:
            core = base.unwrapped
            hops_per_env = core._touchdown_count.sum().float() / max(args.num_envs, 1)
            minutes = (datetime.now() - wall_t0).total_seconds() / 60.0
            print(f"[ROLLOUT] frames={frame_count} min={minutes:.1f} "
                  f"collected={collected}/{args.transitions_per_update} "
                  f"hops/env={hops_per_env.item():.1f} "
                  f"hit={(core._target_hit_count.sum() / core._touchdown_count.sum().clamp_min(1)).item():.4f}",
                  flush=True)

        count = collected
        if count < args.transitions_per_update:
            continue
        batch = {key: torch.cat(value, dim=0)[:args.transitions_per_update] for key, value in storage.items()}
        storage = {key: [] for key in storage}
        collected = 0
        with torch.no_grad():
            bootstrap = (1.0 - batch["done"]) * batch["discount"] * model.value(batch["next_obs"])
            target = (batch["return"] + bootstrap).clamp(-20.0, 20.0)
            advantage = target - model.value(batch["obs"])
            advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1.0e-6)
        indices = torch.arange(args.transitions_per_update, device=device)
        for _ in range(4):
            indices = indices[torch.randperm(indices.numel(), device=device)]
            for ids in indices.split(256):
                distribution = model.distribution(batch["obs"][ids])
                logp = distribution.log_prob(batch["action"][ids]).sum(-1)
                ratio = (logp - batch["logp"][ids]).exp()
                surrogate = torch.minimum(ratio * advantage[ids], ratio.clamp(0.8, 1.2) * advantage[ids])
                value_loss = (model.value(batch["obs"][ids]) - target[ids]).square().mean()
                if bias_tensor is not None:
                    anchor = (model.mean(batch["obs"][ids]) - bias_tensor).square().mean()
                else:
                    anchor = model.mean(batch["obs"][ids]).square().mean()
                policy_loss = -(batch["weight"][ids] * surrogate).sum() / batch["weight"][ids].sum().clamp_min(1.0)
                loss = policy_loss + 0.5 * value_loss + args.anchor_scale * anchor
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        update += 1
        # Exploration anneal: wide while the route is easy, tight by the time
        # the full 0.50-0.80 / 0.80-1.00 m, 180 deg ranges are commanded.
        progress = (update - start_update) / max(args.updates, 1)
        std = args.std_start + (args.std_end - args.std_start) * progress
        with torch.no_grad():
            model.log_std.copy_(torch.full_like(model.log_std, math.log(std)))
        lr = args.lr * 0.1 + 0.5 * (args.lr - args.lr * 0.1) * (1.0 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        save(log_dir / f"model_{update}.pt", model, optimizer, update)
        core = base.unwrapped
        pairs = core._pair_attempts.sum().float().clamp_min(1.0)
        cond = core._conditional_second_attempts.sum().float().clamp_min(1.0)
        hits = core._touchdown_count.sum().float().clamp_min(1.0)
        prepared = core._prepared_landing_count.sum().float() / hits
        attitude_error = core._touchdown_attitude_error_sum.sum() / hits
        next_velocity_error = core._touchdown_next_velocity_error_sum.sum() / hits
        short_count = train_split_count[0].clamp_min(1.0)
        long_count = train_split_count[1].clamp_min(1.0)
        short_hit = train_split_hit[0] / short_count
        long_hit = train_split_hit[1] / long_count
        short_error = train_split_error[0] / short_count
        long_error = train_split_error[1] / long_count
        correction_mode = "none" if args.passive_airborne else args.correction_mode
        motor_correction = torch.as_tensor(
            0.0 if correction_mode == "none" else args.motor_correction_limit,
            device=device,
        )
        print(f"[SEMIMDP] update={update}/{start_update + args.updates} "
              f"transitions={args.transitions_per_update} "
              f"pair={(core._pair_hits.sum()/pairs).item():.4f} "
              f"conditional={(core._conditional_second_hits.sum()/cond).item():.4f} "
              f"hit={(core._target_hit_count.sum()/hits).item():.4f} "
              f"error={(core._touchdown_error_sum.sum()/hits).item():.4f} "
              f"short_hit={short_hit.item():.4f} short_err={short_error.item():.4f} "
              f"long_hit={long_hit.item():.4f} long_err={long_error.item():.4f} "
              f"prepared={prepared.item():.4f} "
              f"att_err={attitude_error.item():.4f} "
              f"next_v_err={next_velocity_error.item():.4f} "
              f"corr={correction_mode} "
              f"motor_limit={motor_correction.item():.3f} "
              f"std={std:.4f} lr={lr:.2e} "
              f"min={(datetime.now() - wall_t0).total_seconds()/60.0:.1f}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
