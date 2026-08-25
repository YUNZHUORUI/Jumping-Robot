"""Clone route contexts across env groups and search second-hop actions.

Each group executes an identical first hop.  At its first touchdown the group
fans out over candidate second-hop actions.  Touchdown-state components and
the complete 69-D context are saved for supervised ranking/distillation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--planner_checkpoint", required=True)
parser.add_argument("--output", default="outputs/second_hop_search/candidates.pt")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--candidates", type=int, default=8)
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=1800)
parser.add_argument("--candidate_scale", type=float, default=0.20)
parser.add_argument("--resample_candidates_each_trial", action="store_true")
parser.add_argument("--trial_seed_stride", type=int, default=1009)
parser.add_argument("--search_phase", choices=("first", "second"), default="second")
parser.add_argument("--target_tolerance", type=float, default=0.10)
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
parser.add_argument("--action_scorer", default=None,
                    help="Hit-only second-hop scorer used after first-hop candidate evaluation")
parser.add_argument("--seed", type=int, default=123)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not (0.0 < args.short_radius_min <= args.short_radius_max):
    parser.error("--short_radius_min/max must satisfy 0 < min <= max")
if not (0.0 < args.long_radius_min <= args.long_radius_max):
    parser.error("--long_radius_min/max must satisfy 0 < min <= max")
if args.target_tolerance <= 0.0:
    parser.error("--target_tolerance must be positive")
args.headless = True
args.rendering_mode = "performance"
app = AppLauncher(args).app

import gymnasium as gym
import isaaclab as _il
import torch
from torch import nn
from torch.distributions import Normal

_RL = os.path.join(os.path.dirname(_il.__file__), "source", "isaaclab_rl")
if _RL not in sys.path:
    sys.path.insert(0, _RL)
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import Quadhopper_Planner_Random  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg  # noqa: E402
from Quadhopper_Planner_Random.pair_objective import touchdown_components, touchdown_score  # noqa: E402
from Quadhopper_Planner_Random.second_hop_scorer import SecondHopScorer  # noqa: E402
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg  # noqa: E402
from Quadhopper_Planner_Random.two_hop_state_planner_wrapper import TeacherTwoHopStatePlannerVecEnv  # noqa: E402


class HopActorCritic(nn.Module):
    def __init__(self, obs_dim=69, action_dim=4):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, action_dim))
        self.second_hop_adapter = nn.Sequential(nn.Linear(obs_dim, 64), nn.ELU(), nn.Linear(64, action_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), -3.0))

    def mean(self, obs):
        return self.actor(obs) + obs[:, -2:-1] * self.second_hop_adapter(obs)


def load_model_state_compat(model, state):
    current = model.state_dict()
    for key in ("actor.0.weight", "critic.0.weight"):
        if state[key].shape != current[key].shape:
            expanded = torch.zeros_like(current[key])
            expanded[:, :state[key].shape[1]] = state[key]
            state[key] = expanded
    missing, unexpected = model.load_state_dict(state, strict=False)
    if any(not key.startswith("second_hop_adapter.") for key in missing) or unexpected:
        raise ValueError(f"Incompatible planner checkpoint: missing={missing}, unexpected={unexpected}")


def make_env():
    cfg = PlannerRandomTwoHopEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.debug_vis = False
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
    cfg.short_hop_radius_min = args.short_radius_min
    cfg.short_hop_radius_max = args.short_radius_max
    cfg.long_hop_radius_min = args.long_radius_min
    cfg.long_hop_radius_max = args.long_radius_max
    cfg.max_turn_angle_deg = 180.0
    cfg.distance_curriculum_iterations = 0.0
    cfg.curriculum_by_hops = True
    cfg.target_tolerance = args.target_tolerance
    cfg.planner_landing_xy_velocity_scale = 0.0
    cfg.anticipatory_velocity_blend = 0.0
    cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    base = RslRlVecEnvWrapper(gym.make("Quadhopper-Planner-Random-Two-Hop-Direct-v0", cfg=cfg))
    runner = OnPolicyRunner(base, PlannerCircularPPORunnerCfg().to_dict(), log_dir=None, device=args.device)
    runner.load(str(Path(args.teacher_checkpoint).resolve()), load_optimizer=False)
    teacher = runner.alg.policy
    teacher.eval()
    return base, TeacherTwoHopStatePlannerVecEnv(
        base,
        teacher,
        velocity_feedback_gain=args.velocity_feedback_gain,
        attitude_feedback_gain=args.attitude_feedback_gain,
        motor_correction_limit=args.motor_correction_limit,
        correction_mode=args.correction_mode,
        landing_correction_height=args.landing_correction_height,
        state_reward_scale=args.state_reward_scale,
        expert_anchor_scale=0.0,
    )


def clone_routes(core, group_size):
    origins = core._terrain.env_origins[:, :2]
    for start in range(0, args.num_envs, group_size):
        ids = torch.arange(start, min(start + group_size, args.num_envs), device=core.device)
        leader = ids[0]
        rel_anchor = core.commands.anchor_w[leader] - origins[leader]
        rel_targets = core.commands.targets_w[leader] - origins[leader]
        core.commands.anchor_w[ids] = origins[ids] + rel_anchor
        core.commands.targets_w[ids] = origins[ids, None, :] + rel_targets
        core.commands.route_index[ids] = core.commands.route_index[leader].clone()
        core.commands.cycle_index[ids] = core.commands.cycle_index[leader].clone()


def main():
    if args.num_envs % args.candidates:
        raise ValueError("--num_envs must be divisible by --candidates")
    torch.manual_seed(args.seed)
    base, env = make_env()
    model = HopActorCritic().to(args.device)
    data = torch.load(Path(args.planner_checkpoint).resolve(), map_location=args.device, weights_only=False)
    load_model_state_compat(model, data["model_state_dict"])
    model.eval()
    rows = {key: [] for key in (
        "context_id", "candidate_id", "observation", "action", "position",
        "velocity", "attitude", "spring", "score", "hit", "second_hit", "pair_hit",
    )}
    scorer = None
    scorer_offsets = None
    if args.action_scorer:
        scorer_data = torch.load(Path(args.action_scorer).resolve(), map_location=args.device, weights_only=False)
        scorer = SecondHopScorer().to(args.device)
        scorer.load_state_dict(scorer_data["model_state_dict"])
        scorer.eval()
        scorer_offsets = scorer_data["candidate_offsets"].to(args.device)

    def select_second(observations):
        with torch.no_grad():
            base_action = model.mean(observations)
            if scorer is None:
                return base_action
            candidates = (base_action[:, None, :] + scorer_offsets[None, :, :]).clamp(-1.0, 1.0)
            expanded = observations[:, None, :].expand(-1, len(scorer_offsets), -1)
            logits, _ = scorer(expanded.reshape(-1, observations.shape[-1]), candidates.reshape(-1, 4))
            best = logits.reshape(len(observations), len(scorer_offsets)).argmax(dim=1)
            return candidates[torch.arange(len(observations), device=args.device), best]
    candidate_offsets = torch.zeros(args.candidates, 4, device=args.device)
    if args.candidates > 1:
        candidate_offsets[1:] = args.candidate_scale * (
            2.0 * torch.rand(args.candidates - 1, 4, device=args.device) - 1.0
        )

    for trial in range(args.trials):
        if args.resample_candidates_each_trial:
            # The same seed controls route reset and candidate generation for
            # reproducibility, while the stride gives every trial a genuinely
            # different route/candidate distribution.
            torch.manual_seed(args.seed + trial * args.trial_seed_stride)
            candidate_offsets.zero_()
            if args.candidates > 1:
                candidate_offsets[1:] = args.candidate_scale * (
                    2.0 * torch.rand(args.candidates - 1, 4, device=args.device) - 1.0
                )
        obs, _ = env.reset()
        clone_routes(base.unwrapped, args.candidates)
        obs = env.get_observations()["policy"]
        with torch.no_grad():
            held = model.mean(obs)
        second_obs = torch.zeros_like(obs)
        second_action = torch.zeros(args.num_envs, 4, device=args.device)
        first_hit_cache = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        component_cache = {
            key: torch.zeros(args.num_envs, device=args.device)
            for key in ("position", "velocity", "attitude", "spring", "score")
        }
        searching = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        finished = torch.zeros_like(searching)
        if args.search_phase == "first":
            group_candidate = torch.arange(args.num_envs, device=args.device) % args.candidates
            second_obs.copy_(obs)
            second_action.copy_((held + candidate_offsets[group_candidate]).clamp(-1.0, 1.0))
            held.copy_(second_action)
        for _ in range(args.max_steps):
            obs_dict, _, dones, _ = env.step(held)
            obs = obs_dict["policy"]
            core = base.unwrapped
            touchdown = core._touchdown_event
            first = touchdown & ((core.commands.route_index % 2) == 1) & (~searching)
            if torch.any(first):
                ids = torch.where(first)[0]
                if args.search_phase == "second":
                    group_candidate = ids % args.candidates
                    with torch.no_grad():
                        action = (model.mean(obs[ids]) + candidate_offsets[group_candidate]).clamp(-1.0, 1.0)
                    second_obs[ids] = obs[ids]
                    second_action[ids] = action
                    held[ids] = action
                else:
                    spring_id = core._spring_joint_id
                    components = touchdown_components(
                        core._landing_error[ids], core._touchdown_next_velocity_error[ids],
                        core._touchdown_attitude_error[ids],
                        torch.linalg.norm(core._robot.data.root_ang_vel_b[ids], dim=1),
                        core._robot.data.joint_pos[ids, spring_id],
                        core._robot.data.joint_vel[ids, spring_id],
                    )
                    for key in ("position", "velocity", "attitude", "spring"):
                        component_cache[key][ids] = components[key]
                    component_cache["score"][ids] = touchdown_score(components)
                    first_hit_cache[ids] = core._target_hit_event[ids]
                    held[ids] = select_second(obs[ids])
                searching[ids] = True
            second = touchdown & searching & (~finished) & ((core.commands.route_index % 2) == 0)
            if torch.any(second):
                ids = torch.where(second)[0]
                spring_id = core._spring_joint_id
                if args.search_phase == "second":
                    components = touchdown_components(
                        core._landing_error[ids], core._touchdown_next_velocity_error[ids],
                        core._touchdown_attitude_error[ids],
                        torch.linalg.norm(core._robot.data.root_ang_vel_b[ids], dim=1),
                        core._robot.data.joint_pos[ids, spring_id],
                        core._robot.data.joint_vel[ids, spring_id],
                    )
                    score = touchdown_score(components)
                    first_labels = torch.ones(len(ids), device=args.device)
                else:
                    components = {key: component_cache[key][ids] for key in ("position", "velocity", "attitude", "spring")}
                    score = component_cache["score"][ids]
                    first_labels = first_hit_cache[ids].float()
                second_labels = core._target_hit_event[ids].float()
                rows["context_id"].append(
                    (trial * (args.num_envs // args.candidates) + ids // args.candidates).cpu()
                )
                rows["candidate_id"].append((ids % args.candidates).cpu())
                rows["observation"].append(second_obs[ids].cpu())
                rows["action"].append(second_action[ids].cpu())
                for key in ("position", "velocity", "attitude", "spring"):
                    rows[key].append(components[key].cpu())
                rows["score"].append(score.cpu())
                rows["hit"].append(first_labels.cpu())
                rows["second_hit"].append(second_labels.cpu())
                rows["pair_hit"].append((first_labels * second_labels).cpu())
                finished[ids] = True
            if torch.all(finished):
                break
        print(f"[SEARCH] trial={trial + 1}/{args.trials} completed={finished.sum().item()}/{args.num_envs}", flush=True)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **{key: torch.cat(value) for key, value in rows.items()},
        "candidate_offsets": candidate_offsets.cpu(),
        "infos": {
            "search_phase": args.search_phase,
            "candidate_scale": args.candidate_scale,
            "target_tolerance": args.target_tolerance,
            "short_radius_min": args.short_radius_min,
            "short_radius_max": args.short_radius_max,
            "long_radius_min": args.long_radius_min,
            "long_radius_max": args.long_radius_max,
            "correction_mode": args.correction_mode,
            "motor_correction_limit": args.motor_correction_limit,
        },
    }, output)
    print(f"[SEARCH] saved {len(torch.cat(rows['score']))} candidates to {output}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
