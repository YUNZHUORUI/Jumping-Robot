"""Play a planner-conditioned circular or random-route checkpoint."""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play planner-conditioned circular hopping")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Fixed evaluation seed so checkpoints see the same waypoint sequence.",
)
parser.add_argument("--height_stage", choices=("low", "high", "alternate"), default="alternate")
parser.add_argument("--height_high", type=float, default=1.0)
parser.add_argument("--height_low", type=float, default=0.7)
parser.add_argument(
    "--route", choices=("circle", "random_two_hop"), default="circle"
)
parser.add_argument(
    "--distance_stage",
    choices=("direction", "medium", "short", "bridge", "full", "custom"),
    default="full",
)
parser.add_argument("--short_radius_min", type=float, default=0.5)
parser.add_argument("--short_radius_max", type=float, default=0.8)
parser.add_argument("--long_radius_min", type=float, default=0.8)
parser.add_argument("--long_radius_max", type=float, default=1.0)
parser.add_argument("--landing_compensation", type=float, default=0.0)
parser.add_argument("--relative_next_hop", action="store_true")
parser.add_argument("--max_turn_angle", type=float, default=180.0)
parser.add_argument("--landing_velocity_scale", type=float, default=1.0)
parser.add_argument("--target_tolerance", type=float, default=0.10)
parser.add_argument("--anticipatory", action="store_true")
parser.add_argument(
    "--continuity",
    action="store_true",
    help="Use the settled-touchdown two-hop continuity references from v32 training.",
)
parser.add_argument("--anticipatory_speed", type=float, default=0.30)
parser.add_argument("--anticipatory_tilt_deg", type=float, default=6.0)
parser.add_argument("--landing_correction_gain", type=float, default=0.0)
parser.add_argument(
    "--continuous_queue",
    action="store_true",
    help="Keep the waypoint queue continuous; this is now the default for random two-hop routes.",
)
parser.add_argument(
    "--pair_restart_queue",
    action="store_true",
    help="Legacy mode: after a long hop, restart a fresh short/long pair around the measured touchdown.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after this many environment steps and print the latest episode metrics.",
)
parser.add_argument("--no_debug_vis", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if (args_cli.anticipatory or args_cli.continuity) and not args_cli.relative_next_hop:
    parser.error("--anticipatory/--continuity requires --relative_next_hop")
if args_cli.anticipatory and args_cli.continuity:
    parser.error("--anticipatory and --continuity are separate reference modes")
if args_cli.continuous_queue and args_cli.pair_restart_queue:
    parser.error("--continuous_queue and --pair_restart_queue are mutually exclusive")
args_cli.continuous_queue = not args_cli.pair_restart_queue
DISTANCE_STAGES = {
    "direction": (0.22, 0.30, 0.22, 0.30),
    "medium": (0.30, 0.50, 0.30, 0.50),
    "short": (0.50, 0.80, 0.50, 0.80),
    "bridge": (0.50, 0.80, 0.65, 0.85),
    "full": (0.50, 0.80, 0.80, 1.00),
}
if args_cli.route == "random_two_hop" and args_cli.distance_stage != "custom":
    (
        args_cli.short_radius_min,
        args_cli.short_radius_max,
        args_cli.long_radius_min,
        args_cli.long_radius_max,
    ) = DISTANCE_STAGES[args_cli.distance_stage]

PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = {
    "low": "quadhopper_planner_circular_v15_descend_to_070",
    "high": "quadhopper_planner_circular_v15_high_100",
    "alternate": "quadhopper_planner_circular_v15_alternate_070_100",
}
if args_cli.route == "random_two_hop":
    low_cm = round(args_cli.height_low * 100.0)
    high_cm = round(args_cli.height_high * 100.0)
    height_label = (
        f"alternate_{low_cm:03d}_{high_cm:03d}"
        if args_cli.height_stage == "alternate"
        else f"fixed_{round((args_cli.height_low if args_cli.height_stage == 'low' else args_cli.height_high) * 100.0):03d}"
    )
    experiment = (
        f"quadhopper_planner_random_two_hop_v24_{args_cli.distance_stage}_{height_label}"
    )
else:
    experiment = EXPERIMENTS[args_cli.height_stage]
LOG_ROOT = PROJECT_DIR / "logs/rsl_rl" / experiment


def latest_checkpoint() -> Path | None:
    candidates = list(LOG_ROOT.glob("*/model_*.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


checkpoint = Path(args_cli.checkpoint).expanduser() if args_cli.checkpoint else latest_checkpoint()
if checkpoint is None:
    parser.error(f"No planner circular checkpoint found under {LOG_ROOT}")
checkpoint = checkpoint.resolve()
if not checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys

import gymnasium as gym
import isaaclab as _il
import torch

_ISAACLAB_RL = os.path.join(os.path.dirname(_il.__file__), "source", "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)

import Quadhopper_Planner_Circular  # noqa: F401
import Quadhopper_Planner_Random  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Planner_Circular.planner_circular_env import PlannerCircularEnvCfg
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg


def main():
    env_cfg = (
        PlannerRandomTwoHopEnvCfg()
        if args_cli.route == "random_two_hop"
        else PlannerCircularEnvCfg()
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    env_cfg.debug_vis = not args_cli.no_debug_vis
    env_cfg.force_full_planner = True
    # Evaluation must expose policy quality rather than observation RNG.
    env_cfg.observation_noise_std = 0.0
    env_cfg.alternate_target_heights = args_cli.height_stage == "alternate"
    env_cfg.alternate_height_high = args_cli.height_high
    env_cfg.alternate_height_low = args_cli.height_low
    env_cfg.target_height = (
        args_cli.height_low if args_cli.height_stage == "low" else args_cli.height_high
    )
    env_cfg.target_tolerance = args_cli.target_tolerance
    # Evaluation is always the final fixed command, never the training schedule.
    env_cfg.fixed_height_curriculum = False
    env_cfg.symmetric_height_tracking = True
    env_cfg.require_apex_tolerance_for_hit = True
    if args_cli.route == "random_two_hop":
        env_cfg.short_hop_radius_min = args_cli.short_radius_min
        env_cfg.short_hop_radius_max = args_cli.short_radius_max
        env_cfg.long_hop_radius_min = args_cli.long_radius_min
        env_cfg.long_hop_radius_max = args_cli.long_radius_max
        env_cfg.planner_landing_compensation_m = args_cli.landing_compensation
        env_cfg.relative_next_hop_observation = args_cli.relative_next_hop
        env_cfg.max_turn_angle_deg = args_cli.max_turn_angle
        env_cfg.planner_landing_xy_velocity_scale = args_cli.landing_velocity_scale
        env_cfg.restart_two_hop_pair = not args_cli.continuous_queue
        # Landing feedback is an independent receding-horizon reference and
        # can be evaluated without enabling either terminal-velocity mode.
        env_cfg.online_landing_correction_gain = args_cli.landing_correction_gain
        if args_cli.anticipatory:
            env_cfg.anticipatory_velocity_blend = 1.0
            env_cfg.anticipatory_speed_max = args_cli.anticipatory_speed
            env_cfg.anticipatory_tilt_rad = math.radians(
                args_cli.anticipatory_tilt_deg
            )
            env_cfg.anticipation_start_phase = 0.50
            env_cfg.prepared_attitude_tolerance_rad = math.radians(10.0)
            env_cfg.prepared_velocity_tolerance = max(
                0.35, args_cli.anticipatory_speed + 0.25
            )
        if args_cli.continuity:
            env_cfg.planner_landing_xy_velocity_scale = 0.0
            env_cfg.anticipatory_velocity_blend = 0.0
            env_cfg.anticipatory_tilt_rad = math.radians(
                args_cli.anticipatory_tilt_deg
            )
            env_cfg.anticipation_start_phase = 0.60
            env_cfg.prepared_attitude_tolerance_rad = math.radians(10.0)
            env_cfg.prepared_velocity_tolerance = 0.55
        print(
            "[PLAY] Random two-hop geometry: "
            f"first=[{args_cli.short_radius_min:.2f}, {args_cli.short_radius_max:.2f}] m "
            "about the current position; "
            f"second=[{args_cli.long_radius_min:.2f}, {args_cli.long_radius_max:.2f}] m "
            f"about P_t; max turn={args_cli.max_turn_angle:.1f} deg; "
            f"planner compensation={args_cli.landing_compensation:.3f} m; "
            f"landing-v scale={args_cli.landing_velocity_scale:.2f}; "
            f"success tolerance={args_cli.target_tolerance:.2f} m; "
            f"queue={'continuous' if args_cli.continuous_queue else 'pair-restart'}"
        )
    env_cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    output_name = "planner_random_two_hop" if args_cli.route == "random_two_hop" else "planner_circular"
    env_cfg.csv_log_path = str(PROJECT_DIR / f"outputs/{output_name}/on_quadhopper_sim.csv")
    task_id = (
        "Quadhopper-Planner-Random-Two-Hop-Direct-v0"
        if args_cli.route == "random_two_hop"
        else "Quadhopper-Planner-Circular-Direct-v0"
    )
    env = RslRlVecEnvWrapper(gym.make(task_id, cfg=env_cfg))
    runner = OnPolicyRunner(env, PlannerCircularPPORunnerCfg().to_dict(), log_dir=None, device=args_cli.device)
    print(f"[PLAY] Loading planner {args_cli.route} checkpoint: {checkpoint}")
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args_cli.device)
    obs, _ = env.reset()
    if args_cli.route == "random_two_hop":
        base_env = env.unwrapped
        p_t, p_t1 = base_env.commands.lookahead()
        first_distance = torch.linalg.norm(
            p_t - base_env.commands.anchor_w, dim=1
        )
        second_distance = torch.linalg.norm(p_t1 - p_t, dim=1)
        print(
            "[PLAY] Sampled distances: "
            f"first={first_distance.min().item():.3f}--{first_distance.max().item():.3f} m, "
            f"second={second_distance.min().item():.3f}--{second_distance.max().item():.3f} m"
        )
    step_count = 0
    latest_log = {}
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, extras = env.step(actions)
        step_count += 1
        if "log" in extras:
            latest_log = extras["log"]
        if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
            break
    if latest_log:
        print(f"[EVAL] steps={step_count}")
        for key in sorted(latest_log):
            if key.startswith("Metrics/"):
                value = latest_log[key]
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().mean().item()
                print(f"[EVAL] {key}={value}")
    if args_cli.max_steps is not None and args_cli.route == "random_two_hop":
        base_env = env.unwrapped
        touchdown_count = torch.sum(base_env._touchdown_count).float()
        short_touchdown_count = torch.sum(base_env._short_touchdown_count).float()
        long_touchdown_count = torch.sum(base_env._long_touchdown_count).float()
        per_env_hit_rate = base_env._target_hit_count.float() / torch.clamp(
            base_env._touchdown_count.float(), min=1.0
        )
        per_env_max_streak = base_env._max_consecutive_hits.float()
        valid_apex = base_env._settled_apex_valid
        direct_metrics = {
            "episode_touchdown_error_m": (
                torch.sum(base_env._touchdown_error_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_along_error_m": (
                torch.sum(base_env._touchdown_along_error_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_lateral_abs_error_m": (
                torch.sum(base_env._touchdown_lateral_abs_error_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_attitude_error_rad": (
                torch.sum(base_env._touchdown_attitude_error_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_next_velocity_error_mps": (
                torch.sum(base_env._touchdown_next_velocity_error_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_next_velocity_projection_mps": (
                torch.sum(base_env._touchdown_next_velocity_projection_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "touchdown_next_velocity_lateral_abs_mps": (
                torch.sum(base_env._touchdown_next_velocity_lateral_abs_sum)
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "prepared_landing_rate": (
                torch.sum(base_env._prepared_landing_count).float()
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "target_hit_rate": (
                torch.sum(base_env._target_hit_count).float()
                / torch.clamp(touchdown_count, min=1.0)
            ),
            "target_hit_rate_min": torch.min(per_env_hit_rate),
            "target_hit_rate_p10": torch.quantile(per_env_hit_rate, 0.10),
            "target_hit_rate_median": torch.median(per_env_hit_rate),
            "short_touchdown_error_m": (
                torch.sum(base_env._short_touchdown_error_sum)
                / torch.clamp(short_touchdown_count, min=1.0)
            ),
            "short_target_hit_rate": (
                torch.sum(base_env._short_target_hit_count).float()
                / torch.clamp(short_touchdown_count, min=1.0)
            ),
            "long_touchdown_error_m": (
                torch.sum(base_env._long_touchdown_error_sum)
                / torch.clamp(long_touchdown_count, min=1.0)
            ),
            "long_target_hit_rate": (
                torch.sum(base_env._long_target_hit_count).float()
                / torch.clamp(long_touchdown_count, min=1.0)
            ),
            "last_apex_height_m": (
                torch.mean(base_env._settled_apex_height[valid_apex])
                if torch.any(valid_apex)
                else torch.tensor(0.0, device=base_env.device)
            ),
            "last_apex_error_m": (
                torch.mean(base_env._settled_apex_error[valid_apex])
                if torch.any(valid_apex)
                else torch.tensor(0.0, device=base_env.device)
            ),
            "max_consecutive_hits": torch.mean(
                per_env_max_streak
            ),
            "max_consecutive_hits_min": torch.min(per_env_max_streak),
            "max_consecutive_hits_median": torch.median(per_env_max_streak),
            "route_completion": torch.mean(
                (
                    base_env._consecutive_hits
                    >= base_env.commands.steps_per_revolution
                ).float()
            ),
            "successful_waypoints": torch.mean(
                base_env._successful_cycles.float()
            ),
        }
        print(f"[EVAL-DIRECT] steps={step_count}", flush=True)
        for key, value in direct_metrics.items():
            print(f"[EVAL-DIRECT] {key}={value.item()}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
