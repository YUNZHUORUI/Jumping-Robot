"""Train two-cycle planner-conditioned circular or random-route hopping."""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train Quadhopper planner circular task")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="37-D stable, 42-D legacy planner, or 43-D planner checkpoint"
)
parser.add_argument(
    "--height_stage",
    choices=("low", "high", "alternate"),
    default="low",
    help="Curriculum stage: fixed 0.70 m, fixed 1.00 m, or alternating commands.",
)
parser.add_argument("--resume_optimizer", action="store_true")
parser.add_argument(
    "--accuracy_finetune",
    action="store_true",
    help="Fine-tune first-attempt landing accuracy; terminate an episode on each miss.",
)
parser.add_argument(
    "--streak_finetune",
    action="store_true",
    help="Fine-tune consecutive hits with balanced phases while retaining recovery trajectories.",
)
parser.add_argument(
    "--anticipatory_finetune",
    action="store_true",
    help="Fine-tune a two-hop plan whose Pt terminal velocity and attitude prepare Pt+1.",
)
parser.add_argument(
    "--continuity_finetune",
    action="store_true",
    help="Curriculum for consecutive two-hop hits with next-target landing preparation.",
)
parser.add_argument(
    "--first_attempt_finetune",
    action="store_true",
    help="Train a rolling route where every touchdown advances, including misses.",
)
parser.add_argument(
    "--pair_polish",
    action="store_true",
    help="Low-noise final polish emphasizing consecutive first-attempt pairs.",
)
parser.add_argument("--smooth_distance_curriculum", action="store_true")
parser.add_argument("--distance_curriculum_iterations", type=float, default=100.0)
parser.add_argument(
    "--direct_variable_height",
    action="store_true",
    help="Train periodic alternating heights directly from a stable baseline without a fixed-height specialist.",
)
parser.add_argument(
    "--expand_variable_height",
    action="store_true",
    help="Expand an existing alternating-height policy to a wider periodic height pair.",
)
parser.add_argument("--height_high", type=float, default=1.0)
parser.add_argument("--height_low", type=float, default=0.7)
parser.add_argument(
    "--route",
    choices=("circle", "random_two_hop"),
    default="circle",
    help="Ground waypoint generator. random_two_hop alternates 0.5--0.8 m and 0.8--1.0 m hops.",
)
parser.add_argument(
    "--distance_stage",
    choices=("direction", "medium", "short", "bridge", "full", "custom"),
    default="full",
    help="Random-route distance curriculum stage; custom uses the four explicit radius flags.",
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
parser.add_argument("--anticipatory_speed", type=float, default=0.30)
parser.add_argument("--anticipatory_tilt_deg", type=float, default=6.0)
parser.add_argument("--anticipatory_reward_scale", type=float, default=1.0)
parser.add_argument("--landing_correction_gain", type=float, default=0.0)
parser.add_argument("--tolerance_curriculum_start", type=float, default=0.15)
parser.add_argument("--tolerance_curriculum_iterations", type=float, default=60.0)
parser.add_argument(
    "--low_curriculum_iterations",
    type=float,
    default=300.0,
    help="Deprecated compatibility option; fixed 0.70 m training does not use a height curriculum.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.pair_polish and not args_cli.first_attempt_finetune:
    parser.error("--pair_polish requires --first_attempt_finetune")
if args_cli.smooth_distance_curriculum and (
    not args_cli.first_attempt_finetune or args_cli.distance_stage != "full"
):
    parser.error(
        "--smooth_distance_curriculum requires --first_attempt_finetune --distance_stage full"
    )
if sum(
    int(flag)
    for flag in (
        args_cli.accuracy_finetune,
        args_cli.streak_finetune,
        args_cli.anticipatory_finetune,
        args_cli.continuity_finetune,
        args_cli.first_attempt_finetune,
    )
) > 1:
    parser.error("accuracy, streak, and anticipatory fine-tuning are separate stages")
DISTANCE_STAGES = {
    "direction": (0.22, 0.30, 0.22, 0.30, 0.12),
    "medium": (0.30, 0.50, 0.30, 0.50, 0.10),
    "short": (0.50, 0.80, 0.50, 0.80, 0.08),
    "bridge": (0.50, 0.80, 0.65, 0.85, 0.06),
    "full": (0.50, 0.80, 0.80, 1.00, 0.06),
}
if args_cli.route == "random_two_hop" and args_cli.distance_stage != "custom":
    (
        args_cli.short_radius_min,
        args_cli.short_radius_max,
        args_cli.long_radius_min,
        args_cli.long_radius_max,
        _,
    ) = DISTANCE_STAGES[args_cli.distance_stage]
args_cli.headless = True
args_cli.rendering_mode = "performance"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys
from datetime import datetime

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

from Quadhopper_Planner_Circular.checkpoint_migration import (
    absolute_next_to_relative_state_dict,
    migrate_stable_checkpoint,
)
from Quadhopper_Planner_Circular.planner_circular_env import PlannerCircularEnvCfg
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg


PROJECT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = {
    "low": "quadhopper_planner_circular_v17_fixed_070_from_stable",
    "high": "quadhopper_planner_circular_v15_high_100",
    "alternate": "quadhopper_planner_circular_v15_alternate_070_100",
}
EXPERIMENT = EXPERIMENTS[args_cli.height_stage]
if args_cli.direct_variable_height:
    if args_cli.height_stage != "alternate":
        raise ValueError("--direct_variable_height requires --height_stage alternate")
    low_cm = round(args_cli.height_low * 100.0)
    high_cm = round(args_cli.height_high * 100.0)
    EXPERIMENT = (
        f"quadhopper_planner_circular_v21_direct_alternate_{low_cm:03d}_{high_cm:03d}"
    )
if args_cli.expand_variable_height:
    if args_cli.height_stage != "alternate":
        raise ValueError("--expand_variable_height requires --height_stage alternate")
    if args_cli.direct_variable_height or args_cli.accuracy_finetune:
        raise ValueError("--expand_variable_height is a separate curriculum stage")
    low_cm = round(args_cli.height_low * 100.0)
    high_cm = round(args_cli.height_high * 100.0)
    EXPERIMENT = (
        f"quadhopper_planner_circular_v22_expand_alternate_{low_cm:03d}_{high_cm:03d}"
    )
if args_cli.accuracy_finetune:
    if args_cli.direct_variable_height:
        raise ValueError("Use direct variable-height learning before --accuracy_finetune")
    if args_cli.height_stage == "low":
        EXPERIMENT = "quadhopper_planner_circular_v18_fixed_070_accuracy"
    elif args_cli.height_stage == "alternate":
        low_cm = round(args_cli.height_low * 100.0)
        high_cm = round(args_cli.height_high * 100.0)
        EXPERIMENT = (
            f"quadhopper_planner_circular_v20_alternate_{low_cm:03d}_{high_cm:03d}_height_accuracy"
        )
    elif args_cli.height_stage == "high" and args_cli.route == "random_two_hop":
        # The random-route block below replaces this placeholder with its
        # dedicated v25 full/distance-stage namespace.  Fixed 1 m route
        # training has already solved height and may fine-tune XY accuracy.
        EXPERIMENT = "quadhopper_planner_circular_v20_fixed_100_accuracy"
    else:
        raise ValueError(
            "--accuracy_finetune supports low/alternate stages, plus fixed-high random_two_hop"
        )
fixed_high_soft_accuracy = False
fixed_high_streak = False
fixed_high_anticipatory = False
fixed_high_continuity = False
fixed_high_first_attempt = False
fixed_high_pair_polish = False
fixed_high_smooth_distance = False
if args_cli.route == "random_two_hop":
    if args_cli.direct_variable_height or args_cli.expand_variable_height:
        raise ValueError(
            "random_two_hop is a separate route-transfer stage; do not combine it with another curriculum flag"
        )
    low_cm = round(args_cli.height_low * 100.0)
    high_cm = round(args_cli.height_high * 100.0)
    height_label = (
        f"alternate_{low_cm:03d}_{high_cm:03d}"
        if args_cli.height_stage == "alternate"
        else f"fixed_{round((args_cli.height_low if args_cli.height_stage == 'low' else args_cli.height_high) * 100.0):03d}"
    )
    fixed_high_soft_accuracy = (
        args_cli.accuracy_finetune and args_cli.height_stage == "high"
    )
    fixed_high_streak = args_cli.streak_finetune and args_cli.height_stage == "high"
    fixed_high_anticipatory = (
        args_cli.anticipatory_finetune and args_cli.height_stage == "high"
    )
    fixed_high_continuity = (
        args_cli.continuity_finetune and args_cli.height_stage == "high"
    )
    fixed_high_first_attempt = (
        args_cli.first_attempt_finetune and args_cli.height_stage == "high"
    )
    fixed_high_pair_polish = fixed_high_first_attempt and args_cli.pair_polish
    fixed_high_smooth_distance = (
        fixed_high_first_attempt and args_cli.smooth_distance_curriculum
    )
    route_version = (
        "v36"
        if fixed_high_smooth_distance
        else "v35"
        if fixed_high_pair_polish
        else "v34"
        if fixed_high_first_attempt
        else "v32"
        if fixed_high_continuity
        else "v31"
        if fixed_high_anticipatory
        else "v30"
        if args_cli.relative_next_hop and fixed_high_streak
        else "v29"
        if args_cli.relative_next_hop
        else "v28"
        if fixed_high_streak
        else "v26"
        if fixed_high_soft_accuracy
        else "v25"
        if args_cli.accuracy_finetune
        else "v24"
    )
    accuracy_suffix = (
        "_relative_next_smooth_distance"
        if fixed_high_smooth_distance
        else "_relative_next_pair_polish"
        if fixed_high_pair_polish
        else "_relative_next_first_attempt"
        if fixed_high_first_attempt
        else "_relative_next_continuity"
        if fixed_high_continuity
        else "_relative_next_anticipatory"
        if fixed_high_anticipatory
        else "_relative_next_recovery_streak"
        if args_cli.relative_next_hop and fixed_high_streak
        else "_relative_next_accuracy"
        if args_cli.relative_next_hop and fixed_high_soft_accuracy
        else "_relative_next"
        if args_cli.relative_next_hop
        else "_recovery_streak"
        if fixed_high_streak
        else "_soft_accuracy"
        if fixed_high_soft_accuracy
        else "_accuracy"
        if args_cli.accuracy_finetune
        else ""
    )
    EXPERIMENT = (
        f"quadhopper_planner_random_two_hop_{route_version}_"
        f"{args_cli.distance_stage}{accuracy_suffix}_"
        f"turn_{round(args_cli.max_turn_angle):03d}_"
        f"tol_{round(100 * args_cli.target_tolerance):02d}_"
        f"spd_{round(100 * args_cli.anticipatory_speed):03d}_"
        f"tilt_{round(args_cli.anticipatory_tilt_deg):02d}_"
        f"arw_{round(100 * args_cli.anticipatory_reward_scale):03d}_"
        f"lcg_{round(100 * args_cli.landing_correction_gain):03d}_{height_label}"
    )


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = PROJECT_DIR / "logs" / "rsl_rl" / EXPERIMENT / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint = None
    checkpoint_data = None
    input_width = None
    source_uses_relative_next = False
    curriculum_iteration_offset = 0.0
    if args_cli.checkpoint:
        source_checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
        checkpoint_data = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        input_width = checkpoint_data["model_state_dict"]["memory_a.rnn.weight_ih_l0"].shape[1]
        source_uses_relative_next = any(
            "relative_next" in part for part in source_checkpoint.parts
        ) or any(
            "quadhopper_planner_random_two_hop_v31" in part
            for part in source_checkpoint.parts
        )
        if input_width == 42 and "quadhopper_planner_circular_v4" in source_checkpoint.parts:
            # The v4 policy has already completed the height/path curriculum.
            # Preserve that behavior while learning the new joint-horizon and
            # touchdown-precision objective.
            curriculum_iteration_offset = 400.0
        elif input_width == 42 and (
            "quadhopper_planner_circular_v5" in source_checkpoint.parts
            or "quadhopper_planner_circular_v6" in source_checkpoint.parts
            or "quadhopper_planner_circular_v7" in source_checkpoint.parts
            or "quadhopper_planner_circular_v8" in source_checkpoint.parts
            or "quadhopper_planner_circular_v9" in source_checkpoint.parts
            or "quadhopper_planner_circular_v10" in source_checkpoint.parts
            or "quadhopper_planner_circular_v11_variable_height" in source_checkpoint.parts
        ):
            # Every v5 run starts from a v4 policy which had already reached
            # the full-planner phase. Never regress to the stationary-apex
            # curriculum when resuming a short v5 fine-tuning run.
            curriculum_iteration_offset = max(400.0, float(checkpoint_data.get("iter", 0)))

    env_cfg = (
        PlannerRandomTwoHopEnvCfg()
        if args_cli.route == "random_two_hop"
        else PlannerCircularEnvCfg()
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.debug_vis = False
    env_cfg.alternate_target_heights = args_cli.height_stage == "alternate"
    env_cfg.alternate_height_high = args_cli.height_high
    env_cfg.alternate_height_low = args_cli.height_low
    env_cfg.target_height = (
        args_cli.height_low if args_cli.height_stage == "low" else args_cli.height_high
    )
    if args_cli.target_tolerance <= 0.0:
        raise ValueError("--target_tolerance must be positive")
    env_cfg.target_tolerance = args_cli.target_tolerance
    # The low stage is a fixed-height specialist: command 0.70 m from the
    # first rollout.  Variable/descending-height curricula are deferred until
    # this fixed target is learned reliably.
    env_cfg.fixed_height_curriculum = False
    env_cfg.height_curriculum_start = 1.30
    env_cfg.height_curriculum_end = args_cli.height_low
    env_cfg.height_curriculum_iterations = args_cli.low_curriculum_iterations
    env_cfg.height_curriculum_iteration_offset = (
        float(checkpoint_data.get("iter", 0))
        if input_width == 43 and EXPERIMENT in source_checkpoint.parts
        else 0.0
    )
    env_cfg.symmetric_height_tracking = True
    env_cfg.require_apex_tolerance_for_hit = True
    env_cfg.force_full_planner = input_width in (42, 43)
    if args_cli.route == "random_two_hop":
        env_cfg.short_hop_radius_min = args_cli.short_radius_min
        env_cfg.short_hop_radius_max = args_cli.short_radius_max
        env_cfg.long_hop_radius_min = args_cli.long_radius_min
        env_cfg.long_hop_radius_max = args_cli.long_radius_max
        env_cfg.planner_landing_compensation_m = args_cli.landing_compensation
        env_cfg.relative_next_hop_observation = args_cli.relative_next_hop
        env_cfg.max_turn_angle_deg = args_cli.max_turn_angle
        env_cfg.planner_landing_xy_velocity_scale = args_cli.landing_velocity_scale
        if args_cli.smooth_distance_curriculum:
            if args_cli.distance_curriculum_iterations <= 0.0:
                raise ValueError("--distance_curriculum_iterations must be positive")
            env_cfg.distance_curriculum_iterations = (
                args_cli.distance_curriculum_iterations
            )
    if args_cli.expand_variable_height:
        if input_width != 43:
            raise ValueError("--expand_variable_height requires a trained 43-D planner checkpoint")
        env_cfg.apex_event_reward_scale = 150.0
        env_cfg.apex_error_penalty_scale = -350.0
        env_cfg.apex_shortfall_penalty_scale = -200.0
        env_cfg.airborne_overshoot_penalty_scale = -200.0
        env_cfg.height_progress_reward_scale = 80.0
    if args_cli.accuracy_finetune:
        env_cfg.terminate_on_target_miss = True
        env_cfg.target_miss_penalty_scale = -150.0
        env_cfg.landing_error_penalty_scale = -250.0
        env_cfg.landing_precision_reward_scale = 120.0
        env_cfg.landing_precision_width = 0.04
        env_cfg.projected_landing_penalty_scale = -80.0
        if args_cli.route == "random_two_hop" and args_cli.height_stage == "high":
            # Full-distance fixed-height transfer is already near the target
            # basin.  A terminal miss and very large penalties destroyed that
            # policy within 50 updates, so use a conservative precision stage
            # that preserves recovery data while tightening XY guidance.
            env_cfg.terminate_on_target_miss = False
            env_cfg.target_miss_penalty_scale = -90.0
            env_cfg.landing_error_penalty_scale = -160.0
            env_cfg.landing_precision_reward_scale = 100.0
            env_cfg.landing_precision_width = 0.05
            env_cfg.projected_landing_penalty_scale = -50.0
        if args_cli.height_stage in ("low", "high"):
            # Fixed-height specialists already have good height control, so
            # this stage can devote most of its update budget to XY accuracy.
            env_cfg.apex_event_reward_scale = 50.0
            env_cfg.apex_error_penalty_scale = -120.0
            env_cfg.height_progress_reward_scale = 30.0
        else:
            # Alternating commands must first learn that 0.70 and 0.80 m are
            # distinct tasks.  Strong symmetric tracking prevents the policy
            # from compromising at one average apex while landing accurately.
            env_cfg.apex_event_reward_scale = 150.0
            env_cfg.apex_error_penalty_scale = -350.0
            env_cfg.apex_shortfall_penalty_scale = -200.0
            env_cfg.airborne_overshoot_penalty_scale = -200.0
            env_cfg.height_progress_reward_scale = 80.0
    if args_cli.streak_finetune:
        if args_cli.route != "random_two_hop" or args_cli.height_stage != "high":
            raise ValueError(
                "--streak_finetune currently requires random_two_hop with fixed high height"
            )
        # A terminal miss made the rollout distribution collapse onto fresh
        # failures and removed the post-hit states needed for continuity.
        # Keep recovery trajectories: a miss clears the streak in the event
        # logic, while the policy continues to observe later flight states.
        env_cfg.terminate_on_target_miss = False
        env_cfg.target_hit_reward_scale = 160.0
        env_cfg.target_miss_penalty_scale = -100.0
        env_cfg.landing_error_penalty_scale = -180.0
        env_cfg.landing_precision_reward_scale = 120.0
        env_cfg.landing_precision_width = 0.045
        env_cfg.projected_landing_penalty_scale = -60.0
        env_cfg.streak_progress_reward_scale = 400.0
        env_cfg.circle_complete_reward_scale = 4000.0
        env_cfg.apex_event_reward_scale = 50.0
        env_cfg.apex_error_penalty_scale = -120.0
        env_cfg.height_progress_reward_scale = 30.0
    if args_cli.anticipatory_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
        ):
            raise ValueError(
                "--anticipatory_finetune requires fixed-high random_two_hop and --relative_next_hop"
            )
        env_cfg.anticipatory_velocity_blend = 1.0
        env_cfg.anticipatory_speed_max = args_cli.anticipatory_speed
        env_cfg.anticipatory_tilt_rad = math.radians(args_cli.anticipatory_tilt_deg)
        env_cfg.anticipation_start_phase = 0.50
        if args_cli.anticipatory_reward_scale < 0.0:
            raise ValueError("--anticipatory_reward_scale must be non-negative")
        if args_cli.landing_correction_gain < 0.0:
            raise ValueError("--landing_correction_gain must be non-negative")
        anticipation_reward_scale = args_cli.anticipatory_reward_scale
        env_cfg.anticipatory_attitude_reward_scale = (
            40.0 * anticipation_reward_scale
        )
        env_cfg.anticipatory_attitude_penalty_scale = (
            -60.0 * anticipation_reward_scale
        )
        env_cfg.anticipatory_velocity_penalty_scale = (
            -80.0 * anticipation_reward_scale
        )
        env_cfg.prepared_landing_reward_scale = 300.0 * anticipation_reward_scale
        env_cfg.prepared_attitude_tolerance_rad = math.radians(10.0)
        env_cfg.prepared_velocity_tolerance = max(
            0.35, args_cli.anticipatory_speed + 0.25
        )
        env_cfg.online_landing_correction_gain = args_cli.landing_correction_gain
        env_cfg.attitude_penalty_scale = -12.0
        env_cfg.terminate_on_target_miss = False
        env_cfg.target_hit_reward_scale = 160.0
        env_cfg.target_miss_penalty_scale = -100.0
        env_cfg.landing_error_penalty_scale = -180.0
        env_cfg.landing_precision_reward_scale = 120.0
        env_cfg.landing_precision_width = 0.05
        env_cfg.projected_landing_penalty_scale = -60.0
        env_cfg.streak_progress_reward_scale = 250.0
        env_cfg.apex_event_reward_scale = 50.0
        env_cfg.apex_error_penalty_scale = -120.0
        env_cfg.height_progress_reward_scale = 30.0
    if args_cli.continuity_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
        ):
            raise ValueError(
                "--continuity_finetune requires fixed-high random_two_hop and --relative_next_hop"
            )
        if args_cli.tolerance_curriculum_start < args_cli.target_tolerance:
            raise ValueError(
                "--tolerance_curriculum_start must be >= --target_tolerance"
            )
        if args_cli.tolerance_curriculum_iterations <= 0.0:
            raise ValueError("--tolerance_curriculum_iterations must be positive")
        # Keep the current touchdown settled and accurate.  P_(t+1) affects
        # the late-flight attitude reference and the pair-success objective,
        # but does not demand a destabilizing horizontal impact velocity.
        env_cfg.planner_landing_xy_velocity_scale = 0.0
        env_cfg.anticipatory_velocity_blend = 0.0
        env_cfg.anticipatory_tilt_rad = math.radians(args_cli.anticipatory_tilt_deg)
        env_cfg.anticipation_start_phase = 0.60
        env_cfg.anticipatory_attitude_reward_scale = 10.0
        env_cfg.anticipatory_attitude_penalty_scale = -15.0
        env_cfg.anticipatory_velocity_penalty_scale = 0.0
        env_cfg.prepared_landing_reward_scale = 60.0
        env_cfg.pair_hit_reward_scale = 300.0
        env_cfg.prepared_attitude_tolerance_rad = math.radians(10.0)
        env_cfg.prepared_velocity_tolerance = 0.55
        env_cfg.online_landing_correction_gain = args_cli.landing_correction_gain
        env_cfg.target_tolerance_curriculum_start = (
            args_cli.tolerance_curriculum_start
        )
        env_cfg.target_tolerance_curriculum_iterations = (
            args_cli.tolerance_curriculum_iterations
        )
        env_cfg.terminate_on_target_miss = False
        env_cfg.target_hit_reward_scale = 180.0
        env_cfg.target_miss_penalty_scale = -120.0
        env_cfg.landing_error_penalty_scale = -220.0
        env_cfg.landing_precision_reward_scale = 140.0
        env_cfg.landing_precision_width = 0.045
        env_cfg.projected_landing_penalty_scale = -75.0
        env_cfg.streak_progress_reward_scale = 1200.0
        env_cfg.circle_complete_reward_scale = 5000.0
        env_cfg.apex_event_reward_scale = 50.0
        env_cfg.apex_error_penalty_scale = -120.0
        env_cfg.height_progress_reward_scale = 30.0
    if args_cli.first_attempt_finetune:
        if (
            args_cli.route != "random_two_hop"
            or args_cli.height_stage != "high"
            or not args_cli.relative_next_hop
        ):
            raise ValueError(
                "--first_attempt_finetune requires fixed-high random_two_hop and --relative_next_hop"
            )
        # The random-route environment advances after every touchdown.  These
        # terms therefore optimize first-attempt accuracy and two-hop pairs;
        # there is no recovery distribution left to exploit.
        env_cfg.terminate_on_target_miss = False
        env_cfg.target_hit_reward_scale = 250.0
        env_cfg.target_miss_penalty_scale = -180.0
        env_cfg.landing_error_penalty_scale = -250.0
        env_cfg.landing_precision_reward_scale = 150.0
        env_cfg.landing_precision_width = 0.05
        env_cfg.projected_landing_penalty_scale = -80.0
        env_cfg.pair_hit_reward_scale = 400.0 if args_cli.pair_polish else 100.0
        env_cfg.streak_progress_reward_scale = (
            1200.0 if args_cli.pair_polish else 300.0
        )
        env_cfg.circle_complete_reward_scale = 5000.0
        env_cfg.apex_event_reward_scale = 50.0
        env_cfg.apex_error_penalty_scale = -120.0
        env_cfg.height_progress_reward_scale = 30.0
    env_cfg.curriculum_iteration_offset = curriculum_iteration_offset
    env_cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    output_name = "planner_random_two_hop" if args_cli.route == "random_two_hop" else "planner_circular"
    env_cfg.csv_log_path = str(PROJECT_DIR / f"outputs/{output_name}/on_quadhopper_sim.csv")
    task_id = (
        "Quadhopper-Planner-Random-Two-Hop-Direct-v0"
        if args_cli.route == "random_two_hop"
        else "Quadhopper-Planner-Circular-Direct-v0"
    )
    env = RslRlVecEnvWrapper(gym.make(task_id, cfg=env_cfg))

    runner_cfg = PlannerCircularPPORunnerCfg()
    runner_cfg.experiment_name = EXPERIMENT
    if args_cli.direct_variable_height:
        # The 37-D stable policy has no circular or height-conditioned behavior
        # to preserve.  Use the normal planner learning rate and save densely
        # enough to select the point before any late-stage regression.
        runner_cfg.save_interval = 25
    if args_cli.expand_variable_height:
        runner_cfg.algorithm.learning_rate = 5.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-4
        runner_cfg.save_interval = 25
    if args_cli.height_stage == "low" and input_width in (42, 43):
        # Preserve a transferred planner policy with conservative updates.
        # A 37-D stable baseline has no circular behavior to protect and uses
        # the normal planner PPO settings to learn the task from scratch.
        runner_cfg.algorithm.learning_rate = 1.0e-4
        runner_cfg.algorithm.entropy_coef = 2.0e-4
        runner_cfg.save_interval = 50
    if args_cli.accuracy_finetune:
        runner_cfg.algorithm.learning_rate = 5.0e-5
        runner_cfg.algorithm.entropy_coef = 1.0e-4
        runner_cfg.save_interval = 25
    if args_cli.route == "random_two_hop":
        # Route transfer should preserve the accepted v22 height controller
        # while learning the much broader direction/distance distribution.
        runner_cfg.algorithm.learning_rate = 5.0e-5
        runner_cfg.algorithm.entropy_coef = 1.0e-4
        runner_cfg.save_interval = 25
    if fixed_high_soft_accuracy:
        # Balanced phase adaptation is a distribution-transfer step, not a
        # fresh route-learning run.  Use conservative updates and dense
        # checkpoints so deterministic validation can select pre-regression.
        runner_cfg.algorithm.learning_rate = 2.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-5
        runner_cfg.save_interval = 10
    if args_cli.streak_finetune:
        runner_cfg.algorithm.learning_rate = 1.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-5
        runner_cfg.save_interval = 10
    if args_cli.anticipatory_finetune:
        runner_cfg.algorithm.learning_rate = 1.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-5
        runner_cfg.save_interval = 10
    if args_cli.continuity_finetune:
        runner_cfg.algorithm.learning_rate = 1.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-5
        runner_cfg.save_interval = 10
    if args_cli.first_attempt_finetune:
        runner_cfg.algorithm.learning_rate = 5.0e-6 if args_cli.pair_polish else 1.0e-5
        runner_cfg.algorithm.entropy_coef = 5.0e-5
        runner_cfg.save_interval = 10
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(log_dir), device=args_cli.device)
    if args_cli.checkpoint:
        if input_width in (37, 42):
            migrated = migrate_stable_checkpoint(
                source_checkpoint, log_dir / "initial_policy_43d.pt"
            )
            print(f"[INFO] Migrating {input_width}-D policy to 43-D: {migrated}")
            runner.load(str(migrated), load_optimizer=False)
        elif input_width == 43:
            if args_cli.resume_optimizer and EXPERIMENT in source_checkpoint.parts:
                print(f"[INFO] Exactly resuming 43-D {args_cli.height_stage} stage")
                runner.load(str(source_checkpoint), load_optimizer=True)
            else:
                transfer_checkpoint = log_dir / "initial_policy_43d_transfer.pt"
                transferred_state = checkpoint_data["model_state_dict"].copy()
                if args_cli.relative_next_hop and not source_uses_relative_next:
                    transferred_state = absolute_next_to_relative_state_dict(
                        transferred_state
                    )
                elif source_uses_relative_next and not args_cli.relative_next_hop:
                    raise ValueError(
                        "A relative-next checkpoint requires --relative_next_hop"
                    )
                transferred_state["std"] = torch.full_like(transferred_state["std"], 0.15)
                torch.save(
                    {
                        "model_state_dict": transferred_state,
                        "iter": 0,
                        "infos": {
                            "source_checkpoint": str(source_checkpoint),
                            "height_stage": args_cli.height_stage,
                            "route": args_cli.route,
                            "relative_next_hop": args_cli.relative_next_hop,
                            "optimizer_reset": True,
                        },
                    },
                    transfer_checkpoint,
                )
                print(f"[INFO] Transferring 43-D policy with optimizer reset: {source_checkpoint}")
                runner.load(str(transfer_checkpoint), load_optimizer=False)
        else:
            raise ValueError(f"Unsupported checkpoint observation width: {input_width}")
    if args_cli.route == "random_two_hop":
        fixed_action_std = (
            # The rolling first-attempt task changes the post-touchdown state
            # distribution substantially and needs enough exploration to
            # discover the longer-hop action profile.  Other precision stages
            # retain the converged 0.01 checkpoint noise.
            0.01
            if args_cli.pair_polish
            else 0.02
            if args_cli.smooth_distance_curriculum
            else 0.04
            if args_cli.first_attempt_finetune
            else 0.01
            if args_cli.streak_finetune or args_cli.anticipatory_finetune or args_cli.continuity_finetune
            else 0.01
            if args_cli.accuracy_finetune
            else DISTANCE_STAGES[args_cli.distance_stage][4]
            if args_cli.distance_stage != "custom"
            else 0.08
        )
        if not hasattr(runner.alg.policy, "std"):
            raise RuntimeError("The random-route curriculum requires scalar action std")
        with torch.no_grad():
            runner.alg.policy.std.fill_(fixed_action_std)
        runner.alg.policy.std.requires_grad_(False)
        print(
            f"[INFO] Distance stage {args_cli.distance_stage}: "
            f"short=[{args_cli.short_radius_min:.2f}, {args_cli.short_radius_max:.2f}] m, "
            f"long=[{args_cli.long_radius_min:.2f}, {args_cli.long_radius_max:.2f}] m, "
            f"fixed action std={fixed_action_std:.2f}"
        )
    iterations = args_cli.iterations if args_cli.iterations is not None else runner_cfg.max_iterations
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
