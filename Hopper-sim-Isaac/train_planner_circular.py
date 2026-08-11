"""Train two-cycle planner-conditioned circular hopping."""

import argparse
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
    "--low_curriculum_iterations",
    type=float,
    default=300.0,
    help="Deprecated compatibility option; fixed 0.70 m training does not use a height curriculum.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Planner_Circular.checkpoint_migration import migrate_stable_checkpoint
from Quadhopper_Planner_Circular.planner_circular_env import PlannerCircularEnvCfg
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg


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
    else:
        raise ValueError("--accuracy_finetune supports low or alternate height stages")


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = PROJECT_DIR / "logs" / "rsl_rl" / EXPERIMENT / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint = None
    checkpoint_data = None
    input_width = None
    curriculum_iteration_offset = 0.0
    if args_cli.checkpoint:
        source_checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
        checkpoint_data = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        input_width = checkpoint_data["model_state_dict"]["memory_a.rnn.weight_ih_l0"].shape[1]
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

    env_cfg = PlannerCircularEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.debug_vis = False
    env_cfg.alternate_target_heights = args_cli.height_stage == "alternate"
    env_cfg.alternate_height_high = args_cli.height_high
    env_cfg.alternate_height_low = args_cli.height_low
    env_cfg.target_height = (
        args_cli.height_low if args_cli.height_stage == "low" else args_cli.height_high
    )
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
        if args_cli.height_stage == "low":
            # Fixed 0.70 m already has good height control, so this specialist
            # stage can devote most of its update budget to XY accuracy.
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
    env_cfg.curriculum_iteration_offset = curriculum_iteration_offset
    env_cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    env_cfg.csv_log_path = str(PROJECT_DIR / "outputs/planner_circular/on_quadhopper_sim.csv")
    env = RslRlVecEnvWrapper(gym.make("Quadhopper-Planner-Circular-Direct-v0", cfg=env_cfg))

    runner_cfg = PlannerCircularPPORunnerCfg()
    runner_cfg.experiment_name = EXPERIMENT
    if args_cli.direct_variable_height:
        # The 37-D stable policy has no circular or height-conditioned behavior
        # to preserve.  Use the normal planner learning rate and save densely
        # enough to select the point before any late-stage regression.
        runner_cfg.save_interval = 25
    if args_cli.expand_variable_height:
        runner_cfg.algorithm.learning_rate = 5.0e-5
        runner_cfg.algorithm.entropy_coef = 1.0e-4
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
                transferred_state["std"] = torch.full_like(transferred_state["std"], 0.15)
                torch.save(
                    {
                        "model_state_dict": transferred_state,
                        "iter": 0,
                        "infos": {
                            "source_checkpoint": str(source_checkpoint),
                            "height_stage": args_cli.height_stage,
                            "optimizer_reset": True,
                        },
                    },
                    transfer_checkpoint,
                )
                print(f"[INFO] Transferring 43-D policy with optimizer reset: {source_checkpoint}")
                runner.load(str(transfer_checkpoint), load_optimizer=False)
        else:
            raise ValueError(f"Unsupported checkpoint observation width: {input_width}")
    iterations = args_cli.iterations if args_cli.iterations is not None else runner_cfg.max_iterations
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
