"""Train two-cycle planner-conditioned circular hopping."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train Quadhopper planner circular task")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Stable 37-D or planner 42-D checkpoint")
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
EXPERIMENT = "quadhopper_planner_circular_v10"


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
        ):
            # Every v5 run starts from a v4 policy which had already reached
            # the full-planner phase. Never regress to the stationary-apex
            # curriculum when resuming a short v5 fine-tuning run.
            curriculum_iteration_offset = max(400.0, float(checkpoint_data.get("iter", 0)))

    env_cfg = PlannerCircularEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.debug_vis = False
    env_cfg.curriculum_iteration_offset = curriculum_iteration_offset
    env_cfg.power_model_path = str(PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
    env_cfg.csv_log_path = str(PROJECT_DIR / "outputs/planner_circular/on_quadhopper_sim.csv")
    env = RslRlVecEnvWrapper(gym.make("Quadhopper-Planner-Circular-Direct-v0", cfg=env_cfg))

    runner_cfg = PlannerCircularPPORunnerCfg()
    runner_cfg.experiment_name = EXPERIMENT
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=str(log_dir), device=args_cli.device)
    if args_cli.checkpoint:
        if input_width == 37:
            migrated = migrate_stable_checkpoint(
                source_checkpoint, log_dir / "initial_policy_42d.pt"
            )
            print(f"[INFO] Migrating stable 37-D policy to 42-D: {migrated}")
            runner.load(str(migrated), load_optimizer=False)
        elif input_width == 42:
            if "quadhopper_planner_circular_v10" in source_checkpoint.parts:
                print(f"[INFO] Exactly resuming 42-D v10 checkpoint: {source_checkpoint}")
                runner.load(str(source_checkpoint), load_optimizer=True)
            elif (
                "quadhopper_planner_circular_v4" in source_checkpoint.parts
                or "quadhopper_planner_circular_v5" in source_checkpoint.parts
                or "quadhopper_planner_circular_v6" in source_checkpoint.parts
                or "quadhopper_planner_circular_v7" in source_checkpoint.parts
                or "quadhopper_planner_circular_v8" in source_checkpoint.parts
                or "quadhopper_planner_circular_v9" in source_checkpoint.parts
            ):
                transfer_checkpoint = log_dir / "initial_policy_for_v10.pt"
                transferred_state = checkpoint_data["model_state_dict"].copy()
                # Do not inherit v7's runaway exploration std (>1.17).
                transferred_state["std"] = torch.full_like(transferred_state["std"], 0.2)
                torch.save(
                    {
                        "model_state_dict": transferred_state,
                        "iter": 0,
                        "infos": {"source_checkpoint": str(source_checkpoint)},
                    },
                    transfer_checkpoint,
                )
                print(f"[INFO] Initializing v10 nominal policy with std=0.2 and no optimizer: {source_checkpoint}")
                runner.load(str(transfer_checkpoint), load_optimizer=False)
            else:
                raise ValueError(
                    "Only v4-v9 policy transfer or exact v10 resume is supported for 42-D checkpoints."
                )
        else:
            raise ValueError(f"Unsupported checkpoint observation width: {input_width}")
    iterations = args_cli.iterations if args_cli.iterations is not None else runner_cfg.max_iterations
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
