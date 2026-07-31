import argparse
from datetime import datetime
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train circular waypoint hopping with RSL-RL PPO")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=None, help="Override PPO training iterations")
parser.add_argument("--hop_distance", type=float, default=None, help="Override circular waypoint chord distance")
parser.add_argument("--target_tolerance", type=float, default=None, help="Override landing target tolerance")
parser.add_argument("--apex_height_ref", type=float, default=None, help="Override desired per-hop apex height above takeoff")
parser.add_argument("--flight_time_ref", type=float, default=None, help="Override desired per-hop flight time")
parser.add_argument("--planning_horizon_hops", type=int, default=None, help="Override waypoint lookahead count in observations")
parser.add_argument("--min_target_hop_height", type=float, default=None, help="Override minimum hop height for target hits")
parser.add_argument("--min_flight_time_for_hit", type=float, default=None, help="Override minimum flight time required for target hits")
parser.add_argument("--max_under_arc_error_for_hit", type=float, default=None, help="Override allowed below-reference arc error for target hits")
parser.add_argument("--max_stance_time", type=float, default=None, help="Override maximum allowed settled stance time")
parser.add_argument("--stance_root_height", type=float, default=None, help="Override root-height contact proxy threshold")
parser.add_argument("--contact_vz_threshold", type=float, default=None, help="Override vertical velocity threshold for root-height contact")
parser.add_argument("--liftoff_vz_threshold", type=float, default=None, help="Override vertical velocity threshold for liftoff detection")
parser.add_argument("--max_missed_landings_per_target", type=int, default=None, help="Override allowed missed landings before reset")
parser.add_argument("--max_successful_hops", type=int, default=None, help="Override successful hops required per episode")
parser.add_argument("--episode_length_s", type=float, default=None, help="Override episode length in seconds")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume training from a model checkpoint")
parser.add_argument(
    "--task_preset",
    choices=["default", "strict_short", "track_feasible", "track_precise"],
    default="default",
    help="Apply a known task setting before individual CLI overrides",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
args_cli.rendering_mode = "performance"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab as _il
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ISAACLAB_SOURCE = os.path.join(os.path.dirname(_il.__file__), "source")
_ISAACLAB_RL = os.path.join(_ISAACLAB_SOURCE, "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import CircularHopper_Isaac
from CircularHopper_Isaac.circular_hopper_env import CircularHopperEnvCfg
from CircularHopper_Isaac.rsl_rl_ppo_cfg import CircularHopperPPORunnerCfg


def find_latest_checkpoint(experiment_name: str) -> str | None:
    log_root = os.path.join(os.path.dirname(__file__), "logs", "rsl_rl", experiment_name)
    if not os.path.exists(log_root):
        return None
    candidates = []
    for root, _, files in os.walk(log_root):
        for name in files:
            if name.endswith(".pt"):
                path = os.path.join(root, name)
                candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def find_migrated_checkpoint() -> str | None:
    path = os.path.join(
        os.path.dirname(__file__),
        "logs",
        "rsl_rl",
        "circular_hopper_horizon",
        "migrated",
        "migrated_horizon_model_12489.pt",
    )
    return path if os.path.exists(path) else None


def main():
    env_cfg = CircularHopperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.task_preset == "strict_short":
        env_cfg.hop_distance = 0.14
        env_cfg.target_tolerance = 0.18
        env_cfg.apex_height_ref = 0.26
        env_cfg.flight_time_ref = 0.55
        env_cfg.min_target_hop_height = 0.14
        env_cfg.min_flight_time_for_hit = 0.22
        env_cfg.max_under_arc_error_for_hit = 0.12
        env_cfg.max_missed_landings_per_target = 4
        env_cfg.max_stance_time = 0.75
        env_cfg.max_successful_hops = 8
        env_cfg.episode_length_s = 12
    elif args_cli.task_preset == "track_feasible":
        env_cfg.hop_distance = 0.14
        env_cfg.target_tolerance = 0.18
        env_cfg.apex_height_ref = 0.18
        env_cfg.flight_time_ref = 0.55
        env_cfg.min_target_hop_height = 0.10
        env_cfg.min_flight_time_for_hit = 0.20
        env_cfg.max_under_arc_error_for_hit = 0.08
        env_cfg.max_missed_landings_per_target = 4
        env_cfg.max_stance_time = 0.75
        env_cfg.max_successful_hops = 8
        env_cfg.episode_length_s = 12
    elif args_cli.task_preset == "track_precise":
        env_cfg.hop_distance = 0.14
        env_cfg.target_tolerance = 0.12
        env_cfg.apex_height_ref = 0.18
        env_cfg.flight_time_ref = 0.55
        env_cfg.min_target_hop_height = 0.10
        env_cfg.min_flight_time_for_hit = 0.20
        env_cfg.max_under_arc_error_for_hit = 0.06
        env_cfg.max_missed_landings_per_target = 4
        env_cfg.max_stance_time = 0.75
        env_cfg.max_successful_hops = 8
        env_cfg.episode_length_s = 12
    if args_cli.hop_distance is not None:
        env_cfg.hop_distance = args_cli.hop_distance
    if args_cli.target_tolerance is not None:
        env_cfg.target_tolerance = args_cli.target_tolerance
    if args_cli.apex_height_ref is not None:
        env_cfg.apex_height_ref = args_cli.apex_height_ref
    if args_cli.flight_time_ref is not None:
        env_cfg.flight_time_ref = args_cli.flight_time_ref
    if args_cli.planning_horizon_hops is not None:
        env_cfg.planning_horizon_hops = args_cli.planning_horizon_hops
        env_cfg.observation_space = 40 + 2 * env_cfg.planning_horizon_hops
    if args_cli.min_target_hop_height is not None:
        env_cfg.min_target_hop_height = args_cli.min_target_hop_height
    if args_cli.min_flight_time_for_hit is not None:
        env_cfg.min_flight_time_for_hit = args_cli.min_flight_time_for_hit
    if args_cli.max_under_arc_error_for_hit is not None:
        env_cfg.max_under_arc_error_for_hit = args_cli.max_under_arc_error_for_hit
    if args_cli.max_stance_time is not None:
        env_cfg.max_stance_time = args_cli.max_stance_time
    if args_cli.stance_root_height is not None:
        env_cfg.stance_root_height = args_cli.stance_root_height
    if args_cli.contact_vz_threshold is not None:
        env_cfg.contact_vz_threshold = args_cli.contact_vz_threshold
    if args_cli.liftoff_vz_threshold is not None:
        env_cfg.liftoff_vz_threshold = args_cli.liftoff_vz_threshold
    if args_cli.max_missed_landings_per_target is not None:
        env_cfg.max_missed_landings_per_target = args_cli.max_missed_landings_per_target
    if args_cli.max_successful_hops is not None:
        env_cfg.max_successful_hops = args_cli.max_successful_hops
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s

    print(
        "[INFO] env=circular-hopper "
        f"obs={env_cfg.observation_space} actions={env_cfg.action_space} "
        f"radius={env_cfg.circle_radius} hop_distance={env_cfg.hop_distance} "
        f"horizon={env_cfg.planning_horizon_hops} "
        f"flight_time_ref={env_cfg.flight_time_ref} "
        f"max_hops={env_cfg.max_successful_hops} episode_length_s={env_cfg.episode_length_s}"
    )

    env = gym.make("circular-hopper", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = CircularHopperPPORunnerCfg()
    if args_cli.max_iterations is not None:
        runner_cfg.max_iterations = args_cli.max_iterations
    print(f"[INFO] runner_cfg={runner_cfg.__class__.__name__} experiment={runner_cfg.experiment_name}")
    log_dir = os.path.join(
        os.path.dirname(__file__),
        "logs",
        "rsl_rl",
        runner_cfg.experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    checkpoint = args_cli.checkpoint
    if checkpoint == "latest":
        checkpoint = find_latest_checkpoint(runner_cfg.experiment_name)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found under logs/rsl_rl/{runner_cfg.experiment_name}")
    elif checkpoint == "migrated":
        checkpoint = find_migrated_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError("No migrated horizon checkpoint found")
    if checkpoint is not None:
        print(f"[INFO] loading checkpoint for continued training: {checkpoint}")
        runner.load(checkpoint, load_optimizer=("migrated_horizon" not in os.path.basename(checkpoint)))
        if "migrated_horizon" in os.path.basename(checkpoint):
            print("[INFO] migrated checkpoint detected; starting iteration counter from 0")
            runner.current_learning_iteration = 0
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
