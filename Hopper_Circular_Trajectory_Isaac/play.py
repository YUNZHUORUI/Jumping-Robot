import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play circular waypoint hopping policy")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to visualize")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
parser.add_argument("--max_successful_hops", type=int, default=999, help="Do not auto-end play after only 4 target hits")
parser.add_argument("--print_dones", action="store_true", help="Print reset reasons while playing")
parser.add_argument("--hop_distance", type=float, default=None, help="Override circular waypoint chord distance")
parser.add_argument("--target_tolerance", type=float, default=None, help="Override landing target tolerance")
parser.add_argument("--apex_height_ref", type=float, default=None, help="Override desired per-hop apex height above takeoff")
parser.add_argument("--flight_time_ref", type=float, default=None, help="Override desired per-hop flight time")
parser.add_argument("--planning_horizon_hops", type=int, default=None, help="Override waypoint lookahead count in observations")
parser.add_argument("--print_actions", action="store_true", help="Print policy action statistics every 100 steps")
parser.add_argument("--print_state", action="store_true", help="Print root/contact/target state every 100 steps")
parser.add_argument("--done_print_limit", type=int, default=12, help="Maximum number of done lines to print")
parser.add_argument(
    "--task_preset",
    choices=["default", "strict_short", "track_feasible", "track_precise"],
    default="default",
    help="Apply a known task setting before individual CLI overrides",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab as _il
import gymnasium as gym
import torch

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


def find_latest_checkpoint():
    log_root = os.path.join(os.path.dirname(__file__), "logs", "rsl_rl", "circular_hopper_horizon")
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


def find_migrated_checkpoint():
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
    env = None
    try:
        if args_cli.checkpoint in (None, "latest"):
            checkpoint = find_latest_checkpoint()
        elif args_cli.checkpoint == "migrated":
            checkpoint = find_migrated_checkpoint()
        else:
            checkpoint = args_cli.checkpoint
        if checkpoint is None:
            print("[ERROR] No checkpoint found. Pass --checkpoint path/to/model.pt")
            return
        print(f"[INFO] Loading checkpoint: {checkpoint}")

        env_cfg = CircularHopperEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.episode_length_s = 30.0
        env_cfg.debug_vis = True
        env_cfg.max_successful_hops = args_cli.max_successful_hops
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
        print(
            "[INFO] play cfg "
            f"hop_distance={env_cfg.hop_distance} "
            f"target_tolerance={env_cfg.target_tolerance} "
            f"apex_height_ref={env_cfg.apex_height_ref} "
            f"flight_time_ref={env_cfg.flight_time_ref} "
            f"planning_horizon_hops={env_cfg.planning_horizon_hops} "
            f"max_hops={env_cfg.max_successful_hops}"
        )

        raw_env = gym.make("circular-hopper", cfg=env_cfg)
        env = RslRlVecEnvWrapper(raw_env)

        runner_cfg = CircularHopperPPORunnerCfg()
        runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device="cuda:0")

        obs = env.get_observations()
        step = 0
        done_prints = 0
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
            if args_cli.print_actions and step % 100 == 0:
                print(
                    "[ACTION] "
                    f"mean={actions.mean().item():.3f} "
                    f"abs_mean={actions.abs().mean().item():.3f} "
                    f"min={actions.min().item():.3f} "
                    f"max={actions.max().item():.3f}"
                )
            if args_cli.print_state and step % 100 == 0:
                base_env = raw_env.unwrapped
                root_pos = base_env._robot.data.root_pos_w[0]
                vz = base_env._robot.data.root_lin_vel_w[0, 2]
                joint_pos = base_env._robot.data.joint_pos[0, base_env._spring_joint_id]
                target_dist = torch.linalg.norm(base_env.commands.target_pos_w[0] - root_pos[:2])
                phase = torch.clamp(base_env._time_since_liftoff[0] / base_env.planner.flight_time_ref[0], 0.0, 1.0)
                planned_z = (
                    base_env.planner.takeoff_pos_w[0, 2]
                    + 4.0 * phase * (1.0 - phase) * base_env.cfg.apex_height_ref
                )
                print(
                    "[STATE] "
                    f"z={root_pos[2].item():.3f} "
                    f"z_ref={planned_z.item():.3f} "
                    f"vz={vz.item():.3f} "
                    f"joint={joint_pos.item():.4f} "
                    f"touching={bool(base_env._touching[0])} "
                    f"phase={int(base_env._phase[0].item())} "
                    f"target_dist={target_dist.item():.3f} "
                    f"under_arc={base_env._hop_max_under_arc_error[0].item():.3f} "
                    f"hop_height={base_env._hop_max_height[0].item():.3f} "
                    f"hops={int(base_env.commands.successful_hops[0].item())}"
                )
            obs, _, dones, _ = env.step(actions)
            step += 1
            if args_cli.print_dones and torch.any(dones) and done_prints < args_cli.done_print_limit:
                base_env = raw_env.unwrapped
                done_ids = torch.nonzero(dones, as_tuple=False).flatten()
                for env_id in done_ids[:8].tolist():
                    reasons = []
                    if bool(base_env._debug_done_completed[env_id]):
                        reasons.append("completed")
                    if bool(base_env._debug_done_tilt[env_id]):
                        reasons.append("tilt")
                    if bool(base_env._debug_done_height[env_id]):
                        reasons.append("height")
                    if bool(base_env._debug_done_workspace[env_id]):
                        reasons.append("workspace")
                    if bool(base_env._debug_done_stance_stall[env_id]):
                        reasons.append("stance_stall")
                    if bool(base_env._debug_done_repeated_miss[env_id]):
                        reasons.append("repeated_miss")
                    if not reasons:
                        reasons.append("timeout")
                    hops = int(base_env.commands.successful_hops[env_id].item())
                    err = float(base_env._last_landing_error[env_id].item())
                    print(f"[DONE] env={env_id} reason={'+'.join(reasons)} hops={hops} last_landing_error={err:.3f}")
                    done_prints += 1
                    if done_prints >= args_cli.done_print_limit:
                        break
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
