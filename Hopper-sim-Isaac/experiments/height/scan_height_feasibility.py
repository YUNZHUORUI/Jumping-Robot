"""Measure how collective action bias changes v10 apex and landing accuracy."""

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Scan Quadhopper collective-height authority")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--biases",
    type=float,
    nargs="+",
    default=[0.0, -0.05, -0.10, -0.15, -0.20, -0.30, -0.40],
)
parser.add_argument("--touchdowns", type=int, default=12)
parser.add_argument("--warmup_touchdowns", type=int, default=2)
parser.add_argument("--max_steps", type=int, default=30000)
parser.add_argument(
    "--output",
    type=str,
    default="outputs/height_feasibility/collective_bias_scan.csv",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.rendering_mode = "performance"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import statistics
import sys

import gymnasium as gym
import isaaclab as _il
import torch

_ISAACLAB_RL = os.path.join(os.path.dirname(_il.__file__), "source", "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import Quadhopper_Planner_Circular  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Planner_Circular.planner_circular_env import PlannerCircularEnvCfg
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg

def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def main():
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    biases = torch.tensor(args_cli.biases, device=args_cli.device)
    num_envs = len(args_cli.biases)
    env_cfg = PlannerCircularEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.debug_vis = False
    env_cfg.force_full_planner = True
    env_cfg.observation_noise_std = 0.0
    env_cfg.alternate_target_heights = False
    env_cfg.power_model_path = str(
        PROJECT_DIR / "Quadhopper_Stable/model/quadhopper_memory_power.pt"
    )
    env_cfg.csv_log_path = str(
        PROJECT_DIR / "outputs/height_feasibility/raw_environment.csv"
    )

    gym_env = gym.make("Quadhopper-Planner-Circular-Direct-v0", cfg=env_cfg)
    core = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env)
    runner = OnPolicyRunner(
        env,
        PlannerCircularPPORunnerCfg().to_dict(),
        log_dir=None,
        device=args_cli.device,
    )
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args_cli.device)

    apexes: list[list[float]] = [[] for _ in args_cli.biases]
    landing_errors: list[list[float]] = [[] for _ in args_cli.biases]
    hits: list[list[float]] = [[] for _ in args_cli.biases]
    clipping: list[list[float]] = [[] for _ in args_cli.biases]
    touchdown_seen = [0 for _ in args_cli.biases]

    obs, _ = env.reset()
    steps = 0
    while simulation_app.is_running() and steps < args_cli.max_steps:
        with torch.inference_mode():
            teacher_actions = policy(obs)
            biased_actions = teacher_actions + biases[:, None]
            clipping_now = ((biased_actions < -1.0) | (biased_actions > 1.0)).float().mean(dim=1)
            obs, _, _, _ = env.step(biased_actions)

        apex_ids = core._apex_event.nonzero(as_tuple=False).flatten().tolist()
        for env_id in apex_ids:
            if (
                touchdown_seen[env_id] >= args_cli.warmup_touchdowns
                and len(apexes[env_id]) < args_cli.touchdowns
            ):
                apexes[env_id].append(float(core._cycle_max_z[env_id].item()))

        touchdown_ids = core._touchdown_event.nonzero(as_tuple=False).flatten().tolist()
        for env_id in touchdown_ids:
            if (
                touchdown_seen[env_id] >= args_cli.warmup_touchdowns
                and len(landing_errors[env_id]) < args_cli.touchdowns
            ):
                landing_errors[env_id].append(float(core._landing_error[env_id].item()))
                hits[env_id].append(float(core._target_hit_event[env_id].item()))
                clipping[env_id].append(float(clipping_now[env_id].item()))
            touchdown_seen[env_id] += 1

        steps += 1
        if min(len(values) for values in landing_errors) >= args_cli.touchdowns:
            break

    rows = []
    for env_id, bias in enumerate(args_cli.biases):
        row = {
            "collective_action_bias": bias,
            "equivalent_motor_u_bias": 0.5 * bias,
            "apex_samples": len(apexes[env_id]),
            "touchdown_samples": len(landing_errors[env_id]),
            "mean_apex_m": _mean(apexes[env_id]),
            "min_apex_m": min(apexes[env_id], default=float("nan")),
            "max_apex_m": max(apexes[env_id], default=float("nan")),
            "mean_landing_error_m": _mean(landing_errors[env_id]),
            "hit_rate": _mean(hits[env_id]),
            "action_clip_fraction": _mean(clipping[env_id]),
        }
        rows.append(row)

    output = (PROJECT_DIR / args_cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[HEIGHT SCAN] steps={steps} output={output}")
    print(
        "bias     apex[m]  landing_err[m]  hit_rate  clips  samples\n"
        "----------------------------------------------------------"
    )
    for row in rows:
        print(
            f"{row['collective_action_bias']:>+6.2f}  "
            f"{row['mean_apex_m']:>7.3f}  "
            f"{row['mean_landing_error_m']:>14.3f}  "
            f"{row['hit_rate']:>8.3f}  "
            f"{row['action_clip_fraction']:>5.3f}  "
            f"{row['touchdown_samples']:>7d}"
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
