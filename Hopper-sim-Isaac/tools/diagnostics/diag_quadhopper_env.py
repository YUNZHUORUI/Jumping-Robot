"""Minimal quadhopper-env diagnostic: no wrappers, no trainer, no teacher.

Creates the PlannerRandomTwoHopEnv like the semimdp trainer does, then steps
it with zero motor actions for 300 frames.  If this asserts, the problem is
the scene/physx/state layer; if it steps cleanly, the wrapper or trainer
layer introduces the trigger.
"""
import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=300)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Quadhopper_Planner_Random  # noqa: E402,F401
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg  # noqa: E402

cfg = PlannerRandomTwoHopEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.debug_vis = False
cfg.force_full_planner = True
cfg.randomize_dynamics = True
cfg.randomize_action_delay = True
cfg.target_height = 1.0
cfg.alternate_target_heights = False
cfg.fixed_height_curriculum = False
cfg.require_apex_tolerance_for_hit = True
cfg.relative_next_hop_observation = True
cfg.short_hop_radius_min, cfg.short_hop_radius_max = 0.50, 0.80
cfg.long_hop_radius_min, cfg.long_hop_radius_max = 0.80, 1.00
cfg.max_turn_angle_deg = 180.0
cfg.distance_curriculum_iterations = 0.0
cfg.target_tolerance = 0.10
cfg.curriculum_by_hops = True
cfg.power_model_path = str(ROOT / "Quadhopper_Stable/model/quadhopper_memory_power.pt")
cfg.csv_log_path = str(ROOT / "outputs/planner_random_two_hop_semimdp/training.csv")

env = gym.make("Quadhopper-Planner-Random-Two-Hop-Direct-v0", cfg=cfg)
print("[DIAG] env created", flush=True)
obs, _ = env.reset()
print("[DIAG] reset OK", flush=True)
actions = torch.zeros(args.num_envs, 4, device=cfg.sim.device)
for i in range(args.steps):
    obs, rew, term, trunc, extras = env.step(actions)
    if i % 50 == 0:
        core = env.unwrapped
        td = int(core._touchdown_count.sum().item())
        print(f"[DIAG] step {i} OK touchdowns={td} rew={rew.mean().item():.4f}", flush=True)
print("[DIAG] ALL OK - raw quadhopper env healthy", flush=True)
env.close()
simulation_app.close()
