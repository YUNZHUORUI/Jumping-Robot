"""Bisect #2: raw quadhopper env + frozen v36 teacher + state planner wrapper.

If this asserts while the raw env is healthy, the trigger is introduced by
the teacher inference or the wrapper's motor correction.
"""
import argparse
import os
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
import isaaclab as _il  # noqa: E402
import torch  # noqa: E402

_ISAACLAB_RL = os.path.join(os.path.dirname(_il.__file__), "source", "isaaclab_rl")
if _ISAACLAB_RL not in sys.path:
    sys.path.insert(0, _ISAACLAB_RL)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Quadhopper_Planner_Random  # noqa: E402,F401
from Quadhopper_Planner_Random.random_two_hop_env import PlannerRandomTwoHopEnvCfg  # noqa: E402
from Quadhopper_Planner_Random.two_hop_state_planner_wrapper import TeacherTwoHopStatePlannerVecEnv  # noqa: E402
from Quadhopper_Planner_Circular.rsl_rl_ppo_cfg import PlannerCircularPPORunnerCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
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

base = RslRlVecEnvWrapper(gym.make("Quadhopper-Planner-Random-Two-Hop-Direct-v0", cfg=cfg))
runner = OnPolicyRunner(base, PlannerCircularPPORunnerCfg().to_dict(), log_dir=None, device=args.device)
teacher_path = Path(
    "logs/rsl_rl/quadhopper_planner_random_two_hop_v36_full_relative_next_smooth_distance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_fixed_100/2026-08-23_12-24-40/model_90.pt"
).expanduser().resolve()
runner.load(str(teacher_path), load_optimizer=False)
teacher = runner.alg.policy
teacher.eval()
for parameter in teacher.parameters():
    parameter.requires_grad_(False)
env = TeacherTwoHopStatePlannerVecEnv(base, teacher, expert_anchor_scale=0.0)
print("[DIAG2] wrapper created", flush=True)
obs, _ = env.reset()
print("[DIAG2] reset OK", flush=True)
actions = torch.zeros(args.num_envs, 4, device=cfg.sim.device)
for i in range(args.steps):
    obs, rew, dones, extras = env.step(actions)
    if i % 50 == 0:
        core = base.unwrapped
        td = int(core._touchdown_count.sum().item())
        print(f"[DIAG2] step {i} OK touchdowns={td} rew={rew.mean().item():.4f}", flush=True)
print("[DIAG2] ALL OK - env+wrapper+teacher healthy", flush=True)
env.close()
simulation_app.close()
