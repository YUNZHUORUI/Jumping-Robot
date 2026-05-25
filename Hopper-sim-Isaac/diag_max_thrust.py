"""诊断：绕过 policy，强制最大油门 ~2s，记录 z_pos / joint_pos / motor_u。
如果 z_pos 显著上升 → 物理 OK，是奖励/探索问题。
如果 z_pos 不动 → 物理问题（spring/motor 模型、joint sign）。

运行: bash -c "ISAAC=/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64; \$ISAAC/python.sh diag_max_thrust.py --headless"
"""
import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--duration_s", type=float, default=2.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Quadhopper_Isaac  # noqa: F401  registers gym env
import gymnasium as gym
from Quadhopper_Isaac.quadhopper_env import QuadhopperEnvCfg

cfg = QuadhopperEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.episode_length_s = args.duration_s + 1.0

env = gym.make("myhopper", cfg=cfg)
unwrapped = env.unwrapped

print(f"[diag] sim dt={cfg.sim.dt}, decimation={cfg.decimation}, step_dt={unwrapped.step_dt}")
print(f"[diag] body_id={unwrapped._body_id}, spring_joint_id={unwrapped._spring_joint_id}")

# 检查 joint 属性
print(f"\n[diag] joint stiffness: {unwrapped._robot.data.joint_stiffness}")
print(f"[diag] joint damping:   {unwrapped._robot.data.joint_damping}")
print(f"[diag] joint pos limits: {unwrapped._robot.data.joint_pos_limits}")
print(f"[diag] num bodies: {unwrapped._robot.num_bodies}, body names: {unwrapped._robot.body_names}")
_leg_idx = unwrapped._robot.find_bodies("SpringLeg")[0][0]
print(f"[diag] leg body index: {_leg_idx}")

# 让弹簧静止稳定一会儿，看 rest joint_pos
print("\n[diag] === phase 1: zero action, 500 steps = 5s ===")
obs, _ = env.reset()
zero_action = torch.zeros(args.num_envs, 4, device=unwrapped.device)
for i in range(500):
    obs, rew, term, trunc, info = env.step(zero_action)
    if i % 50 == 0 or i >= 480:
        z = unwrapped._robot.data.root_pos_w[:, 2]
        jp = unwrapped._robot.data.joint_pos[:, unwrapped._spring_joint_id]
        is_c = (jp > 0.002)
        vz = unwrapped._robot.data.root_lin_vel_w[:, 2]
        print(f"  step {i:3d}: z=[{z.min():.3f},{z.max():.3f}] "
              f"jp=[{jp.min():+.4f},{jp.max():+.4f}] "
              f"vz=[{vz.min():+.2f},{vz.max():+.2f}] is_contact={is_c.sum().item()}/{is_c.numel()}")

# 现在切到最大油门
print("\n[diag] === phase 2: action=+1.0 (max thrust) ===")
max_action = torch.ones(args.num_envs, 4, device=unwrapped.device)
steps = int(args.duration_s / unwrapped.step_dt)
for i in range(steps):
    obs, rew, term, trunc, info = env.step(max_action)
    if i % 10 == 0 or i < 10:
        z = unwrapped._robot.data.root_pos_w[:, 2]
        jp = unwrapped._robot.data.joint_pos[:, unwrapped._spring_joint_id]
        u = unwrapped._motor_u
        thrust_each = unwrapped._thrust[:, 0, 2]
        print(f"  step {i:3d}: z=[{z.min():.3f}, {z.max():.3f}], "
              f"joint_pos=[{jp.min():.4f}, {jp.max():.4f}], "
              f"u_mean={u.mean():.3f}, F_total={thrust_each.mean():.3f}N")

z_final = unwrapped._robot.data.root_pos_w[:, 2]
print(f"\n[diag] === RESULT ===")
print(f"  final z: min={z_final.min():.3f}m  max={z_final.max():.3f}m  mean={z_final.mean():.3f}m")
if z_final.max() > 1.0:
    print("  ✓ PHYSICS OK: 最大油门能把 hopper 推到 >1m，问题在 policy/reward/exploration")
elif z_final.max() > 0.6:
    print("  ⚠ MARGINAL: 能稍微抬起来但不高，TWR 偏低")
else:
    print("  ✗ PHYSICS BROKEN: 最大油门也推不起来，需要查 spring/motor/asset")

env.close()
simulation_app.close()
