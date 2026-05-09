# 推理脚本：加载训练好的策略，可视化无人机飞行
# 运行: python.bat play.py
# 指定模型: python.bat play.py --checkpoint logs/rsl_rl/myquadcopter/2026-05-09_20-51-47/model_999.pt

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play MyQuadcopter policy")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to visualize")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab as _il
_ISAACLAB_SOURCE = os.path.join(os.path.dirname(_il.__file__), "source")
_p = os.path.join(_ISAACLAB_SOURCE, "isaaclab_rl")
if _p not in sys.path:
    sys.path.insert(0, _p)

import Ruigang_smi
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from Ruigang_smi.my_quadcopter_env import QuadcopterEnvCfg
from Ruigang_smi.rsl_rl_ppo_cfg import QuadcopterPPORunnerCfg


def find_latest_checkpoint():
    log_root = os.path.join(os.path.dirname(__file__), "logs", "rsl_rl", "myquadcopter")
    if not os.path.exists(log_root):
        return None
    runs = sorted(os.listdir(log_root))
    if not runs:
        return None
    latest = os.path.join(log_root, runs[-1])
    pts = sorted([f for f in os.listdir(latest) if f.endswith(".pt")],
                 key=lambda x: int(x.replace("model_", "").replace(".pt", "")))
    return os.path.join(latest, pts[-1]) if pts else None


def main():
    checkpoint = args_cli.checkpoint or find_latest_checkpoint()
    if checkpoint is None:
        print("[ERROR] No checkpoint found. Please specify --checkpoint path/to/model.pt")
        return
    print(f"[INFO] Loading checkpoint: {checkpoint}")

    env_cfg = QuadcopterEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 30.0

    env = gym.make("myquadcopter", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = QuadcopterPPORunnerCfg()
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(checkpoint)

    policy = runner.get_inference_policy(device="cuda:0")

    obs = env.get_observations()
    print("[INFO] Running policy. Press Ctrl+C to stop.")
    try:
        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    except KeyboardInterrupt:
        pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
