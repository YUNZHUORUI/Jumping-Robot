"""导出 .pt checkpoint 为 policy.onnx (含 empirical_normalizer + LSTM 状态接口)。

用法:
    bash -c "ISAAC=/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64; \$ISAAC/python.sh export_onnx.py"
    # 默认导出最新 checkpoint；指定 --checkpoint logs/.../model_2997.pt 选其他

输出: 与 checkpoint 同目录的 policy.onnx
  Inputs : obs (1,37), h_in (1,1,256), c_in (1,1,256)   ← LSTM 1 层、hidden=256
  Outputs: actions (1,4), h_out, c_out
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model_*.pt; default = latest")
parser.add_argument("--filename", type=str, default="policy.onnx")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 导出不用渲染，强制 headless
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Quadhopper_Isaac  # noqa: F401
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.exporter import export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner

from Quadhopper_Isaac.quadhopper_env import QuadhopperEnvCfg
from Quadhopper_Isaac.rsl_rl_ppo_cfg import QuadhopperPPORunnerCfg


def find_latest_checkpoint():
    log_root = os.path.join(os.path.dirname(__file__), "logs", "rsl_rl", "myhopper")
    runs = sorted([d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))])
    if not runs:
        return None
    latest = os.path.join(log_root, runs[-1])
    pts = sorted(
        [f for f in os.listdir(latest) if f.startswith("model_") and f.endswith(".pt")],
        key=lambda x: int(x.replace("model_", "").replace(".pt", "")),
    )
    return os.path.join(latest, pts[-1]) if pts else None


def main():
    checkpoint = args.checkpoint or find_latest_checkpoint()
    if checkpoint is None or not os.path.exists(checkpoint):
        print(f"[ERROR] checkpoint not found: {checkpoint}")
        return
    print(f"[INFO] checkpoint: {checkpoint}")

    cfg = QuadhopperEnvCfg()
    cfg.scene.num_envs = 1
    env = gym.make("myhopper", cfg=cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = QuadhopperPPORunnerCfg()
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(checkpoint)

    out_dir = os.path.dirname(checkpoint)
    policy = runner.alg.policy
    # 这个 rsl_rl 版本：normalizer 内嵌在 policy.actor_obs_normalizer
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    print(f"[INFO] normalizer: {type(normalizer).__name__ if normalizer else 'None'}")
    print(f"[INFO] policy: {type(policy).__name__}, recurrent={policy.is_recurrent}")
    if policy.is_recurrent:
        rnn = policy.memory_a.rnn
        print(f"[INFO] RNN: type={type(rnn).__name__}, num_layers={rnn.num_layers}, "
              f"hidden_size={rnn.hidden_size}, input_size={rnn.input_size}")

    export_policy_as_onnx(
        policy, out_dir, normalizer=normalizer, filename=args.filename, verbose=False
    )
    print(f"[INFO] saved → {os.path.join(out_dir, args.filename)}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
