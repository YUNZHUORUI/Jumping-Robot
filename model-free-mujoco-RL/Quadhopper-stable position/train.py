import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
import torch.nn as nn
from env import PogoDroneEnv
import os
import shutil

# 🔥 1. 强力清理：删掉所有以前的记忆
if os.path.exists("vec_normalize_hop.pkl"):
    os.remove("vec_normalize_hop.pkl")
if os.path.exists("logs_hop"):
    shutil.rmtree("logs_hop")
if os.path.exists("models_hop"):
    shutil.rmtree("models_hop")


def make_env(rank):
    def _init():
        return PogoDroneEnv()

    return _init


if __name__ == "__main__":
    num_cpu = 8
    print("🔥 启动训练：强制跳跃模式 (No Flying Allowed)...")

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # Clip Reward 设大一点，因为我们现在的惩罚很重
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=100., gamma=0.99)
    env = VecMonitor(env, "logs_hop")

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            activation_fn=nn.Tanh,
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1,
        learning_rate=3e-4,
        batch_size=2048,
        n_steps=2048,
    )

    # 至少训练 100万步
    try:
        model.learn(total_timesteps=3_000_000)
    except KeyboardInterrupt:
        print("⚠️ 训练中断，保存中...")

    model.save("pogo_hop_final")
    env.save("vec_normalize_hop.pkl")
    print("✅ 模型已保存：pogo_hop_final.zip")