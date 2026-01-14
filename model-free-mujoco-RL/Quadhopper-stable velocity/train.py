import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn
from env import PogoDroneEnv
import os
import shutil

# 🔥 必须清理
if os.path.exists("vec_normalize_natural.pkl"):
    os.remove("vec_normalize_natural.pkl")
if os.path.exists("models_natural"):
    shutil.rmtree("models_natural")

def make_env(rank):
    def _init():
        return PogoDroneEnv()
    return _init

if __name__ == "__main__":
    num_cpu = 8
    print(f"🔥 启动 V9 训练 (Natural SLIP / No Air-Thrust)...")
    print("目标：彻底消除空中加速，实现自然弹跳")

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=100., gamma=0.99)
    env = VecMonitor(env, "logs")

    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=dict(activation_fn=nn.Tanh, net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=512, ent_coef=0.01, device="auto"
    )

    checkpoint_callback = CheckpointCallback(save_freq=200000 // num_cpu, save_path='./models_natural/', name_prefix='pogo_v9')

    try:
        model.learn(total_timesteps=3_500_000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("停止训练...")

    model.save("pogo_natural_final")
    env.save("vec_normalize_natural.pkl")
    print("✅ 训练完成")