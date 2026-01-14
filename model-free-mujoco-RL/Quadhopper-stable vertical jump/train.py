import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn
from env import PogoDroneEnv
import os
import shutil

# 清理
if os.path.exists("vec_normalize_vertical.pkl"):
    os.remove("vec_normalize_vertical.pkl")
# 注意：如果你想接着之前的模型训练，不要删 models_vertical 文件夹
if os.path.exists("models_vertical"):
    shutil.rmtree("models_vertical")


def make_env(rank):
    def _init():
        return PogoDroneEnv()

    return _init


if __name__ == "__main__":
    num_cpu = 8  # M1 Pro 建议 8 核

    print(f"🔥 启动垂直跳跃特训 V2 (Strict Vertical Hopper)...")

    # 创建并行环境
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # 归一化是必须的
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=100., gamma=0.99)
    env = VecMonitor(env, "logs")

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            activation_fn=nn.Tanh,
            net_arch=dict(pi=[256, 256], vf=[256, 256])  # 网络结构
        ),
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        ent_coef=0.01,  # 增加探索，防止一开始就学会“趴着不动”
        device="auto"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=500000 // num_cpu,
        save_path='./models_vertical/',
        name_prefix='pogo_strict'
    )

    print("🚀 开始训练... 重点解决下落翻车和悬停作弊问题")

    try:
        # 建议训练 300万步，大概需要 15-20 分钟
        model.learn(total_timesteps=3_000_000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("🛑 中断保存...")

    model.save("pogo_vertical_final")
    env.save("vec_normalize_vertical.pkl")
    print("✅ 训练完成")