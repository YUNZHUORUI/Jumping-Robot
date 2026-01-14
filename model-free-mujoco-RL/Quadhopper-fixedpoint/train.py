import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sb3_contrib import RecurrentPPO  # <--- 核心：引入带记忆的 PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from drone_env import DroneHoverEnv


def plot_training_results(log_dir, save_name="training_result_lstm.png"):
    """绘图函数"""
    if not os.path.exists(os.path.join(log_dir, "monitor.csv")):
        print("无数据可画")
        return

    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if len(x) == 0: return

    window_size = 50
    y_smooth = pd.Series(y).rolling(window=window_size).mean() if len(y) >= window_size else y

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, alpha=0.3, color='gray', label='Episode Reward')
    plt.plot(x, y_smooth, color='red', linewidth=2, label=f'Moving Average (LSTM)')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('RecurrentPPO Training Curve')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(os.getcwd(), save_name)
    plt.savefig(save_path)
    print(f"图表已保存: {save_path}")


def train():
    models_dir = "models/RecurrentPPO"
    os.makedirs(models_dir, exist_ok=True)

    # 使用临时目录记录日志，跑完自动删
    with tempfile.TemporaryDirectory() as tmp_log_dir:
        print(f"正在初始化环境...")
        env = DroneHoverEnv(xml_file='quadhopper-model/scene.xml')
        env = Monitor(env, tmp_log_dir)

        # === 定义 LSTM 网络 ===
        # enable_critic_lstm: Critic 是否也使用 LSTM (建议 True)
        # lstm_hidden_size: 记忆单元的大小，256 足够记住复杂的物理惯性
        policy_kwargs = dict(
            net_arch=[],  # MLP层，LSTM前不需要太深
            enable_critic_lstm=True,
            lstm_hidden_size=256,
            n_lstm_layers=1
        )

        print("正在初始化 RecurrentPPO 模型...")
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,  # LSTM 训练时 batch_size 不宜过大
            gamma=0.99,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tmp_log_dir
        )

        print("开始训练 (LSTM模式)...")
        # 建议训练步数：LSTM 收敛比普通 MLP 稍慢，建议 50w - 100w
        total_timesteps = 1000000

        model.learn(total_timesteps=total_timesteps)

        # 保存
        model.save(f"{models_dir}/final_model")
        print("模型已保存。")

        try:
            plot_training_results(tmp_log_dir)
        except Exception as e:
            print(f"绘图出错: {e}")

    print("训练结束，临时数据已清理。")


if __name__ == "__main__":
    train()