from utils.Env.JumpingRobot_Env import JumpingRobot_Env
from utils.PPO.Actor_Critic_Discrete import Actor_Critic
from utils.Config.Config import *
import torch
import matplotlib.pyplot as plt
import numpy as np

# 确保参数正确
maximum_step = PPO_Config.PPOParam.maximum_step
episode = PPO_Config.PPOParam.episode
train = Env_Config.EnvParam.train

AC = Actor_Critic(PPO_Config, Env_Config)
if not train:
    try:
        AC.load_each_epi_model()
        print("Model loaded successfully.")
    except:
        print("No model found, running with random weights.")

env = JumpingRobot_Env(Env_Config, Robot_Config, PPO_Config)

if not train:
    plt.ion()
    plt.figure(figsize=(12, 6))

for epi in range(episode):
    print(f"=================== Episode: {epi} ===================")
    env.reset_all()  # 这里会应用新的初始化逻辑

    for step in range(maximum_step):
        state = env.get_current_observations()
        action_index, force = AC.sample_action(state)

        if not train:
            # 提取第一个机器人的状态
            x = state[0, 0].item()
            y = state[0, 1].item()
            tht = state[0, 2].item()
            target_x = state[0, 6].item()
            target_y = state[0, 7].item()

            # 机器人身体可视化
            body_len = 0.5
            x_head = x + body_len * np.sin(tht)
            y_head = y + body_len * np.cos(tht)
            x_tail = x - body_len * np.sin(tht)
            y_tail = y - body_len * np.cos(tht)

            # 腿部可视化 (示意)
            leg_len = 0.3
            # 假设腿是刚性的且向下延伸 (相对于身体)
            # 或者按照图示，腿在下方

            plt.clf()

            # 画目标
            plt.plot(target_x, target_y, 'ro', markersize=15, label='Target')
            plt.axvline(x=target_x, color='r', linestyle='--', alpha=0.3)

            # 画机器人
            plt.plot([x_tail, x_head], [y_tail, y_head], 'b-', linewidth=4, label='Body')
            plt.plot(x, y, 'bo', markersize=8)  # 质心

            # 画推力方向 (简单示意)
            # 如果有推力，画出箭头
            f_val = force[0].cpu().numpy()
            if f_val[0] > 0:
                plt.arrow(x_tail, y_tail, 0.2 * np.sin(tht), 0.2 * np.cos(tht), color='orange', width=0.05)
            if f_val[1] > 0:
                plt.arrow(x_head, y_head, 0.2 * np.sin(tht), 0.2 * np.cos(tht), color='orange', width=0.05)

            # 文本信息
            plt.text(x, y + 0.5, f"Tht: {np.degrees(tht):.1f}°\nTarget Tht: -20°", ha='center')

            plt.axhline(0, color='black', linewidth=2)
            plt.xlim(x - 2, target_x + 2)
            plt.ylim(-1, 6)
            plt.gca().set_aspect('equal')
            plt.legend()
            plt.title(f"Episode: {epi}, Step: {step}")

            plt.draw()
            plt.pause(0.01)

        env.step(force)
        next_state = env.get_next_observations()
        reward, over = env.compute_reward()

        if train:
            AC.store_experience(state, action_index, next_state, reward, over, step)

        # 重置那些已经结束的 agent
        done_indices = torch.nonzero(over.flatten()).flatten()
        if len(done_indices) > 0:
            env.reset(done_indices)

    if train:
        AC.update()
        env.print_reward_sum()

if not train:
    plt.ioff()
    plt.show()