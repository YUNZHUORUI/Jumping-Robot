from Config import Config
from Env import JumpingRobot_Env
from ppo_agent import Actor_Critic
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import torch


def train():
    cfg = Config()
    env = JumpingRobot_Env(cfg, cfg.RobotParam, cfg.PPOParam)
    agent = Actor_Critic(cfg)

    # 增加训练轮数
    episodes = 3000
    max_step = cfg.PPOParam.maximum_step

    # 用于计算滑动平均奖励
    reward_history = []

    print(f"Start Training for {episodes} episodes...")

    for epi in range(episodes):
        env.reset_all()
        episode_reward = 0

        # --- Episode 循环 ---
        for step in range(max_step):
            state = env.get_current_observations()
            action_idx, action_val = agent.sample_action(state)

            env.step(action_val)

            next_state = env.get_next_observations()
            reward, over = env.compute_reward()

            # 存储经验
            agent.store_experience(state, action_idx, next_state, reward, over, step)

            episode_reward += reward.mean().item()

            if over.all(): break
        # -------------------

        # 记录奖励
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-50:]) if len(reward_history) > 0 else episode_reward

        # 更新策略 (传入当前的 epi 索引)
        agent.update(epi, episodes, avg_reward)

        # 每 10 轮打印一次
        if epi % 10 == 0:
            print(f"Episode: {epi}, Reward: {episode_reward:.2f}, AvgReward: {avg_reward:.2f}")

        # 每 500 轮保存一次模型
        if epi % 500 == 0 and epi > 0:
            torch.save(agent.actor.state_dict(), f'model/actor_{epi}.pth')

    return agent, cfg


def run_animation(agent, cfg):
    print("Generating Animation...")
    # 动画展示时不训练
    cfg.EnvParam.train = False

    env = JumpingRobot_Env(cfg, cfg.RobotParam, cfg.PPOParam)
    env.reset_all()

    # 尝试加载最佳模型
    try:
        agent.actor.load_state_dict(torch.load('model/actor_best.pth'))
        print("Loaded Best Model for Animation.")
    except:
        print("Warning: Could not load best model, using current model.")

    # 数据记录列表
    traj_x, traj_y = [], []
    body_x, body_y, body_theta, body_l = [], [], [], []
    phases = []

    max_frames = 400
    for _ in range(max_frames):
        state = env.get_current_observations()

        # 测试时取样
        _, action_val = agent.sample_action(state)
        env.step(action_val)

        # 记录用于画图的数据 (CPU numpy)
        # q: [x, y, theta, l]
        bx = env.q[0, 0]
        by = env.q[0, 1]
        bth = env.q[0, 2]
        bl = env.q[0, 3]  # 获取实时腿长

        traj_x.append(bx)
        traj_y.append(by)
        body_x.append(bx)
        body_y.append(by)
        body_theta.append(bth)
        body_l.append(bl)
        phases.append(env.phase[0])

        if env.over[0]: break

    # --- 绘图部分 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, 8)
    ax.set_ylim(-0.5, 4.5)  # 稍微调整视野
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title("Hopper Simulation: Body Beam & Leg Dynamics")

    # 初始化绘图对象
    # 1. 地面和目标
    ax.axhline(0, color='black', linewidth=2)
    ax.plot([2.0, 4.0], [0.0, 0.0], 'gX', markersize=10, label='Targets')

    # 2. 轨迹线
    line, = ax.plot([], [], 'b--', alpha=0.3, label='Trajectory')

    # 3. 机器人部件
    # Beam (横梁/身体)
    beam_line, = ax.plot([], [], lw=4, color='black', label='Body Beam')
    left_dot, = ax.plot([], [], 'bo', markersize=6)  # Beam 左端点
    right_dot, = ax.plot([], [], 'bo', markersize=6)  # Beam 右端点

    # Leg (腿)
    leg_line, = ax.plot([], [], 'k-', linewidth=2, label='Leg')
    body_dot, = ax.plot([], [], 'ro', markersize=8, label='COM')  # 质心

    # 4. 文字信息 (修复了之前的 bbox 语法错误)
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='gray'))

    def init():
        beam_line.set_data([], [])
        left_dot.set_data([], [])
        right_dot.set_data([], [])
        leg_line.set_data([], [])
        body_dot.set_data([], [])
        line.set_data([], [])
        info_text.set_text("")
        return beam_line, left_dot, right_dot, leg_line, body_dot, line, info_text

    def update(frame):
        # 更新轨迹
        line.set_data(traj_x[:frame], traj_y[:frame])

        # 获取当前帧数据
        bx, by = body_x[frame], body_y[frame]
        theta = body_theta[frame]
        l = body_l[frame]  # 动态腿长
        phase = phases[frame]

        # 1. 身体 Beam 的几何计算 (可视化为与腿垂直的横梁)
        # lc 是半长
        lc = cfg.RobotParam.lc

        # 假设 Beam 垂直于腿 (T字型或者十字型结构)
        # 腿的角度是 theta (0是垂直向上, 顺时针为正)
        # Beam 的角度就是 theta + 90度
        beam_angle = theta + np.pi / 2

        dx_beam = lc * np.sin(beam_angle)
        dy_beam = lc * np.cos(beam_angle)  # 注意坐标系: y轴向上为正

        # 计算 Beam 左右端点 (因为 theta=0 是 y轴，sin/cos 需要对应调整)
        # theta=0 -> beam_angle=90 -> sin=1, cos=0 -> dx=lc, dy=0 (水平) -> 正确

        # 左端点 (减去向量)
        bx_left = bx - dx_beam
        by_left = by + dy_beam  # 这里符号取决于你的视觉偏好，通常是对称的

        # 右端点 (加上向量)
        bx_right = bx + dx_beam
        by_right = by - dy_beam

        beam_line.set_data([bx_left, bx_right], [by_left, by_right])
        left_dot.set_data([bx_left], [by_left])
        right_dot.set_data([bx_right], [by_right])

        # 2. 腿部 Geometry 计算
        # theta = 0 时，腿垂直向下?
        # 根据之前的 Env 定义: theta=0 是垂直向上(Y正)。
        # 所以腿指向地面应该是反方向，或者 Env 里定义 theta 是偏离 y 轴的角度
        # 你的 Env 代码中：x + l*sin(theta), y - l*cos(theta)。
        # 这意味着 theta=0 时，sin=0, cos=1 -> (x, y-l)。即垂直向下。符合直觉。

        lx_tip = bx + l * np.sin(theta)
        ly_tip = by - l * np.cos(theta)

        leg_line.set_data([bx, lx_tip], [by, ly_tip])
        body_dot.set_data([bx], [by])

        # 3. 更新文字
        info_str = (f'Frame: {frame}\n'
                    f'Phase: {phase}\n'
                    f'Pos: ({bx:.2f}, {by:.2f})\n'
                    f'Angle: {np.degrees(theta):.1f}°\n'
                    f'Leg Len: {l:.3f}m')
        info_text.set_text(info_str)

        return beam_line, left_dot, right_dot, leg_line, body_dot, line, info_text

    # 生成动画
    if len(traj_x) > 0:
        ani = FuncAnimation(fig, update, frames=len(traj_x), init_func=init, blit=True, interval=30)
        try:
            ani.save('hopper_sim.gif', writer=PillowWriter(fps=30))
            print("Animation saved as 'hopper_sim.gif'")
        except Exception as e:
            print(f"Animation save failed: {e}")
            plt.show()
    else:
        print("Simulation failed immediately, no animation generated.")


if __name__ == "__main__":
    trained_agent, config = train()
    run_animation(trained_agent, config)