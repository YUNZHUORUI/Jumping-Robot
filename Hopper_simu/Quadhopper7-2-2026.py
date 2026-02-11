import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import freeze_support
import torch as th
import os
import matplotlib
matplotlib.use('Agg')  # 关键！解决 Retina 屏幕尺寸爆炸问题
import matplotlib.pyplot as plt
import imageio

'''now we first use $y = x \tan(\alpha) - \frac{gx^2}{2v_0^2 \cos^2(\alpha)}$ to solve the
solve parabolas. we select the parabolas with departure angle between 20 deg to 40 deg. Then
let PPO to follow the trajectory.
'''


class QuadhopperTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # ==================== 1. 物理参数 ====================
        self.dt = 0.002
        self.m = 1.21
        self.l0 = 1.0  # 腿长
        self.lc = 0.177  # 重心到电机距离
        self.J = 0.15  # 转动惯量
        self.g = 9.81
        self.max_thrust = 30.0
        self.ground_y = 0.0
        self.k_spring = 150.0
        self.c_damping = 1.0

        # ==================== 2. 目标与空间 ====================
        # 设置一系列目标点
        self.targets = np.array([3.0, 7.0, 12.0, 18.0])
        self.target_tolerance = 0.3  # 目标半径

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: [x, y, theta, l, dx, dy, dtheta, dl, dist_x, target_idx, contact, y_err, dy_err]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.current_target_idx = 0
        self.steps = 0
        self.max_episode_steps = 3000

        # 轨迹参数
        self.traj_a = 0.0
        self.traj_b_local = 0.0
        self.traj_x0 = 0.0
        self.traj_y0 = 0.0
        self.traj_valid = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # 初始随机状态
        init_theta = np.random.uniform(math.radians(-35), math.radians(-25))
        # 确保不会一出生就埋在地里
        leg_proj = self.l0 * math.cos(init_theta)
        init_y = max(1.4, leg_proj + 0.2)

        self.q = np.array([0.0, init_y, init_theta, self.l0], dtype=np.float64)

        # 初始给予一定的水平速度，模拟连续跳跃中的状态
        self.dq = np.array([
            np.random.uniform(2.5, 4.5),  # dx
            np.random.uniform(-0.5, 1.0),  # dy
            0.0, 0.0
        ], dtype=np.float64)

        self.current_target_idx = 0
        self.steps = 0

        # 立即规划第一跳
        self._plan_trajectory()

        return self._get_obs(), {}

    def get_foot_pos(self):
        """计算脚尖坐标"""
        x, y, theta = self.q[0], self.q[1], self.q[2]
        x_f = x + self.l0 * math.sin(theta)
        y_f = y - self.l0 * math.cos(theta)
        return np.array([x_f, y_f])

    def _plan_trajectory(self):
        """
        核心算法：根据目标距离，在 20~40 度倾角范围内规划抛物线
        y = x * tan(alpha) - g*x^2 / (2 * v0^2 * cos^2(alpha))
        """
        if self.current_target_idx >= len(self.targets):
            self.traj_valid = False
            return

        x_curr, y_curr = self.q[0], self.q[1]
        x_target_foot = self.targets[self.current_target_idx]

        # 我们控制的是质心(COM)，但目标是脚(Foot)。
        # 假设落地时机器人稍微前倾 theta approx -15度
        estimated_land_theta = math.radians(-15)
        offset_x = self.l0 * math.sin(estimated_land_theta)

        # 质心的目标位置 = 脚的目标位置 - 偏移量
        x_target_com = x_target_foot - offset_x

        # 计算相对距离
        dx = x_target_com - x_curr
        dy = 0.8 - y_curr  # 假设目标高度是离地 0.8m (压缩状态)

        if dx <= 0.2:
            self.traj_valid = False
            return  # 距离太近，不用规划，直接落下

        # ================= 角度选择策略 =================
        # 题目要求：departure angle between 20 deg to 40 deg (相对垂直方向)
        # 即 launch_alpha (相对水平) = 90 - (20~40) = 50~70 度
        # 我们随机选择一个角度，或者根据距离选择最优角
        tilt_deg = np.random.uniform(20, 40)
        launch_alpha = math.radians(90 - tilt_deg)

        tan_a = math.tan(launch_alpha)
        cos_a = math.cos(launch_alpha)

        # 检查判别式 (确保能到达)
        # 抛物线公式变换求 v0:
        # dy = dx * tan_a - (g * dx^2) / (2 * v0^2 * cos_a^2)
        # => (g * dx^2) / (2 * v0^2 * cos_a^2) = dx * tan_a - dy
        # => v0^2 = (g * dx^2) / (2 * cos_a^2 * (dx * tan_a - dy))

        denominator = dx * tan_a - dy
        if denominator <= 0.01:
            # 如果目标太高或角度太低，强制提高角度
            launch_alpha = math.radians(75)
            tan_a = math.tan(launch_alpha)
            cos_a = math.cos(launch_alpha)
            denominator = dx * tan_a - dy

        v0_sq = (self.g * dx ** 2) / (2 * (cos_a ** 2) * denominator)

        # 存储局部坐标系下的参数
        # y_local = a * x_local^2 + b * x_local
        self.traj_a = -self.g / (2 * v0_sq * cos_a ** 2)
        self.traj_b_local = tan_a
        self.traj_x0 = x_curr
        self.traj_y0 = y_curr
        self.traj_valid = True

        # print(f"Plan: Dist={dx:.2f}, Angle={math.degrees(launch_alpha):.1f}, V0={math.sqrt(v0_sq):.2f}")

    def get_trajectory_state(self, x_current):
        """
        计算当前的理想高度(y)和理想垂直速度(dy)
        """
        if not self.traj_valid:
            # 如果没有轨迹（比如落地后），目标维持现状或准备姿态
            return 1.0, 0.0

        dx = x_current - self.traj_x0

        # 如果已经超过目标点，不再强行跟随抛物线（避免插入地下）
        if dx < 0: dx = 0

        # 1. 理想高度
        y_ideal = self.traj_a * (dx ** 2) + self.traj_b_local * dx + self.traj_y0

        # 2. 理想斜率 (dy/dx)
        slope_ideal = 2 * self.traj_a * dx + self.traj_b_local

        # 3. 理想垂直速度 (dy/dt) = (dy/dx) * (dx/dt)
        # 【关键优化】：使用当前的水平速度 self.dq[0] 来计算目标的垂直速度
        # 这样即使水平速度慢了，机器人也会相应地减缓垂直升降，而不是盲目跟随
        current_vx = max(0.1, self.dq[0])  # 防止除零或负数
        dy_ideal = slope_ideal * current_vx

        return y_ideal, dy_ideal

    def _get_obs(self):
        foot_pos = self.get_foot_pos()
        target_x = self.targets[self.current_target_idx] if self.current_target_idx < len(self.targets) else foot_pos[0]
        dist_x = target_x - foot_pos[0]

        is_touching = 1.0 if foot_pos[1] <= self.ground_y else 0.0

        # 轨迹误差计算
        y_ideal, dy_ideal = self.get_trajectory_state(self.q[0])

        y_error = self.q[1] - y_ideal
        dy_error = self.dq[1] - dy_ideal  # 垂直速度误差

        # 归一化/Clip防止数值爆炸
        obs = np.concatenate([
            self.q,  # [x, y, theta, l]
            self.dq,  # [dx, dy, dtheta, dl]
            [dist_x, float(self.current_target_idx), is_touching, y_error, dy_error]
        ])
        return np.clip(obs, -20.0, 20.0).astype(np.float32)

    def step(self, action):
        # Action: [Left Thrust, Right Thrust] (0~1)
        u1, u2 = action[0], action[1]

        # 物理计算
        F1 = u1 * self.max_thrust
        F2 = u2 * self.max_thrust
        F_total = F1 + F2
        tau = (F2 - F1) * self.lc

        x, y, theta = self.q[0], self.q[1], self.q[2]
        dx, dy, dtheta = self.dq[0], self.dq[1], self.dq[2]
        s, c = math.sin(theta), math.cos(theta)

        # 弹簧与地面接触力
        foot_x_virt = x + self.l0 * s
        foot_y_virt = y - self.l0 * c
        F_spring_x, F_spring_y = 0.0, 0.0
        touching = False

        if foot_y_virt < self.ground_y:
            touching = True
            l_curr = y / max(c, 0.01)
            compression = self.l0 - l_curr
            if compression > 0:
                comp_rate = -(dx * s - dy * c)
                F_mag = self.k_spring * compression + self.c_damping * comp_rate
                F_mag = max(0.0, F_mag)
                F_spring_x = -F_mag * s
                F_spring_y = F_mag * c

        # 动力学积分 (Symplectic Euler approx)
        ddx = (-F_total * s + F_spring_x) / self.m
        ddy = (F_total * c + F_spring_y) / self.m - self.g
        ddtheta = tau / self.J

        self.dq[0] += ddx * self.dt
        self.dq[1] += ddy * self.dt
        self.dq[2] += ddtheta * self.dt
        self.q[0] += self.dq[0] * self.dt
        self.q[1] += self.dq[1] * self.dt
        self.q[2] += self.dq[2] * self.dt

        # ==================== 奖励函数优化 ====================
        reward = 0.0
        terminated = False
        truncated = False

        foot_pos = self.get_foot_pos()
        dist_to_target = abs(foot_pos[0] - self.targets[self.current_target_idx])

        # 1. 轨迹跟踪奖励 (空中时生效)
        if not touching and self.traj_valid:
            y_ideal, dy_ideal = self.get_trajectory_state(self.q[0])

            # 高度误差惩罚
            error_y = abs(self.q[1] - y_ideal)
            reward += 1.0 * np.exp(-10.0 * error_y ** 2)

            # 速度误差惩罚 (如果应该下落却在上升，重罚)
            error_dy = abs(self.dq[1] - dy_ideal)
            reward -= 0.1 * error_dy

        # 2. 节能/推力惩罚
        # 鼓励在抛物线顶端关机滑行
        thrust_cost = 0.05 * (u1 ** 2 + u2 ** 2)
        reward -= thrust_cost

        # 3. 姿态稳定奖励
        reward -= 0.02 * abs(self.q[2])  # 保持直立
        reward -= 0.01 * abs(self.dq[2])  # 减少旋转

        # 4. 落地与目标判定
        if touching:
            # 只有当垂直速度较小(真的踩稳了)时才结算
            if self.dq[1] > -1.0:
                if dist_to_target < self.target_tolerance:
                    reward += 100.0  # 命中目标大奖
                    print(f"🎯 Hit Target {self.current_target_idx} (Dist: {dist_to_target:.2f})")

                    self.current_target_idx += 1
                    if self.current_target_idx >= len(self.targets):
                        terminated = True
                        reward += 200.0  # 通关
                    else:
                        self._plan_trajectory()  # 规划下一跳
                elif dist_to_target < 1.0:
                    # 没踩中红心，但在附近，给一点小奖励
                    reward += 10.0 * (1.0 - dist_to_target)

                # 落地后如果不重新规划(比如没得跳了)，就在这里结束
                if self.dq[0] < 0.1 and not self.traj_valid:
                    pass

        # 5. 失败判定
        # 倒地 或 飞太高 或 往回飞
        if abs(self.q[2]) > 1.4 or self.q[1] > 6.0 or self.q[0] < -1.0:
            terminated = True
            reward -= 50.0

        # 飞过头判定
        if foot_pos[0] > self.targets[self.current_target_idx] + 2.0:
            terminated = True
            reward -= 20.0

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    freeze_support()

    MODEL_PATH = "ppo_quadhopper_v2"
    MODE = "train"  # 选项: "train", "test"

    if MODE == "train":
        print("🚀 开始训练...")
        # 并行环境加速训练
        env = make_vec_env(QuadhopperTargetEnv, n_envs=8, vec_env_cls=SubprocVecEnv)

        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            ent_coef=0.01,  # 稍微增加探索
            gamma=0.99,
            device="auto"
        )

        model.learn(total_timesteps=2_000_000)
        model.save(MODEL_PATH)
        print("✅ 模型已保存")

    elif MODE == "test":
        print(f"🎬 加载模型并测试: {MODEL_PATH}")

        # 如果模型不存在，为了演示代码逻辑，创建一个临时模型
        if os.path.exists(MODEL_PATH + ".zip"):
            model = PPO.load(MODEL_PATH)
        else:
            print("⚠️ 未找到模型文件，使用未训练模型演示...")
            model = PPO("MlpPolicy", QuadhopperTargetEnv())

        env = QuadhopperTargetEnv()
        obs, _ = env.reset()

        # 数据记录
        history = {
            'x': [], 'y': [], 'target_y': [],
            'thrust_l': [], 'thrust_r': [],
            'step': []
        }

        frames = []

        for i in range(600):  # 测试 600 步
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # 记录数据
            history['step'].append(i)
            history['x'].append(obs[0])
            history['y'].append(obs[1])
            history['thrust_l'].append(action[0])
            history['thrust_r'].append(action[1])

            # 获取理想轨迹用于对比
            traj_y, _ = env.get_trajectory_state(obs[0])
            history['target_y'].append(traj_y)

            # --- 绘图生成 GIF 帧 ---
            if i % 2 == 0:  # 每2步存一帧
                fig = plt.figure(figsize=(10, 5), dpi=80)
                ax = fig.add_subplot(111)

                # 视角跟随
                cx = obs[0]
                ax.set_xlim(cx - 3, cx + 7)
                ax.set_ylim(-1, 5)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

                # 画地面和目标
                ax.axhline(0, color='k', lw=2)
                for tid, tx in enumerate(env.targets):
                    color = 'g' if tid == env.current_target_idx else 'gray'
                    ax.plot(tx, 0, 'x', color=color, markersize=10, markeredgewidth=3)

                # 画理想轨迹
                if env.traj_valid:
                    tx = np.linspace(env.traj_x0, env.targets[env.current_target_idx], 30)
                    dx_arr = tx - env.traj_x0
                    ty = env.traj_a * dx_arr ** 2 + env.traj_b_local * dx_arr + env.traj_y0
                    valid_mask = ty > -0.5
                    ax.plot(tx[valid_mask], ty[valid_mask], 'r--', alpha=0.5, label='Trajectory')

                # 画机器人
                x, y, theta = obs[0], obs[1], obs[2]
                foot_pos = env.get_foot_pos()

                # 机身
                body_x = [x - 0.2 * math.cos(theta), x + 0.2 * math.cos(theta)]
                body_y = [y - 0.2 * math.sin(theta), y + 0.2 * math.sin(theta)]
                ax.plot(body_x, body_y, 'k-', lw=6)
                # 腿
                ax.plot([x, foot_pos[0]], [y, foot_pos[1]], 'b-', lw=3)
                # 脚
                ax.plot(foot_pos[0], foot_pos[1], 'ro')

                # 状态文字
                ax.text(cx - 2.5, 4.5, f"Step: {i}", fontsize=12)
                ax.text(cx - 2.5, 4.2, f"Thrust: L={action[0]:.2f} R={action[1]:.2f}", fontsize=10, color='blue')

                # --- 关键修复 ---
                fig.canvas.draw()
                # 直接获取 RGBA 数组
                image = np.array(fig.canvas.buffer_rgba())
                # 转为 RGB (去掉 Alpha 通道)
                image = image[:, :, :3]

                frames.append(image)
                plt.close(fig)

            if done or truncated:
                print("Episode finished.")
                break

        # 保存 GIF
        imageio.mimsave('quadhopper_simulation.gif', frames, fps=30)
        print("✅ GIF 保存成功: quadhopper_simulation.gif")

        # ==================== 绘制推力分析图 (你的需求) ====================
        print("📊 正在绘制推力分析图...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 上图：高度跟踪情况
        ax1.plot(history['step'], history['y'], 'k-', label='Actual COM Y')
        ax1.plot(history['step'], history['target_y'], 'r--', label='Target Trajectory Y')
        ax1.set_ylabel("Height (m)")
        ax1.set_title("Trajectory Tracking Performance")
        ax1.legend()
        ax1.grid(True)

        # 下图：推力输出
        ax1_limit = ax1.get_xlim()
        ax2.plot(history['step'], history['thrust_l'], 'b-', alpha=0.7, label='Left Motor')
        ax2.plot(history['step'], history['thrust_r'], 'm-', alpha=0.7, label='Right Motor')
        ax2.set_ylabel("Thrust Input (0-1)")
        ax2.set_xlabel("Simulation Step")
        ax2.set_title("Motor Thrust Output")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('thrust_analysis.png')
        print("✅ 分析图保存成功: thrust_analysis.png")