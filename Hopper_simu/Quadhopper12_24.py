import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_checker import check_env
import torch as th
from multiprocessing import freeze_support  # Windows 需要这个，但 macOS 无害
import os


class QuadhopperTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ==================== 物理参数 ====================
        self.dt = 0.005
        self.m = 1.21
        self.l0 = 1.0
        self.lc = 0.177
        self.J = 0.15
        self.g = 9.81
        self.max_thrust = 30.0
        self.ground_y = 0.0

        # ==================== 目标设置 ====================
        # 稍微拉开距离，测试它的长距离跳跃能力
        self.targets = np.array([3.0, 7.0])
        self.target_tolerance = 0.15  # 收紧判定范围 (之前是 0.20)

        # Action: [Left Thrust, Right Thrust]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: 增加 explicit distance
        # [x, y, theta, l, dx, dy, dtheta, dl, dist_to_target, target_idx]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.current_target_idx = 0
        self.steps = 0
        self.max_episode_steps = 2000

        # 记录上一步的距离，用于计算 Potential Reward
        self.prev_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # 初始向右倾斜 (-45 ~ -20度)，准备起跑
        init_theta = np.random.uniform(math.radians(-45), math.radians(-20))
        leg_vertical_height = self.l0 * math.cos(init_theta)

        # 初始高度
        init_y = max(1.3, leg_vertical_height + 0.2)

        self.q = np.array([0.0, init_y, init_theta, self.l0], dtype=np.float64)

        # 初始给一点点向前的速度，帮助启动
        self.dq = np.array([
            np.random.uniform(0.0, 4.0),  # dx
            np.random.uniform(0.0, 4.0),  # dy (向下)
            0.0, 0.0
        ], dtype=np.float64)

        self.current_target_idx = 0
        self.steps = 0

        # 初始化距离记录
        foot_x = self.get_foot_pos()[0]
        self.prev_dist = abs(foot_x - self.targets[self.current_target_idx])

        return self._get_obs(), {}

    def _get_obs(self):
        # 显式计算距离，让神经网络直接看到"还有多远"
        dist_to_next = self.targets[self.current_target_idx] - self.get_foot_pos()[0]
        return np.concatenate([
            self.q, self.dq,
            [dist_to_next, float(self.current_target_idx)]
        ]).astype(np.float32)

    def get_foot_pos(self):
        x, y, theta = self.q[0], self.q[1], self.q[2]
        x_f = x + self.l0 * math.sin(theta)
        y_f = y - self.l0 * math.cos(theta)
        return np.array([x_f, y_f])

    def step(self, action):
        # ==================== 物理计算 ====================
        u1 = np.clip(action[0], 0.0, 1.0)
        u2 = np.clip(action[1], 0.0, 1.0)
        F1 = u1 * self.max_thrust
        F2 = u2 * self.max_thrust
        F_total = F1 + F2
        tau = (F2 - F1) * self.lc

        theta = self.q[2]
        s, c = math.sin(theta), math.cos(theta)

        ddx = (-F_total * s) / self.m
        ddy = (F_total * c) / self.m - self.g
        ddtheta = tau / self.J

        self.dq[0] += ddx * self.dt
        self.dq[1] += ddy * self.dt
        self.dq[2] += ddtheta * self.dt
        self.q[0] += self.dq[0] * self.dt
        self.q[1] += self.dq[1] * self.dt
        self.q[2] += self.dq[2] * self.dt

        # ==================== 状态更新 ====================
        foot_pos = self.get_foot_pos()
        foot_x, foot_y = foot_pos[0], foot_pos[1]

        target_x = self.targets[self.current_target_idx]
        dist_to_target = abs(foot_x - target_x)

        reward = 0.0
        terminated = False
        truncated = False

        # ==================== 核心奖励设计 (Reward Shaping) ====================

        # 1. 基础生存与距离引导 (Potential Field)
        # 如果这一步比上一步离目标更近，给正奖励；反之给负奖励。
        # 这比直接给距离绝对值更有效，能引导梯度。
        progress = self.prev_dist - dist_to_target
        reward += 10.0 * progress
        self.prev_dist = dist_to_target

        # 2. 严厉的能耗惩罚 (Energy Efficiency)
        # 我们惩罚 action 的平方，但惩罚系数加大
        # 此外，如果高度 > 1.5m (弹道顶点附近)，额外加重推力惩罚，强迫它在空中"闭嘴"，只靠惯性
        thrust_cost = np.sum(action ** 2)
        if self.q[1] > 1.5:
            reward -= 0.5 * thrust_cost  # 高空推力惩罚 x 5
        else:
            reward -= 0.1 * thrust_cost  # 低空正常惩罚

        # 3. 姿态稳定惩罚
        # 惩罚过大的角速度和角度，防止它在空中乱翻
        reward -= 0.05 * abs(self.dq[2])
        reward -= 0.1 * abs(self.q[2])

        # ==================== 触地与弹跳逻辑 ====================
        if foot_y <= self.ground_y and self.dq[1] < 0:

            # A. 精准落点奖励 (Gaussian Kernel)
            # 只有非常接近目标时，这个值才会接近 1。
            # sigma=0.5 意味着误差在 0.5m 以内才有明显分数
            accuracy_bonus = 20.0 * np.exp(-(dist_to_target ** 2) / 0.15)
            reward += accuracy_bonus

            # B. 目标达成逻辑
            if dist_to_target < self.target_tolerance:
                reward += 100.0  # 踩中大奖
                print(
                    f"🎯 Hit Target {self.current_target_idx} | Error: {dist_to_target:.3f}m | Thrust Avg: {np.mean(action):.2f}")

                if self.current_target_idx == 1:
                    terminated = True
                    reward += 500.0  # 通关大奖
                else:
                    self.current_target_idx += 1
                    # 重置距离记录，防止下一个目标的距离突变导致巨大的负 progress
                    self.prev_dist = abs(foot_x - self.targets[self.current_target_idx])

            # C. 物理反弹 (Raibert Hopping Physics)
            # 修正：让水平速度的变化更加物理化
            # 我们不再强制赋值 self.dq[0]，而是模拟地面摩擦力和反弹力

            # 垂直反弹：能量补充 (Pogo Stick 核心)
            # 设定一个固定的弹跳高度目标
            h_bounce = 2.0
            v_rebound = math.sqrt(2 * self.g * h_bounce)
            self.dq[1] = v_rebound

            # 水平反弹：
            # 之前的代码: self.dq[0] -= 2.0 * theta (硬编码)
            # 改进: 依然保留这个机制，因为它是单腿机器人的控制核心(把腿往前伸，就能减速/向后跳)
            # 但我们减小系数，让 AI 更多依赖空中的姿态调整，而不是依靠近地面的"魔法修正"
            # 这里的逻辑是：Forward Lean (theta < 0) -> Adds forward velocity
            # Backward Lean (theta > 0) -> Reduces forward velocity
            self.dq[0] -= 3.0 * theta

            # 角速度阻尼 (模拟触地瞬间的稳定化)
            self.dq[2] *= 0.5

            # 防止穿模
            self.q[1] = self.ground_y + self.l0 * math.cos(theta) + 0.01

        # ==================== 终止条件 ====================
        # 1. 摔倒
        if self.q[1] < 0.2 or abs(self.q[2]) > math.radians(70):
            terminated = True
            reward -= 50.0  # 摔倒惩罚

        # 2. 飞天 (超过 4m 判负)
        if self.q[1] > 4.0:
            terminated = True
            reward -= 100.0

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        info = {"current_target": self.current_target_idx}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass


if __name__ == '__main__':
    freeze_support()

    MODEL_PATH = "ppo_quadhopper_precise"
    TOTAL_TIMESTEPS = 3_000_000

    mode = "train_new"  # 建议使用 train_new 重新适应新的奖励函数

    if mode == "train_new":
        print("🚀 开始高精度训练...")
        # 增加环境数量加速训练
        vec_env = make_vec_env(QuadhopperTargetEnv, n_envs=12, vec_env_cls=SubprocVecEnv)

        model = PPO(
            "MlpPolicy", vec_env, verbose=1,
            # 网络稍微加宽，以处理更复杂的精细控制
            policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            gamma=0.99,
            # 熵系数(ent_coef)稍微调高，增加探索，防止过早陷入"飞过去"的局部最优
            ent_coef=0.01,
            tensorboard_log="./quadhopper_log/"
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(MODEL_PATH)
        print("✅ 训练完成")

    elif mode == "test":
        print("🎬 生成演示动画...")
        import matplotlib.pyplot as plt
        import matplotlib
        import imageio

        matplotlib.use('Agg')

        if not os.path.exists(MODEL_PATH + ".zip"):
            print("找不到模型，请先训练！")
            exit()

        model = PPO.load(MODEL_PATH)
        env = QuadhopperTargetEnv()
        obs, _ = env.reset()

        frames = []
        WIDTH, HEIGHT = 1000, 500

        print("正在渲染...")
        for i in range(800):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
            ax = fig.add_subplot(111)

            # 动态跟随视角
            cx = obs[0]  # Body X
            ax.set_xlim(cx - 4, cx + 6)
            ax.set_ylim(-1.0, 4.0)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            # 绘制目标
            targets = env.targets
            for t_idx, tx in enumerate(targets):
                color = 'r' if t_idx == info['current_target'] else 'gray'
                # 画出目标点
                ax.plot(tx, 0, marker='x', color=color, markersize=10, markeredgewidth=3)
                # 画出判定范围
                ax.add_patch(
                    plt.Rectangle((tx - env.target_tolerance, -0.05), env.target_tolerance * 2, 0.1, color=color,
                                  alpha=0.3))

            # 绘制机器人
            x, y, theta = obs[0], obs[1], obs[2]
            lc, l0 = env.lc, env.l0

            # 机身
            xl = x - lc * math.cos(theta)
            yl = y - lc * math.sin(theta)
            xr = x + lc * math.cos(theta)
            yr = y + lc * math.sin(theta)

            # 腿
            foot_x = x + l0 * math.sin(theta)
            foot_y = y - l0 * math.cos(theta)

            ax.plot([-100, 100], [0, 0], 'k-', lw=2)  # 地面
            ax.plot([xl, xr], [yl, yr], 'k-', lw=5)  # 机身
            ax.plot([x, foot_x], [y, foot_y], 'b-', lw=3)  # 腿

            # 喷气效果 (Visualizing Thrust)
            thrust_scale = 1.0
            if action[0] > 0.05:  # 左电机喷气
                ax.arrow(xl, yl, 0, -action[0] * thrust_scale, head_width=0.1, color='orange')
            if action[1] > 0.05:  # 右电机喷气
                ax.arrow(xr, yr, 0, -action[1] * thrust_scale, head_width=0.1, color='orange')

            ax.set_title(f"Step {i} | Vel X: {obs[4]:.2f} | Dist: {obs[8]:.2f} | Thrust: {np.mean(action):.2f}")

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape((HEIGHT, WIDTH, 3))
            frames.append(image)
            plt.close(fig)

            if done:
                print("Reach Goal or Fail")
                break

        imageio.mimsave('quadhopper_precise.gif', frames, fps=50)
        print("✅ GIF Saved")
