import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import freeze_support
import torch as th
import os


class QuadhopperTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ==================== 1. 物理参数 (纯物理模式) ====================
        # 时间步长: 纯弹簧物理对时间步长很敏感，太大会导致弹簧爆炸
        # 建议 0.002 或 0.001。为了训练速度我们尝试 0.002
        self.dt = 0.002

        self.m = 1.21  # 质量 (kg)
        self.l0 = 1.0  # 腿原长 (m)
        self.lc = 0.177  # 力臂 (m)
        self.J = 0.15  # 转动惯量 (kg*m^2)
        self.g = 9.81
        self.max_thrust = 30.0
        self.ground_y = 0.0

        # === 新增：弹簧阻尼参数 ===
        # Stiffness (k): 决定了弹簧有多硬。
        # 质量1.2kg，重力12N。k=200意味着静止站立压缩 6cm。
        # 动态跳跃冲击力大，我们需要大一点的 k 防止触底。
        self.k_spring = 150.0

        # Damping (c): 决定了能量消耗的速度（防止无限震荡）。
        # 过小=跳跳球停不下来；过大=像陷入泥潭。
        self.c_damping = 1.0

        # ==================== 2. 目标与空间 ====================
        self.targets = np.array([3.0, 7.0])
        self.target_tolerance = 0.25  # 纯物理控制更难，稍微放宽一点

        # Action: [Left Thrust, Right Thrust]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation:
        # [x, y, theta, l, dx, dy, dtheta, dl, dist_to_target, target_idx, touch_ground_bool]
        # 我加了一维 touch_ground_bool，这对 AI 很有用，让它知道什么时候"脚踏实地"了
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        self.current_target_idx = 0
        self.steps = 0
        self.max_episode_steps = 4000  # dt变小了，步数上限要增加
        self.prev_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # 初始前倾 (-45 ~ -20度)
        init_theta = np.random.uniform(math.radians(-40), math.radians(-20))

        # 初始高度: 稍微高一点，给它势能
        # 腿的垂直投影长度
        leg_proj = self.l0 * math.cos(init_theta)
        init_y = max(1.4, leg_proj + 0.3)

        self.q = np.array([0.0, init_y, init_theta, self.l0], dtype=np.float64)

        self.dq = np.array([
            np.random.uniform(1.0, 4.0),  # dx
            np.random.uniform(1.0, 4.5),  # dy (给一个向下的初速度，帮助第一次压缩)
            0.0, 0.0
        ], dtype=np.float64)

        self.current_target_idx = 0
        self.steps = 0

        foot_x = self.get_foot_pos()[0]
        self.prev_dist = abs(foot_x - self.targets[self.current_target_idx])

        return self._get_obs(), {}

    def get_foot_pos(self):
        # 几何计算
        x, y, theta = self.q[0], self.q[1], self.q[2]
        x_f = x + self.l0 * math.sin(theta)
        y_f = y - self.l0 * math.cos(theta)
        return np.array([x_f, y_f])

    def _get_obs(self):
        dist_to_next = self.targets[self.current_target_idx] - self.get_foot_pos()[0]

        # 计算脚是否触地 (用于 observation)
        foot_y = self.get_foot_pos()[1]
        is_touching = 1.0 if foot_y <= self.ground_y else 0.0

        return np.concatenate([
            self.q, self.dq,
            [dist_to_next, float(self.current_target_idx), is_touching]
        ]).astype(np.float32)

    def step(self, action):
        # 1. 提取推力控制
        u1 = np.clip(action[0], 0.0, 1.0)
        u2 = np.clip(action[1], 0.0, 1.0)
        F_thrust_1 = u1 * self.max_thrust
        F_thrust_2 = u2 * self.max_thrust
        F_thrust_total = F_thrust_1 + F_thrust_2
        tau_thrust = (F_thrust_2 - F_thrust_1) * self.lc

        # 2. 状态变量
        x, y, theta = self.q[0], self.q[1], self.q[2]
        dx, dy, dtheta = self.dq[0], self.dq[1], self.dq[2]
        s, c = math.sin(theta), math.cos(theta)

        # ==========================================================
        # 3. 核心物理引擎更新 (Spring-Damper Physics)
        # ==========================================================

        # 计算脚的虚拟位置（假设腿是刚体时的位置）
        foot_x_virtual = x + self.l0 * s
        foot_y_virtual = y - self.l0 * c

        # 弹簧力初始化
        F_spring_x = 0.0
        F_spring_y = 0.0
        touching_ground = False

        # 如果脚穿过地面 (foot_y < 0)，说明弹簧被压缩了
        if foot_y_virtual < self.ground_y:
            touching_ground = True

            # A. 计算压缩量 (Compression)
            # 几何关系：机身高度 y，腿角度 theta。
            # 腿实际需要伸出的长度 L_needed = y / cos(theta)
            # 实际上，因为 foot_y_virtual < 0，压缩量大约是 -foot_y_virtual / cos(theta)
            # 为了数值稳定性，我们使用向量法：

            # 当前腿长向量 (Body -> Foot_Ground)
            # 假设脚钉在地面上 (foot_x_virtual, 0)
            # 腿现在的实际长度 l_curr (Body到地面的距离)
            l_curr = y / max(c, 0.01)  # 防止除以0

            compression = self.l0 - l_curr

            if compression > 0:
                # B. 计算压缩速度 (Compression Rate)
                # v_compression = -v_body_projected_on_leg
                # 速度向下的分量越大，压缩越快
                # 简单近似: 垂直速度 dy 在腿方向的投影
                # v_radial = dx * sin(theta) - dy * cos(theta)  <-- 这是一个几何投影
                # 但主要能量来自 dy
                compression_rate = -(dx * s - dy * c)

                # C. 胡克定律 + 阻尼
                F_spring_mag = self.k_spring * compression + self.c_damping * compression_rate
                F_spring_mag = max(0.0, F_spring_mag)  # 弹簧只能推，不能拉 (绳子效应)

                # D. 力的分解 (施加给机身)
                # 弹簧力沿着腿的方向，从脚指向机身
                # 腿向量方向: (-sin, cos)
                # Fx = -F_mag * sin(theta)  (注意正负号：theta<0时，sin<0，Fx>0，向前推，正确)
                # Fy = F_mag * cos(theta)
                F_spring_x = -F_spring_mag * s
                F_spring_y = F_spring_mag * c

        # 4. 牛顿欧拉方程
        # X轴加速度: (推力水平分量 + 弹簧水平分量) / m
        # 推力 F_total 垂直于机臂。机臂倾斜 theta，推力方向为 (-sin, cos) 的垂线 -> (-sin, cos)? No.
        # Body frame y-axis is thrust direction.
        # Rotated by theta: (-sin(theta), cos(theta))
        ddx = (-F_thrust_total * s + F_spring_x) / self.m

        # Y轴加速度: (推力垂直分量 + 弹簧垂直分量) / m - g
        ddy = (F_thrust_total * c + F_spring_y) / self.m - self.g

        # 角加速度: 力矩 / I
        # 弹簧力通过质心吗？在这个简化模型中，腿连接在质心，所以弹簧力没有力矩。
        # 只有推力差产生力矩。
        ddtheta = tau_thrust / self.J

        # 5. 欧拉积分 (Euler Integration)
        self.dq[0] += ddx * self.dt
        self.dq[1] += ddy * self.dt
        self.dq[2] += ddtheta * self.dt
        self.q[0] += self.dq[0] * self.dt
        self.q[1] += self.dq[1] * self.dt
        self.q[2] += self.dq[2] * self.dt

        # ==================== 奖励函数 (Reward Shaping) ====================
        foot_pos = self.get_foot_pos()
        dist_to_target = abs(foot_pos[0] - self.targets[self.current_target_idx])
        reward = 0.0
        terminated = False
        truncated = False

        # (1) 进度奖励 (PPO 核心驱动力)
        reward += 10.0 * (self.prev_dist - dist_to_target)
        self.prev_dist = dist_to_target

        # (2) 生存奖励 (鼓励不要摔倒)
        reward += 0.05

        # (3) 物理能效惩罚
        # 空中尽量少喷气，利用弹簧势能
        thrust_penalty = 0.1
        if not touching_ground:
            thrust_penalty = 0.5  # 空中喷气更贵
        reward -= thrust_penalty * np.sum(action ** 2)

        # (4) 姿态惩罚 (防止剧烈旋转)
        reward -= 0.1 * abs(self.q[2])
        reward -= 0.02 * abs(self.dq[2])

        # (5) 目标奖励
        # 只有脚踩在地上，且距离够近才算赢
        if touching_ground and dist_to_target < self.target_tolerance:
            # 检查速度，防止是"摔"过去的，要求是稳稳踩住或跳过去
            # 这里放宽一点，只要踩中就行
            reward += 50.0
            print(f"🎯 Hit T{self.current_target_idx} (Phys)! Dist: {dist_to_target:.2f}")

            if self.current_target_idx == 1:
                terminated = True
                reward += 500.0
            else:
                self.current_target_idx += 1
                self.prev_dist = abs(foot_pos[0] - self.targets[self.current_target_idx])

        # ==================== 终止条件 ====================
        # 摔倒判定
        if self.q[1] < 0.2:  # 机身太低 (腿折了)
            terminated = True
            reward -= 50.0
        if abs(self.q[2]) > math.radians(80):  # 翻车
            terminated = True
            reward -= 50.0
        if self.q[1] > 4.5:  # 飞太高
            terminated = True
            reward -= 50.0

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        info = {"current_target": self.current_target_idx}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

if __name__ == '__main__':
    freeze_support()

    MODEL_PATH = "ppo_quadhopper_targets"  # 不带 .zip 后缀
    TOTAL_TIMESTEPS = 5_000_000

    # ==================== 选择模式 ====================
    mode = "train_new"  # "train_new" / "continue" / "test"

    if mode == "train_new":
        # 从零开始全新训练
        print("🚀 从零开始全新训练...")
        vec_env = make_vec_env(QuadhopperTargetEnv, n_envs=12, vec_env_cls=SubprocVecEnv)

        model = PPO(
            "MlpPolicy", vec_env, verbose=1,
            policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])),
            learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            tensorboard_log="./quadhopper_ppo_tensorboard/"
        )

        # 可选评估回调
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=320, verbose=1)
        eval_env = DummyVecEnv([lambda: QuadhopperTargetEnv()])
        eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/best_model/",
                                     log_path="./logs/", eval_freq=10000,
                                     deterministic=True, render=False,
                                     callback_on_new_best=callback_on_best)

        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
        model.save(MODEL_PATH)
        print(f"✅ 新训练完成，模型已保存到 {MODEL_PATH}.zip")

    elif mode == "continue":
        # 加载已有模型继续训练
        if not os.path.exists(MODEL_PATH + ".zip"):
            raise FileNotFoundError(f"找不到模型文件 {MODEL_PATH}.zip，请先训练或检查路径")

        print("🔄 加载已有模型，继续训练...")
        vec_env = make_vec_env(QuadhopperTargetEnv, n_envs=12, vec_env_cls=SubprocVecEnv)
        model = PPO.load(MODEL_PATH, env=vec_env)

        model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=False)  # 重要：不重置步数计数
        model.save(MODEL_PATH)
        print(f"✅ 继续训练完成，模型已更新保存到 {MODEL_PATH}.zip")

    elif mode == "test":
        # 只加载模型测试 + 录制动画
        if not os.path.exists(MODEL_PATH + ".zip"):
            raise FileNotFoundError(f"找不到模型文件 {MODEL_PATH}.zip，请先训练")

        print("🎬 加载模型进行测试和录制动画...")
        import matplotlib.pyplot as plt
        import imageio

        eval_env = DummyVecEnv([lambda: QuadhopperTargetEnv()])
        model = PPO.load(MODEL_PATH)

        obs = eval_env.reset()
        images = []
        print("正在录制动画（最多1000步）...")
        import matplotlib.pyplot as plt
        import matplotlib
        import imageio
        import numpy as np

        # 强制使用非交互后端，避免 macosx 后端问题
        matplotlib.use('Agg')  # 关键！无窗口、无交互、纯离屏渲染

        WIDTH, HEIGHT = 1000, 500  # 我们想要的固定像素尺寸

        for i in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

            # ... (前文代码不变)

            # 创建固定大小的图
            fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
            ax = fig.add_subplot(111)

            # 调整视野范围 (因为腿长变成了1.0，跳得更高了，视野要放大)
            ax.set_xlim(obs[0][0] - 3, obs[0][0] + 7)  # 视野放宽
            ax.set_ylim(-1.5, 3.5)  # 高度放宽 (腿长1m + 跳跃高度)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            x, y, theta = obs[0][0], obs[0][1], obs[0][2]

            # === 使用新的物理尺寸进行绘制 ===
            # 获取环境中的真实尺寸 (如果无法直接获取，手动填入 0.177 和 1.0)
            draw_lc = 0.177  # 对应 self.lc
            draw_l0 = 1.0  # 对应 self.l0

            # 绘制机臂 (Body)
            xl = x - draw_lc * math.cos(theta)
            yl = y - draw_lc * math.sin(theta)
            xr = x + draw_lc * math.cos(theta)
            yr = y + draw_lc * math.sin(theta)

            # 绘制腿 (Leg)
            # 注意: 这里使用你在 get_foot_pos 中的逻辑
            foot_x = x + draw_l0 * math.sin(theta)
            foot_y = y - draw_l0 * math.cos(theta)

            ax.plot([-10, 20], [0, 0], 'k-', lw=2)  # 地面
            ax.plot([xl, xr], [yl, yr], 'k-', lw=4)  # 机身 (加粗)
            ax.plot([x, foot_x], [y, foot_y], 'b-', lw=2)  # 腿
            ax.plot([foot_x], [foot_y], 'ro', markersize=8)  # 脚 (圆球加大)
            ax.plot([xl], [yl], 'go', markersize=6)  # 左电机
            ax.plot([xr], [yr], 'go', markersize=6)  # 右电机

            current_target_x = 4.0 if info[0]['current_target'] == 1 else 2.0
            ax.set_title(f"Step {i} | Vel X: {obs[0][4]:.2f} m/s | Next Target: {current_target_x:.1f} m")

            # 渲染到固定大小的 buffer
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape((HEIGHT, WIDTH, 3))  # 直接强制 reshape 到我们想要的尺寸

            images.append(image)
            plt.close(fig)

            if done[0]:
                print("Episode 结束")
                break

        imageio.mimsave('quadhopper_trained_demo_25.gif', images, fps=50)
        print("✅ 动画已成功保存为 'quadhopper_trained_demo.gif‘")