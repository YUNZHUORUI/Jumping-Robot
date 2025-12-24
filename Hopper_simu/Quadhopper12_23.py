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

        # ==========================================
        # 1. 物理参数 (基于 cf2.xml 更新)
        # ==========================================
        self.dt = 0.005

        # 质量: Body(1.1) + Foot(0.11)
        self.m = 1.21

        # 腿长: 基于 pos="0 0 -1.0"
        self.l0 = 1.0

        # 力臂 (半宽): 基于 motor_site pos="0.1768"
        self.lc = 0.177

        # 转动惯量: 基于 diaginertia="0.15 ..."
        self.J = 0.15

        self.g = 9.81

        # 推力: 单电机15N * 2 (双桨合并) = 30N
        self.max_thrust = 30.0

        # 弹簧参数 (虽然当前step用的是Raibert跳跃, 但建议存着)
        self.k_spring = 150.0
        self.c_damping = 1.0

        self.ground_y = 0.0

        # ==========================================
        # 2. 目标与空间设置 (保持逻辑不变)
        # ==========================================
        # 两个固定目标 (因为腿变长了1.0m, 步幅变大，建议稍微拉远目标距离)
        self.targets = np.array([2.0, 4.0])
        self.target_tolerance = 0.20  # 适当放宽判定范围

        # Action: 左右油门 [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.current_target_idx = 0
        self.steps = 0
        self.max_episode_steps = 2000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # 1. 随机初始化角度 (Theta)
        init_theta = np.random.uniform(math.radians(-60), math.radians(-30))

        # 2. 计算最小安全高度 (Min Safe Height)
        # 腿完全伸直时的垂直长度 = l0 * cos(theta)
        leg_vertical_height = self.l0 * math.cos(init_theta)

        # 3. 随机初始化高度 (Y)
        # 我们需要在腿长基础上，再加一点"悬空高度"，让它落下来积蓄能量
        init_y = np.random.uniform(1.2, 1.5)

        # 确保无论如何 y 都要比腿长高 (防止插地)
        init_y = max(init_y, leg_vertical_height + 0.1)

        self.q = np.array([
            0.0,  # x: 从 0 开始
            init_y,  # y: 修改后的高度
            init_theta,  # theta: 修改后的前倾角度
            self.l0  # l: 腿长 1.0
        ], dtype=np.float64)

        # 4. 初始化速度
        # 可以稍微给一点向下的初速度，模拟"扔下来"的感觉，或者设为0
        self.dq = np.array([
            np.random.uniform(0.0, 5.0),  # dx: 稍微给一点点向前的初速度
            np.random.uniform(0.0, 5.0),  # dy: 稍微给一点向下的初速度(砸向地面)
            0.0,  # dtheta
            0.0  # dl
        ], dtype=np.float64)

        self.current_target_idx = 0
        self.steps = 0

        return self._get_obs(), {}

    def _get_obs(self):
        rel_x_to_next = self.targets[self.current_target_idx] - self.q[0]
        return np.concatenate([
            self.q, self.dq,
            [rel_x_to_next, float(self.current_target_idx)]
        ]).astype(np.float32)

    def get_foot_pos(self):
        x, y, theta = self.q[0], self.q[1], self.q[2]
        x_f = x + self.l0 * math.sin(theta)
        y_f = y - self.l0 * math.cos(theta)
        return np.array([x_f, y_f])

    def step(self, action):
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

        foot_pos = self.get_foot_pos()
        foot_y = foot_pos[1]
        foot_x = foot_pos[0]

        reward = 0.0
        terminated = False
        truncated = False

        # ==========================================
        # 奖励函数修改 (核心修改区域)
        # ==========================================

        # 1. 存活奖励 (保持不变)
        reward += 0.01

        # 2. 距离惩罚/奖励 (改用更平滑的引导)
        # 之前是 +0.2 * dx，这会鼓励它无脑冲。改为鼓励"靠近目标"
        dist_to_target = abs(foot_x - self.targets[self.current_target_idx])
        reward += (1.0 / (dist_to_target + 0.1)) * 0.1

        # 3. [新增] 高度惩罚 (Ceiling Penalty)
        # 强迫它不要飞太高。理想跳跃高度约为 1.5m - 2.0m (机身高度)
        # 如果超过 2.5m，给予巨大惩罚
        if self.q[1] > 2.5:
            reward -= 1.0  # 每一步都扣，很疼

        # 4. [修改] 能耗惩罚 (Energy Penalty)
        # 大幅增加油门成本，迫使它利用被动弹跳而不是主动飞行
        # 之前是 0.001，现在改为 0.05 (增加50倍)
        reward -= 0.5 * np.sum(action ** 2)

        # 5. [新增] 姿态惩罚 (Stability)
        # 稍微加强对倾斜的惩罚，防止乱翻
        reward -= 2.0 * abs(self.q[2])

        # ==========================================
        # 触地逻辑 (Raibert Pogo Physics)
        # ==========================================
        if foot_y <= self.ground_y and self.dq[1] < 0:  # 触地

            # 只有触地时，能获得"反弹奖励"
            # 这鼓励 AI 主动去触地，因为这是获得动量最"便宜"的方式(Raibert模型免费给速度)
            reward += 5.0

            if dist_to_target < self.target_tolerance:
                reward += 100.0  # 到达奖励
                print(f"🎯 踩中目标! Dist: {dist_to_target:.3f}")
                if self.current_target_idx == 1:
                    terminated = True
                    reward += 500.0  # 最终大奖
                else:
                    self.current_target_idx += 1

            # Raibert 物理 (瞬间反弹)
            # 这是一个"魔法"物理，不消耗电能就能获得向上的速度
            # AI 会发现：用电机飞很贵(扣分)，撞地反弹免费(给分且给速度)，所以它会选择跳。
            h_target = 1.5  # 目标弹跳高度
            v_launch = math.sqrt(2 * self.g * h_target)
            self.dq[1] = v_launch

            # 触地时修正水平速度和角速度 (模拟摩擦和控制)
            self.dq[0] -= 2.0 * theta
            self.dq[2] -= 5.0 * theta

            # 防止穿模
            self.q[1] = self.ground_y + self.l0 * math.cos(theta) + 0.01

        # ==========================================
        # 终止条件 (Termination)
        # ==========================================
        # 1. 摔倒 (高度太低 或 角度太大)
        # 注意：因为腿长1.0m，机身高度 < 0.5 肯定已经摔了
        if self.q[1] < 0.5 or abs(self.q[2]) > math.radians(60):
            terminated = True
            reward -= 50.0  # 摔倒惩罚

        # 2. [新增] 飞出天际 (Fly Away)
        # 如果飞得太高(比如超过4米)，直接判负并结束
        if self.q[1] > 4.0:
            terminated = True
            reward -= 100.0
            print("❌ 飞太高了，判定为失控")

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        info = {"current_target": self.current_target_idx}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    freeze_support()

    MODEL_PATH = "ppo_quadhopper_targets"  # 不带 .zip 后缀
    TOTAL_TIMESTEPS = 5_000_000

    # ==================== 选择模式 ====================
    mode = "test"  # "train_new" / "continue" / "test"

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

        imageio.mimsave('quadhopper_trained_demo.gif', images, fps=50)
        print("✅ 动画已成功保存为 'quadhopper_trained_demo.gif‘")
