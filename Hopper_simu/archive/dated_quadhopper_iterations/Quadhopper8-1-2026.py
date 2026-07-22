import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch as th
from multiprocessing import freeze_support
import os


class QuadhopperTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ==================== 物理参数 ====================
        self.dt = 0.005
        self.m = 1.21
        self.l0 = 1.0  # 腿长
        self.lc = 0.177  # 质心到推力点的距离
        self.J = 0.15  # 转动惯量
        self.g = 9.81
        self.max_thrust = 30.0  # 稍微增加推力上限，因为现在需要克服阻尼做功
        self.ground_y = 0.0

        # ==================== 接触模型参数 (用户指定) ====================
        self.k_ground = 150.0  # Stiffness (N/m) - 较软的弹簧
        self.c_ground = 1.0  # Damping (N*s/m) - 能量耗散
        self.mu = 0.8  # 地面摩擦系数 (Rubber on Concrete)

        # ==================== 目标设置 ====================
        self.targets = np.array([3.0, 7.0])
        self.target_tolerance = 0.20

        # Action: [Left Thrust, Right Thrust]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.current_target_idx = 0
        self.steps = 0
        self.max_episode_steps = 2000
        self.prev_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # 初始向右倾斜 (-30 度左右)
        init_theta = np.random.uniform(math.radians(-35), math.radians(-25))

        # 初始高度 (必须高一点，防止k=150太软直接触底)
        init_y = 1.6

        self.q = np.array([0.0, init_y, init_theta, self.l0], dtype=np.float64)

        # 初始给向前的速度
        self.dq = np.array([3.0, 1.0, 0.0, 0.0], dtype=np.float64)

        self.current_target_idx = 0
        self.steps = 0

        foot_x = self.get_foot_pos()[0]
        self.prev_dist = abs(foot_x - self.targets[self.current_target_idx])

        return self._get_obs(), {}

    def _get_obs(self):
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

    def get_foot_velocity(self):
        # 雅可比矩阵法计算脚尖速度
        # v_foot = v_com + omega x r
        theta = self.q[2]
        dx, dy, dtheta = self.dq[0], self.dq[1], self.dq[2]

        # r = [l sin(theta), -l cos(theta)]
        # dr/dt = [l cos(theta)*dtheta, l sin(theta)*dtheta]
        vx_f = dx + self.l0 * math.cos(theta) * dtheta
        vy_f = dy + self.l0 * math.sin(theta) * dtheta
        return vx_f, vy_f

    def step(self, action):
        # 1. 施加推力
        u1 = np.clip(action[0], 0.0, 1.0)
        u2 = np.clip(action[1], 0.0, 1.0)
        F_thrust_1 = u1 * self.max_thrust
        F_thrust_2 = u2 * self.max_thrust
        F_thrust_total = F_thrust_1 + F_thrust_2
        tau_thrust = (F_thrust_2 - F_thrust_1) * self.lc

        theta = self.q[2]
        s, c = math.sin(theta), math.cos(theta)

        # 2. 计算地面反作用力 (Ground Reaction Force - GRF)



        foot_pos = self.get_foot_pos()
        foot_y = foot_pos[1]

        Fx_ground = 0.0
        Fy_ground = 0.0

        is_contact = False

        if foot_y < self.ground_y:
            is_contact = True
            penetration = self.ground_y - foot_y
            vx_foot, vy_foot = self.get_foot_velocity()

            # --- 垂直方向 (Spring-Damper) ---
            # F = k * x - c * v
            # 注意：vy_foot 向下为负，所以 -c*vy 产生向上的阻尼力
            normal_force = self.k_ground * penetration - self.c_ground * vy_foot
            normal_force = max(0.0, normal_force)  # 地面不能产生拉力
            Fy_ground = normal_force

            # --- 水平方向 (Friction) ---
            # 简单的粘滞摩擦 + 库仑摩擦限幅
            # 先假设完全粘滞 (不打滑): F = -k_f * v
            friction_force = -50.0 * vx_foot

            # 库仑限制: |Ff| <= mu * Fn
            max_friction = self.mu * normal_force
            Fx_ground = np.clip(friction_force, -max_friction, max_friction)

        # 3. 动力学方程
        # Fx_total = F_thrust_x + F_ground_x
        # Fy_total = F_thrust_y + F_ground_y - mg

        # 推力在世界坐标系的投影
        # 推力是沿脊柱向上的: [-sin(theta), cos(theta)]
        Fx_thrust_world = -F_thrust_total * s
        Fy_thrust_world = F_thrust_total * c

        ddx = (Fx_thrust_world + Fx_ground) / self.m
        ddy = (Fy_thrust_world + Fy_ground) / self.m - self.g

        # 4. 力矩计算
        # 推力产生的力矩已经算出: tau_thrust
        # 地面力产生的力矩: tau = r x F
        # r (从质心到脚) = [l sin, -l cos]
        # tau_ground = rx * Fy - ry * Fx
        tau_ground = 0.0
        if is_contact:
            rx = self.l0 * s
            ry = -self.l0 * c
            tau_ground = rx * Fy_ground - ry * Fx_ground

        ddtheta = (tau_thrust + tau_ground) / self.J

        # 5. 积分 (Euler)
        self.dq[0] += ddx * self.dt
        self.dq[1] += ddy * self.dt
        self.dq[2] += ddtheta * self.dt
        self.q[0] += self.dq[0] * self.dt
        self.q[1] += self.dq[1] * self.dt
        self.q[2] += self.dq[2] * self.dt

        # ==================== 状态更新与奖励 ====================

        # 重新获取更新后的位置
        foot_pos_new = self.get_foot_pos()
        dist_to_target = abs(foot_pos_new[0] - self.targets[self.current_target_idx])

        reward = 0.0
        terminated = False
        truncated = False

        # Reward 1: 向目标前进 (Potential Field)
        progress = self.prev_dist - dist_to_target
        reward += 10.0 * progress
        self.prev_dist = dist_to_target

        # Reward 2: 生存奖励 (鼓励它保持在空中或站立，不要为了避免能耗直接趴下)
        reward += 0.05

        # Reward 3: 能耗惩罚 (稍微降低一点，因为现在必须用推力来对抗阻尼)
        reward -= 0.5 * np.sum(action ** 2)

        # Reward 4: 姿态惩罚 (防止旋转过快)
        reward -= 0.01 * (self.dq[2] ** 2)

        # 判定是否踩中目标 (仅在接触地面且非常接近目标时)
        # 既然是物理接触，我们通过检测 foot_y < 0.05 来判定
        if foot_pos_new[1] < 0.1 and dist_to_target < self.target_tolerance:
            # 只有速度不太快的时候才算踩稳了 (可选)
            if abs(self.dq[0]) < 2.0:
                reward += 15.0
                print(f"🎯 Hit T{self.current_target_idx} | Dist: {dist_to_target:.2f}")

                if self.current_target_idx < len(self.targets) - 1:
                    self.current_target_idx += 1
                    self.prev_dist = abs(foot_pos_new[0] - self.targets[self.current_target_idx])
                else:
                    reward += 50.0
                    terminated = True

        # 失败判定
        # 1. 身体触地 (即 y 太低，且不是脚在支撑)
        # 简单的判定：如果重心 y < 0.2 (腿长1.0，所以这肯定倒了)
        if self.q[1] < 0.3:
            terminated = True
            reward -= 10.0

        # 2. 角度过大 (摔倒)
        if abs(self.q[2]) > math.radians(80):
            terminated = True
            reward -= 10.0

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {"current_target": self.current_target_idx}


        def render(self):
            pass


if __name__ == '__main__':
    freeze_support()

    MODEL_PATH = "ppo_quadhopper_physics"
    TOTAL_TIMESTEPS = 3_000_000

    # 切换模式： "train_new" 或 "test"
    mode = "train_new"

    if mode == "train_new":
        print("🚀 开始物理引擎训练 (Spring k=150, Damping c=1)...")
        # 必须多开环境，因为物理控制比魔法跳跃难得多，需要更多样本
        vec_env = make_vec_env(QuadhopperTargetEnv, n_envs=16, vec_env_cls=SubprocVecEnv)

        model = PPO(
            "MlpPolicy", vec_env, verbose=1,
            policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256])),
            learning_rate=3e-4,
            batch_size=256,  # Batch size 加大稳定梯度
            n_steps=2048,
            gamma=0.995,  # Gamma 提高，看重更长远的未来(因为跳跃周期变长了)
            tensorboard_log="./quadhopper_log/"
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(MODEL_PATH)
        print("✅ 训练完成")

    elif mode == "test":
        print("🎬 生成物理演示动画...")
        import matplotlib.pyplot as plt
        import matplotlib
        import imageio

        matplotlib.use('Agg')

        # 修复 plotting 时的 DPI
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.rcParams['savefig.dpi'] = 100

        if not os.path.exists(MODEL_PATH + ".zip"):
            print("⚠️ 警告：找不到该物理模型，将尝试加载旧模型或请先训练。")
            # 这里为了演示方便，如果找不到新模型，你可以临时注释掉退出，看看未经训练的行为(会直接摔倒)
            if not os.path.exists(MODEL_PATH + ".zip"): exit()

        model = PPO.load(MODEL_PATH)
        env = QuadhopperTargetEnv()
        obs, _ = env.reset()

        frames = []
        WIDTH, HEIGHT = 1000, 500

        print("正在渲染...")
        for i in range(600):  # 物理模拟可能需要更长的时间
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
            ax = fig.add_subplot(111)

            cx = obs[0]
            ax.set_xlim(cx - 3, cx + 5)  # 视角紧跟
            ax.set_ylim(-0.5, 3.5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            # 绘制目标
            for t_idx, tx in enumerate(env.targets):
                color = 'r' if t_idx == info['current_target'] else 'gray'
                ax.plot(tx, 0, marker='x', color=color, markersize=10, markeredgewidth=3)
                ax.add_patch(
                    plt.Rectangle((tx - env.target_tolerance, -0.02), env.target_tolerance * 2, 0.04, color=color,
                                  alpha=0.2))

            # 绘制机器人
            x, y, theta = obs[0], obs[1], obs[2]
            lc, l0 = env.lc, env.l0

            # 绘制真实的弹簧压缩效果 (仅视觉)
            # 计算脚实际在哪里
            foot_x_real = x + l0 * math.sin(theta)
            foot_y_real = y - l0 * math.cos(theta)

            # 如果脚陷地了，视觉上让它停在地面，表现出腿被压缩
            visual_foot_y = max(0.0, foot_y_real)

            xl = x - lc * math.cos(theta)
            yl = y - lc * math.sin(theta)
            xr = x + lc * math.cos(theta)
            yr = y + lc * math.sin(theta)

            ax.plot([-100, 100], [0, 0], 'k-', lw=2)
            ax.plot([xl, xr], [yl, yr], 'k-', lw=5)  # Body
            ax.plot([x, foot_x_real], [y, visual_foot_y], 'b-', lw=3)  # Leg

            # 绘制地面反作用力 (可选，调试用)
            if foot_y_real < 0:
                ax.plot(foot_x_real, 0, 'ro', markersize=5)  # 接触点

            # Thrust Visual
            if action[0] > 0.05:
                ax.arrow(xl, yl, 0, -action[0], head_width=0.08, color='orange')
            if action[1] > 0.05:
                ax.arrow(xr, yr, 0, -action[1], head_width=0.08, color='orange')

            ax.set_title(f"Step {i} | H: {y:.2f} | Theta: {math.degrees(theta):.1f}°")

            fig.canvas.draw()
            # ---------------- 修复后的图像获取代码 ----------------
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape((HEIGHT, WIDTH, 4))
            image = image[:, :, :3]
            # ---------------------------------------------------

            frames.append(image)
            plt.close(fig)

            if done:
                print("Episode Finished")
                break

        imageio.mimsave('quadhopper_physics.gif', frames, fps=50)
        print("✅ GIF Saved")