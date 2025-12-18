import torch
import numpy as np
import math


class JumpingRobot_Env:
    def __init__(self, Env_Config, Robot_Config, PPO_Config):
        self.cfg_env = Env_Config.EnvParam
        self.cfg_robot = Robot_Config
        self.device = self.cfg_env.device

        self.dt = self.cfg_env.dt
        self.agents_num = self.cfg_env.agents_num
        self.max_step = PPO_Config.maximum_step

        # 物理参数
        self.m1 = self.cfg_robot.m1
        self.m2 = self.cfg_robot.m2
        self.Jm = self.cfg_robot.Jm
        self.lc = self.cfg_robot.lc
        self.l0 = self.cfg_robot.l0
        self.k = self.cfg_robot.k
        self.g = self.cfg_robot.g

        # 目标点
        self.target_x = 2.0
        self.target_z = 0.0  # Ground

        # 状态变量
        self.q = np.zeros((self.agents_num, 4))
        self.dq = np.zeros((self.agents_num, 4))
        self.phase = ["flight"] * self.agents_num

        # 奖励追踪
        self.tracking_reward_sum = 0
        self.over = torch.zeros((self.agents_num, 1), device=self.device, dtype=torch.bool)

        self.reset_all()

    def reset_all(self):
        for i in range(self.agents_num):
            # --- 修改 1: 初始条件设定 (pi/6 到 pi/3) ---

            # 1. 随机生成发射角度 (Launch Angle)
            # 这里的角度是相对于地面(水平)的夹角
            launch_angle = np.random.uniform(math.pi / 6, math.pi / 3)

            # 2. 随机生成发射速度大小 (Velocity Magnitude)
            # 根据经验，要跳到 2米远，速度大概在 4~6 m/s 之间
            v_mag = np.random.uniform(4.5, 6.0)

            # 3. 分解出初始速度向量 (vx, vy)
            vx0 = v_mag * np.cos(launch_angle)
            vy0 = v_mag * np.sin(launch_angle)

            # 4. 设定初始位置
            # 让机器人从 x < 0 的地方起跳，模拟助跑后的腾空
            x0 = np.random.uniform(-1.0, -0.5)

            # 高度不能太低，否则一步就落地了；也不能太高
            y0 = np.random.uniform(0.6, 1.0)

            # 5. 初始姿态 (Theta)
            # 身体的姿态通常和速度方向接近，或者稍微抬头
            # 这里设为与 launch_angle 接近，加一点点随机扰动
            # 注意：在我们的坐标系里，0是垂直向上，顺时针为正(向右倒)。
            # 速度向右(vx>0)，说明身体可能需要稍微前倾(theta < 0)或者保持直立。
            # 为了简化，我们让身体大概保持在 "垂直" 附近，稍微有点初速度
            theta0 = np.random.uniform(math.radians(-10), math.radians(10))
            dtheta0 = np.random.uniform(-1.0, 1.0)  # 稍微给点旋转角速度

            # 设置状态
            self.q[i] = [x0, y0, theta0, self.l0]  # 初始腿长必须是 l0
            self.dq[i] = [vx0, vy0, dtheta0, 0.0]  # 初始腿伸缩速度为 0
            self.phase[i] = "flight"

    def _solve_dynamics(self, i, F1, F2):
        x, y, theta, l = self.q[i]
        dx, dy, dtheta, dl = self.dq[i]

        c, s = math.cos(theta), math.sin(theta)

        # --- Mass Matrix M ---
        M = np.array([
            [self.m1 + self.m2, 0.0, l * self.m1 * c, self.m1 * s],
            [0.0, self.m1 + self.m2, -l * self.m1 * s, self.m1 * c],
            [l * self.m1 * c, -l * self.m1 * s, self.Jm + (l ** 2) * self.m1, 0],
            [self.m1 * s, self.m1 * c, 0, self.m1]
        ], dtype=np.float64)

        # --- Coriolis/Centrifugal B ---
        B = np.array([
            - self.m1 * (dtheta ** 2) * l * s,
            - self.m1 * (dtheta ** 2) * l * c,
            - 2.0 * self.m1 * l * dtheta * (dy * c + dx * s),
            (dtheta ** 2) * l * self.m1 + 2 * dtheta * dx * self.m1 * c - 2 * dtheta * dy * self.m1 * s
        ], dtype=np.float64)

        # --- Gravity G ---
        G = np.array([
            0.0,
            (self.m1 + self.m2) * self.g,
            - self.g * l * self.m1 * s,
            self.m1 * self.g * c
        ], dtype=np.float64)

        # --- External Force F ---

        # >> 修改 2: 腿部弹簧/保持力逻辑 <<

        F_leg_direction = 0.0

        if self.phase[i] == "stance":
            # 支撑相：主弹簧工作，遵循胡克定律推地
            F_leg_direction = self.k * (self.l0 - l) - 10.0 * dl  # 加一点物理阻尼防止震荡
        else:
            # 腾空相 (Flight)：
            # 必须加一个“虚拟弹簧”或“机械限位”，把腿拉回 l0。
            # 否则离心力 (m*l*omega^2) 会把腿甩出去。
            k_holding = 5000.0  # 非常硬的保持弹簧
            d_holding = 100.0  # 强阻尼
            F_leg_direction = k_holding * (self.l0 - l) - d_holding * dl

        F_vec = np.array([
            (F1 + F2) * s,
            (F1 + F2) * c,
            (F1 - F2) * self.lc,
            F_leg_direction  # 这个力作用在腿伸缩方向
        ], dtype=np.float64)

        # --- Damping (空气阻力/关节摩擦) ---
        damping_coeff = 0.05
        F_vec -= damping_coeff * self.dq[i]

        # --- 求解加速度 ---
        # 简化处理：不再解复杂的约束矩阵 W，因为我们通过状态机手动切换 Phase
        # 如果是 Stance，地面约束隐含在 F_leg_direction 和 y>=0 的硬约束里

        try:
            ddq = np.linalg.solve(M, F_vec - B - G)
        except np.linalg.LinAlgError:
            ddq = np.zeros(4)

        return ddq

    def step(self, force_tensor):
        forces = force_tensor.cpu().numpy()

        # 定义物理子步数：AI 动一次，物理算 10 次
        # 这样物理精度提高了 10 倍，不再容易爆炸
        n_substeps = 10
        dt_sub = self.dt / n_substeps

        for i in range(self.agents_num):
            F1, F2 = forces[i]

            # --- 物理循环 (Sub-stepping) ---
            for _ in range(n_substeps):
                # 1. 状态保护 (防止除零)
                if self.q[i, 3] < 0.1:
                    self.q[i, 3] = 0.1
                    if self.dq[i, 3] < 0: self.dq[i, 3] = 0

                # 2. 计算力矩逻辑 (Flight Phase 改力矩)
                current_F1, current_F2 = F1, F2
                if self.phase[i] == "flight":
                    desired_torque = (F1 - F2) * self.lc
                    # 重新计算产生纯力矩的力
                    current_F1 = desired_torque / (2 * self.lc)
                    current_F2 = - desired_torque / (2 * self.lc)

                # 3. 求解动力学
                ddq = self._solve_dynamics(i, current_F1, current_F2)

                # 4. NaN 保护 (如果这一小步炸了，就回退或归零)
                if np.isnan(ddq).any() or np.isinf(ddq).any():
                    ddq = np.zeros(4)

                    # 5. 欧拉积分 (使用 dt_sub 而不是 self.dt)
                self.dq[i] += ddq * dt_sub

                # === 速度硬限幅 (防止无限加速) ===
                self.dq[i] = np.clip(self.dq[i], -20.0, 20.0)
                # ============================

                self.q[i] += self.dq[i] * dt_sub

                # 6. 地面碰撞检测与修正
                if self.q[i, 1] < 0:
                    self.q[i, 1] = 0
                    if self.dq[i, 1] < 0: self.dq[i, 1] = 0

                    # 简单状态机切换：落地瞬间进入 Stance
                    if self.phase[i] == "flight":
                        self.phase[i] = "stance"

            # --- 物理循环结束 ---

            # 简单的离地检测 (如果在 Stance 且腿伸长了 -> Flight)
            if self.phase[i] == "stance" and self.q[i, 1] > 0.1:  # 简化判定
                # 这里可以加更复杂的判定，比如弹簧伸长到原长
                pass

    def get_current_observations(self):
        obs = []
        for i in range(self.agents_num):
            dx_to_target = self.target_x - self.q[i, 0]
            dy_to_target = self.target_z - self.q[i, 1]
            o = [
                dx_to_target, dy_to_target, self.q[i, 2],
                self.dq[i, 0], self.dq[i, 1], self.dq[i, 2],
                self.q[i, 3], self.dq[i, 3]
            ]


            o = np.clip(o, -10.0, 10.0)
            # -----------------------------

            obs.append(o)
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def get_next_observations(self):
        return self.get_current_observations()

    def compute_reward(self):
        rewards = torch.zeros(self.agents_num, 1, device=self.device)
        self.over = torch.zeros(self.agents_num, 1, device=self.device, dtype=torch.bool)
        desired_angle = math.radians(-30)
        for i in range(self.agents_num):
            x, y, theta = self.q[i, 0], self.q[i, 1], self.q[i, 2]
            dx, dy, dtheta = self.dq[i, 0], self.dq[i, 1], self.dq[i, 2]

            # --- 基础奖励 ---
            rewards[i] -= 0.01  # 生存奖励 (只要不倒就有分)
            dist = np.sqrt((self.q[i, 0] - self.target_x) ** 2 + (self.q[i, 1] - self.target_z) ** 2)
            rewards[i] = -dist

            if self.phase[i] == "flight":
                # 1. 角度误差惩罚
                angle_error = abs(theta - desired_angle)
                r_angle = np.exp(-5.0 * angle_error)  # 误差越小，奖励越接近 1

                # 2. 角速度惩罚 (Damping)
                # 我们希望它调整到角度后停住，而不是疯狂旋转
                r_spin = -0.1 * (dtheta ** 2)

                # 3. 最高点 (Apex) 强化奖励
                # 当垂直速度接近 0 时，说明在最高点，此时姿态最重要
                if abs(dy) < 0.1:
                    rewards[i] += 2.0 * r_angle  # 在最高点若角度正确，给大奖

                # 综合空中奖励
                rewards[i] += 1.0 * r_angle + r_spin

            # --- 失败判定 ---
            # 角度太歪 (摔倒)
            if abs(theta) > math.radians(45):
                self.over[i] = True
                rewards[i] -= 5.0


            # 越界
            if x > 6.0 or x < -2.0:
                self.over[i] = True
                rewards[i] -= 10.0

        self.tracking_reward_sum += rewards.mean().item()
        return rewards, self.over.float()

    def print_reward_sum(self):
        # print(f"Reward Sum: {self.tracking_reward_sum:.4f}")
        self.tracking_reward_sum = 0