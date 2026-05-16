import torch
import math


class JumpingRobot_Env:
    def __init__(self, Env_Config, Robot_Config, PPO_Config):
        """初始化环境变量"""
        self.dt = Env_Config.EnvParam.dt
        self.sub_step = Env_Config.EnvParam.sub_step
        self.device = Env_Config.EnvParam.device
        self.train = Env_Config.EnvParam.train
        self.agents_num = (Env_Config.EnvParam.agents_num - Env_Config.EnvParam.agents_num_in_play) * self.train + \
                          Env_Config.EnvParam.agents_num_in_play

        """初始化机器人变量"""
        self.actuator_num = Robot_Config.ActuatorParam.actuator_num
        self.max_thrust = Robot_Config.ActuatorParam.max_thrust

        """初始化初始条件参数"""
        self.r_min_sq = Robot_Config.InitialState.r_min_sq
        self.r_max_sq = Robot_Config.InitialState.r_max_sq
        self.theta_min_rad = math.radians(Robot_Config.InitialState.theta_min_deg)
        self.theta_max_rad = math.radians(Robot_Config.InitialState.theta_max_deg)
        self.target_attack_angle_rad = math.radians(Robot_Config.InitialState.target_attack_angle_deg)

        self.target_dist_min = Robot_Config.InitialState.target_distance_min
        self.target_dist_max = Robot_Config.InitialState.target_distance_max

        # 物理常数
        self.g = 9.81

        # 状态容器
        self.x = torch.zeros((self.agents_num, 1), device=self.device)
        self.y = torch.zeros((self.agents_num, 1), device=self.device)
        self.tht = torch.zeros((self.agents_num, 1), device=self.device)
        self.x_dot = torch.zeros((self.agents_num, 1), device=self.device)
        self.y_dot = torch.zeros((self.agents_num, 1), device=self.device)
        self.tht_dot = torch.zeros((self.agents_num, 1), device=self.device)

        self.target_x = torch.zeros((self.agents_num, 1), device=self.device)
        self.target_y = torch.zeros((self.agents_num, 1), device=self.device)

        # 辅助状态
        self.apex_reached = torch.zeros((self.agents_num, 1), device=self.device, dtype=torch.bool)

        self.max_step = PPO_Config.PPOParam.maximum_step
        self.tracking_reward_sum = 0
        self.attitude_reward_sum = 0
        self.termination_reward_sum = 0

    def  late_ballistic_velocity(self, x0, y0, target_x, target_y, launch_theta):
        """
        根据初始位置、目标位置和发射角度，计算所需的初速度 v0
        公式: v0 = sqrt( g * dx^2 / (2 * cos(theta)^2 * (dx * tan(theta) - dy)) )
        """
        delta_x = target_x - x0
        delta_y = target_y - y0  # 注意：这里 delta_z 对应 y 轴高度差

        # 避免除以零和负数开根号
        # 必须满足 dx * tan(theta) > dy 才能到达
        denom_term = delta_x * torch.tan(launch_theta) - delta_y
        valid_mask = denom_term > 0.01  # 确保分母为正且不过小

        # 对无效的情况给一个默认速度，后续会被 reset 逻辑处理或者忽略
        safe_denom = torch.where(valid_mask, denom_term, torch.ones_like(denom_term))

        v0_sq = (self.g * delta_x ** 2) / (2 * torch.cos(launch_theta) ** 2 * safe_denom)
        v0 = torch.sqrt(torch.abs(v0_sq))

        # 如果计算出的情况无效（例如目标太高打不到），则 v0 设为 0（或者处理为重置）
        v0 = v0 * valid_mask.float()

        return v0

    def reset(self, index):
        if len(index) == 0:
            return

        num_reset = len(index)

        # 1. 在扇形区域内生成初始位置 (x0, y0)
        # 使用极坐标生成 r 和 angle，再转回直角坐标，或者简单的拒绝采样。
        # 这里为了简单和并行化，假设扇形是在第一象限附近
        # r^2 < x^2 + y^2 < R^2
        r = torch.sqrt(torch.rand(num_reset, 1, device=self.device) * (self.r_max_sq - self.r_min_sq) + self.r_min_sq)
        # 假设扇形在 0 到 90 度之间的一段，或者根据具体需求设定。
        # 既然是跳向目标，假设初始位置在原点附近，目标在远方
        # 这里我们设置初始位置在 (0,0) 附近的一个小的起始区域，或者按照题目要求的扇形
        # 题目图示似乎是起跳瞬间的状态。
        # 让我们设定：x0 在 0 附近，y0 在 0 附近，或者按照约束
        phi_pos = torch.rand(num_reset, 1, device=self.device) * (math.pi / 2)  # 0-90度分布位置
        self.x[index] = r * torch.cos(phi_pos)
        self.y[index] = r * torch.sin(phi_pos)

        # 2. 生成目标位置
        # 目标在 x 轴正方向远处
        dist_target = self.target_dist_min + torch.rand(num_reset, 1, device=self.device) * (
                    self.target_dist_max - self.target_dist_min)
        # 目标通常在地面 y=0 或者某个特定高度
        self.target_x[index] = self.x[index] + dist_target  # 目标在前方
        self.target_y[index] = torch.zeros(num_reset, 1, device=self.device)  # 目标在地面

        # 3. 设定发射角 theta (Leg inclination / Launch angle)
        # 题目: 30 < theta_0 < 60
        launch_theta = self.theta_min_rad + torch.rand(num_reset, 1, device=self.device) * (
                    self.theta_max_rad - self.theta_min_rad)
        self.tht[index] = launch_theta  # 身体/腿的初始倾角等于发射角

        # 4. 计算弹道速度 v0
        v0 = self.calculate_ballistic_velocity(self.x[index], self.y[index], self.target_x[index], self.target_y[index],
                                               launch_theta)

        # 5. 分解速度
        self.x_dot[index] = v0 * torch.cos(launch_theta)
        self.y_dot[index] = v0 * torch.sin(launch_theta)
        self.tht_dot[index] = torch.zeros(num_reset, 1, device=self.device)  # 初始角速度为0

        # 重置标志位
        self.apex_reached[index] = False

    def reset_all(self):
        indices = torch.arange(self.agents_num, device=self.device)
        self.reset(indices)

    def step(self, action_ctrl):
        real_dt = self.dt / self.sub_step

        # 动作处理: action_ctrl 是 Actor 输出的控制量
        # 假设 action_ctrl 是 [0, 1] 之间的连续值，表示推力比例
        # 乘以 max_thrust 得到实际力
        # 题目: "只给他两个thrust为0.314N"
        force = action_ctrl * self.max_thrust

        F1 = force[:, 0].view(-1, 1)
        F2 = force[:, 1].view(-1, 1)

        for i in range(self.sub_step):
            # 动力学更新 (保持原有的四轴动力学方程)
            # 注意: 这里的 tht 是身体倾角。
            # 当腿弹射结束后，物体在空中，动力学受重力和推力影响

            sin_tht = torch.sin(self.tht)
            cos_tht = torch.cos(self.tht)

            # 修正后的动力学方程 (基于 Config 中隐含的惯性参数)
            # x_acc = (F_total_x) / m
            # y_acc = (F_total_y) / m - g
            # tht_acc = Torque / I

            # 原代码中的系数看起来像是在特定质量/惯量下的简化值
            # 10/3, 5 等系数保留不动，假设它们对应了 m=0.3kg 左右的模型

            x_dotdot = (10 * F1 * sin_tht) / 3 - 5 * F1 * cos_tht + \
                       (10 * F2 * sin_tht) / 3 + 5 * F2 * cos_tht + \
                       (2 * sin_tht * self.tht_dot ** 2) / 3  # 这一项看起来像是哥氏力或离心力耦合，保留

            y_dotdot = 5 * F1 * sin_tht + (10 * F1 * cos_tht) / 3 - \
                       5 * F2 * sin_tht + (10 * F2 * cos_tht) / 3 + \
                       (2 * cos_tht * self.tht_dot ** 2) / 3 - 9.81

            tht_dotdot = (15 * F1) / 2 - (15 * F2) / 2

            self.x_dot += x_dotdot * real_dt
            self.y_dot += y_dotdot * real_dt
            self.tht_dot += tht_dotdot * real_dt
            self.x += self.x_dot * real_dt
            self.y += self.y_dot * real_dt
            self.tht += self.tht_dot * real_dt

        # 简单的地面碰撞检测 (Touchdown)
        # 如果 y < 0, 认为落地
        # 实际仿真中可能需要更复杂的接触模型，但在RL训练初期，可以直接视为终止或重置
        pass

    def get_current_observations(self):
        # 增加 apex_reached 状态让网络知道是否已经过了最高点
        # 增加 target 信息
        current_state = torch.concatenate((
            self.x, self.y, self.tht,
            self.x_dot, self.y_dot, self.tht_dot,
            self.target_x, self.target_y,
            # 可以加入相对目标的差值，帮助网络泛化
            self.target_x - self.x,
            self.target_y - self.y
        ), dim=-1)
        return current_state

    def get_next_observations(self):
        # 逻辑同上
        next_state = torch.concatenate((
            self.x, self.y, self.tht,
            self.x_dot, self.y_dot, self.tht_dot,
            self.target_x, self.target_y,
            self.target_x - self.x,
            self.target_y - self.y
        ), dim=-1)
        return next_state

    def compute_reward(self):
        # 1. 检测是否到达最高点 (Vertical velocity crosses zero from positive to negative)
        # 由于是离散步，简单判断 y_dot 变小且接近0
        # 或者仅仅根据当前高度

        # 2. 目标追踪奖励 (Distance to target)
        dist_to_target = torch.sqrt((self.x - self.target_x) ** 2 + (self.y - self.target_y) ** 2)
        tracking_reward = -1.0 * dist_to_target

        # 3. 姿态调整奖励 (Attitude Adjustment)
        # 目标: 在落地前将 tht 调整到 target_attack_angle (-20度)
        # 越接近地面，姿态奖励权重越高
        height_factor = torch.exp(-2.0 * torch.abs(self.y))  # 高度越低，因子越大(接近1)
        angle_error = torch.abs(self.tht - self.target_attack_angle_rad)
        attitude_reward = -2.0 * angle_error * height_factor

        # 4. 终止/成功判定
        # 成功: 落地 (y<=0) 且 距离目标近 且 姿态正确
        at_ground = self.y <= 0.05
        near_target = dist_to_target < 0.5
        good_attitude = angle_error < 0.2  # 约10度误差内

        success = at_ground & near_target & good_attitude

        # 失败: 飞太远、翻车、反向飞
        fail = (torch.abs(self.x) > 10) | (torch.abs(self.tht) > 2.0) | (self.x < -1)

        self.over = success | fail

        reward = tracking_reward * 0.1 + attitude_reward * 0.5

        # 稀疏奖励
        reward += 100.0 * success.float()
        reward -= 10.0 * fail.float()

        # 额外惩罚: 落地时如果姿态不对
        bad_landing = at_ground & (~good_attitude)
        reward -= 20.0 * bad_landing.float()

        self.tracking_reward_sum += tracking_reward.mean().item()
        self.attitude_reward_sum += attitude_reward.mean().item()

        return reward, self.over.float()

    def print_reward_sum(self):
        print(
            f"Track: {self.tracking_reward_sum:.2f} | Att: {self.attitude_reward_sum:.2f} | Term: {self.termination_reward_sum:.2f}")
        self.tracking_reward_sum = 0
        self.attitude_reward_sum = 0
        self.termination_reward_sum = 0