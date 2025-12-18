class Env_Config:
    class EnvParam:  # 训练环境的参数
        agents_num = 64
        agents_num_in_play = 1  # 实际显示的agent数量
        dt = 0.01  # 减小dt以提高物理精度
        sub_step = 10  # 增加子步数以获得更平滑的积分
        train = 0  # 0为推理/可视化模式，1为训练模式
        device = 'cpu'
        # device = 'cuda' # 如果有NVIDIA显卡


class Robot_Config:
    class ActuatorParam:  # 机器人的参数
        actuator_num = 2  # 两个螺旋桨
        max_thrust = 0.314  # [N] 最大推力
        # 腿部/身体参数可以根据需要添加，目前主要体现在动力学方程中

    class InitialState:
        # 扇形区域参数
        r_min_sq = 0.5 ** 2  # r^2 下限
        r_max_sq = 2.0 ** 2  # R^2 上限

        # 发射角范围 [度]
        theta_min_deg = 30
        theta_max_deg = 60

        # 目标攻击角 (落地时的理想姿态)
        target_attack_angle_deg = -20

        # 目标位置范围 (相对于原点)
        target_distance_min = 3.0
        target_distance_max = 5.0


class PPO_Config:
    class CriticParam:  # Critic 神经网络 参数
        state_dim = 10  # 增加状态维度: [x, y, tht, dx, dy, dtht, tar_x, tar_y, accumulated_action, apex_reached]
        critic_layers_num = 256
        critic_lr = 3e-4
        critic_update_frequency = 200

    class ActorParam:  # Actor 神经网络 参数
        action_scale = 1
        # 动作选择:
        # 0: [0, 0] 关闭
        # 1: [1, 1] 全推力 (用于主要的飞行对抗重力/姿态调整)
        # 2: [1, 0] 左推
        # 3: [0, 1] 右推
        # 这里简化为离散动作，或者保持原来的 continuous * scale。
        # 根据"只给他两个thrust为0.314N"，假设是离散的开关控制或者连续的比例控制。
        # 这里沿用原来的结构，假设输出是 continuous 0-1 之间的系数，然后乘以 max_thrust
        action_choice = [0, 1]
        act_layers_num = 256
        actuator_num = Robot_Config.ActuatorParam.actuator_num
        actor_lr = 3e-4
        actor_update_frequency = 100

    class PPOParam:  # 强化学习 PPO算法 参数
        gamma = 0.99
        lam = 0.95
        epsilon = 0.2
        maximum_step = 200  # 增加步数以覆盖完整的跳跃周期
        episode = 2000
        entropy_coef = 0.01
        batch_size = 4096