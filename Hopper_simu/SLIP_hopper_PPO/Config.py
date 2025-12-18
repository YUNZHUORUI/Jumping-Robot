import torch


class Config:
    def __init__(self):
        # 强制使用 CPU，因为您是在 macOS 上
        self.device = torch.device("cpu")

        # --- 环境参数 ---
        self.EnvParam = self.EnvParamConfig(self.device)
        # --- 机器人参数 ---
        self.RobotParam = self.RobotParamConfig()
        # --- PPO 参数 ---
        self.PPOParam = self.PPOParamConfig()
        # --- 网络参数 ---
        self.ActorParam = self.ActorParamConfig()
        self.CriticParam = self.CriticParamConfig()

    class EnvParamConfig:
        def __init__(self, device):
            self.device = device
            self.dt = 0.005  # 仿真步长 (s)
            self.sub_step = 1  # 物理子步数 (简化处理)
            self.train = True  # 是否训练模式
            self.agents_num = 1  # 您的 Mac 可能只跑得动少量并行环境，这里设为 1 用于演示
            self.agents_num_in_play = 1

    class RobotParamConfig:
        def __init__(self):
            class InitialState:
                def __init__(self):
                    # 初始范围 (在 reset 中具体实现扇形区域逻辑)
                    self.initial_x_range = 0.0
                    self.initial_y_range = 2.0
                    self.initial_theta_range = 0.0
                    self.initial_x_dot_range = 0.0
                    self.initial_y_dot_range = 0.0
                    self.initial_theta_dot = 0.0
                    self.initial_target_x_range = 2.0
                    self.initial_target_y_range = 0.0

            class ActuatorParam:
                def __init__(self):
                    self.actuator_num = 2  # 两个推力 F1, F2

            # 物理参数
            self.m1 = 0.0363  # Body mass
            self.m2 = 0.01  # Leg mass (approx)
            self.Jm = 0.02  # Inertia
            self.lc = 0.25  # Lever arm
            self.l0 = 0.5  # Natural leg length
            self.k = 2000.0  # Spring stiffness
            self.g = 9.81  # Gravity
            self.thrust_max = 0.314

            self.InitialState = InitialState()
            self.ActuatorParam = ActuatorParam()

    class PPOParamConfig:
        def __init__(self):
            self.maximum_step = 400  # 每个episode最大步数
            self.gamma = 0.99
            self.lam = 0.95
            self.epsilon = 0.2
            self.entropy_coef = 0.01
            self.batch_size = 256  # 小一点的 batch size 适应单环境

    class ActorParamConfig:
        def __init__(self):
            # 状态维度: [x, y, theta, dx, dy, dtheta, target_x, target_y] = 8
            self.state_dim = 8
            self.act_layers_num = 64
            self.actor_update_frequency = 5
            self.actuator_num = 2
            self.actor_lr = 3e-4
            # 离散动作选择：推力大小 (牛顿)
            self.action_choice = [0.0, 5.0, 10.0]
            self.action_scale = 1.0

    class CriticParamConfig:
        def __init__(self):
            self.state_dim = 8
            self.critic_layers_num = 64
            self.critic_update_frequency = 5
            self.critic_lr = 1e-3