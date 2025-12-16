import torch


class JumpingRobot_Env:
    def __init__(self, Env_Config, Robot_Config, PPO_Config):
        """初始化环境变量"""
        """
        
        警告！！！所有物理变量没法在这里定义，如果你要更改某一些物理变量，请重新推导动力学方程并更改step函数内的动力学更新代码
        
        """

        self.dt = Env_Config.EnvParam.dt
        self.sub_step = Env_Config.EnvParam.sub_step
        self.device = Env_Config.EnvParam.device
        self.train = Env_Config.EnvParam.train
        self.agents_num = (Env_Config.EnvParam.agents_num - Env_Config.EnvParam.agents_num_in_play) * self.train + \
                          Env_Config.EnvParam.agents_num_in_play

        """初始化机器人变量"""
        self.actuator_num = Robot_Config.ActuatorParam.actuator_num

        """初始化机器人位姿参数"""
        self.initial_x_range = Robot_Config.InitialState.initial_x_range
        self.initial_y_range = Robot_Config.InitialState.initial_y_range
        self.initial_theta_range = Robot_Config.InitialState.initial_theta_range
        self.initial_x_dot_range = Robot_Config.InitialState.initial_x_dot_range
        self.initial_y_dot_range = Robot_Config.InitialState.initial_y_dot_range
        self.initial_theta_dot = Robot_Config.InitialState.initial_theta_dot

        self.initial_target_x_range = Robot_Config.InitialState.initial_target_x_range
        self.initial_target_y_range = Robot_Config.InitialState.initial_target_y_range


        self.x = self.initial_x_range*(2*torch.rand((self.agents_num, 1), device=self.device)-1)
        self.y = self.initial_y_range*torch.rand((self.agents_num, 1), device=self.device) # 位置不能为负数，否则会穿地
        self.tht = self.initial_theta_range*(2*torch.rand((self.agents_num, 1), device=self.device)-1)
        self.x_dot = self.initial_x_dot_range*(2*torch.rand((self.agents_num, 1), device=self.device)-1)
        self.y_dot = self.initial_y_dot_range * torch.rand((self.agents_num, 1), device=self.device)+4
        self.tht_dot = self.initial_theta_dot*(2*torch.rand((self.agents_num, 1), device=self.device)-1)

        self.target_x = self.initial_target_x_range *(2*torch.rand((self.agents_num, 1), device=self.device)-1)
        self.target_y = self.initial_target_y_range *torch.rand((self.agents_num, 1), device=self.device) # 位置不能为负数，否则会穿地

        """初始化额外机器人参数"""
        self.action = torch.zeros((self.agents_num, self.actuator_num), device=self.device)  # 动作

        """奖励和"""
        self.max_step = PPO_Config.PPOParam.maximum_step
        self.tracking_reward_sum = 0
        self.torque_reward_sum = 0
        self.termination_reward_sum = 0

    """-------------------以上均为初始化代码-----------------------"""
    """-------------------以上均为初始化代码-----------------------"""
    """-------------------以上均为初始化代码-----------------------"""

    """-------------------以下均为环境运行代码-----------------------"""
    """-------------------以下均为环境运行代码-----------------------"""
    """-------------------以下均为环境运行代码-----------------------"""

    def reset(self, index):
        # 从随机位置开始
        self.x[index] = self.initial_x_range * (2 * torch.rand(device=self.device, size=(len(index), 1)) - 1)
        self.y[index] = self.initial_y_range * torch.rand(device=self.device, size=(len(index), 1))  # 位置不能为负数，否则会穿地
        self.tht[index] = self.initial_theta_range * (2 * torch.rand(device=self.device, size=(len(index), 1)) - 1)
        self.x_dot[index] = self.initial_x_dot_range * (2 * torch.rand(device=self.device, size=(len(index), 1)) - 1)
        # self.y_dot[index] = self.initial_y_dot_range * (2 * torch.rand(device=self.device, size=(len(index), 1)) - 1)
        self.y_dot[index] = self.initial_y_dot_range * torch.rand(device=self.device, size=(len(index), 1))+4
        self.tht_dot[index] = self.initial_theta_dot * (2 * torch.rand(device=self.device, size=(len(index), 1)) - 1)
        self.target_x[index] = self.initial_target_x_range * (2 * torch.rand(device=self.device, size=(len(index), 1)) - 1)
        self.target_y[index] = self.initial_target_y_range * torch.rand(device=self.device, size=(len(index), 1))  # 位置不能为负数，否则会穿地

    def reset_all(self):
        self.x = self.initial_x_range * (2 * torch.rand((self.agents_num, 1), device=self.device) - 1)
        self.y = self.initial_y_range * torch.rand((self.agents_num, 1), device=self.device)  # 位置不能为负数，否则会穿地
        self.tht = self.initial_theta_range * (2 * torch.rand((self.agents_num, 1), device=self.device) - 1)
        self.x_dot = self.initial_x_dot_range * (2 * torch.rand((self.agents_num, 1), device=self.device) - 1)
        # self.y_dot = self.initial_y_dot_range * (2 * torch.rand((self.agents_num, 1), device=self.device) - 1)
        self.y_dot = self.initial_y_dot_range *torch.rand((self.agents_num, 1), device=self.device)+4
        self.tht_dot = self.initial_theta_dot * (2 * torch.rand((self.agents_num, 1), device=self.device) - 1)

        self.target_x = self.initial_target_x_range * (2 * torch.rand((self.agents_num, 1), device=self.device) - 1)
        self.target_y = self.initial_target_y_range * torch.rand((self.agents_num, 1), device=self.device)  # 位置不能为负数，否则会穿地

    """机器人状态更新"""

    def step(self, force):
        real_dt = self.dt / self.sub_step
        self.action = force
        F1 = force[:, 0].view(-1, 1)
        F2 = force[:, 1].view(-1, 1)
        for i in range(self.sub_step):
            # 动力学更新 - 根据提供的公式修正
            x_dotdot = (10 * F1 * torch.sin(self.tht)) / 3 - 5 * F1 * torch.cos(self.tht) + \
                       (10 * F2 * torch.sin(self.tht)) / 3 + 5 * F2 * torch.cos(self.tht) + \
                       (2 * torch.sin(self.tht) * self.tht_dot ** 2) / 3

            y_dotdot = 5 * F1 * torch.sin(self.tht) + (10 * F1 * torch.cos(self.tht)) / 3 - \
                       5 * F2 * torch.sin(self.tht) + (10 * F2 * torch.cos(self.tht)) / 3 + \
                       (2 * torch.cos(self.tht) * self.tht_dot ** 2) / 3 - 9.81
            tht_dotdot = (15 * F1) / 2 - (15 * F2) / 2

            self.x_dot += x_dotdot * real_dt
            self.y_dot += y_dotdot * real_dt
            self.tht_dot += tht_dotdot * real_dt
            self.x += self.x_dot * real_dt
            self.y += self.y_dot * real_dt
            self.tht += self.tht_dot * real_dt



    def get_current_observations(self):
        current_state = torch.concatenate((self.x,
                                           self.y,
                                           self.tht,
                                           self.x_dot,
                                           self.y_dot,
                                           self.tht_dot,
                                           self.target_x,
                                           self.target_y), dim=-1)
        self.current_tht = self.tht.clone()
        self.current_x = self.x.clone()
        self.current_y = self.y.clone()



        # #——————————————————————获取额外机器人状态结束————————————————————————————————##

        return current_state

    def get_next_observations(self):

        self.next_x = self.x.clone()
        self.next_y = self.y.clone()
        self.next_tht = self.tht.clone()
        self.next_x_dot = self.x_dot.clone()
        self.next_y_dot = self.y_dot.clone()
        self.next_tht_dot = self.tht_dot.clone()

        next_state = torch.concatenate((self.next_x,
                                           self.next_y,
                                           self.next_tht,
                                           self.next_x_dot,
                                           self.next_y_dot,
                                           self.next_tht_dot,
                                        self.target_x,
                                        self.target_y
                                        ), dim=-1)

        return next_state

    """-------------------以上均为环境运行代码-----------------------"""
    """-------------------以上均为环境运行代码-----------------------"""
    """-------------------以上均为环境运行代码-----------------------"""

    """-------------------以下均为奖励计算代码-----------------------"""
    """-------------------以下均为奖励计算代码-----------------------"""
    """-------------------以下均为奖励计算代码-----------------------"""


    def target_tracking_reward(self):
        # prev_p = torch.concatenate((self.current_x,self.current_y),dim=-1)
        # next_p = torch.concatenate((self.next_x,self.next_y),dim=-1)
        # target_p = torch.concatenate((self.target_x,self.target_y),dim=-1)
        #
        # potential1 = (prev_p - target_p).norm(dim=-1,keepdim=True)
        # potential2 = (next_p - target_p).norm(dim=-1, keepdim=True)

        potential1 = torch.abs(self.current_x - self.target_x)
        potential2 =  torch.abs(self.next_x -self.target_x)

        target_tracking_reward = 10* (potential1-potential2)

        potential1 = torch.abs(self.current_tht - 0)
        potential2 =  torch.abs(self.next_tht -0)
        target_tracking_reward += 10 * (potential1 - potential2)
        return target_tracking_reward


    def effort_penalty_reward(self):
        joint_effort_reward = -0.0 * torch.abs(self.action.sum(dim=1, keepdim=True))
        return joint_effort_reward

    def termination_reward(self):
        over1 = self.next_y < 0
        over2 = self.next_y >10
        over3 = torch.abs(self.next_x)>6
        over4 = torch.abs(self.tht)>0.6

        self.over = over2 | over3 | over4
        reward = -1 * self.over.float()



        success_over = ((torch.abs(self.next_x-self.target_x)<0.1) &  (torch.abs(self.next_y-self.target_y)<0.1) &
                        (self.next_y_dot<0))



        reward += 10 * success_over.float()


        self.over = self.over | success_over | over1

        return reward

    def compute_reward(self):
        reward = 0
        reward += 1 * self.target_tracking_reward()
        reward += 1 * self.effort_penalty_reward()
        reward += 1 * self.termination_reward()
        reward += 0.2
        self.tracking_reward_sum += self.target_tracking_reward().mean().item() / self.max_step
        self.torque_reward_sum += self.effort_penalty_reward().mean().item() / self.max_step
        self.termination_reward_sum += self.termination_reward().mean().item() / self.max_step

        return reward, self.over.float()

    def print_reward_sum(self):
        print(f"target_tracking_reward_sum: {self.tracking_reward_sum:.4f}")
        print(f"torque_reward_sum: {self.torque_reward_sum:.4f}")
        print(f"termination_reward_sum: {self.termination_reward_sum:.4f}")
        self.tracking_reward_sum = 0
        self.torque_reward_sum = 0
        self.termination_reward_sum = 0
