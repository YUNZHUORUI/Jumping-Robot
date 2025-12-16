class Env_Config:
    class EnvParam:  # 训练环境的参数
        agents_num = 2000
        agents_num_in_play = 1
        dt = 0.03
        sub_step = 4
        train = 0
        device = 'cuda'
class Robot_Config:
    class ActuatorParam:  # 机器人的参数
        actuator_num = 2

    class InitialState:
        initial_x_range = 0.5
        initial_y_range = 0.5
        initial_theta_range = (15)/180*3.1415926
        initial_x_dot_range = 3
        initial_y_dot_range = 5
        initial_theta_dot = 0.5

        initial_target_x_range = 2
        initial_target_y_range = 2


class PPO_Config:
    class CriticParam:  # Critic 神经网络 参数
        state_dim = 8
        critic_layers_num = 128
        critic_lr = 5e-4
        critic_update_frequency = 300

    class ActorParam:  # Actor 神经网络 参数
        action_scale = 1
        action_choice = [0,1]
        act_layers_num = 128
        actuator_num = Robot_Config.ActuatorParam.actuator_num
        actor_lr = 5e-4
        actor_update_frequency = 100

    class PPOParam:  # 强化学习 PPO算法 参数
        gamma = 0.99
        lam = 0.95
        epsilon = 0.2
        maximum_step = 100
        episode = 500
        entropy_coef = 0  # positive means std increase, else decrease
        batch_size = 20000

