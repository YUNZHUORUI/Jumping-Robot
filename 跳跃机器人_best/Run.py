from utils.Env.JumpingRobot_Env import JumpingRobot_Env
from utils.PPO.Actor_Critic_Discrete import Actor_Critic
from utils.Config.Config import *

maximum_step = PPO_Config.PPOParam.maximum_step
episode = PPO_Config.PPOParam.episode
train = Env_Config.EnvParam.train
AC = Actor_Critic(PPO_Config, Env_Config)
if not train:
    # AC.load_best_model()
    AC.load_each_epi_model()
env = JumpingRobot_Env(Env_Config, Robot_Config, PPO_Config)
import torch
import matplotlib.pyplot as plt
import numpy as np

for epi in range(episode):
    print(f"===================episode: {epi}===================")
    angle = []
    for step in range(maximum_step):
        """获取当前状态"""
        state = env.get_current_observations()

        """做动作"""
        action_index,force = AC.sample_action(state)



        if not train:
            x = state[0,0].item()
            y = state[0,1].item()
            tht = state[0,2].item()


            target_x = state[0,-2].item()
            target_y = state[0,-1].item()

            x_beam = [x,x+np.sin(tht)]
            y_beam = [y,y+np.cos(tht)]

            plt.plot(x,y, '*', markersize=11)
            plt.plot(x_beam, y_beam, 'b', markersize=11)



            plt.plot(target_x, target_y, 'r*', markersize=10)


            plt.xlim(-10, 10)
            plt.ylim(-0.2, 8)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show(block= False)
            plt.pause(0.01)
            plt.clf()

        """更新环境"""
        env.step(force)
        """获取下一个状态"""
        next_state = env.get_next_observations()

        """计算奖励 判断是否结束"""
        reward, over = env.compute_reward()

        """存储经验"""
        if train:
            AC.store_experience(state,
                                action_index,
                                next_state,
                                reward,
                                over,
                                step)

        """重置挂掉的机器人"""
        env.reset(torch.nonzero(over.flatten()).flatten())
    """每个回合结束后训练一次"""
    if train:
        AC.update()
        env.print_reward_sum()
