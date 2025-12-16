import os
import copy
import matplotlib.animation as animation
from pathlib import Path

# ================= 1. 环境设置与依赖导入 =================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.patches as patches
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 启用3060极限性能优化
cudnn.benchmark = True
torch.cuda.empty_cache()

# ================= 2. 全局物理参数 =================
Pool_length = 0.6
OBSTACLES = [(-0.3, 0.3), (0.3, 0.3), (0, -0.3), (0, 0)]
R_obstacle = 0.1
fish_length = 0.05 / 2
TARGET_RADIUS = 0.02
TARGET_SPEED_INIT = 0.02  # 基础速度

# 探索关闭阈值：当成功率极高时，关闭随机性，巩固策略
EXPLORATION_STOP_THRESHOLD = 93.0

# 强制指定保存路径
gif_save_folder = Path(r"C:\Users\田进宁\OneDrive - GTIIT\桌面\GIF")
gif_save_folder.mkdir(parents=True, exist_ok=True)
fail_gif_folder = gif_save_folder / "fail_cases"
fail_gif_folder.mkdir(parents=True, exist_ok=True)


# ================= 3. 辅助函数 =================
def generate_suitable_pos(range_val, dense=100):
    """生成合法的初始坐标候选列表"""
    x_l = np.linspace(-range_val, range_val, dense)
    y_l = np.linspace(-range_val, range_val, dense)
    suitable_pos = []
    for x in x_l:
        for y in y_l:
            flag = True
            for obs in OBSTACLES:
                dist = np.sqrt((x - obs[0]) ** 2 + (y - obs[1]) ** 2)
                if dist < R_obstacle + 0.05:
                    flag = False
                    break
            if flag:
                suitable_pos.append((x, y))
    return suitable_pos


def generate_target_pos(dense=100):
    """生成合法的目标位置候选列表"""
    x_list = np.linspace(-0.6, 0.6, dense)
    y_list = np.linspace(-0.6, 0.6, dense)
    suitable_target_pos = []
    for x in x_list:
        for y in y_list:
            flag = True
            for obs in OBSTACLES:
                dist = np.sqrt((x - obs[0]) ** 2 + (y - obs[1]) ** 2)
                if dist < R_obstacle + TARGET_RADIUS + 0.01:
                    flag = False
                    break
            if flag:
                suitable_target_pos.append((x, y))
    return suitable_target_pos


# ================= 4. 配置类（动态生成实例，避免全局污染） =================
class Env_Config:
    def __init__(self, agents_num=8000, train=1):
        self.agents_num = agents_num
        self.agents_num_in_play = 1
        self.dt = 1
        self.sub_step = None
        self.train = train
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Robot_Config:
    def __init__(self, initial_pos_list, initial_target_pos_list):
        self.ActuatorParam = type('ActuatorParam', (), {'actuator_num': 1})
        self.InitialState = type('InitialState', (), {
            'initial_pos_list': initial_pos_list,
            'initial_theta_range': 3.14,
            'initial_target_pos_list': initial_target_pos_list
        })
        self.ObstacleParam = type('ObstacleParam', (), {
            'R_obstacle': 0.1,
            'obstacle_pos': [[-0.3, 0.3], [0.3, 0.3], [0, -0.3]]
        })


class PPO_Config:
    def __init__(self):
        self.CriticParam = type('CriticParam', (), {
            'state_dim': 7,
            'critic_layers_num': 256,
            'critic_lr': 4e-4,
            'critic_update_frequency': 100
        })
        self.ActorParam = type('ActorParam', (), {
            'action_scale': 0,
            'action_choice': [0, 1, 2],
            'act_layers_num': 256,
            'actuator_num': 1,
            'actor_lr': 4e-4,
            'actor_update_frequency': 100
        })
        self.PPOParam = type('PPOParam', (), {
            'gamma': 0.99,
            'lam': 0.99,
            'epsilon': 0.1,
            'maximum_step': 30,
            'episode': 70,
            'entropy_coef': 0.01,
            'batch_size': 8000,
            'drop_threshold': 0.1,
            'history_model_num': 56
        })


# ================= 5. PPO核心类 =================
class Agent_State_Buffer:
    def __init__(self, state_dim, actuator_num, agent_num, max_step, device):
        self.state_dim = state_dim
        self.actuator_num = actuator_num
        self.agent_num = agent_num
        self.max_step = max_step
        self.device = device
        self.state_buffer = torch.zeros((max_step, agent_num, state_dim), device=self.device)
        self.action_index_buffer = torch.zeros((max_step, agent_num, actuator_num), device=self.device,
                                               dtype=torch.long)
        self.next_state_buffer = torch.zeros((max_step, agent_num, state_dim), device=self.device)
        self.reward_buffer = torch.zeros((max_step, agent_num, 1), device=self.device)
        self.over_buffer = torch.zeros((max_step, agent_num, 1), device=self.device)
        self.GAE_buffer = torch.zeros((max_step, agent_num, 1), device=self.device)

    def compute_GAE(self, critic_net, gamma, lam):
        with torch.no_grad():
            target_value = self.reward_buffer + (1 - self.over_buffer) * gamma * critic_net(self.next_state_buffer)
            current_value = critic_net(self.state_buffer)
        GAE = (target_value - current_value)
        advantage = 0
        index = -1
        for delta in GAE.flip(0):
            advantage = gamma * lam * advantage * (1 - self.over_buffer[index]) + delta
            self.GAE_buffer[index] = advantage
            index += -1
        if self.GAE_buffer.std() > 1e-6:
            self.GAE_buffer = (self.GAE_buffer - self.GAE_buffer.mean()) / self.GAE_buffer.std()

    def store_state(self, current_state, current_step):
        self.state_buffer[current_step] = current_state

    def store_action_index(self, current_action_index, current_step):
        self.action_index_buffer[current_step] = current_action_index.long()

    def store_next_state(self, next_state_, current_step):
        self.next_state_buffer[current_step] = next_state_

    def store_reward(self, current_reward, current_step):
        self.reward_buffer[current_step] = current_reward

    def store_over(self, current_over, current_step):
        self.over_buffer[current_step] = current_over


class Actor(torch.nn.Module):
    def __init__(self, state_dim, num_layers, actuator_num, action_choice, action_scale=1):
        super(Actor, self).__init__()
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.actuator_num = actuator_num
        self.action_choice = action_choice
        self.action_scale = action_scale
        self.fc1_x = torch.nn.Linear(self.state_dim, self.num_layers)
        self.fc2_x = torch.nn.Linear(self.num_layers, self.num_layers)
        self.fc3_x = torch.nn.Linear(self.num_layers, self.num_layers)
        self.fc4_x = torch.nn.Linear(self.num_layers, actuator_num * len(action_choice))

    def forward(self, input_):
        x_ = torch.nn.functional.elu(self.fc1_x(input_))
        x_ = torch.nn.functional.elu(self.fc2_x(x_))
        x_ = torch.nn.functional.elu(self.fc3_x(x_))
        x_ = torch.nn.functional.elu(self.fc4_x(x_))
        output = torch.nn.functional.softmax(x_.reshape(input_.shape[0], self.actuator_num, len(self.action_choice)),
                                             dim=-1)
        return output


class Critic(torch.nn.Module):
    def __init__(self, state_dim, num_layers):
        super(Critic, self).__init__()
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.fc1_x = torch.nn.Linear(self.state_dim, self.num_layers)
        self.fc2_x = torch.nn.Linear(self.num_layers, self.num_layers)
        self.fc3_x = torch.nn.Linear(self.num_layers, self.num_layers)
        self.fc4_x = torch.nn.Linear(self.num_layers, 1)

    def forward(self, input_):
        # ========== 修正这里 ==========
        x_ = torch.nn.functional.elu(self.fc1_x(input_))  # 第一层：input_ → x_
        x_ = torch.nn.functional.elu(self.fc2_x(x_))      # 第二层：x_（而非input_）
        x_ = torch.nn.functional.elu(self.fc3_x(x_))      # 第三层：x_（而非input_）
        output = self.fc4_x(x_)
        return output


class Actor_Critic:
    def __init__(self, ppo_config, env_config):
        self.agent_num = env_config.agents_num
        self.device = env_config.device
        self.maximum_step = ppo_config.PPOParam.maximum_step
        self.train = env_config.train

        # 动态计算Buffer大小
        current_agents = (
                                     env_config.agents_num - env_config.agents_num_in_play) * self.train + env_config.agents_num_in_play
        self.agent_num = current_agents

        # PPO参数
        self.gamma = ppo_config.PPOParam.gamma
        self.lam = ppo_config.PPOParam.lam
        self.epsilon = ppo_config.PPOParam.epsilon
        self.entropy_coef = ppo_config.PPOParam.entropy_coef
        self.batch_size = ppo_config.PPOParam.batch_size
        self.drop_threshold = ppo_config.PPOParam.drop_threshold
        self.history_model_num = ppo_config.PPOParam.history_model_num
        self.loss_fn = torch.nn.MSELoss()

        # 网络参数
        self.state_dim = ppo_config.CriticParam.state_dim
        self.critic_num_layers = ppo_config.CriticParam.critic_layers_num
        self.critic_lr = ppo_config.CriticParam.critic_lr
        self.critic_update_frequency = ppo_config.CriticParam.critic_update_frequency
        self.action_scale = ppo_config.ActorParam.action_scale
        self.actor_num_layers = ppo_config.ActorParam.act_layers_num
        self.actor_update_frequency = ppo_config.ActorParam.actor_update_frequency
        self.actuator_num = ppo_config.ActorParam.actuator_num
        self.actor_lr = ppo_config.ActorParam.actor_lr
        self.action_choice = torch.tensor(ppo_config.ActorParam.action_choice, device=self.device)

        # 初始化网络
        self.actor = Actor(self.state_dim, self.actor_num_layers, self.actuator_num, self.action_choice,
                           self.action_scale).to(self.device)
        self.critic = Critic(self.state_dim, self.critic_num_layers).to(self.device)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # 经验缓冲区
        self.Buffer = Agent_State_Buffer(self.state_dim, self.actuator_num, self.agent_num, self.maximum_step,
                                         self.device)

        # 模型跟踪
        self.top_rewards = [-float('inf'), -float('inf')]
        self.top_success_rates = [-float('inf'), -float('inf')]
        self.history_models = []
        self.current_best_success_rate = 0.0

        # 探索控制
        self.explore = self.train
        self.exploration_stopped = False

        # 测试模式直接加载模型
        if not self.train:
            self.load_best_model()
            print("Testing Mode: Loaded trained model")

    def decay_lr(self, progress):
        factor = 1.0 - progress * 0.66
        new_actor_lr = self.actor_lr * factor
        new_critic_lr = self.critic_lr * factor
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_critic_lr

    def boost_for_new_stage(self):
        self.entropy_coef = 0.025
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = 4e-4
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = 4e-4
        print(f"🚀 New Stage Boost! Entropy set to {self.entropy_coef}, LR reset to 4e-4")

    def sample_action(self, state_):
        with torch.no_grad():
            action_prob = self.actor(state_)

        if self.train and self.explore:
            action_index_ = torch.multinomial(action_prob.view(-1, len(self.action_choice)), 1)
        else:
            action_index_ = torch.argmax(action_prob, dim=-1, keepdim=True)

        action_output = self.action_choice[action_index_]
        current_batch_size = state_.shape[0]
        action_index_ = action_index_.view(current_batch_size, self.actuator_num)
        action_output = action_output.view(current_batch_size, self.actuator_num)

        return action_index_, self.action_scale * action_output

    def stop_exploration(self):
        if self.explore and self.train:
            self.explore = False
            self.exploration_stopped = True
            self.entropy_coef = 0.001
            print(f"\n🛑 Exploration Disabled! Success Rate ≥ {EXPLORATION_STOP_THRESHOLD}%")

    def store_experience(self, state_, action, next_state_, reward_, over_, current_step):
        self.Buffer.store_state(state_, current_step)
        self.Buffer.store_action_index(action, current_step)
        self.Buffer.store_next_state(next_state_, current_step)
        self.Buffer.store_reward(reward_, current_step)
        self.Buffer.store_over(over_, current_step)

    def check_success_rate_drop(self, current_success_rate):
        if self.current_best_success_rate == 0.0:
            self.current_best_success_rate = current_success_rate
            return False
        drop_ratio = (self.current_best_success_rate - current_success_rate) / self.current_best_success_rate
        if drop_ratio > self.drop_threshold:
            print(f"\n⚠️ Success rate dropped by {drop_ratio * 100:.1f}% (over {self.drop_threshold * 100}%)!")
            return True
        return False

    def recover_prev_prev_model(self, back_steps=2):
        reward_models = [m for m in self.history_models if m['type'] == 'reward']
        if len(reward_models) >= back_steps:
            target_idx = max(0, len(reward_models) - back_steps)
            target_model = reward_models[target_idx]
            self.actor.load_state_dict(torch.load(target_model['actor_path'], map_location=self.device))
            self.critic.load_state_dict(torch.load(target_model['critic_path'], map_location=self.device))
            self.current_best_success_rate = target_model['success_rate']
            print(f"✅ Recovered to model at index {target_idx}")
            return True
        elif len(reward_models) >= 1:
            target_model = reward_models[0]
            self.actor.load_state_dict(torch.load(target_model['actor_path'], map_location=self.device))
            self.critic.load_state_dict(torch.load(target_model['critic_path'], map_location=self.device))
            print(f"✅ Recovered to earliest model")
            return True
        else:
            return False

    def update(self, current_success_rate=None):
        buffer = self.Buffer
        state_ = buffer.state_buffer.view(-1, self.state_dim)
        action_index_ = buffer.action_index_buffer.view(-1, self.actuator_num).unsqueeze(2)
        next_state_ = buffer.next_state_buffer.view(-1, self.state_dim)
        reward_ = buffer.reward_buffer.view(-1, 1)
        over_ = buffer.over_buffer.view(-1, 1)
        reward_sum = reward_.mean().item()

        if current_success_rate and self.check_success_rate_drop(current_success_rate):
            back_steps = max(2, len(self.history_models) - 56) if len(self.history_models) > 56 else 2
            self.recover_prev_prev_model(back_steps=back_steps)
            reward_models = [m for m in self.history_models if m['type'] == 'reward']
            if len(reward_models) >= back_steps:
                target_idx = max(0, len(reward_models) - back_steps)
                reward_sum = reward_models[target_idx]['reward']
                current_success_rate = reward_models[target_idx]['success_rate']

        with torch.no_grad():
            action_prob = self.actor(state_)
            old_prob = action_prob.gather(index=action_index_.long(), dim=-1)
            old_prob = old_prob.log().sum(dim=1)

        # 更新Critic
        for _ in range(self.critic_update_frequency):
            idx = torch.randperm(len(state_), device=state_.device)[:self.batch_size]
            s_batch, ns_batch, r_batch, o_batch = state_[idx], next_state_[idx], reward_[idx], over_[idx]
            value = self.critic(s_batch)
            with torch.no_grad():
                next_value = self.critic(ns_batch)
                target_value = r_batch + self.gamma * next_value * (1 - o_batch)
            critic_loss = self.loss_fn(value, target_value)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # 计算GAE
        buffer.compute_GAE(self.critic, self.gamma, self.lam)
        GAE = buffer.GAE_buffer.view(-1, 1)

        # 更新Actor
        for _ in range(self.actor_update_frequency):
            idx = torch.randperm(len(state_), device=state_.device)[:self.batch_size]
            s_batch, a_batch, ns_batch = state_[idx], action_index_[idx], next_state_[idx]
            gae_batch, old_prob_batch = GAE[idx], old_prob[idx]

            action_prob = self.actor(s_batch)
            new_prob = action_prob.gather(index=a_batch.long(), dim=-1)
            new_prob = new_prob.log().sum(dim=1)

            ratio = torch.exp(new_prob - old_prob_batch)
            surr1 = ratio * gae_batch
            surr2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * gae_batch
            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # 打印状态
        explore_status = "✅ Explore" if self.explore else "❌ No Explore"
        print(f"Reward: {reward_sum:.4f} | Success: {current_success_rate:.2f}% | {explore_status}")

        # 保存最优模型
        if reward_sum > self.top_rewards[0]:
            old_first = copy.deepcopy({
                'type': 'reward', 'rank': 1,
                'actor_path': f'model/actor_best_1_reward.pth',
                'critic_path': f'model/critic_best_1_reward.pth',
                'success_rate': self.top_success_rates[0],
                'reward': self.top_rewards[0]
            })
            old_second = copy.deepcopy({
                'type': 'reward', 'rank': 2,
                'actor_path': f'model/actor_best_2_reward.pth',
                'critic_path': f'model/critic_best_2_reward.pth',
                'success_rate': self.top_success_rates[1],
                'reward': self.top_rewards[1]
            })
            self.top_rewards[1] = self.top_rewards[0]
            self.top_rewards[0] = reward_sum
            self.save_best_model(rank=1, suffix='_reward')
            self.save_best_model(rank=2, suffix='_reward')
            if old_first['reward'] != -float('inf'):
                self.history_models.append(old_first)
            if old_second['reward'] != -float('inf'):
                self.history_models.append(old_second)
            if len(self.history_models) > self.history_model_num:
                self.history_models.pop(0)
            if current_success_rate:
                self.current_best_success_rate = max(self.current_best_success_rate, current_success_rate)
            print(f"→ Saved Reward Best 1&2")
        elif reward_sum > self.top_rewards[1] + 0.1:
            old_second = copy.deepcopy({
                'type': 'reward', 'rank': 2,
                'actor_path': f'model/actor_best_2_reward.pth',
                'critic_path': f'model/critic_best_2_reward.pth',
                'success_rate': self.top_success_rates[1],
                'reward': self.top_rewards[1]
            })
            self.top_rewards[1] = reward_sum
            self.save_best_model(rank=2, suffix='_reward')
            if old_second['reward'] != -float('inf'):
                self.history_models.append(old_second)
            if len(self.history_models) > self.history_model_num:
                self.history_models.pop(0)
            print(f"→ Saved Reward Best 2")

        if current_success_rate:
            if current_success_rate > self.top_success_rates[0]:
                self.top_success_rates[1] = self.top_success_rates[0]
                self.top_success_rates[0] = current_success_rate
                self.save_best_model(rank=1, suffix='_success')
                self.save_best_model(rank=2, suffix='_success')
                self.history_models.append({
                    'type': 'success', 'rank': 1,
                    'actor_path': f'model/actor_best_1_success.pth',
                    'critic_path': f'model/critic_best_1_success.pth',
                    'success_rate': current_success_rate,
                    'reward': reward_sum
                })
                if len(self.history_models) > self.history_model_num:
                    self.history_models.pop(0)
                print(f"→ Saved Success Best 1&2")
            elif current_success_rate > self.top_success_rates[1]:
                self.top_success_rates[1] = current_success_rate
                self.save_best_model(rank=2, suffix='_success')
                self.history_models.append({
                    'type': 'success', 'rank': 2,
                    'actor_path': f'model/actor_best_2_success.pth',
                    'critic_path': f'model/critic_best_2_success.pth',
                    'success_rate': current_success_rate,
                    'reward': reward_sum
                })
                if len(self.history_models) > self.history_model_num:
                    self.history_models.pop(0)
                print(f"→ Saved Success Best 2")
            self.prev_success_rate = current_success_rate

    def save_best_model(self, rank=1, suffix=''):
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(self.actor.state_dict(), f'model/actor_best_{rank}{suffix}.pth')
        torch.save(self.critic.state_dict(), f'model/critic_best_{rank}{suffix}.pth')

    def load_best_model(self, rank=1, suffix='_reward'):
        actor_path = f'model/actor_best_{rank}{suffix}.pth'
        critic_path = f'model/critic_best_{rank}{suffix}.pth'
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            print(f"Loaded Best Model {rank}{suffix} successfully!")
            return True
        else:
            print(f"Best Model {rank}{suffix} not found. Please train first!")
            return False


# ================= 6. 环境类 (GPU加速 & 安全检查) =================
class Fish_Env:
    def __init__(self, env_config, robot_config, ppo_config):
        self.dt = env_config.dt
        self.device = env_config.device
        self.train = env_config.train
        self.agent_num = (
                                     env_config.agents_num - env_config.agents_num_in_play) * self.train + env_config.agents_num_in_play
        self.actuator_num = robot_config.ActuatorParam.actuator_num

        # 初始位置和目标位置
        self.initial_pos_list = torch.FloatTensor(robot_config.InitialState.initial_pos_list).to(self.device)
        self.initial_theta_range = robot_config.InitialState.initial_theta_range
        self.initial_target_pos_list = torch.FloatTensor(robot_config.InitialState.initial_target_pos_list).to(
            self.device)

        # 智能体状态
        self.pos = torch.zeros((self.agent_num, 2), device=self.device)
        self.tht = torch.zeros((self.agent_num, 1), device=self.device)
        self.target_pos = torch.zeros((self.agent_num, 2), device=self.device)
        self.target_vel = torch.zeros((self.agent_num, 2), device=self.device)
        self.time = torch.zeros((self.agent_num, 1), device=self.device)

        # 障碍物参数
        self.obstacle_pos = torch.FloatTensor(robot_config.ObstacleParam.obstacle_pos).to(self.device)
        self.R_obstacle = robot_config.ObstacleParam.R_obstacle
        self.max_step = ppo_config.PPOParam.maximum_step

    def update_initial_pos_list(self, new_pos_list):
        self.initial_pos_list = torch.FloatTensor(new_pos_list).to(self.device)

    # 核心安全检测：预测8步
    def check_start_safety(self, indices, steps=8):
        current_pos = self.pos[indices].clone()
        current_tht = self.tht[indices].clone()
        batch_size = len(indices)
        survived_any = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for action_idx in [0, 1, 2]:
            sim_pos = current_pos.clone()
            sim_tht = current_tht.clone()
            crashed = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            for _ in range(steps):
                turn_left = (action_idx == 0)
                turn_right = (action_idx == 2)
                orientation = torch.cat((-torch.sin(sim_tht), torch.cos(sim_tht)), dim=-1)
                sim_tht += self.dt * 0.5 * (float(turn_left) - float(turn_right))
                sim_tht = (sim_tht + 3.14) % (2 * 3.14) - 3.14
                speed = 0.06 if (turn_left or turn_right) else 0.1
                sim_pos += self.dt * speed * orientation

                obs_crash = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                for obs in self.obstacle_pos:
                    dist = (sim_pos - obs).norm(dim=-1)
                    obs_crash |= (dist < self.R_obstacle)
                wall_crash = (torch.abs(sim_pos[:, 0]) > Pool_length) | (torch.abs(sim_pos[:, 1]) > Pool_length)
                crashed |= (obs_crash | wall_crash)
            survived_any |= (~crashed)
        return survived_any

    def reset(self, index=None, reset_all=False, custom_speed=None):
        if reset_all:
            index = torch.arange(self.agent_num, device=self.device)

        retry_indices = index
        max_retries = 100
        for _ in range(max_retries):
            if len(retry_indices) == 0:
                break
            rand_indices = torch.randint(0, self.initial_pos_list.shape[0], (retry_indices.shape[0],),
                                         device=self.device)
            self.pos[retry_indices] = self.initial_pos_list[rand_indices]
            self.tht[retry_indices] = (torch.rand((retry_indices.shape[0], 1),
                                                  device=self.device) - 0.5) * 2 * self.initial_theta_range
            is_safe = self.check_start_safety(retry_indices)
            retry_indices = retry_indices[~is_safe]

        # 重置目标位置和速度
        self.target_pos[index] = self.initial_target_pos_list[
            torch.randint(0, self.initial_target_pos_list.shape[0], (index.shape[0],), device=self.device)]
        theta = torch.rand((index.shape[0], 1), device=self.device) * 2 * torch.pi
        speed_val = custom_speed if custom_speed is not None else TARGET_SPEED_INIT
        self.target_vel[index, 0] = speed_val * torch.cos(theta).squeeze()
        self.target_vel[index, 1] = speed_val * torch.sin(theta).squeeze()
        self.time[index] = 0

    def update_target(self):
        self.target_pos += self.target_vel * self.dt
        x_out = torch.abs(self.target_pos[:, 0]) > Pool_length - TARGET_RADIUS
        self.target_vel[x_out, 0] *= -1
        self.target_pos[x_out, 0] = torch.clamp(self.target_pos[x_out, 0], -Pool_length + TARGET_RADIUS,
                                                Pool_length - TARGET_RADIUS)
        y_out = torch.abs(self.target_pos[:, 1]) > Pool_length - TARGET_RADIUS
        self.target_vel[y_out, 1] *= -1
        self.target_pos[y_out, 1] = torch.clamp(self.target_pos[y_out, 1], -Pool_length + TARGET_RADIUS,
                                                Pool_length - TARGET_RADIUS)

        # 目标与障碍物碰撞检测
        for obs in self.obstacle_pos:
            obs_pos = obs.unsqueeze(0)
            dist = torch.norm(self.target_pos - obs_pos, dim=1)
            collide = dist < self.R_obstacle + TARGET_RADIUS
            if collide.any():
                normal = (self.target_pos[collide] - obs_pos) / dist[collide].unsqueeze(1)
                vel_dot_normal = (self.target_vel[collide] * normal).sum(dim=1, keepdim=True)
                self.target_vel[collide] = self.target_vel[collide] - 2 * vel_dot_normal * normal
                self.target_pos[collide] = obs_pos + normal * (self.R_obstacle + TARGET_RADIUS + 1e-4)

    def step(self, action_index_):
        self.update_target()
        turn_left = (action_index_ == 0)
        turn_forward = (action_index_ == 1)
        turn_right = (action_index_ == 2)
        orientation = torch.cat((-torch.sin(self.tht), torch.cos(self.tht)), dim=-1)
        self.tht += self.dt * 0.5 * (-turn_right.float() + turn_left.float())
        self.tht = (self.tht + 3.14) % (2 * 3.14) - 3.14
        self.pos += self.dt * (0.06 * (turn_left | turn_right).float() + 0.1 * turn_forward.float()) * orientation
        self.time += 1

    def get_current_observations(self):
        self.current_pos = self.pos.clone()
        self.current_tht = self.tht.clone()
        return torch.cat((self.pos, self.tht, self.target_pos, self.target_vel), dim=-1)

    def get_next_observations(self):
        self.next_pos = self.pos.clone()
        self.next_tht = self.tht.clone()
        return torch.cat((self.pos, self.tht, self.target_pos, self.target_vel), dim=-1)

    def target_tracking_reward(self):
        vec_to_target = self.target_pos - self.current_pos
        dist_to_target = vec_to_target.norm(dim=-1, keepdim=True)
        vec_to_target = vec_to_target / (dist_to_target + 1e-6)
        orientation = torch.cat((-torch.sin(self.current_tht), torch.cos(self.current_tht)), dim=-1)
        heading_reward = (orientation * vec_to_target).sum(dim=-1, keepdim=True)
        potential1 = (self.current_pos - self.target_pos).norm(dim=-1, keepdim=True)
        potential2 = (self.next_pos - self.target_pos).norm(dim=-1, keepdim=True)
        dist_reward = 1.0 * (potential1 - potential2)
        return dist_reward + 0.05 * heading_reward

    def termination_reward(self):
        over1 = (self.next_pos - self.obstacle_pos[0]).norm(dim=-1, keepdim=True) < self.R_obstacle
        over2 = (self.next_pos - self.obstacle_pos[1]).norm(dim=-1, keepdim=True) < self.R_obstacle
        over3 = (self.next_pos - self.obstacle_pos[2]).norm(dim=-1, keepdim=True) < self.R_obstacle
        over4 = torch.abs(self.next_pos[:, 0]).view(-1, 1) > 0.6
        over5 = torch.abs(self.next_pos[:, 1]).view(-1, 1) > 0.6
        crash_over = over1 | over2 | over3 | over4 | over5
        reward_ = -50 * crash_over.float()
        success_over = (self.next_pos - self.target_pos).norm(dim=-1, keepdim=True) < (TARGET_RADIUS + 0.03)
        reward_ += 50 * success_over.float()
        self.over_ = crash_over | success_over
        return reward_

    def compute_reward(self):
        reward_ = self.target_tracking_reward() + self.termination_reward() + 0.1
        return reward_, (self.over_ | (self.time > self.max_step * 2)).float()

    def get_failure_type(self):
        obs_crash = torch.zeros(self.agent_num, dtype=torch.bool, device=self.device)
        for obs in self.obstacle_pos:
            obs_crash |= ((self.next_pos - obs).norm(dim=-1) < self.R_obstacle)
        if obs_crash.any():
            return "撞障碍物"
        wall_crash = (torch.abs(self.next_pos[:, 0]) > 0.6) | (torch.abs(self.next_pos[:, 1]) > 0.6)
        if wall_crash.any():
            return "撞墙"
        if (self.time > self.max_step * 2).any():
            return "超时"
        return "追不上目标"


# ================= 7. 动画函数 =================
def animate_case(traj, obstacle_pos, R_obstacle, is_success=True, save_filename="animation.gif", fps=5):
    traj_x, traj_y, traj_theta = traj['x'], traj['y'], traj['theta']
    target_x, target_y = traj['target_x'], traj['target_y']
    failure_type = traj.get('failure_type', '')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-Pool_length - 0.1, Pool_length + 0.1)
    ax.set_ylim(-Pool_length - 0.1, Pool_length + 0.1)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.plot([-Pool_length, Pool_length, Pool_length, -Pool_length, -Pool_length],
            [-Pool_length, -Pool_length, Pool_length, Pool_length, -Pool_length], 'k-', linewidth=1.5)
    for obs in obstacle_pos:
        ax.add_patch(patches.Circle(obs, R_obstacle, color='magenta', alpha=0.6))

    trajectory_line, = ax.plot([], [], 'b-' if is_success else 'r-', linewidth=2)
    fish_pos, = ax.plot([], [], 'ro', markersize=8)
    target_pos_plot, = ax.plot([], [], 'yo', markersize=6)
    target_line, = ax.plot([], [], 'y--', linewidth=1)
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)
    fail_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=10, color='red')
    arrow_list = []

    def update(frame):
        nonlocal arrow_list
        trajectory_line.set_data(traj_x[:frame + 1], traj_y[:frame + 1])
        fish_pos.set_data([traj_x[frame]], [traj_y[frame]])
        for arrow in arrow_list:
            arrow.remove()
        arrow_list.clear()
        dx, dy = fish_length * np.cos(traj_theta[frame]), fish_length * np.sin(traj_theta[frame])
        arrow_list.append(ax.arrow(traj_x[frame], traj_y[frame], dx, dy, head_width=0.02, fc='red', ec='red'))
        target_pos_plot.set_data([target_x[frame]], [target_y[frame]])
        target_line.set_data(target_x[:frame + 1], target_y[:frame + 1])
        step_text.set_text(f'Step: {frame}/{len(traj_x) - 1}')
        if not is_success:
            fail_text.set_text(f'Failure: {failure_type}')
        if frame == len(traj_x) - 1:
            if is_success:
                ax.text(traj_x[frame], traj_y[frame], 'SUCCESS!', fontsize=12, color='green')
            else:
                ax.text(traj_x[frame], traj_y[frame], 'FAIL!', fontsize=12, color='red')
        return [trajectory_line, fish_pos, target_pos_plot, target_line, step_text, fail_text] + arrow_list

    ani = animation.FuncAnimation(fig, update, frames=len(traj_x), blit=False, interval=1000 / fps)
    ani.save(str(save_filename), writer='pillow', fps=fps)
    plt.close(fig)
    print(f"✅ Animation saved to {save_filename}")


# ================= 8. 主程序 =================
if __name__ == "__main__":
    # ========== 1. 训练阶段 ==========
    print("===== Start Training (Standard Fixed-Step PPO) =====")

    # 初始化训练配置
    train_env_config = Env_Config(agents_num=8000, train=1)
    ppo_config = PPO_Config()
    train_initial_pos = generate_suitable_pos(0.4, dense=100)
    train_target_pos = generate_target_pos(dense=100)
    train_robot_config = Robot_Config(train_initial_pos, train_target_pos)

    # 初始化训练环境和智能体
    ac_train = Actor_Critic(ppo_config, train_env_config)
    env_train = Fish_Env(train_env_config, train_robot_config, ppo_config)

    # 训练参数
    SUCCESS_THRESHOLD = 80.0
    CURRENT_RANGE = 0.4
    MAX_RANGE = 0.6
    RANGE_BREAKPOINT = 0.5
    STEP_SMALL = 0.02
    STEP_TINY = 0.01
    success_rates = []
    consecutive_success_cnt = 0
    current_target_speed = TARGET_SPEED_INIT

    # 训练循环
    for epi in tqdm(range(ppo_config.PPOParam.episode), desc="Training"):
        progress = epi / ppo_config.PPOParam.episode
        ac_train.decay_lr(progress)

        # 动态调整目标速度
        if epi < 40:
            current_target_speed = 0.015
        else:
            current_target_speed = 0.02

        # 重置环境
        env_train.reset(reset_all=True, custom_speed=current_target_speed)
        success_agents = torch.zeros(env_train.agent_num, dtype=torch.bool, device=env_train.device)
        total_agents = env_train.agent_num

        # 单轮步数循环
        for step in range(ppo_config.PPOParam.maximum_step):
            state = env_train.get_current_observations()
            action_index, _ = ac_train.sample_action(state)
            env_train.step(action_index)
            next_state = env_train.get_next_observations()
            reward, over = env_train.compute_reward()

            # 统计成功
            dist_to_target = (env_train.next_pos - env_train.target_pos).norm(dim=-1)
            current_success = dist_to_target < (TARGET_RADIUS + 0.03)
            success_agents |= current_success

            # 存储经验
            ac_train.store_experience(state, action_index, next_state, reward, over, step)

            # 部分重置完成的智能体
            done_indices = torch.nonzero(over.flatten()).flatten()
            if len(done_indices) > 0:
                env_train.reset(index=done_indices, reset_all=False, custom_speed=current_target_speed)

        # 计算成功率
        success_rate = (success_agents.sum().item() / total_agents) * 100
        success_rates.append(success_rate)

        # 关闭探索（达到阈值）
        if success_rate >= EXPLORATION_STOP_THRESHOLD and not ac_train.exploration_stopped:
            ac_train.stop_exploration()

        # 课程学习：扩展初始位置范围
        if success_rate >= SUCCESS_THRESHOLD:
            consecutive_success_cnt += 1
        else:
            consecutive_success_cnt = 0

        if consecutive_success_cnt >= 3 and CURRENT_RANGE < MAX_RANGE:
            step_size = STEP_SMALL if CURRENT_RANGE <= RANGE_BREAKPOINT else STEP_TINY
            CURRENT_RANGE = min(CURRENT_RANGE + step_size, MAX_RANGE)
            new_initial_pos = generate_suitable_pos(CURRENT_RANGE, dense=100)
            env_train.update_initial_pos_list(new_initial_pos)
            if not ac_train.exploration_stopped:
                ac_train.boost_for_new_stage()
            print(f"\n[Curriculum] Range expanded to ±{CURRENT_RANGE:.3f}")
            consecutive_success_cnt = 0

        # 更新模型
        ac_train.update(current_success_rate=success_rate)

    # 保存训练成功率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(success_rates, color='blue', label='Success Rate')
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title('Training Success Rate Curve')
    plt.legend()
    plt.savefig(str(gif_save_folder / 'training_success_real.png'))
    plt.close()

    # ========== 2. 测试阶段 ==========
    print("\n===== Start Final Testing =====")

    # 初始化测试配置（独立实例，不污染训练配置）
    test_env_config = Env_Config(agents_num=1, train=0)
    test_initial_pos = generate_suitable_pos(CURRENT_RANGE, dense=200)  # 更高密度
    test_target_pos = generate_target_pos(dense=200)
    test_robot_config = Robot_Config(test_initial_pos, test_target_pos)

    # 初始化测试环境和智能体
    ac_test = Actor_Critic(ppo_config, test_env_config)
    env_test = Fish_Env(test_env_config, test_robot_config, ppo_config)

    # 测试参数
    test_success_count = 0
    test_trajectories = []
    test_target_speed = 0.02  # 与训练后期一致

    # 测试循环
    for _ in tqdm(range(200), desc="Testing"):
        env_test.reset(reset_all=True, custom_speed=test_target_speed)
        init_state = env_test.get_current_observations()
        traj = {
            'x': [init_state[0, 0].item()],
            'y': [init_state[0, 1].item()],
            'theta': [init_state[0, 2].item()],
            'target_x': [init_state[0, 3].item()],
            'target_y': [init_state[0, 4].item()],
            'success': False,
            'failure_type': ''
        }

        # 测试步数翻倍
        for _ in range(ppo_config.PPOParam.maximum_step * 2):
            state = env_test.get_current_observations()
            action_index, _ = ac_test.sample_action(state)
            env_test.step(action_index)
            next_s = env_test.get_next_observations()

            # 记录轨迹
            traj['x'].append(next_s[0, 0].item())
            traj['y'].append(next_s[0, 1].item())
            traj['theta'].append(next_s[0, 2].item())
            traj['target_x'].append(next_s[0, 3].item())
            traj['target_y'].append(next_s[0, 4].item())

            # 检查终止条件
            _, over = env_test.compute_reward()
            if over[0]:
                if (env_test.next_pos - env_test.target_pos).norm() < (TARGET_RADIUS + 0.03):
                    test_success_count += 1
                    traj['success'] = True
                else:
                    traj['failure_type'] = env_test.get_failure_type()
                break

        test_trajectories.append(traj)

    # 输出测试结果
    test_success_rate = (test_success_count / 200) * 100
    print(f"Test Result: {test_success_count}/200 Success ({test_success_rate:.2f}%)")

    # 保存成功/失败案例动画
    success_cases = [t for t in test_trajectories if t['success']]
    if success_cases:
        animate_case(success_cases[0], OBSTACLES, R_obstacle, is_success=True,
                     save_filename=gif_save_folder / "test_success.gif")

    fail_cases = [t for t in test_trajectories if not t['success']]
    if fail_cases:
        animate_case(fail_cases[0], OBSTACLES, R_obstacle, is_success=False,
                     save_filename=fail_gif_folder / "test_fail.gif")