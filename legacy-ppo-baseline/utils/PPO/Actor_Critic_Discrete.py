import torch
from ..PPO.Buffer_discrete import *

""" PPO算法的Actor-Critic网络结构
Actor: 输入状态 输出动作的均值和标准差"""


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
        x = torch.nn.functional.elu(self.fc1_x(input_))
        x = torch.nn.functional.elu(self.fc2_x(x))
        x = torch.nn.functional.elu(self.fc3_x(x))
        x = torch.nn.functional.elu(self.fc4_x(x))
        output = torch.nn.functional.softmax(x.reshape(-1, self.actuator_num, len(self.action_choice)), dim=-1)

        return output


""" Critic: 输入状态 输出状态值函数"""


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
        x = torch.nn.functional.elu(self.fc1_x(input_))
        x = torch.nn.functional.elu(self.fc2_x(x))
        x = torch.nn.functional.elu(self.fc3_x(x))
        output = self.fc4_x(x)
        return output


""" Actor-Critic类 包含Actor和Critic网络 以及相关的优化器和经验回放缓冲区"""


class Actor_Critic:
    def __init__(self, PPO_Config, Env_Config, index=0):
        """ 初始化Actor-Critic网络
        Args:
            PPO_Config: PPO算法的配置参数 (类型: 配置类)
            Env_Config: 环境的配置参数 (类型: 配置类)
            index: 该AC的索引 (类型: int, 默认值: 0)
        """
        # Env parameter
        self.agent_num = Env_Config.EnvParam.agents_num
        self.device = Env_Config.EnvParam.device
        self.maximum_step = PPO_Config.PPOParam.maximum_step
        self.train = Env_Config.EnvParam.train
        self.index = index
        self.agent_num = (Env_Config.EnvParam.agents_num - Env_Config.EnvParam.agents_num_in_play) * self.train + \
                          Env_Config.EnvParam.agents_num_in_play
        # PPO parameter
        self.gamma = PPO_Config.PPOParam.gamma
        self.lam = PPO_Config.PPOParam.lam
        self.epsilon = PPO_Config.PPOParam.epsilon
        self.entropy_coef = PPO_Config.PPOParam.entropy_coef
        self.batch_size = PPO_Config.PPOParam.batch_size
        self.loss_fn = torch.nn.MSELoss()

        # state parameter
        self.state_dim = PPO_Config.CriticParam.state_dim
        self.critic_num_layers = PPO_Config.CriticParam.critic_layers_num
        self.critic_update_frequency = PPO_Config.CriticParam.critic_update_frequency
        self.critic_lr = PPO_Config.CriticParam.critic_lr

        # actor parameter
        self.action_scale = PPO_Config.ActorParam.action_scale
        self.actor_num_layers = PPO_Config.ActorParam.act_layers_num
        self.actor_update_frequency = PPO_Config.ActorParam.actor_update_frequency
        self.actuator_num = PPO_Config.ActorParam.actuator_num
        self.actor_lr = PPO_Config.ActorParam.actor_lr
        self.action_choice = torch.tensor(PPO_Config.ActorParam.action_choice,device=self.device)



        # 初始化网络
        self.actor = Actor(self.state_dim,
                           self.actor_num_layers,
                           self.actuator_num, self.action_choice,

                           self.action_scale,
                           ).to(self.device)

        self.critic = Critic(self.state_dim,
                             self.critic_num_layers).to(self.device)

        # 初始化优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.Buffer = Agent_State_Buffer(self.state_dim,
                                         self.actuator_num,
                                         self.agent_num,
                                         self.maximum_step,
                                         self.device)

        self.initial_reward_sum = -999


    def sample_action(self, state):

        with torch.no_grad():
            action_prob = self.actor(state)


        action_index = torch.multinomial(action_prob.view(-1, len(self.action_choice)), 1)
        if not self.train:
            action_index = torch.argmax(action_prob,dim=-1)
        action_output = self.action_choice[action_index]
        action_index = action_index.view(self.agent_num, self.actuator_num)
        action_output = action_output.view(self.agent_num, self.actuator_num)



        return action_index, self.action_scale*action_output

    def store_experience(self, state, action, next_state, reward, over,
                         current_step):
        """ 存储经验到缓冲区
        Args:
            state: 当前状态 (类型: torch.tensor, 形状: [agent_num, state_dim])
            action: 当前动作 (类型: torch.tensor, 形状: [agent  _num, actuator_num])
            reward: 当前奖励 (类型: torch.tensor, 形状: [agent_num, 1])
            over: 当前是否结束 (类型: torch.tensor, 形状: [agent_num, 1])
            current_step: 当前时间步 (类型: int)
        """
        self.Buffer.store_state(state, current_step)
        self.Buffer.store_action_index(action, current_step)
        self.Buffer.store_next_state(next_state, current_step)
        self.Buffer.store_reward(reward, current_step)
        self.Buffer.store_over(over, current_step)

    def update(self):
        # 获取经验数据
        buffer = self.Buffer
        state = buffer.state_buffer.view(-1, self.state_dim)
        action_index = buffer.action_index_buffer.view(-1, self.actuator_num).unsqueeze(2)
        next_state = buffer.next_state_buffer.view(-1, self.state_dim)
        reward = buffer.reward_buffer.view(-1, 1)
        over = buffer.over_buffer.view(-1, 1)
        reward_sum = reward.mean().item()

        # 计算旧策略概率
        with torch.no_grad():
            action_prob = self.actor(state)
            old_prob = action_prob.gather(index=action_index.long(), dim=-1)
            old_prob = old_prob.log().sum(dim=1)
        # Critic更新
        for _ in range(self.critic_update_frequency):
            idx = torch.randperm(len(state), device=state.device)[:self.batch_size]
            s_batch, ns_batch, r_batch, o_batch = state[idx], next_state[idx], reward[idx], over[idx]

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

        # Actor更新
        for _ in range(self.actor_update_frequency):
            idx = torch.randperm(len(state), device=state.device)[:self.batch_size]
            s_batch, a_batch, ns_batch = state[idx], action_index[idx], next_state[idx]
            gae_batch, old_prob_batch = GAE[idx], old_prob[idx]

            # 计算新策略
            action_prob = self.actor(s_batch)
            new_prob = action_prob.gather(index=a_batch.long(), dim=-1)
            new_prob = new_prob.log().sum(dim=1)

            # PPO损失
            ratio = torch.exp(new_prob - old_prob_batch)  # 括号里是对数
            surr1 = ratio * gae_batch

            surr2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * gae_batch

            # 对调并取反结束
            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        print(f"Experience Collected: {len(state)}, Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
        print("reward:", reward_sum)

        self.save_each_epi_model()
        if reward_sum > self.initial_reward_sum:
            self.initial_reward_sum = reward_sum
            self.save_best_model()
            print(f"Best Model Saved")

    def save_best_model(self):
        torch.save(self.actor.state_dict(), f'model/actor{self.index}.pth')
        torch.save(self.critic.state_dict(), f'model/critic{self.index}.pth')

    def save_each_epi_model(self):
        torch.save(self.actor.state_dict(), f'model/actor{self.index}_f.pth')
        torch.save(self.critic.state_dict(), f'model/critic{self.index}_f.pth')

    def load_best_model(self):
        self.actor.load_state_dict(torch.load(f'model/actor{self.index}.pth'))
        self.critic.load_state_dict(torch.load(f'model/critic{self.index}.pth'))

    def load_each_epi_model(self):
        self.actor.load_state_dict(torch.load(f'model/actor{self.index}_f.pth'))
        self.critic.load_state_dict(torch.load(f'model/critic{self.index}_f.pth'))
