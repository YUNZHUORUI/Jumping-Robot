import torch
import os
import numpy as np
from networks import Actor, Critic
from buffer import Agent_State_Buffer


class Actor_Critic:
    def __init__(self, Config, index=0):
        self.cfg = Config
        env_p = Config.EnvParam
        ppo_p = Config.PPOParam
        crit_p = Config.CriticParam
        act_p = Config.ActorParam

        self.device = env_p.device
        self.index = index
        self.train = env_p.train

        self.agent_num = env_p.agents_num

        self.gamma = ppo_p.gamma
        self.lam = ppo_p.lam
        self.epsilon = ppo_p.epsilon
        self.batch_size = ppo_p.batch_size
        self.loss_fn = torch.nn.MSELoss()

        self.actuator_num = act_p.actuator_num
        self.action_choice = torch.tensor(act_p.action_choice, device=self.device)
        self.action_scale = act_p.action_scale

        self.actor = Actor(act_p.state_dim, act_p.act_layers_num,
                           self.actuator_num, act_p.action_choice).to(self.device)
        self.critic = Critic(crit_p.state_dim, crit_p.critic_layers_num).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=act_p.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=crit_p.critic_lr)

        self.Buffer = Agent_State_Buffer(act_p.state_dim, self.actuator_num,
                                         self.agent_num, ppo_p.maximum_step, self.device)

        if not os.path.exists('model'):
            os.makedirs('model')

        self.best_reward = -float('inf')

    def sample_action(self, state):
        # 1. 检查输入状态是否包含 NaN (防御性编程)
        if torch.isnan(state).any() or torch.isinf(state).any():
            # print("Warning: NaN in state detected during sampling. Replacing with zeros.")
            state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.no_grad():
            action_prob = self.actor(state)

        # 2. 检查输出概率 NaN/Inf
        if torch.isnan(action_prob).any() or torch.isinf(action_prob).any():
            # 只有当真的出现 NaN 时才打印，避免刷屏
            # print("Warning: NaN/Inf detected in action probabilities. Resetting to uniform.")
            action_prob = torch.ones_like(action_prob) / action_prob.size(-1)

        # 3. 极小值保护 (防止概率为0导致采样报错)
        action_prob = torch.clamp(action_prob, min=1e-8)
        # 重新归一化
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)

        if self.train:
            try:
                action_index = torch.multinomial(action_prob.view(-1, len(self.action_choice)), 1)
            except RuntimeError as e:
                print(f"RuntimeError in multinomial: {e}")
                action_index = torch.randint(0, len(self.action_choice),
                                             (action_prob.size(0) * action_prob.size(1), 1)).to(self.device)
        else:
            action_index = torch.argmax(action_prob, dim=-1, keepdim=True)

        flat_indices = action_index.squeeze()
        if flat_indices.dim() == 0:
            flat_indices = flat_indices.unsqueeze(0)

        action_vals = self.action_choice[flat_indices]
        action_index = action_index.view(self.agent_num, self.actuator_num)
        action_output = action_vals.view(self.agent_num, self.actuator_num)

        return action_index, self.action_scale * action_output

    def store_experience(self, state, action, next_state, reward, over, step):
        self.Buffer.store_state(state, step)
        self.Buffer.store_action_index(action, step)
        self.Buffer.store_next_state(next_state, step)
        self.Buffer.store_reward(reward, step)
        self.Buffer.store_over(over, step)

    def update(self, update_idx, num_updates, avg_reward):
        buffer = self.Buffer
        buffer.compute_GAE(self.critic, self.gamma, self.lam)

        state = buffer.state_buffer
        action_index = buffer.action_index_buffer.unsqueeze(2)
        next_state = buffer.next_state_buffer
        reward = buffer.reward_buffer
        over = buffer.over_buffer
        GAE = buffer.GAE_buffer

        # 检查 State 中是否有 NaN，如果有则跳过整个 update，保护权重
        if torch.isnan(state).any():
            print("Warning: NaN detected in training buffer. Skipping update to protect model.")
            buffer.clear()
            return

        with torch.no_grad():
            action_prob = self.actor(state)
            # --- 关键修复: 这里必须加 1e-10，否则 log(0) = -inf，导致后续 ratio = inf ---
            old_prob = action_prob.gather(index=action_index.long(), dim=-1)
            old_prob = (old_prob + 1e-10).log().sum(dim=1)

        indices = torch.randperm(len(state), device=self.device)

        for start in range(0, len(state), self.batch_size):
            end = start + self.batch_size
            idx = indices[start:end]

            s_b, a_b, ns_b, r_b, o_b, gae_b, old_p_b = \
                state[idx], action_index[idx], next_state[idx], reward[idx], over[idx], GAE[idx], old_prob[idx]

            # --- Critic Update ---
            val = self.critic(s_b)
            with torch.no_grad():
                next_val = self.critic(ns_b)
                target = r_b + self.gamma * next_val * (1 - o_b)

            c_loss = self.loss_fn(val, target)
            self.critic_optimizer.zero_grad()
            c_loss.backward()

            # 熔断检查 1: Critic 梯度
            c_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            if torch.isnan(c_grad_norm) or torch.isinf(c_grad_norm):
                # print("Warning: NaN gradient in Critic. Skipping step.")
                self.critic_optimizer.zero_grad()  # 清空坏梯度
            else:
                self.critic_optimizer.step()

            # --- Actor Update ---
            act_prob = self.actor(s_b)
            # 这里原本有 1e-10，保持住
            new_prob = (act_prob.gather(index=a_b.long(), dim=-1) + 1e-10).log().sum(dim=1)

            ratio = torch.exp(new_prob - old_p_b)

            # 额外的数值保护: 限制 ratio 不超过 100，防止 extreme importance weight
            ratio = torch.clamp(ratio, 0.0, 100.0)

            surr1 = ratio * gae_b.squeeze()
            surr2 = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * gae_b.squeeze()
            a_loss = -torch.min(surr1, surr2).mean()

            # 增加 Entropy Loss (鼓励探索，防止过早收敛到 0 概率)
            entropy = -(act_prob * (act_prob + 1e-10).log()).sum(dim=-1).mean()
            # 从 Config 读取 entropy_coef，这里硬编码默认值 0.01 防止报错
            entropy_coef = 0.01

            total_a_loss = a_loss - entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            total_a_loss.backward()

            # 熔断检查 2: Actor 梯度
            a_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            if torch.isnan(a_grad_norm) or torch.isinf(a_grad_norm):
                # print("Warning: NaN gradient in Actor. Skipping step.")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()

        buffer.clear()

        print(f"[Update {update_idx + 1}/{num_updates}] steps={len(state)} avg_reward={avg_reward:.2f}")

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            print(f"Best Model saving, average reward = {self.best_reward:.2f}")
            torch.save(self.actor.state_dict(), f'model/actor_best.pth')
            torch.save(self.critic.state_dict(), f'model/critic_best.pth')