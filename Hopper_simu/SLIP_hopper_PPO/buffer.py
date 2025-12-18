import torch


class Agent_State_Buffer:
    def __init__(self, state_dim, actuator_num, agent_num, max_step, device):
        self.device = device
        self.state_dim = state_dim
        self.actuator_num = actuator_num
        self.agent_num = agent_num
        self.max_step = max_step
        self.ptr = 0

        self.state_buffer = torch.zeros((max_step * agent_num, state_dim), device=device)
        self.action_index_buffer = torch.zeros((max_step * agent_num, actuator_num), device=device)
        self.next_state_buffer = torch.zeros((max_step * agent_num, state_dim), device=device)
        self.reward_buffer = torch.zeros((max_step * agent_num, 1), device=device)
        self.over_buffer = torch.zeros((max_step * agent_num, 1), device=device)
        self.GAE_buffer = torch.zeros((max_step * agent_num, 1), device=device)

    def clear(self):
        self.ptr = 0

    def store_state(self, state, step):
        idx_start = step * self.agent_num
        idx_end = idx_start + self.agent_num
        self.state_buffer[idx_start:idx_end] = state

    def store_action_index(self, action, step):
        idx_start = step * self.agent_num
        idx_end = idx_start + self.agent_num
        self.action_index_buffer[idx_start:idx_end] = action

    def store_next_state(self, next_state, step):
        idx_start = step * self.agent_num
        idx_end = idx_start + self.agent_num
        self.next_state_buffer[idx_start:idx_end] = next_state

    def store_reward(self, reward, step):
        idx_start = step * self.agent_num
        idx_end = idx_start + self.agent_num
        self.reward_buffer[idx_start:idx_end] = reward

    def store_over(self, over, step):
        idx_start = step * self.agent_num
        idx_end = idx_start + self.agent_num
        self.over_buffer[idx_start:idx_end] = over

    def compute_GAE(self, critic_net, gamma, lam):
        rewards = self.reward_buffer.view(self.max_step, self.agent_num)
        overs = self.over_buffer.view(self.max_step, self.agent_num)
        states = self.state_buffer.view(self.max_step, self.agent_num, self.state_dim)
        next_states = self.next_state_buffer.view(self.max_step, self.agent_num, self.state_dim)

        with torch.no_grad():
            values = critic_net(states.view(-1, self.state_dim)).view(self.max_step, self.agent_num)
            next_values = critic_net(next_states.view(-1, self.state_dim)).view(self.max_step, self.agent_num)

        deltas = rewards + gamma * next_values * (1.0 - overs) - values

        gae = 0
        gae_tensor = torch.zeros_like(deltas)

        for t in reversed(range(self.max_step)):
            mask = 1.0 - overs[t]
            gae = deltas[t] + gamma * lam * mask * gae
            gae_tensor[t] = gae

        self.GAE_buffer = gae_tensor.view(-1, 1)