import torch
import torch.nn as nn
import numpy as np

# --- 辅助函数：正交初始化 ---
# 这能防止训练一开始梯度就消失或爆炸
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    import numpy as np
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, state_dim, num_layers, actuator_num, action_choice, action_scale=1):
        super(Actor, self).__init__()

        # 使用 LayerNorm 防止数值爆炸
        self.fc1 = layer_init(nn.Linear(state_dim, num_layers))
        self.ln1 = nn.LayerNorm(num_layers)

        self.fc2 = layer_init(nn.Linear(num_layers, num_layers))
        self.ln2 = nn.LayerNorm(num_layers)

        self.fc3 = layer_init(nn.Linear(num_layers, num_layers))
        self.ln3 = nn.LayerNorm(num_layers)

        # 输出层 std 设置小一点 (0.01)，保证初始策略接近均匀分布
        self.fc4 = layer_init(nn.Linear(num_layers, actuator_num * len(action_choice)), std=0.01)

        self.actuator_num = actuator_num
        self.action_len = len(action_choice)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.fc1(x))  # Tanh 比 ELU 在 PPO 中通常更稳定
        x = self.ln1(x)

        x = torch.nn.functional.tanh(self.fc2(x))
        x = self.ln2(x)

        x = torch.nn.functional.tanh(self.fc3(x))
        x = self.ln3(x)

        logits = self.fc4(x)

        # Reshape to [Batch, Actuator, Action_Options]
        logits = logits.reshape(-1, self.actuator_num, self.action_len)

        # Softmax 在 logits 很大时会产生 NaN，LayerNorm 已经缓解了这个问题
        # 但为了双重保险，我们还是加上数值稳定性保护
        output = torch.nn.functional.softmax(logits, dim=-1)
        return output


class Critic(nn.Module):
    def __init__(self, state_dim, num_layers):
        super(Critic, self).__init__()

        self.fc1 = layer_init(nn.Linear(state_dim, num_layers))
        self.ln1 = nn.LayerNorm(num_layers)

        self.fc2 = layer_init(nn.Linear(num_layers, num_layers))
        self.ln2 = nn.LayerNorm(num_layers)

        self.fc3 = layer_init(nn.Linear(num_layers, num_layers))
        self.ln3 = nn.LayerNorm(num_layers)

        self.fc4 = layer_init(nn.Linear(num_layers, 1), std=1.0)

    def forward(self, x):
        x = torch.nn.functional.tanh(self.fc1(x))
        x = self.ln1(x)

        x = torch.nn.functional.tanh(self.fc2(x))
        x = self.ln2(x)

        x = torch.nn.functional.tanh(self.fc3(x))
        x = self.ln3(x)

        output = self.fc4(x)
        return output