"""Learned state-action scorer used to select a second-hop command."""

from __future__ import annotations

import torch
from torch import nn


class SecondHopScorer(nn.Module):
    def __init__(self, obs_dim: int = 69, action_dim: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 2),
        )

    def forward(self, observation, action):
        output = self.network(torch.cat((observation, action), dim=-1))
        return output[..., 0], output[..., 1]
