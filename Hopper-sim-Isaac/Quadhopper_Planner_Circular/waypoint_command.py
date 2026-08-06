from __future__ import annotations

import math

import torch


class TwoCycleCircularCommand:
    """Circular task command exposing exactly the next two hop targets."""

    def __init__(self, num_envs: int, device: str, radius: float, hop_distance: float):
        self.num_envs = num_envs
        self.device = device
        self.radius = float(radius)
        self.hop_distance = float(hop_distance)
        ratio = min(self.hop_distance / (2.0 * self.radius), 0.95)
        self.step_angle = 2.0 * math.asin(ratio)
        self.steps_per_revolution = int(math.ceil((2.0 * math.pi) / self.step_angle))
        self.center_w = torch.zeros(num_envs, 2, device=device)
        self.angle = torch.zeros(num_envs, device=device)
        self.direction = torch.ones(num_envs, device=device)
        self.cycle_index = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor, env_origins: torch.Tensor, random_phase: bool):
        self.center_w[env_ids] = env_origins[env_ids, :2]
        if random_phase:
            self.angle[env_ids] = torch.rand(len(env_ids), device=self.device) * (2.0 * math.pi)
        else:
            self.angle[env_ids] = 0.0
        self.direction[env_ids] = 1.0
        self.cycle_index[env_ids] = 0

    def _position(self, angle: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            center = self.center_w
        else:
            center = self.center_w[env_ids]
        return center + self.radius * torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)

    def start_points(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            return self._position(self.angle)
        return self._position(self.angle[env_ids], env_ids)

    def lookahead(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            angle = self.angle
            direction = self.direction
        else:
            angle = self.angle[env_ids]
            direction = self.direction[env_ids]
        p_t = self._position(angle + direction * self.step_angle, env_ids)
        p_t1 = self._position(angle + direction * (2.0 * self.step_angle), env_ids)
        return p_t, p_t1

    def advance(self, env_ids: torch.Tensor):
        self.angle[env_ids] += self.direction[env_ids] * self.step_angle
        self.cycle_index[env_ids] += 1
