from __future__ import annotations

import math

import torch


class CircularWaypointCommand:
    """Vectorized circular waypoint generator for hopper landing targets."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        radius: float,
        hop_distance: float,
        fixed_ccw: bool = True,
        segment_horizon: int = 5,
    ):
        self.num_envs = num_envs
        self.device = device
        self.radius = torch.full((num_envs,), radius, device=device)
        self.hop_distance = hop_distance
        self.fixed_ccw = fixed_ccw
        self.segment_horizon = segment_horizon

        self.center_w = torch.zeros(num_envs, 2, device=device)
        self.path_angle = torch.zeros(num_envs, device=device)
        self.direction = torch.ones(num_envs, device=device)
        self.segment_start_angle = torch.zeros(num_envs, device=device)
        self.segment_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.segment_angles = torch.zeros(num_envs, segment_horizon + 1, device=device)
        self.segment_waypoints_w = torch.zeros(num_envs, segment_horizon + 1, 2, device=device)
        self.current_start_pos_w = torch.zeros(num_envs, 2, device=device)
        self.target_pos_w = torch.zeros(num_envs, 2, device=device)
        self.hop_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.successful_hops = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor, env_origins: torch.Tensor):
        n = len(env_ids)
        self.center_w[env_ids] = env_origins[env_ids, :2]
        self.path_angle[env_ids] = torch.rand(n, device=self.device) * (2.0 * math.pi)
        if self.fixed_ccw:
            self.direction[env_ids] = 1.0
        else:
            signs = torch.randint(0, 2, (n,), device=self.device, dtype=torch.long) * 2 - 1
            self.direction[env_ids] = signs.float()
        self.hop_index[env_ids] = 0
        self.successful_hops[env_ids] = 0
        self.segment_start_angle[env_ids] = self.path_angle[env_ids]
        self.segment_index[env_ids] = 0
        self._generate_segment(env_ids)
        self._refresh_current_targets(env_ids)

    def _step_angle(self, env_ids: torch.Tensor) -> torch.Tensor:
        radius = self.radius[env_ids]
        chord = torch.clamp(
            torch.full_like(radius, self.hop_distance),
            max=(2.0 * radius * 0.95),
        )
        return 2.0 * torch.asin(chord / (2.0 * radius))

    def position_at(self, env_ids: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        center = self.center_w[env_ids]
        radius = self.radius[env_ids]
        return torch.stack(
            (center[:, 0] + radius * torch.cos(angle), center[:, 1] + radius * torch.sin(angle)),
            dim=-1,
        )

    def _generate_segment(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        offsets = torch.arange(self.segment_horizon + 1, device=self.device).float()
        dphi = self._step_angle(env_ids)
        angles = (
            self.segment_start_angle[env_ids, None]
            + self.direction[env_ids, None] * dphi[:, None] * offsets[None, :]
        )
        flat_env_ids = env_ids.repeat_interleave(self.segment_horizon + 1)
        self.segment_angles[env_ids] = angles
        self.segment_waypoints_w[env_ids] = self.position_at(flat_env_ids, angles.reshape(-1)).reshape(
            len(env_ids), self.segment_horizon + 1, 2
        )

    def _refresh_current_targets(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        local = self.segment_index[env_ids]
        self.current_start_pos_w[env_ids] = self.segment_waypoints_w[env_ids, local]
        self.target_pos_w[env_ids] = self.segment_waypoints_w[env_ids, local + 1]
        self.path_angle[env_ids] = self.segment_angles[env_ids, local + 1]

    def advance(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self.hop_index[env_ids] += 1
        self.successful_hops[env_ids] += 1
        self.segment_index[env_ids] += 1

        finished_segment = self.segment_index[env_ids] >= self.segment_horizon
        if torch.any(finished_segment):
            finished_ids = env_ids[finished_segment]
            self.segment_start_angle[finished_ids] = self.segment_angles[finished_ids, -1]
            self.segment_index[finished_ids] = 0
            self._generate_segment(finished_ids)

        self._refresh_current_targets(env_ids)

    def horizon_waypoints(self, horizon: int) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device)
        local = self.segment_index
        offsets = torch.arange(1, horizon + 1, device=self.device)
        indices = local[:, None] + offsets[None, :]
        waypoints = torch.empty(self.num_envs, horizon, 2, device=self.device)

        in_segment = indices <= self.segment_horizon
        clamped = torch.clamp(indices, max=self.segment_horizon)
        waypoints[in_segment] = self.segment_waypoints_w[
            env_ids[:, None].expand_as(clamped)[in_segment],
            clamped[in_segment],
        ]

        if torch.any(~in_segment):
            future_env_ids = env_ids[:, None].expand_as(indices)[~in_segment]
            extra_steps = (indices[~in_segment] - self.segment_horizon).float()
            dphi = self._step_angle(future_env_ids)
            angles = (
                self.segment_angles[future_env_ids, -1]
                + self.direction[future_env_ids] * dphi * extra_steps
            )
            waypoints[~in_segment] = self.position_at(future_env_ids, angles)
        return waypoints

    def tangent_w(self) -> torch.Tensor:
        return torch.stack(
            (-torch.sin(self.path_angle) * self.direction, torch.cos(self.path_angle) * self.direction),
            dim=-1,
        )

    def radial_error(self, root_xy_w: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(root_xy_w - self.center_w, dim=-1) - self.radius
