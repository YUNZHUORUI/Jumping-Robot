from __future__ import annotations

import torch


class BallisticHopPlanner:
    """Per-hop reference velocities for discrete landing targets."""

    def __init__(self, num_envs: int, device: str, flight_time: float, apex_height: float):
        self.flight_time_ref = torch.full((num_envs,), flight_time, device=device)
        self.apex_height_ref = torch.full((num_envs,), apex_height, device=device)
        self.vz_ref = torch.full((num_envs,), (2.0 * 9.81 * apex_height) ** 0.5, device=device)
        self.takeoff_pos_w = torch.zeros(num_envs, 3, device=device)
        self.target_pos_w = torch.zeros(num_envs, 3, device=device)
        self.v_xy_ref_w = torch.zeros(num_envs, 2, device=device)

    def reset(self, env_ids: torch.Tensor, root_pos_w: torch.Tensor, target_xy_w: torch.Tensor):
        self.replan(env_ids, root_pos_w, target_xy_w)

    def replan(self, env_ids: torch.Tensor, root_pos_w: torch.Tensor, target_xy_w: torch.Tensor):
        if len(env_ids) == 0:
            return
        self.takeoff_pos_w[env_ids] = root_pos_w[env_ids]
        self.target_pos_w[env_ids, :2] = target_xy_w[env_ids]
        self.target_pos_w[env_ids, 2] = root_pos_w[env_ids, 2]
        self.v_xy_ref_w[env_ids] = (target_xy_w[env_ids] - root_pos_w[env_ids, :2]) / self.flight_time_ref[env_ids, None]

    def set_fixed_xy_plan(
        self,
        env_ids: torch.Tensor,
        start_xy_w: torch.Tensor,
        target_xy_w: torch.Tensor,
        base_z_w: torch.Tensor,
    ):
        if len(env_ids) == 0:
            return
        self.takeoff_pos_w[env_ids, :2] = start_xy_w
        self.takeoff_pos_w[env_ids, 2] = base_z_w
        self.target_pos_w[env_ids, :2] = target_xy_w
        self.target_pos_w[env_ids, 2] = base_z_w
        self.v_xy_ref_w[env_ids] = (target_xy_w - start_xy_w) / self.flight_time_ref[env_ids, None]
