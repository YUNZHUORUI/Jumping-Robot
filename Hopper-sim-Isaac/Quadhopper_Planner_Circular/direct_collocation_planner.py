from __future__ import annotations

import torch


class DirectCollocationHopPlanner:
    """Joint numerical planner for the next two complete jump cycles.

    Both cycle trajectories are decision variables in one KKT system. Each
    segment enforces takeoff, apex, landing and velocity constraints while the
    objective minimizes third finite differences. The result is a rolling
    two-cycle horizon, not a one-hop curve with P_(t+1) used only as a hint.
    """

    def __init__(self, num_envs: int, device: str, nodes: int, duration: float):
        if nodes < 9 or nodes % 2 == 0:
            raise ValueError("planner_nodes must be an odd integer >= 9")
        self.num_envs = num_envs
        self.device = device
        self.nodes = nodes
        self.duration = float(duration)
        self.dt = self.duration / (nodes - 1)
        self.mid = nodes // 2
        self.positions_w = torch.zeros(num_envs, nodes, 3, device=device)
        self.velocities_w = torch.zeros_like(self.positions_w)
        self.next_positions_w = torch.zeros_like(self.positions_w)
        self.next_velocities_w = torch.zeros_like(self.positions_w)
        self._kkt = self._build_joint_kkt(device)

    def _segment_hessian_and_constraints(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.nodes
        d3 = torch.zeros(n - 3, n, device=device)
        row = torch.arange(n - 3, device=device)
        d3[row, row] = -1.0
        d3[row, row + 1] = 3.0
        d3[row, row + 2] = -3.0
        d3[row, row + 3] = 1.0
        hessian = d3.T @ d3 + 1.0e-5 * torch.eye(n, device=device)

        constraints = torch.zeros(6, n, device=device)
        constraints[0, 0] = 1.0
        constraints[1, self.mid] = 1.0
        constraints[2, -1] = 1.0
        constraints[3, 0:2] = torch.tensor([-1.0, 1.0], device=device) / self.dt
        constraints[4, self.mid - 1 : self.mid + 2] = (
            torch.tensor([-0.5, 0.0, 0.5], device=device) / self.dt
        )
        constraints[5, -2:] = torch.tensor([-1.0, 1.0], device=device) / self.dt
        return hessian, constraints

    def _build_joint_kkt(self, device: str) -> torch.Tensor:
        hessian, constraints = self._segment_hessian_and_constraints(device)
        joint_hessian = torch.block_diag(hessian, hessian)
        joint_constraints = torch.block_diag(constraints, constraints)
        top = torch.cat((joint_hessian, joint_constraints.T), dim=1)
        bottom = torch.cat(
            (joint_constraints, torch.zeros(12, 12, device=device)), dim=1
        )
        return torch.cat((top, bottom), dim=0)

    def _finite_difference_velocity(self, positions: torch.Tensor) -> torch.Tensor:
        velocity = torch.zeros_like(positions)
        velocity[:, 1:-1] = (positions[:, 2:] - positions[:, :-2]) / (2.0 * self.dt)
        velocity[:, 0] = (positions[:, 1] - positions[:, 0]) / self.dt
        velocity[:, -1] = (positions[:, -1] - positions[:, -2]) / self.dt
        return velocity

    def replan(
        self,
        env_ids: torch.Tensor,
        start_pos_w: torch.Tensor,
        start_vel_w: torch.Tensor,
        p_t_xy_w: torch.Tensor,
        p_t1_xy_w: torch.Tensor,
        target_height_w: torch.Tensor,
        landing_height_w: torch.Tensor,
    ):
        if len(env_ids) == 0:
            return

        first_delta = p_t_xy_w - start_pos_w[:, :2]
        second_delta = p_t1_xy_w - p_t_xy_w
        first_speed = torch.linalg.norm(first_delta, dim=1) / self.duration
        second_distance = torch.linalg.norm(second_delta, dim=1)
        second_direction = second_delta / second_distance[:, None].clamp_min(1.0e-6)
        tangent_speed = 0.5 * (first_speed + second_distance / self.duration)
        tangent_v = second_direction * tangent_speed[:, None]

        jump_speed = torch.sqrt(
            2.0 * 9.81 * (target_height_w - landing_height_w).clamp_min(0.02)
        )
        zero_z = torch.zeros(len(env_ids), 1, device=self.device)
        upward_v = torch.cat((tangent_v, jump_speed[:, None]), dim=1)
        apex_v = torch.cat((tangent_v, zero_z), dim=1)
        landing_v = torch.cat((tangent_v, -jump_speed[:, None]), dim=1)

        p_t_ground = torch.cat((p_t_xy_w, landing_height_w[:, None]), dim=1)
        p_t1_ground = torch.cat((p_t1_xy_w, landing_height_w[:, None]), dim=1)
        apex_t = torch.cat(
            (0.5 * (start_pos_w[:, :2] + p_t_xy_w), target_height_w[:, None]), dim=1
        )
        apex_t1 = torch.cat(
            (0.5 * (p_t_xy_w + p_t1_xy_w), target_height_w[:, None]), dim=1
        )

        first_boundary = torch.stack(
            (start_pos_w, apex_t, p_t_ground, start_vel_w, apex_v, landing_v), dim=1
        )
        second_boundary = torch.stack(
            (p_t_ground, apex_t1, p_t1_ground, upward_v, apex_v, landing_v), dim=1
        )
        joint_boundary = torch.cat((first_boundary, second_boundary), dim=1)

        decision_count = 2 * self.nodes
        rhs = torch.zeros(len(env_ids), decision_count + 12, 3, device=self.device)
        rhs[:, decision_count:, :] = joint_boundary
        solution = torch.linalg.solve(self._kkt, rhs)
        first_positions = solution[:, : self.nodes, :]
        second_positions = solution[:, self.nodes : decision_count, :]

        self.positions_w[env_ids] = first_positions
        self.velocities_w[env_ids] = self._finite_difference_velocity(first_positions)
        self.next_positions_w[env_ids] = second_positions
        self.next_velocities_w[env_ids] = self._finite_difference_velocity(second_positions)

    def sample(self, cycle_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase = (cycle_time / self.duration).clamp(0.0, 1.0) * (self.nodes - 1)
        lower = torch.floor(phase).long().clamp(max=self.nodes - 2)
        blend = (phase - lower.float()).unsqueeze(-1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        p0 = self.positions_w[env_ids, lower]
        p1 = self.positions_w[env_ids, lower + 1]
        v0 = self.velocities_w[env_ids, lower]
        v1 = self.velocities_w[env_ids, lower + 1]
        return p0 + blend * (p1 - p0), v0 + blend * (v1 - v0)
