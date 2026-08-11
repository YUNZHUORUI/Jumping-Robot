from __future__ import annotations

import torch


class DirectCollocationHopPlanner:
    """Two-flight, state-dependent reference planner.

    Heights are absolute root-Z commands at the apex.  Each flight duration is
    computed from the measured takeoff/landing height and gravity.  The stance
    phase is intentionally handled by the environment; this planner contains
    flight only and therefore never pretends that ground contact is a smooth
    continuation of a ballistic arc.

    This is a kinematic collocation reference generator, not a full rigid-body
    or contact optimizer.  Motor, spring and attitude feasibility is learned by
    the tracking policy.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        nodes: int,
        gravity: float = 9.81,
        min_flight_duration: float = 0.30,
        max_flight_duration: float = 1.10,
    ):
        if nodes < 9 or nodes % 2 == 0:
            raise ValueError("planner_nodes must be an odd integer >= 9")
        self.num_envs = num_envs
        self.device = device
        self.nodes = nodes
        self.gravity = float(gravity)
        self.min_flight_duration = float(min_flight_duration)
        self.max_flight_duration = float(max_flight_duration)
        self.mid = nodes // 2
        self.positions_w = torch.zeros(num_envs, nodes, 3, device=device)
        self.velocities_w = torch.zeros_like(self.positions_w)
        self.next_positions_w = torch.zeros_like(self.positions_w)
        self.next_velocities_w = torch.zeros_like(self.positions_w)
        self.flight_duration = torch.full(
            (num_envs,), self.min_flight_duration, device=device
        )
        self.next_flight_duration = self.flight_duration.clone()
        self._hessian = self._build_hessian(device)

    def _build_hessian(self, device: str) -> torch.Tensor:
        n = self.nodes
        d3 = torch.zeros(n - 3, n, device=device)
        row = torch.arange(n - 3, device=device)
        d3[row, row] = -1.0
        d3[row, row + 1] = 3.0
        d3[row, row + 2] = -3.0
        d3[row, row + 3] = 1.0
        return d3.T @ d3 + 1.0e-5 * torch.eye(n, device=device)

    def _duration(
        self, takeoff_z: torch.Tensor, apex_z: torch.Tensor, landing_z: torch.Tensor
    ) -> torch.Tensor:
        rise = (apex_z - takeoff_z).clamp_min(0.02)
        fall = (apex_z - landing_z).clamp_min(0.02)
        duration = torch.sqrt(2.0 * rise / self.gravity) + torch.sqrt(
            2.0 * fall / self.gravity
        )
        return duration.clamp(self.min_flight_duration, self.max_flight_duration)

    def _constraints(self, duration: torch.Tensor) -> torch.Tensor:
        batch = len(duration)
        dt = duration / float(self.nodes - 1)
        constraints = torch.zeros(batch, 6, self.nodes, device=self.device)
        constraints[:, 0, 0] = 1.0
        constraints[:, 1, self.mid] = 1.0
        constraints[:, 2, -1] = 1.0
        constraints[:, 3, 0] = -1.0 / dt
        constraints[:, 3, 1] = 1.0 / dt
        constraints[:, 4, self.mid - 1] = -0.5 / dt
        constraints[:, 4, self.mid + 1] = 0.5 / dt
        constraints[:, 5, -2] = -1.0 / dt
        constraints[:, 5, -1] = 1.0 / dt
        return constraints

    def _solve_segment(
        self,
        duration: torch.Tensor,
        start: torch.Tensor,
        apex: torch.Tensor,
        landing: torch.Tensor,
        start_velocity: torch.Tensor,
        apex_velocity: torch.Tensor,
        landing_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(duration)
        constraints = self._constraints(duration)
        hessian = self._hessian.expand(batch, -1, -1)
        zeros = torch.zeros(batch, 6, 6, device=self.device)
        kkt = torch.cat(
            (
                torch.cat((hessian, constraints.transpose(1, 2)), dim=2),
                torch.cat((constraints, zeros), dim=2),
            ),
            dim=1,
        )
        boundary = torch.stack(
            (start, apex, landing, start_velocity, apex_velocity, landing_velocity), dim=1
        )
        rhs = torch.zeros(batch, self.nodes + 6, 3, device=self.device)
        rhs[:, self.nodes :, :] = boundary
        positions = torch.linalg.solve(kkt, rhs)[:, : self.nodes, :]

        dt = duration[:, None, None] / float(self.nodes - 1)
        velocity = torch.zeros_like(positions)
        velocity[:, 1:-1] = (positions[:, 2:] - positions[:, :-2]) / (2.0 * dt)
        velocity[:, 0] = (positions[:, 1] - positions[:, 0]) / dt[:, 0]
        velocity[:, -1] = (positions[:, -1] - positions[:, -2]) / dt[:, 0]
        return positions, velocity

    def replan(
        self,
        env_ids: torch.Tensor,
        start_pos_w: torch.Tensor,
        start_vel_w: torch.Tensor,
        p_t_xy_w: torch.Tensor,
        p_t1_xy_w: torch.Tensor,
        target_height_w: torch.Tensor,
        landing_height_w: torch.Tensor,
        next_target_height_w: torch.Tensor | None = None,
    ):
        if len(env_ids) == 0:
            return
        if next_target_height_w is None:
            next_target_height_w = target_height_w

        duration = self._duration(start_pos_w[:, 2], target_height_w, landing_height_w)
        next_duration = self._duration(
            landing_height_w, next_target_height_w, landing_height_w
        )
        first_xy_velocity = (p_t_xy_w - start_pos_w[:, :2]) / duration[:, None]
        second_xy_velocity = (p_t1_xy_w - p_t_xy_w) / next_duration[:, None]
        first_rise = (target_height_w - start_pos_w[:, 2]).clamp_min(0.02)
        first_fall = (target_height_w - landing_height_w).clamp_min(0.02)
        next_rise = (next_target_height_w - landing_height_w).clamp_min(0.02)
        first_up = torch.sqrt(2.0 * self.gravity * first_rise)
        first_down = torch.sqrt(2.0 * self.gravity * first_fall)
        next_up = torch.sqrt(2.0 * self.gravity * next_rise)
        next_down = next_up
        zero = torch.zeros(len(env_ids), 1, device=self.device)

        p_t_ground = torch.cat((p_t_xy_w, landing_height_w[:, None]), dim=1)
        p_t1_ground = torch.cat((p_t1_xy_w, landing_height_w[:, None]), dim=1)
        apex_t = torch.cat(
            (0.5 * (start_pos_w[:, :2] + p_t_xy_w), target_height_w[:, None]), dim=1
        )
        apex_t1 = torch.cat(
            (0.5 * (p_t_xy_w + p_t1_xy_w), next_target_height_w[:, None]), dim=1
        )
        first_takeoff_v = torch.cat((first_xy_velocity, first_up[:, None]), dim=1)
        # Use measured velocity only when already airborne; at grounded replans
        # the physically meaningful boundary is the required takeoff velocity.
        airborne = start_pos_w[:, 2] > landing_height_w + 0.03
        first_start_v = torch.where(airborne[:, None], start_vel_w, first_takeoff_v)
        first_apex_v = torch.cat((first_xy_velocity, zero), dim=1)
        first_landing_v = torch.cat((first_xy_velocity, -first_down[:, None]), dim=1)
        next_takeoff_v = torch.cat((second_xy_velocity, next_up[:, None]), dim=1)
        next_apex_v = torch.cat((second_xy_velocity, zero), dim=1)
        next_landing_v = torch.cat((second_xy_velocity, -next_down[:, None]), dim=1)

        first_pos, first_vel = self._solve_segment(
            duration,
            start_pos_w,
            apex_t,
            p_t_ground,
            first_start_v,
            first_apex_v,
            first_landing_v,
        )
        next_pos, next_vel = self._solve_segment(
            next_duration,
            p_t_ground,
            apex_t1,
            p_t1_ground,
            next_takeoff_v,
            next_apex_v,
            next_landing_v,
        )
        self.positions_w[env_ids] = first_pos
        self.velocities_w[env_ids] = first_vel
        self.next_positions_w[env_ids] = next_pos
        self.next_velocities_w[env_ids] = next_vel
        self.flight_duration[env_ids] = duration
        self.next_flight_duration[env_ids] = next_duration

    def sample(self, flight_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase = (flight_time / self.flight_duration.clamp_min(1.0e-6)).clamp(0.0, 1.0)
        phase = phase * (self.nodes - 1)
        lower = torch.floor(phase).long().clamp(max=self.nodes - 2)
        blend = (phase - lower.float()).unsqueeze(-1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        p0 = self.positions_w[env_ids, lower]
        p1 = self.positions_w[env_ids, lower + 1]
        v0 = self.velocities_w[env_ids, lower]
        v1 = self.velocities_w[env_ids, lower + 1]
        return p0 + blend * (p1 - p0), v0 + blend * (v1 - v0)
