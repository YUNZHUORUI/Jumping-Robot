"""Batched QuadHopper dynamics that remain entirely on a PyTorch device.

This is the CUDA counterpart of :mod:`Quadhopper.env`.  Every row is an
independent environment; no Gym objects, NumPy arrays, subprocess pipes or
host/device copies are used while collecting a rollout.
"""

from __future__ import annotations

import math

import torch

from .config import ATTITUDE, ENV, PHYSICS, REWARD


_N_SUBSTEPS = 5


class TorchQuadHopperVecEnv:
    """Vectorized SLIP/flight simulator for native CUDA PPO training."""

    observation_dim = 15
    action_dim = 2

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        target_count: int = 1,
        physics_cfg=PHYSICS,
        attitude_cfg=ATTITUDE,
        env_cfg=ENV,
        reward_cfg=REWARD,
        seed: int = 0,
        random_start_probability: float = 0.0,
    ):
        if env_cfg.reset_mode != "ground":
            raise ValueError("Native Torch training currently requires reset_mode='ground'")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.p = physics_cfg
        self.a = attitude_cfg
        self.e = env_cfg
        self.r = reward_cfg
        self.target_count = int(target_count)
        self.max_episode_steps = max(
            self.e.max_episode_steps, self.target_count * 300
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.random_start_probability = float(random_start_probability)

        shape = (self.num_envs,)
        self.q = torch.zeros((self.num_envs, 4), device=self.device)
        self.dq = torch.zeros_like(self.q)
        self.filtered_u = torch.zeros((self.num_envs, 2), device=self.device)
        self.stance = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.prev_touching = torch.zeros_like(self.stance)
        self.stance_substeps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.stance_min_l = torch.full(shape, self.p.leg_length, device=self.device)
        self.consecutive_stance_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.target_idx = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.best_foot_x = torch.zeros(shape, device=self.device)
        self.no_progress_counter = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(shape, device=self.device)
        self.episode_length = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.prev_vy = torch.zeros(shape, device=self.device)
        self.hop_max_height = torch.zeros(shape, device=self.device)

        # Per-environment ballistic plan.
        self.plan_valid = torch.zeros_like(self.stance)
        self.plan_vx = torch.zeros(shape, device=self.device)
        self.plan_vy = torch.zeros(shape, device=self.device)
        self.plan_takeoff_theta = torch.full(shape, math.pi / 4.0, device=self.device)

        self.reset()

    def _all_ids(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device)

    def _target_x(self) -> tuple[torch.Tensor, torch.Tensor]:
        valid = self.target_idx < self.target_count
        target = (self.target_idx.to(self.dtype) + 1.0) * self.e.target_spacing
        return torch.where(valid, target, self.q[:, 0]), valid

    def _com_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        theta, leg = self.q[:, 2], self.q[:, 3]
        dtheta, dl = self.dq[:, 2], self.dq[:, 3]
        s, c = torch.sin(theta), torch.cos(theta)
        com_x = self.q[:, 0] + leg * s
        com_y = self.q[:, 1] + leg * c
        vx = self.dq[:, 0] + dl * s + leg * c * dtheta
        vy = self.dq[:, 1] + dl * c - leg * s * dtheta
        return com_x, com_y, vx, vy

    def _plan(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        theta = self.q[env_ids, 2]
        leg = self.q[env_ids, 3]
        x_curr = self.q[env_ids, 0] + leg * torch.sin(theta)
        y_curr = self.q[env_ids, 1] + leg * torch.cos(theta)
        target_x = (self.target_idx[env_ids].to(self.dtype) + 1.0) * self.e.target_spacing

        phi_td = math.radians(self.r.phi_td_target_deg)
        x_target_com = target_x - self.p.leg_length * math.sin(phi_td)
        y_target = self.p.leg_length * math.cos(phi_td)
        dx = x_target_com - x_curr
        dy = y_target - y_curr
        valid = dx > 0.3

        radius = torch.sqrt(torch.clamp(dx.square() + dy.square(), min=1e-12))
        theta_min = torch.atan2(dy + radius, dx)
        v_min_sq = self.p.gravity * (dy + radius)
        valid &= v_min_sq > 1e-8
        v_min = torch.sqrt(torch.clamp(v_min_sq, min=1e-8))
        vx_min = torch.clamp(v_min * torch.cos(theta_min), min=0.2)
        vy_min = v_min * torch.sin(theta_min)

        apex_min_rel = vy_min.square() / (2.0 * self.p.gravity)
        apex_required = torch.maximum(
            apex_min_rel * max(1.0, self.e.traj_apex_scale),
            torch.clamp(torch.maximum(y_curr, torch.full_like(y_curr, y_target)) - y_curr
                        + max(0.02, self.e.traj_apex_clearance), min=0.0),
        )
        apex_required = torch.maximum(
            apex_required,
            torch.clamp(self.e.traj_apex_height - y_curr, min=0.0),
        )
        vy_nom = torch.sqrt(torch.clamp(2.0 * self.p.gravity * apex_required, min=1e-8))

        disc = vy_nom.square() - 2.0 * self.p.gravity * dy
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        denom = self.p.gravity * torch.clamp(dx, min=1e-6)
        u_slow = (vy_nom + sqrt_disc) / denom
        vx_slow = torch.where(u_slow > 1e-8, 1.0 / u_slow, vx_min)
        vx_nom = torch.where(disc > 0.0, vx_slow, vx_min)
        vx_nom = torch.clamp(vx_nom, self.e.traj_min_vx, self.e.traj_max_vx)
        flight_t = dx / torch.clamp(vx_nom, min=1e-6)
        vy_exact = (dy + 0.5 * self.p.gravity * flight_t.square()) / torch.clamp(flight_t, min=1e-6)
        fallback = vy_exact <= 1e-6
        vx_nom = torch.where(fallback, vx_min, vx_nom)
        vy_nom = torch.where(fallback, vy_min, vy_exact)

        self.plan_valid[env_ids] = valid
        self.plan_vx[env_ids] = torch.where(valid, vx_nom, torch.zeros_like(vx_nom))
        self.plan_vy[env_ids] = torch.where(valid, vy_nom, torch.zeros_like(vy_nom))
        alpha = torch.atan2(vy_nom, vx_nom)
        self.plan_takeoff_theta[env_ids] = torch.where(
            valid, math.pi / 2.0 - alpha, torch.full_like(alpha, math.pi / 4.0)
        )

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = self._all_ids()
        if env_ids.numel() == 0:
            return self.observe()
        n = env_ids.numel()
        lo = math.radians(self.e.ground_init_theta_min_deg)
        hi = math.radians(self.e.ground_init_theta_max_deg)
        theta = lo + (hi - lo) * torch.rand(n, generator=self.generator, device=self.device)
        leg = min(
            max(self.p.leg_length - self.e.ground_init_leg_compression, self.p.leg_min_length),
            self.p.leg_length,
        )

        start_idx = torch.zeros(n, dtype=torch.long, device=self.device)
        if self.random_start_probability > 0.0 and self.target_count > 1:
            random_mask = torch.rand(
                n, generator=self.generator, device=self.device
            ) < self.random_start_probability
            random_idx = torch.randint(
                0, self.target_count, (n,), generator=self.generator, device=self.device
            )
            start_idx = torch.where(random_mask, random_idx, start_idx)

        self.q[env_ids] = 0.0
        self.q[env_ids, 0] = start_idx.to(self.dtype) * self.e.target_spacing
        self.q[env_ids, 1] = self.p.ground_y
        self.q[env_ids, 2] = theta
        self.q[env_ids, 3] = leg
        self.dq[env_ids] = 0.0
        self.filtered_u[env_ids] = 0.0
        self.stance[env_ids] = True
        self.prev_touching[env_ids] = True
        self.stance_substeps[env_ids] = 0
        self.stance_min_l[env_ids] = leg
        self.consecutive_stance_steps[env_ids] = 1
        self.steps[env_ids] = 0
        self.target_idx[env_ids] = start_idx
        self.best_foot_x[env_ids] = self.q[env_ids, 0]
        self.no_progress_counter[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.episode_length[env_ids] = 0
        self.prev_vy[env_ids] = 0.0
        reset_com_y = self.q[env_ids, 1] + self.q[env_ids, 3] * torch.cos(self.q[env_ids, 2])
        self.hop_max_height[env_ids] = reset_com_y
        self._plan(env_ids)
        return self.observe()

    def _thrust(self, command: torch.Tensor) -> torch.Tensor:
        low = self.p.thrust_a * command.square() + self.p.thrust_b * command + self.p.thrust_c
        high = self.p.thrust_offset_high + (
            command - self.p.thrust_breakpoint
        ) * self.p.thrust_slope_high
        force = torch.where(command <= self.p.thrust_breakpoint, low, high)
        return torch.clamp(force, 0.0, self.p.thrust_max_per_motor) * self.p.n_motors_per_side

    def _physics_substep(self, f1: torch.Tensor, f2: torch.Tensor, dt: float) -> None:
        total = f1 + f2
        torque = (f2 - f1) * self.p.cg_to_motor

        # Touchdown impulse and foot pinning.
        touchdown = (~self.stance) & (self.q[:, 1] <= self.p.ground_y)
        theta = self.q[:, 2]
        leg = self.q[:, 3]
        s, c = torch.sin(theta), torch.cos(theta)
        dx, dy, dl = self.dq[:, 0], self.dq[:, 1], self.dq[:, 3]
        j_eff = self.p.inertia + self.p.mass * leg.square()
        radial = dl + s * dx + c * dy
        delta_w = leg * self.p.mass * (c * dx - s * dy) / j_eff
        self.dq[:, 3] = torch.where(touchdown, radial, self.dq[:, 3])
        self.dq[:, 2] = torch.where(
            touchdown,
            self.dq[:, 2] + self.p.impact_angular_restitution * delta_w,
            self.dq[:, 2],
        )
        self.q[:, 1] = torch.where(touchdown, torch.full_like(theta, self.p.ground_y), self.q[:, 1])
        self.dq[:, 0] = torch.where(touchdown, torch.zeros_like(dx), dx)
        self.dq[:, 1] = torch.where(touchdown, torch.zeros_like(dy), dy)
        self.stance |= touchdown
        self.stance_substeps = torch.where(
            touchdown, torch.zeros_like(self.stance_substeps), self.stance_substeps
        )
        self.stance_min_l = torch.where(touchdown, leg, self.stance_min_l)

        stance_before_liftoff = self.stance.clone()
        self.stance_substeps += stance_before_liftoff.long()
        self.stance_min_l = torch.where(
            stance_before_liftoff,
            torch.minimum(self.stance_min_l, self.q[:, 3]),
            self.stance_min_l,
        )
        compression = torch.clamp(self.p.leg_length - self.q[:, 3], 0.0, self.p.stroke_length)
        spring = torch.clamp(
            self.p.k_slip * compression - self.p.c_slip * self.dq[:, 3] + self.p.spring_preload,
            min=0.0,
        )
        f_leg = total + torch.where(stance_before_liftoff, spring, torch.zeros_like(spring))
        liftoff = (
            stance_before_liftoff
            & (self.q[:, 3] >= self.p.leg_length)
            & (self.dq[:, 3] > 0.0)
            & (self.stance_substeps >= self.p.min_stance_substeps)
        )
        self.stance &= ~liftoff

        theta = self.q[:, 2]
        dtheta = self.dq[:, 2]
        theta_err = torch.remainder(
            theta - math.radians(self.r.phi_td_target_deg) + math.pi, 2.0 * math.pi
        ) - math.pi
        tau_att = torch.clamp(
            -self.p.flight_att_kp * theta_err - self.p.flight_att_kd * dtheta,
            -self.p.flight_att_tau_limit,
            self.p.flight_att_tau_limit,
        )
        tau_att = torch.where(~self.stance, tau_att, torch.zeros_like(tau_att))

        # Flight EOM in COM coordinates.
        flight = ~self.stance
        s, c = torch.sin(theta), torch.cos(theta)
        leg_nom = self.p.leg_length
        vx_com = self.dq[:, 0] + self.dq[:, 3] * s + leg_nom * c * dtheta
        vy_com = self.dq[:, 1] + self.dq[:, 3] * c - leg_nom * s * dtheta
        com_x = self.q[:, 0] + leg_nom * s
        com_y = self.q[:, 1] + leg_nom * c
        vx_new = vx_com + total * s / self.p.mass * dt
        vy_new = vy_com + (total * c / self.p.mass - self.p.gravity) * dt
        w_new = dtheta + (torque + tau_att) / self.p.inertia * dt
        theta_new = theta + w_new * dt
        com_x_new = com_x + vx_new * dt
        com_y_new = com_y + vy_new * dt
        s_new, c_new = torch.sin(theta_new), torch.cos(theta_new)
        foot_x_new = com_x_new - leg_nom * s_new
        foot_y_new = com_y_new - leg_nom * c_new
        foot_vx_new = vx_new - leg_nom * c_new * w_new
        foot_vy_new = vy_new + leg_nom * s_new * w_new

        self.q[:, 0] = torch.where(flight, foot_x_new, self.q[:, 0])
        self.q[:, 1] = torch.where(flight, foot_y_new, self.q[:, 1])
        self.q[:, 2] = torch.where(flight, theta_new, self.q[:, 2])
        self.dq[:, 0] = torch.where(flight, foot_vx_new, self.dq[:, 0])
        self.dq[:, 1] = torch.where(flight, foot_vy_new, self.dq[:, 1])
        self.dq[:, 2] = torch.where(flight, w_new, self.dq[:, 2])

        # Stance inverted-pendulum polar EOM.
        stance = self.stance
        theta = self.q[:, 2]
        dtheta = self.dq[:, 2]
        leg = self.q[:, 3]
        dl = self.dq[:, 3]
        s, c = torch.sin(theta), torch.cos(theta)
        ddl = f_leg / self.p.mass - self.p.gravity * c + leg * dtheta.square()
        ddtheta = (
            torque
            + self.p.mass * self.p.gravity * leg * s
            - 2.0 * self.p.mass * leg * dl * dtheta
        ) / (self.p.inertia + self.p.mass * leg.square())
        w_stance = dtheta + ddtheta * dt
        dl_stance = dl + ddl * dt
        theta_stance = theta + w_stance * dt
        leg_stance = torch.clamp(
            leg + dl_stance * dt,
            self.p.leg_min_length,
            self.p.leg_length + self.p.stroke_length,
        )
        at_limit = (
            ((leg_stance <= self.p.leg_min_length) & (dl_stance < 0.0))
            | ((leg_stance >= self.p.leg_length + self.p.stroke_length) & (dl_stance > 0.0))
        )
        dl_stance = torch.where(at_limit, torch.zeros_like(dl_stance), dl_stance)
        self.q[:, 1] = torch.where(stance, torch.full_like(theta, self.p.ground_y), self.q[:, 1])
        self.q[:, 2] = torch.where(stance, theta_stance, self.q[:, 2])
        self.q[:, 3] = torch.where(stance, leg_stance, torch.full_like(leg, leg_nom))
        self.dq[:, 0] = torch.where(stance, torch.zeros_like(dtheta), self.dq[:, 0])
        self.dq[:, 1] = torch.where(stance, torch.zeros_like(dtheta), self.dq[:, 1])
        self.dq[:, 2] = torch.where(stance, w_stance, self.dq[:, 2])
        self.dq[:, 3] = torch.where(stance, dl_stance, torch.zeros_like(dl))

    def _reward(
        self,
        motor: torch.Tensor,
        touching: torch.Tensor,
        touchdown: torch.Tensor,
        liftoff: torch.Tensor,
        apex_event: torch.Tensor,
        target_hit: torch.Tensor,
        all_done: torch.Tensor,
        terminated_bad: torch.Tensor,
        out_of_bounds: torch.Tensor,
        stance_timeout: torch.Tensor,
        target_x: torch.Tensor,
        dx_target: torch.Tensor,
        com_y: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
    ) -> torch.Tensor:
        c = self.r
        reward = torch.zeros(self.num_envs, device=self.device)

        launch = liftoff & self.plan_valid & (dx_target > 0.1)
        speed = torch.sqrt(vx.square() + vy.square())
        speed_nom = torch.sqrt(self.plan_vx.square() + self.plan_vy.square())
        alpha = torch.atan2(vy, vx)
        alpha_nom = torch.atan2(self.plan_vy, self.plan_vx)
        alpha_deg = torch.rad2deg(alpha)
        sector = (alpha_deg >= c.alpha_min_deg) & (alpha_deg <= c.alpha_max_deg)
        v_err = (speed - speed_nom) / torch.clamp(speed_nom, min=0.1)
        reward += torch.where(
            launch & sector & (speed_nom > 0.1),
            c.liftoff_v_weight * torch.exp(-c.liftoff_v_sharpness * v_err.square()),
            torch.zeros_like(reward),
        )
        wrap = lambda x: torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi
        alpha_err = torch.abs(wrap(alpha - alpha_nom))
        center = math.radians(0.5 * (c.alpha_min_deg + c.alpha_max_deg))
        angle_term = torch.where(
            sector,
            c.liftoff_angle_weight * torch.exp(-c.liftoff_angle_sharpness * alpha_err),
            -c.liftoff_angle_weight * 0.3 * torch.abs(wrap(alpha - center)),
        )
        reward += torch.where(launch, angle_term, torch.zeros_like(reward))

        theta, dtheta, leg, dl = self.q[:, 2], self.dq[:, 2], self.q[:, 3], self.dq[:, 3]
        stance_term = c.stance_pendulum_weight * torch.clamp(dtheta, min=0.0)
        stance_term += torch.where(
            theta > 0.0, c.stance_theta_pos_weight * torch.sin(theta), torch.zeros_like(theta)
        )
        compression = torch.clamp((self.p.leg_length - leg) / max(self.p.stroke_length, 1e-6), min=0.0)
        spring_term = torch.where(
            theta < 0.0,
            c.stance_spring_weight * torch.tanh(4.0 * compression),
            c.stance_spring_weight * 0.5 * torch.clamp(dl, min=0.0),
        )
        reward += torch.where(touching, stance_term + spring_term - c.stance_stall_penalty, 0.0)
        reward += torch.where(stance_timeout, -c.stance_timeout_penalty, 0.0)

        progress = torch.where(
            vx >= 0.0,
            c.forward_progress_weight * torch.clamp(vx, max=2.5),
            c.backward_progress_penalty * vx,
        )
        reward += torch.where(dx_target > 0.05, progress, 0.0)

        phi_td = math.radians(c.phi_td_target_deg)
        td_err = torch.abs(wrap(theta - phi_td))
        td_att = torch.where(
            theta < 0.0,
            c.touchdown_weight * torch.exp(-c.touchdown_sharpness * td_err),
            torch.full_like(theta, -c.touchdown_bad_penalty),
        )
        landing = c.landing_proximity_weight * torch.exp(
            -c.landing_proximity_sharpness * (self.q[:, 0] - target_x).square()
        )
        reward += torch.where(touchdown, td_att + landing, 0.0)

        height_err = com_y - c.target_height
        flight_height = c.flight_height_weight * torch.exp(-c.flight_height_sharpness * height_err.square())
        flight_height -= torch.where(
            height_err > 0.0, c.overheight_penalty_weight * height_err.square(), 0.0
        )
        att = c.flight_attitude_weight * torch.exp(
            -c.flight_attitude_sharpness * torch.abs(wrap(theta - phi_td))
        )
        thrust_penalty = -c.flight_thrust_penalty * motor.sum(dim=1)
        reward += torch.where(~touching, flight_height + att + thrust_penalty, 0.0)
        apex_err = com_y - c.target_height
        apex_reward = c.apex_height_weight * torch.exp(
            -c.apex_height_sharpness * apex_err.square()
        )
        apex_reward -= c.apex_height_error_penalty * apex_err.square()
        reward += torch.where(apex_event, apex_reward, 0.0)

        reward += target_hit.to(self.dtype) * c.target_hit_reward
        reward += all_done.to(self.dtype) * c.all_targets_bonus
        reward -= terminated_bad.to(self.dtype) * c.termination_penalty
        reward -= out_of_bounds.to(self.dtype) * c.out_of_bounds_penalty
        return reward

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        motor = torch.clamp(0.5 * (action + 1.0), 0.0, 1.0)
        alpha = self.p.dt / (self.p.motor_tau + self.p.dt)
        self.filtered_u.mul_(1.0 - alpha).add_(motor, alpha=alpha)
        force = self._thrust(self.filtered_u)

        previous = self.prev_touching.clone()
        dt_inner = self.p.dt / _N_SUBSTEPS
        for _ in range(_N_SUBSTEPS):
            self._physics_substep(force[:, 0], force[:, 1], dt_inner)
        touching = self.stance.clone()
        touchdown = touching & ~previous
        liftoff = ~touching & previous

        self.consecutive_stance_steps = torch.where(
            touching, self.consecutive_stance_steps + 1, torch.zeros_like(self.consecutive_stance_steps)
        )
        stance_timeout = self.consecutive_stance_steps >= self.e.max_consecutive_stance_steps
        com_x, com_y, vx, vy = self._com_state()
        self.hop_max_height = torch.maximum(self.hop_max_height, com_y)
        apex_event = (~touching) & (self.prev_vy > 0.0) & (vy <= 0.0)
        target_x, target_valid = self._target_x()
        dx_target = target_x - self.q[:, 0]
        distance = torch.abs(self.q[:, 0] - target_x)

        progressed = self.q[:, 0] > self.best_foot_x + self.e.no_progress_min_delta
        self.best_foot_x = torch.where(progressed, self.q[:, 0], self.best_foot_x)
        self.no_progress_counter = torch.where(
            progressed, torch.zeros_like(self.no_progress_counter), self.no_progress_counter + 1
        )
        no_progress = (
            target_valid
            & (self.no_progress_counter >= self.e.no_progress_steps)
            & (distance > self.e.target_tolerance)
        )

        target_hit = (
            touching
            & (vy > -0.5)
            & target_valid
            & (distance < self.e.target_tolerance)
            & (self.hop_max_height >= self.e.min_target_hop_height)
            & (self.hop_max_height <= self.e.max_target_hop_height)
        )
        self.target_idx += target_hit.long()
        all_targets_done = target_hit & (self.target_idx >= self.target_count)
        next_target = target_hit & ~all_targets_done
        self.hop_max_height = torch.where(
            next_target, com_y, self.hop_max_height
        )
        self._plan(torch.nonzero(next_target, as_tuple=False).squeeze(1))

        terminated_bad = (
            (com_y > self.r.max_height)
            | (com_y < self.p.min_com_height)
            | (torch.abs(self.q[:, 2]) > self.r.max_tilt_rad)
        )
        out_of_bounds = (self.q[:, 0] < -1.0) | (
            target_valid & (self.q[:, 0] > target_x + self.r.max_overshoot)
        )
        terminated = all_targets_done | terminated_bad | out_of_bounds | stance_timeout | no_progress

        reward = self._reward(
            motor, touching, touchdown, liftoff, apex_event, target_hit, all_targets_done,
            terminated_bad | no_progress, out_of_bounds, stance_timeout,
            target_x, dx_target, com_y, vx, vy,
        )
        self.steps += 1
        self.episode_length += 1
        self.episode_return += reward
        truncated = self.steps >= self.max_episode_steps
        done = terminated | truncated
        completed_return = torch.where(done, self.episode_return, torch.full_like(reward, torch.nan))
        completed_length = torch.where(done, self.episode_length, torch.zeros_like(self.episode_length))
        hits = target_hit.sum()
        self.prev_touching = touching
        self.prev_vy = vy

        done_ids = torch.nonzero(done, as_tuple=False).squeeze(1)
        self.reset(done_ids)
        info = {
            "episode_return": completed_return,
            "episode_length": completed_length,
            "target_hits": hits,
            "target_hit_mask": target_hit,
        }
        return self.observe(), reward, done, info

    @torch.no_grad()
    def observe(self) -> torch.Tensor:
        _, com_y, vx, vy = self._com_state()
        target_x, valid = self._target_x()
        dx_target = torch.where(valid, target_x - self.q[:, 0], torch.zeros_like(target_x))
        active_plan = self.plan_valid & self.stance
        vx_def = torch.where(active_plan, self.plan_vx - vx, torch.zeros_like(vx))
        vy_def = torch.where(active_plan, self.plan_vy - vy, torch.zeros_like(vy))
        phi_td = math.radians(self.r.phi_td_target_deg)
        stance_ratio = torch.clamp(
            self.consecutive_stance_steps.to(self.dtype) / max(self.e.max_consecutive_stance_steps, 1),
            0.0,
            1.0,
        )
        obs = torch.stack(
            (
                self.q[:, 2],
                self.dq[:, 2],
                self.q[:, 3] / self.p.leg_length - 1.0,
                self.dq[:, 3],
                vx,
                vy,
                com_y,
                self.r.target_height - com_y,
                vx_def,
                vy_def,
                dx_target,
                self.stance.to(self.dtype),
                torch.zeros_like(vx),  # task phase: do not expose absolute hop index
                self.q[:, 2] - phi_td,
                stance_ratio,
            ),
            dim=1,
        )
        return torch.clamp(obs, -self.e.obs_clip, self.e.obs_clip)
