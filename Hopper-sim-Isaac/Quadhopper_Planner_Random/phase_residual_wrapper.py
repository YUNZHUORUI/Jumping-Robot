"""Phase-conditioned motor residual on top of the frozen two-hop teacher."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class TeacherTwoHopPhaseResidualVecEnv(VecEnv):
    """Learn small motor corrections throughout the whole hop.

    The teacher remains the stability core.  The residual policy sees the
    current/next waypoint geometry and contact-flight phase, then outputs
    collective/roll/pitch/yaw corrections.  Phase-specific limits give it
    enough stance authority to adjust takeoff impulse while keeping the action
    close to the accepted teacher.
    """

    EXTRA_OBSERVATIONS = 4 + 4 + 2 + 1 + 2 + 2 + 2 + 2 + 2 + 3

    def __init__(
        self,
        base_env,
        teacher_model,
        collective_limits=(0.060, 0.040, 0.025, 0.030),
        attitude_limits=(0.050, 0.040, 0.030, 0.040),
        yaw_limits=(0.020, 0.015, 0.012, 0.015),
        residual_slew_rate: float = 0.020,
        residual_penalty_scale: float = 35.0,
        slew_penalty_scale: float = 8.0,
        landing_error_scale_m: float = 0.25,
        safety_gate_error_m: float = 0.08,
    ):
        self.base_env = base_env
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.collective_limits = torch.as_tensor(
            collective_limits, dtype=torch.float32, device=base_env.device
        )
        self.attitude_limits = torch.as_tensor(
            attitude_limits, dtype=torch.float32, device=base_env.device
        )
        self.yaw_limits = torch.as_tensor(
            yaw_limits, dtype=torch.float32, device=base_env.device
        )
        self.residual_slew_rate = float(residual_slew_rate)
        self.residual_penalty_scale = float(residual_penalty_scale)
        self.slew_penalty_scale = float(slew_penalty_scale)
        self.landing_error_scale_m = float(landing_error_scale_m)
        self.safety_gate_error_m = float(safety_gate_error_m)
        if self.landing_error_scale_m <= 0.0:
            raise ValueError("landing_error_scale_m must be positive")
        if self.safety_gate_error_m < 0.0:
            raise ValueError("safety_gate_error_m must be non-negative")
        if min(self.residual_slew_rate, self.residual_penalty_scale, self.slew_penalty_scale) < 0.0:
            raise ValueError("residual regularization terms must be non-negative")

        self.num_envs = base_env.num_envs
        self.num_actions = 4
        self.device = base_env.device
        self.max_episode_length = base_env.max_episode_length
        self._teacher_obs = base_env.get_observations()
        self._teacher_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_motor_residual = torch.zeros_like(self._teacher_actions)
        self._obs = self._residual_observations()

    @property
    def cfg(self):
        return self.base_env.cfg

    @property
    def episode_length_buf(self):
        return self.base_env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.base_env.episode_length_buf = value

    def seed(self, seed: int = -1):
        return self.base_env.seed(seed)

    def _yaw_basis(self) -> tuple[torch.Tensor, torch.Tensor]:
        quat = self.base_env.unwrapped._robot.data.root_quat_w
        w, x, y, z = quat.unbind(dim=1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.cos(yaw), torch.sin(yaw)

    def _world_to_yaw_body(self, vector_w: torch.Tensor) -> torch.Tensor:
        c, s = self._yaw_basis()
        return torch.stack(
            (
                c * vector_w[:, 0] + s * vector_w[:, 1],
                -s * vector_w[:, 0] + c * vector_w[:, 1],
            ),
            dim=1,
        )

    def _phase(self) -> torch.Tensor:
        core = self.base_env.unwrapped
        contact = core._previous_contact
        active = core._cycle_active
        vz = core._robot.data.root_lin_vel_w[:, 2]
        return torch.stack(
            (
                contact.float(),
                ((~contact) & (~active)).float(),
                ((~contact) & active & (vz >= 0.0)).float(),
                ((~contact) & active & (vz < 0.0)).float(),
            ),
            dim=1,
        )

    def _pair_context(self) -> torch.Tensor:
        core = self.base_env.unwrapped
        short = (core.commands.route_index % 2) == 0
        return torch.stack(
            (short.float(), (~short).float(), core._first_hop_hit_for_pair.float()),
            dim=1,
        )

    def _guidance(self):
        core = self.base_env.unwrapped
        pos = core._robot.data.root_pos_w
        vel = core._robot.data.root_lin_vel_w
        gravity = abs(core.sim.cfg.gravity[2]) if core.sim.cfg.gravity is not None else 9.81
        height = torch.clamp(pos[:, 2] - core.cfg.landing_root_height, min=0.0)
        ttl = (vel[:, 2] + torch.sqrt(vel[:, 2].square() + 2.0 * gravity * height)) / gravity
        p_t, p_t1 = core.commands.lookahead()
        projected_xy = pos[:, :2] + vel[:, :2] * ttl[:, None]
        projected_error_b = self._world_to_yaw_body(p_t - projected_xy)
        vxy_b = self._world_to_yaw_body(vel[:, :2])
        current_w = p_t - pos[:, :2]
        current_w = current_w / torch.linalg.norm(current_w, dim=1, keepdim=True).clamp_min(1.0e-6)
        next_w = p_t1 - p_t
        next_w = next_w / torch.linalg.norm(next_w, dim=1, keepdim=True).clamp_min(1.0e-6)
        current_b = self._world_to_yaw_body(current_w)
        next_b = self._world_to_yaw_body(next_w)
        axis_b = self._world_to_yaw_body(core._body_z_axis_w()[:, :2])
        spring = torch.stack(
            (
                core._robot.data.joint_pos[:, core._spring_joint_id],
                core._robot.data.joint_vel[:, core._spring_joint_id],
            ),
            dim=1,
        )
        return projected_error_b, ttl[:, None], vxy_b, current_b, next_b, axis_b, spring

    def _residual_observations(self) -> torch.Tensor:
        projected, ttl, vxy_b, current_b, next_b, axis_b, spring = self._guidance()
        core = self.base_env.unwrapped
        return torch.cat(
            (
                self._teacher_obs["policy"],
                self._teacher_actions,
                self._phase(),
                (projected / self.landing_error_scale_m).clamp(-4.0, 4.0),
                (ttl / core.cfg.max_flight_duration).clamp(0.0, 1.5),
                vxy_b,
                current_b,
                next_b,
                axis_b,
                spring,
                self._pair_context(),
            ),
            dim=1,
        )

    def _observation_dict(self):
        return TensorDict({"policy": self._obs}, batch_size=[self.num_envs])

    @staticmethod
    def mix_high_level_actions(actions: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Map collective/roll/pitch/yaw commands to four motor residuals."""
        collective = actions[:, 0:1] * scales[:, 0:1]
        roll = actions[:, 1:2] * scales[:, 1:2]
        pitch = actions[:, 2:3] * scales[:, 1:2]
        yaw = actions[:, 3:4] * scales[:, 2:3]
        return torch.cat(
            (
                collective + roll + pitch + yaw,
                collective - roll + pitch - yaw,
                collective - roll - pitch + yaw,
                collective + roll - pitch - yaw,
            ),
            dim=1,
        )

    def _phase_scales(self, phase: torch.Tensor, projected_error: torch.Tensor) -> torch.Tensor:
        collective = phase @ self.collective_limits
        attitude = phase @ self.attitude_limits
        yaw = phase @ self.yaw_limits
        scales = torch.stack((collective, attitude, yaw), dim=1)
        if self.safety_gate_error_m > 0.0:
            gate = (projected_error.norm(dim=1) / self.safety_gate_error_m).clamp(0.25, 1.0)
            airborne = (1.0 - phase[:, 0]).clamp(0.0, 1.0)
            scales = scales * (phase[:, 0:1] + airborne[:, None] * gate[:, None])
        return scales

    def reset(self):
        self._teacher_obs, extras = self.base_env.reset()
        with torch.inference_mode():
            self.teacher_model.reset()
        self._teacher_actions.zero_()
        self._previous_motor_residual.zero_()
        self._obs = self._residual_observations()
        return self._observation_dict(), extras

    def get_observations(self):
        self._teacher_obs = self.base_env.get_observations()
        self._obs = self._residual_observations()
        return self._observation_dict()

    def step(self, residual_actions: torch.Tensor):
        phase = self._phase()
        projected, *_ = self._guidance()
        scales = self._phase_scales(phase, projected)
        requested = self.mix_high_level_actions(residual_actions.clamp(-1.0, 1.0), scales)
        delta = (requested - self._previous_motor_residual).clamp(
            -self.residual_slew_rate, self.residual_slew_rate
        )
        motor_residual = self._previous_motor_residual + delta
        with torch.inference_mode():
            teacher_raw = self.teacher_model.act_inference(self._teacher_obs)
        self._teacher_actions = teacher_raw.clamp(-1.0, 1.0)
        motor_actions = teacher_raw + motor_residual

        self._teacher_obs, reward, dones, extras = self.base_env.step(motor_actions)
        residual_cost = self.residual_penalty_scale * motor_residual.square().sum(dim=1)
        slew_cost = self.slew_penalty_scale * delta.square().sum(dim=1)
        reward = reward - residual_cost - slew_cost
        with torch.inference_mode():
            self.teacher_model.reset(dones)
        self._previous_motor_residual = motor_residual
        self._previous_motor_residual[dones.reshape(-1) > 0] = 0.0
        self._obs = self._residual_observations()

        core = self.base_env.unwrapped
        attempts = core._conditional_second_attempts.sum().float()
        pair_attempts = core._pair_attempts.sum().float()
        log = extras.setdefault("log", {})
        log["Metrics/conditional_second_hit_rate"] = core._conditional_second_hits.sum().float() / attempts.clamp_min(1.0)
        log["Metrics/two_hop_pair_success_rate"] = core._pair_hits.sum().float() / pair_attempts.clamp_min(1.0)
        log["Metrics/phase_residual_motor_abs_mean"] = motor_residual.abs().mean()
        log["Metrics/phase_residual_motor_abs_max"] = motor_residual.abs().max()
        log["Metrics/phase_residual_slew_abs_mean"] = delta.abs().mean()
        log["Metrics/phase_residual_regularization"] = residual_cost.mean()
        log["Metrics/phase_residual_slew_regularization"] = slew_cost.mean()
        log["Metrics/phase_residual_stance_duty"] = phase[:, 0].mean()
        log["Metrics/phase_residual_clip_fraction"] = (
            ((motor_actions < -1.0) | (motor_actions > 1.0)).float().mean()
        )
        return self._observation_dict(), reward, dones, extras

    def close(self):
        return self.base_env.close()
