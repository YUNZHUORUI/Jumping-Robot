"""Bounded high-level residual control on top of a frozen two-hop teacher."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class TeacherTwoHopResidualVecEnv(VecEnv):
    """Learn collective/roll/pitch corrections without changing motor physics.

    The frozen teacher still receives its original 43-D observation.  The
    residual receives that observation plus the teacher action, a four-state
    contact/flight phase, predicted touchdown error, time-to-touchdown, and
    the body-Z error relative to the attitude which prepares the next hop.
    Its three high-level outputs are mixed into the existing four direct motor
    commands and tightly bounded before the unchanged actuator layer.
    """

    EXTRA_OBSERVATIONS = 4 + 4 + 2 + 1 + 2

    def __init__(
        self,
        base_env,
        teacher_model,
        collective_scale: float = 0.03,
        attitude_scale: float = 0.05,
        residual_slew_rate: float = 0.02,
        residual_penalty_scale: float = 50.0,
        slew_penalty_scale: float = 10.0,
        landing_error_scale_m: float = 0.25,
        next_tilt_rad: float = 0.105,
    ):
        self.base_env = base_env
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.collective_scale = float(collective_scale)
        self.attitude_scale = float(attitude_scale)
        self.residual_slew_rate = float(residual_slew_rate)
        self.residual_penalty_scale = float(residual_penalty_scale)
        self.slew_penalty_scale = float(slew_penalty_scale)
        self.landing_error_scale_m = float(landing_error_scale_m)
        self.next_tilt_rad = float(next_tilt_rad)
        if min(
            self.collective_scale,
            self.attitude_scale,
            self.residual_slew_rate,
            self.residual_penalty_scale,
            self.slew_penalty_scale,
        ) < 0.0:
            raise ValueError("Residual scales and slew rate must be non-negative")
        if self.landing_error_scale_m <= 0.0:
            raise ValueError("landing_error_scale_m must be positive")

        self.num_envs = base_env.num_envs
        self.num_actions = 3
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

    def _phase(self) -> torch.Tensor:
        core = self.base_env.unwrapped
        contact = core._previous_contact
        active = core._cycle_active
        vz = core._robot.data.root_lin_vel_w[:, 2]
        # contact, takeoff transition, ascent, descent
        return torch.stack(
            (
                contact.float(),
                ((~contact) & (~active)).float(),
                ((~contact) & active & (vz >= 0.0)).float(),
                ((~contact) & active & (vz < 0.0)).float(),
            ),
            dim=1,
        )

    def _guidance(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        core = self.base_env.unwrapped
        pos = core._robot.data.root_pos_w
        vel = core._robot.data.root_lin_vel_w
        gravity = abs(core.sim.cfg.gravity[2]) if core.sim.cfg.gravity is not None else 9.81
        height = torch.clamp(pos[:, 2] - core.cfg.landing_root_height, min=0.0)
        time_to_land = (
            vel[:, 2]
            + torch.sqrt(torch.square(vel[:, 2]) + 2.0 * gravity * height)
        ) / gravity
        p_t, p_t1 = core.commands.lookahead()
        projected_xy = pos[:, :2] + vel[:, :2] * time_to_land[:, None]
        error_w = p_t - projected_xy

        quat = core._robot.data.root_quat_w
        w, x, y, z = quat.unbind(dim=1)
        # Rotate world XY vectors into body yaw frame.  Roll/pitch components
        # are intentionally excluded from waypoint geometry.
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = torch.cos(yaw), torch.sin(yaw)
        error_b = torch.stack(
            (c * error_w[:, 0] + s * error_w[:, 1],
             -s * error_w[:, 0] + c * error_w[:, 1]),
            dim=1,
        ) / self.landing_error_scale_m

        next_direction = p_t1 - p_t
        next_direction = next_direction / torch.linalg.norm(
            next_direction, dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        desired_axis = torch.cat(
            (
                torch.sin(torch.as_tensor(self.next_tilt_rad, device=self.device))
                * next_direction,
                torch.cos(torch.as_tensor(self.next_tilt_rad, device=self.device))
                .expand(self.num_envs, 1),
            ),
            dim=1,
        )
        body_axis = core._body_z_axis_w()
        axis_error_w = desired_axis[:, :2] - body_axis[:, :2]
        axis_error_b = torch.stack(
            (c * axis_error_w[:, 0] + s * axis_error_w[:, 1],
             -s * axis_error_w[:, 0] + c * axis_error_w[:, 1]),
            dim=1,
        )
        normalized_time = (time_to_land / core.cfg.max_flight_duration).clamp(0.0, 1.5)[:, None]
        return error_b.clamp(-4.0, 4.0), normalized_time, axis_error_b

    def _residual_observations(self) -> torch.Tensor:
        landing_error, time_to_land, attitude_error = self._guidance()
        return torch.cat(
            (
                self._teacher_obs["policy"],
                self._teacher_actions.clamp(-1.0, 1.0),
                self._phase(),
                landing_error,
                time_to_land,
                attitude_error,
            ),
            dim=1,
        )

    def _observation_dict(self) -> TensorDict:
        return TensorDict({"policy": self._obs}, batch_size=[self.num_envs])

    @staticmethod
    def mix_high_level_actions(
        actions: torch.Tensor, collective_scale: float, attitude_scale: float
    ) -> torch.Tensor:
        """Map collective/roll/pitch commands to F1,F2,F3,F4 residuals."""
        collective = actions[:, 0:1] * collective_scale
        roll = actions[:, 1:2] * attitude_scale
        pitch = actions[:, 2:3] * attitude_scale
        return torch.cat(
            (
                collective + roll + pitch,
                collective - roll + pitch,
                collective - roll - pitch,
                collective + roll - pitch,
            ),
            dim=1,
        )

    def reset(self):
        self._teacher_obs, extras = self.base_env.reset()
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
        with torch.inference_mode():
            teacher_raw = self.teacher_model.act_inference(self._teacher_obs)
        self._teacher_actions = teacher_raw.clamp(-1.0, 1.0)
        requested = self.mix_high_level_actions(
            residual_actions.clamp(-1.0, 1.0),
            self.collective_scale,
            self.attitude_scale,
        )
        delta = (requested - self._previous_motor_residual).clamp(
            -self.residual_slew_rate, self.residual_slew_rate
        )
        motor_residual = self._previous_motor_residual + delta
        motor_actions = teacher_raw + motor_residual

        self._teacher_obs, reward, dones, extras = self.base_env.step(motor_actions)
        residual_cost = self.residual_penalty_scale * torch.sum(
            torch.square(motor_residual), dim=1
        )
        slew_cost = self.slew_penalty_scale * torch.sum(torch.square(delta), dim=1)
        reward = reward - residual_cost - slew_cost
        self.teacher_model.reset(dones)
        self._previous_motor_residual = motor_residual
        self._previous_motor_residual[dones] = 0.0
        self._obs = self._residual_observations()

        log = extras.setdefault("log", {})
        log["Metrics/residual_motor_abs_mean"] = motor_residual.abs().mean()
        log["Metrics/residual_motor_abs_max"] = motor_residual.abs().max()
        log["Metrics/residual_slew_abs_mean"] = delta.abs().mean()
        log["Metrics/residual_regularization"] = residual_cost.mean()
        log["Metrics/residual_slew_regularization"] = slew_cost.mean()
        log["Metrics/residual_action_clip_fraction"] = (
            ((residual_actions < -1.0) | (residual_actions > 1.0)).float().mean()
        )
        log["Metrics/combined_action_clip_fraction"] = (
            ((motor_actions < -1.0) | (motor_actions > 1.0)).float().mean()
        )
        teacher_saturated = (teacher_raw < -1.0) | (teacher_raw > 1.0)
        combined_saturated = (motor_actions < -1.0) | (motor_actions > 1.0)
        log["Metrics/residual_induced_clip_fraction"] = (
            ((~teacher_saturated) & combined_saturated).float().mean()
        )
        return self._observation_dict(), reward, dones, extras

    def close(self):
        return self.base_env.close()
