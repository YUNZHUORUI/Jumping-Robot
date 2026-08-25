"""RSL-RL VecEnv adapter that freezes a 4-D teacher and learns 1-D collective residual."""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv


class TeacherCollectiveResidualVecEnv(VecEnv):
    """Expose one residual action while stepping the original four-motor environment."""

    def __init__(
        self,
        base_env,
        teacher_model,
        residual_scale: float = 2.0,
        stance_only: bool = True,
        normalize_height_command: bool = False,
        height_command_center: float = 1.075,
        height_command_half_range: float = 0.075,
        height_bias_high: float = 0.0,
        height_bias_low: float = 0.0,
    ):
        self.base_env = base_env
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.residual_scale = float(residual_scale)
        self.stance_only = bool(stance_only)
        self.normalize_height_command = bool(normalize_height_command)
        self.height_command_center = float(height_command_center)
        self.height_command_half_range = float(height_command_half_range)
        self.height_bias_high = float(height_bias_high)
        self.height_bias_low = float(height_bias_low)
        if self.normalize_height_command and self.height_command_half_range <= 0.0:
            raise ValueError("height_command_half_range must be positive")
        self.num_envs = base_env.num_envs
        self.num_actions = 1
        self.device = base_env.device
        self.max_episode_length = base_env.max_episode_length
        self._teacher_obs = base_env.get_observations()
        self._obs = self._residual_observations(self._teacher_obs)

    def _residual_observations(self, teacher_obs: torch.Tensor) -> torch.Tensor:
        """Give the residual a well-scaled height command without changing teacher input."""
        if not self.normalize_height_command:
            return teacher_obs
        residual_obs = teacher_obs.clone()
        target_height_m = teacher_obs[:, -1] * 2.0
        residual_obs[:, -1] = (
            target_height_m - self.height_command_center
        ) / self.height_command_half_range
        return residual_obs

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

    def reset(self):
        self._teacher_obs, extras = self.base_env.reset()
        self._obs = self._residual_observations(self._teacher_obs)
        self.teacher_model.reset()
        return self._obs, extras

    def get_observations(self):
        self._teacher_obs = self.base_env.get_observations()
        self._obs = self._residual_observations(self._teacher_obs)
        return self._obs

    def step(self, residual_actions: torch.Tensor):
        with torch.inference_mode():
            teacher_raw_actions = self.teacher_model.act_inference(self._teacher_obs)
            # This reproduces the action actually executed by the original
            # environment.  The recurrent actor mean is not bounded and v10
            # relied on QuadhopperEnv._pre_physics_step() for this clamp.
            teacher_actions = teacher_raw_actions.clamp(-1.0, 1.0)
        core = self.base_env.unwrapped
        target_height_m = core._height_commands()[0][:, None]
        high_weight = (
            (target_height_m - (self.height_command_center - self.height_command_half_range))
            / (2.0 * self.height_command_half_range)
        ).clamp(0.0, 1.0)
        feedforward_bias = (
            self.height_bias_low
            + high_weight * (self.height_bias_high - self.height_bias_low)
        )
        requested_collective = residual_actions.clamp(-1.0, 1.0) * self.residual_scale
        if self.stance_only:
            joint_pos = core._robot.data.joint_pos[:, core._spring_joint_id]
            stance_mask = (joint_pos > 0.002).float()[:, None]
            collective = requested_collective * stance_mask
        else:
            stance_mask = torch.ones_like(requested_collective)
            collective = requested_collective
        # The calibrated feasibility scan applies a feed-forward bias to the
        # raw actor output and lets the physical environment perform the final
        # clamp.  Clamping before a large negative bias destroys differential
        # attitude authority and is not equivalent near saturation.
        feedforward_teacher_actions = (
            teacher_raw_actions if (self.height_bias_high != 0.0 or self.height_bias_low != 0.0)
            else teacher_actions
        )
        motor_actions = (
            feedforward_teacher_actions
            + feedforward_bias.expand(-1, 4)
            + collective.expand(-1, 4)
        )
        self._teacher_obs, reward, dones, extras = self.base_env.step(motor_actions)
        self._obs = self._residual_observations(self._teacher_obs)
        self.teacher_model.reset(dones)
        extras.setdefault("log", {})
        extras["log"]["Metrics/mean_collective_residual"] = collective.abs().mean()
        extras["log"]["Metrics/signed_collective_residual"] = collective.mean()
        extras["log"]["Metrics/mean_requested_collective_residual"] = (
            requested_collective.abs().mean()
        )
        extras["log"]["Metrics/signed_requested_collective_residual"] = (
            requested_collective.mean()
        )
        extras["log"]["Metrics/height_feedforward_bias"] = feedforward_bias.mean()
        extras["log"]["Metrics/residual_stance_duty"] = stance_mask.mean()
        extras["log"]["Metrics/mean_teacher_action"] = teacher_actions.mean()
        extras["log"]["Metrics/teacher_raw_clip_fraction"] = (
            ((teacher_raw_actions < -1.0) | (teacher_raw_actions > 1.0)).float().mean()
        )
        extras["log"]["Metrics/final_action_clip_fraction"] = (
            ((motor_actions < -1.0) | (motor_actions > 1.0)).float().mean()
        )
        airborne = (~core._previous_contact).float()
        airborne_count = airborne.sum().clamp_min(1.0)
        extras["log"]["Metrics/airborne_motor_u"] = (
            core._motor_u.mean(dim=1) * airborne
        ).sum() / airborne_count
        return self._obs, reward, dones, extras

    def close(self):
        return self.base_env.close()
