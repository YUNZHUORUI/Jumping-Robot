"""High-level touchdown-state planner with a frozen low-level teacher."""

from __future__ import annotations

import math

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class TeacherTwoHopStatePlannerVecEnv(VecEnv):
    """Plan a touchdown state once per hop and track it deterministically.

    Actions are physical commands, not motor commands:
      0: next-direction touchdown velocity residual around the nominal speed
      1: lateral touchdown velocity [-max_lateral_speed, max_lateral_speed]
      2: next-direction body tilt residual around the nominal tilt
      3: lateral body tilt [-max_lateral_tilt, max_lateral_tilt]
    """

    # The final three values make the two-hop Semi-MDP state Markov:
    # [short-hop phase, long-hop phase, first hop already hit].  Without
    # these, the high-level policy had to infer pair eligibility indirectly
    # from geometry and could not specialize its second-hop preparation.
    EXTRA_OBSERVATIONS = 26

    def __init__(
        self,
        base_env,
        teacher_model,
        nominal_forward_speed: float = 0.08,
        nominal_forward_tilt_rad: float = math.radians(6.0),
        max_forward_speed_residual: float = 0.12,
        max_lateral_speed: float = 0.10,
        max_forward_tilt_residual_rad: float = math.radians(4.0),
        max_lateral_tilt_rad: float = math.radians(4.0),
        velocity_feedback_gain: float = 0.04,
        attitude_feedback_gain: float = 0.20,
        motor_correction_limit: float = 0.03,
        correction_mode: str = "descent",
        landing_correction_height: float = 0.18,
        state_reward_scale: float = 80.0,
        expert_anchor_scale: float = 20.0,
        stance_plan_rate: float = 0.10,
    ):
        self.base_env = base_env
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.nominal_forward_speed = float(nominal_forward_speed)
        self.nominal_forward_tilt_rad = float(nominal_forward_tilt_rad)
        self.max_forward_speed_residual = float(max_forward_speed_residual)
        self.max_lateral_speed = float(max_lateral_speed)
        self.max_forward_tilt_residual_rad = float(max_forward_tilt_residual_rad)
        self.max_lateral_tilt_rad = float(max_lateral_tilt_rad)
        self.velocity_feedback_gain = float(velocity_feedback_gain)
        self.attitude_feedback_gain = float(attitude_feedback_gain)
        self.motor_correction_limit = float(motor_correction_limit)
        if correction_mode not in ("none", "descent", "landing"):
            raise ValueError("correction_mode must be one of: none, descent, landing")
        self.correction_mode = correction_mode
        self.landing_correction_height = float(landing_correction_height)
        self.state_reward_scale = float(state_reward_scale)
        self.expert_anchor_scale = float(expert_anchor_scale)
        self.stance_plan_rate = float(stance_plan_rate)

        self.num_envs = base_env.num_envs
        self.num_actions = 4
        self.device = base_env.device
        self.max_episode_length = base_env.max_episode_length
        self._teacher_obs = base_env.get_observations()
        self._teacher_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._latched_command = torch.zeros(self.num_envs, 4, device=self.device)
        # A touchdown starts a new hop whose direction may differ by as much
        # as 180 degrees.  The first command for that hop must not be blended
        # with the previous hop's plan.
        self._needs_new_plan = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_active = self.base_env.unwrapped._cycle_active.clone()
        self._obs = self._planner_observations()

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

    def _yaw_basis(self):
        quat = self.base_env.unwrapped._robot.data.root_quat_w
        w, x, y, z = quat.unbind(dim=1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.cos(yaw), torch.sin(yaw)

    def _world_to_yaw_body(self, vector_w: torch.Tensor) -> torch.Tensor:
        c, s = self._yaw_basis()
        return torch.stack(
            (c * vector_w[:, 0] + s * vector_w[:, 1],
             -s * vector_w[:, 0] + c * vector_w[:, 1]), dim=1
        )

    def _phase(self):
        core = self.base_env.unwrapped
        contact = core._previous_contact
        active = core._cycle_active
        vz = core._robot.data.root_lin_vel_w[:, 2]
        return torch.stack(
            (contact.float(), ((~contact) & (~active)).float(),
             ((~contact) & active & (vz >= 0.0)).float(),
             ((~contact) & active & (vz < 0.0)).float()), dim=1
        )

    def _pair_context(self):
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
        projected_error_b = self._world_to_yaw_body(
            p_t - (pos[:, :2] + vel[:, :2] * ttl[:, None])
        ) / 0.25
        vxy_b = self._world_to_yaw_body(vel[:, :2])
        next_w = p_t1 - p_t
        next_w = next_w / torch.linalg.norm(next_w, dim=1, keepdim=True).clamp_min(1.0e-6)
        next_b = self._world_to_yaw_body(next_w)
        axis_b = self._world_to_yaw_body(core._body_z_axis_w()[:, :2])
        spring = core._robot.data.joint_pos[:, core._spring_joint_id]
        spring_vel = core._robot.data.joint_vel[:, core._spring_joint_id]
        return projected_error_b.clamp(-4.0, 4.0), (ttl / core.cfg.max_flight_duration)[:, None], vxy_b, next_b, axis_b, torch.stack((spring, spring_vel), dim=1)

    def _planner_observations(self):
        projected, ttl, vxy_b, next_b, axis_b, spring = self._guidance()
        return torch.cat((self._teacher_obs["policy"], self._teacher_actions,
                          self._phase(), projected, ttl, vxy_b, next_b,
                          axis_b, spring, self._latched_command,
                          self._pair_context()), dim=1)

    def _obs_dict(self):
        return TensorDict({"policy": self._obs}, batch_size=[self.num_envs])

    def _decode_command(self, actions):
        actions = actions.clamp(-1.0, 1.0)
        return torch.stack(
            (self.nominal_forward_speed + actions[:, 0] * self.max_forward_speed_residual,
             actions[:, 1] * self.max_lateral_speed,
             self.nominal_forward_tilt_rad + actions[:, 2] * self.max_forward_tilt_residual_rad,
             actions[:, 3] * self.max_lateral_tilt_rad), dim=1
        )

    def _motor_correction(self):
        core = self.base_env.unwrapped
        if self.motor_correction_limit <= 0.0 or self.correction_mode == "none":
            return torch.zeros(self.num_envs, 4, device=self.device)
        _, _, vxy_b, next_b, axis_b, _ = self._guidance()
        forward, lateral, tilt_forward, tilt_lateral = self._latched_command.unbind(dim=1)
        desired_v_b = forward[:, None] * next_b + lateral[:, None] * torch.stack((-next_b[:, 1], next_b[:, 0]), dim=1)
        desired_axis_b = tilt_forward[:, None] * next_b + tilt_lateral[:, None] * torch.stack((-next_b[:, 1], next_b[:, 0]), dim=1)
        velocity_error = desired_v_b - vxy_b
        axis_error = desired_axis_b - axis_b
        desired_horizontal_axis = self.velocity_feedback_gain * velocity_error + self.attitude_feedback_gain * axis_error
        roll = -desired_horizontal_axis[:, 1:2]
        pitch = desired_horizontal_axis[:, 0:1]
        correction = torch.cat((roll + pitch, -roll + pitch, -roll - pitch, roll - pitch), dim=1)
        root_pos_z = core._robot.data.root_pos_w[:, 2]
        root_vz = core._robot.data.root_lin_vel_w[:, 2]
        descent = core._cycle_active & (root_vz < 0.0)
        if self.correction_mode == "landing":
            near_ground = (
                root_pos_z
                < core.cfg.landing_root_height + self.landing_correction_height
            )
            spring_contact = (
                core._robot.data.joint_pos[:, core._spring_joint_id] > 0.002
            )
            active_mask = (descent & near_ground) | ((~core._cycle_active) & spring_contact)
        else:
            active_mask = descent
        return torch.where(active_mask[:, None], correction, torch.zeros_like(correction)).clamp(
            -self.motor_correction_limit, self.motor_correction_limit
        )

    def reset(self):
        self._teacher_obs, extras = self.base_env.reset()
        with torch.inference_mode():
            self.teacher_model.reset()
        self._teacher_actions.zero_()
        self._latched_command.zero_()
        self._needs_new_plan.fill_(True)
        self._previous_active = self.base_env.unwrapped._cycle_active.clone()
        self._obs = self._planner_observations()
        return self._obs_dict(), extras

    def get_observations(self):
        self._teacher_obs = self.base_env.get_observations()
        self._obs = self._planner_observations()
        return self._obs_dict()

    def step(self, planner_actions):
        core = self.base_env.unwrapped
        # Causally refine one plan throughout stance, then freeze it in flight.
        # Every PPO action used for learning therefore affects the executed
        # command; this removes the ignored-action credit mismatch in v41.
        stance = ~core._cycle_active
        decoded = self._decode_command(planner_actions)
        alpha = self.stance_plan_rate
        fresh_plan = stance & self._needs_new_plan
        refining = stance & (~self._needs_new_plan)
        self._latched_command[fresh_plan] = decoded[fresh_plan]
        self._latched_command[refining] = (
            (1.0 - alpha) * self._latched_command[refining]
            + alpha * decoded[refining]
        )
        self._needs_new_plan[fresh_plan] = False
        _, _, _, next_b_before, _, _ = self._guidance()
        with torch.inference_mode():
            teacher_raw = self.teacher_model.act_inference(self._teacher_obs)
        self._teacher_actions = teacher_raw.clamp(-1.0, 1.0)
        correction = self._motor_correction()
        self._teacher_obs, reward, dones, extras = self.base_env.step(teacher_raw + correction)
        self._previous_active = core._cycle_active.clone()

        touchdown = core._touchdown_event
        if torch.any(touchdown):
            _, _, vxy_b, _, axis_b, _ = self._guidance()
            next_b = next_b_before
            # This target is independent of the policy command.  The old loss
            # compared the measured state with the policy's own request, so the
            # policy could minimize it by requesting zero preparation.
            desired_v = self.nominal_forward_speed * next_b
            desired_axis = self.nominal_forward_tilt_rad * next_b
            state_error = (vxy_b - desired_v).square().sum(dim=1) + 4.0 * (axis_b - desired_axis).square().sum(dim=1)
            reward = reward - touchdown.float() * self.state_reward_scale * state_error
            self._needs_new_plan[touchdown] = True
        # The analytic expert is action=0 (nominal preparation). Anchor every
        # stance refinement action; all of these actions are actually applied.
        reward = reward - stance.float() * self.expert_anchor_scale * planner_actions.square().sum(dim=1)
        with torch.inference_mode():
            self.teacher_model.reset(dones)
        self._latched_command[dones] = 0.0
        self._needs_new_plan[dones] = True
        self._previous_active[dones] = False
        self._obs = self._planner_observations()
        attempts = core._conditional_second_attempts.sum().float()
        pair_attempts = core._pair_attempts.sum().float()
        log = extras.setdefault("log", {})
        log["Metrics/conditional_second_hit_rate"] = core._conditional_second_hits.sum().float() / attempts.clamp_min(1.0)
        log["Metrics/two_hop_pair_success_rate"] = core._pair_hits.sum().float() / pair_attempts.clamp_min(1.0)
        log["Metrics/state_planner_motor_correction"] = correction.abs().mean()
        log["Metrics/state_planner_motor_clip_fraction"] = (
            torch.zeros((), device=self.device)
            if self.motor_correction_limit <= 0.0
            else (correction.abs() >= self.motor_correction_limit).float().mean()
        )
        log["Metrics/state_planner_update_rate"] = stance.float().mean()
        return self._obs_dict(), reward, dones, extras

    def close(self):
        return self.base_env.close()
