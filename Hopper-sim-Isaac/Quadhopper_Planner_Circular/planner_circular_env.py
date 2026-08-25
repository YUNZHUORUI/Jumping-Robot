from __future__ import annotations

import torch

from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.scene import InteractiveSceneCfg

from Quadhopper_Stable.quadhopper_env import QuadhopperEnv, QuadhopperEnvCfg

from .direct_collocation_planner import DirectCollocationHopPlanner
from .height_schedule import cosine_height
from .waypoint_command import TwoCycleCircularCommand


@configclass
class PlannerCircularEnvCfg(QuadhopperEnvCfg):
    """Stable 37-D contract plus P_t/P_t+1 XY and H_t/H_t+1 commands."""

    observation_space = 43
    # Geometry requires 58 waypoint advances. The learned closed-loop hop
    # cadence is about 2 s (slower than the planner's 0.9 s flight reference),
    # so a real full revolution needs roughly 120 s plus recovery margin.
    episode_length_s = 150.0
    debug_vis = True

    # Final nominal-precision stage matches deterministic single-environment
    # playback. The canonical baseline defaults remain randomized for other
    # tasks and robustness training.
    observation_noise_std = 0.002
    randomize_dynamics = False
    randomize_action_delay = False

    circle_radius = 2.0
    hop_distance = 0.22
    # Heights are absolute root-Z apex commands in world metres.  With the
    # grounded root at 0.38 m, 0.70 m reproduces the old task's ~0.32 m rise.
    target_height = 0.70
    alternate_target_heights = False
    alternate_height_high = 1.00
    alternate_height_low = 0.70
    fixed_height_curriculum = False
    height_curriculum_start = 1.30
    height_curriculum_end = 0.70
    height_curriculum_iterations = 300.0
    height_curriculum_iteration_offset = 0.0
    landing_root_height = 0.38
    target_tolerance = 0.10
    # Optional continuation curriculum.  A value <= 0 disables it.  It is
    # useful for two-hop training: early rollouts advance often enough to
    # expose the policy to post-hit states, then end at the deployment gate.
    target_tolerance_curriculum_start = 0.0
    target_tolerance_curriculum_iterations = 0.0
    # Optional task-reference compensation for systematic radial undershoot.
    # This changes only the planner command, never robot/contact dynamics or
    # the physical waypoint used by the touchdown success test.
    planner_landing_compensation_m = 0.0
    planner_landing_xy_velocity_scale = 1.0
    anticipatory_velocity_blend = 0.0
    anticipatory_speed_max = 0.30
    anticipatory_tilt_rad = 0.0
    anticipation_start_phase = 0.55
    anticipatory_attitude_reward_scale = 0.0
    anticipatory_attitude_penalty_scale = 0.0
    anticipatory_velocity_penalty_scale = 0.0
    prepared_landing_reward_scale = 0.0
    pair_hit_reward_scale = 0.0
    prepared_attitude_tolerance_rad = 0.18
    prepared_velocity_tolerance = 0.45
    online_landing_correction_gain = 0.0
    online_landing_correction_max_m = 0.12
    online_velocity_correction_gain = 0.5
    online_velocity_correction_max_mps = 0.40
    relative_next_hop_observation = False
    min_flight_duration = 0.30
    max_flight_duration = 1.10
    planner_nodes = 25
    circle_vis_points = 128

    apex_tolerance = 0.12
    minimum_valid_apex = 0.58
    require_apex_tolerance_for_hit = False
    # Accuracy fine-tuning can make every miss terminal so the policy cannot
    # learn to use several recovery hops to reach a single waypoint.
    terminate_on_target_miss = False
    # Circular completion historically allowed recovery hops.  Rolling route
    # tasks override this so every physical touchdown consumes exactly one
    # waypoint; a miss breaks the streak but cannot trigger repeated retries.
    advance_route_on_miss = False
    restart_two_hop_pair = False
    randomize_route_phase = True

    # A 2 m radius route needs separation from neighboring tiled worlds.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=6.0)

    planner_position_reward_scale = 12.0
    planner_velocity_reward_scale = 3.0
    planner_xy_reward_scale = 45.0
    planner_z_reward_scale = 15.0
    target_hit_reward_scale = 100.0
    target_miss_penalty_scale = -60.0
    landing_precision_reward_scale = 80.0
    landing_error_penalty_scale = -100.0
    landing_precision_width = 0.06
    projected_landing_reward_scale = 45.0
    projected_landing_penalty_scale = -35.0
    projected_landing_width = 0.08
    descent_velocity_penalty_scale = -8.0
    descent_guidance_max_height = 0.90
    circle_complete_reward_scale = 1500.0
    streak_progress_reward_scale = 50.0
    apex_event_reward_scale = 100.0
    apex_shortfall_penalty_scale = -80.0
    # Legacy stages reward every increase in cycle maximum height and only
    # penalize shortfall.  Variable-height residual training enables the
    # symmetric mode below so overshooting is no longer advantageous.
    symmetric_height_tracking = False
    apex_error_penalty_scale = -250.0
    airborne_overshoot_penalty_scale = -120.0
    height_progress_reward_scale = 80.0
    curriculum_static_apex_iterations = 100.0
    curriculum_full_planner_iterations = 400.0
    curriculum_steps_per_iteration = 256.0
    curriculum_iteration_offset = 0.0
    force_full_planner = False


class PlannerCircularEnv(QuadhopperEnv):
    cfg: PlannerCircularEnvCfg

    def __init__(self, cfg: PlannerCircularEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.commands = self._create_commands()
        self.planner = DirectCollocationHopPlanner(
            self.num_envs,
            self.device,
            self.cfg.planner_nodes,
            gravity=abs(self.sim.cfg.gravity[2]),
            min_flight_duration=self.cfg.min_flight_duration,
            max_flight_duration=self.cfg.max_flight_duration,
        )
        self._cycle_time = torch.zeros(self.num_envs, device=self.device)
        self._cycle_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._previous_contact = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._contact_confirmed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._target_hit_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_miss_event = torch.zeros_like(self._target_hit_event)
        self._successful_cycles = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._consecutive_hits = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._max_consecutive_hits = torch.zeros_like(self._consecutive_hits)
        self._circle_complete_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._planned_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._cycle_takeoff_direction_w = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._cycle_next_direction_w = torch.zeros(
            self.num_envs, 2, device=self.device
        )
        self._cycle_max_z = torch.zeros(self.num_envs, device=self.device)
        self._previous_cycle_max_z = torch.zeros(self.num_envs, device=self.device)
        self._previous_vz = torch.zeros(self.num_envs, device=self.device)
        self._apex_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._apex_error = torch.zeros(self.num_envs, device=self.device)
        self._active_target_height = torch.full(
            (self.num_envs,), self.cfg.target_height, device=self.device
        )
        self._apex_target_height = self._active_target_height.clone()
        # Snapshot only completed flight cycles for reset-time per-command
        # metrics.  Reading live cycle buffers at reset also includes robots
        # that died before liftoff and biases the reported apex toward zero.
        self._settled_apex_height = torch.zeros(self.num_envs, device=self.device)
        self._settled_apex_target = torch.zeros(self.num_envs, device=self.device)
        self._settled_apex_error = torch.zeros(self.num_envs, device=self.device)
        self._settled_apex_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._touchdown_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._landing_error = torch.zeros(self.num_envs, device=self.device)
        self._touchdown_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._touchdown_along_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._touchdown_lateral_abs_error_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._touchdown_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._touchdown_attitude_error_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._touchdown_next_velocity_error_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._touchdown_next_velocity_projection_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._touchdown_next_velocity_lateral_abs_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._prepared_landing_count = torch.zeros_like(self._touchdown_count)
        self._touchdown_attitude_error = torch.zeros(self.num_envs, device=self.device)
        self._touchdown_next_velocity_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self._prepared_landing_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._target_hit_count = torch.zeros_like(self._touchdown_count)
        self._short_touchdown_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._short_touchdown_count = torch.zeros_like(self._touchdown_count)
        self._short_target_hit_count = torch.zeros_like(self._touchdown_count)
        self._long_touchdown_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._long_touchdown_count = torch.zeros_like(self._touchdown_count)
        self._long_target_hit_count = torch.zeros_like(self._touchdown_count)
        # Episode metrics become selection-biased when a miss terminates only
        # the failing environments.  These batch EMAs observe every touchdown
        # across all environments and remain comparable during streak stages.
        self._touchdown_error_ema = torch.tensor(0.0, device=self.device)
        self._target_hit_rate_ema = torch.tensor(0.0, device=self.device)
        self._short_touchdown_error_ema = torch.tensor(0.0, device=self.device)
        self._short_target_hit_rate_ema = torch.tensor(0.0, device=self.device)
        self._long_touchdown_error_ema = torch.tensor(0.0, device=self.device)
        self._long_target_hit_rate_ema = torch.tensor(0.0, device=self.device)
        self._prepared_landing_rate_ema = torch.tensor(0.0, device=self.device)
        self._touchdown_ema_updates = 0
        self._short_ema_updates = 0
        self._long_ema_updates = 0
        for key in (
            "planner_position",
            "planner_velocity",
            "planner_xy",
            "planner_z",
            "target_hit",
            "target_miss",
            "apex_event",
            "apex_shortfall",
            "apex_error",
            "airborne_overshoot",
            "height_progress",
            "landing_precision",
            "landing_error",
            "projected_landing",
            "projected_landing_error",
            "descent_velocity_error",
            "anticipatory_attitude",
            "anticipatory_attitude_error",
            "anticipatory_velocity_error",
            "prepared_landing",
            "pair_hit",
            "circle_complete",
            "streak_progress",
        ):
            self._episode_sums[key] = torch.zeros(self.num_envs, device=self.device)

    def _create_commands(self) -> TwoCycleCircularCommand:
        """Construct the route command source.

        Downstream waypoint tasks override only this factory, leaving the
        proven robot physics, planner, observations, rewards, and PPO contract
        unchanged.
        """
        return TwoCycleCircularCommand(
            self.num_envs, self.device, self.cfg.circle_radius, self.cfg.hop_distance
        )

    def _lookahead_error_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        p_t, p_t1 = self.commands.lookahead()
        root_z = self._robot.data.root_pos_w[:, 2:3]

        def transform(point_xy: torch.Tensor) -> torch.Tensor:
            point_3d = torch.cat((point_xy, root_z), dim=1)
            error_b, _ = subtract_frame_transforms(
                self._robot.data.root_pos_w, self._robot.data.root_quat_w, point_3d
            )
            return error_b[:, :2] / max(self.cfg.hop_distance, 1.0e-3)

        return transform(p_t), transform(p_t1)

    @staticmethod
    def _normalize_xy(vector: torch.Tensor) -> torch.Tensor:
        return vector / torch.linalg.norm(vector, dim=1, keepdim=True).clamp_min(
            1.0e-6
        )

    def _body_z_axis_w(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        quat = self._robot.data.root_quat_w
        if env_ids is not None:
            quat = quat[env_ids]
        w, x, y, z = quat.unbind(dim=1)
        return torch.stack(
            (
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=1,
        )

    def _desired_tilt_axis_w(self, direction_xy: torch.Tensor) -> torch.Tensor:
        tilt = self.cfg.anticipatory_tilt_rad
        horizontal = torch.sin(torch.as_tensor(tilt, device=self.device))
        vertical = torch.cos(torch.as_tensor(tilt, device=self.device))
        return torch.cat(
            (
                horizontal * direction_xy,
                vertical.expand(len(direction_xy), 1),
            ),
            dim=1,
        )

    def _attitude_error_to_direction(
        self, direction_xy: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        desired_axis = self._desired_tilt_axis_w(direction_xy)
        body_axis = self._body_z_axis_w(env_ids)
        alignment = torch.sum(body_axis * desired_axis, dim=1).clamp(-1.0, 1.0)
        return torch.acos(alignment)

    def _update_reference(self):
        reference_pos, reference_vel = self.planner.sample(self._cycle_time)
        apex_pos = self.planner.positions_w[:, self.planner.mid, :]
        equivalent_iteration = (
            self.cfg.curriculum_iteration_offset
            +
            float(getattr(self, "common_step_counter", 0))
            / max(self.cfg.curriculum_steps_per_iteration, 1.0)
        )
        denominator = max(
            self.cfg.curriculum_full_planner_iterations
            - self.cfg.curriculum_static_apex_iterations,
            1.0,
        )
        planner_weight = max(
            0.0,
            min(
                1.0,
                (equivalent_iteration - self.cfg.curriculum_static_apex_iterations)
                / denominator,
            ),
        )
        if self.cfg.force_full_planner:
            planner_weight = 1.0
        # Begin with the exact stable-hopping command: a stationary apex.
        # Gradually introduce the time-parameterized planner reference only
        # after the inherited policy has recovered full-height jumping.
        flight_pos = apex_pos + planner_weight * (reference_pos - apex_pos)
        flight_vel = planner_weight * reference_vel
        if self.cfg.online_landing_correction_gain > 0.0:
            root_pos_w = self._robot.data.root_pos_w
            root_vel_w = self._robot.data.root_lin_vel_w
            gravity = (
                abs(self.sim.cfg.gravity[2])
                if self.sim.cfg.gravity is not None
                else 9.81
            )
            height_to_land = torch.clamp(
                root_pos_w[:, 2] - self.cfg.landing_root_height, min=0.0
            )
            time_to_land = (
                root_vel_w[:, 2]
                + torch.sqrt(
                    torch.square(root_vel_w[:, 2])
                    + 2.0 * gravity * height_to_land
                )
            ) / gravity
            projected_xy = (
                root_pos_w[:, :2] + root_vel_w[:, :2] * time_to_land[:, None]
            )
            p_t, _ = self.commands.lookahead()
            correction = self.cfg.online_landing_correction_gain * (
                p_t - projected_xy
            )
            correction_norm = torch.linalg.norm(
                correction, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            correction = correction * torch.clamp(
                self.cfg.online_landing_correction_max_m / correction_norm,
                max=1.0,
            )
            phase = self._cycle_time / self.planner.flight_duration.clamp_min(1.0e-6)
            correction_mask = (
                self._cycle_active
                & (root_vel_w[:, 2] < -0.05)
                & (phase >= self.cfg.anticipation_start_phase)
            )[:, None]
            correction = torch.where(
                correction_mask, correction, torch.zeros_like(correction)
            )
            velocity_correction = (
                self.cfg.online_velocity_correction_gain
                * correction
                / time_to_land[:, None].clamp_min(0.15)
            )
            velocity_correction_norm = torch.linalg.norm(
                velocity_correction, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            velocity_correction = velocity_correction * torch.clamp(
                self.cfg.online_velocity_correction_max_mps
                / velocity_correction_norm,
                max=1.0,
            )
            flight_pos[:, :2] += correction
            flight_vel[:, :2] += velocity_correction
        # Contact/stance is a separate hybrid phase.  While grounded, command
        # the next landing XY and grounded root height with zero velocity;
        # never sample a fictitious smooth arc through spring compression.
        p_t, _ = self.commands.lookahead()
        stance_pos = torch.cat(
            (
                p_t,
                torch.full(
                    (self.num_envs, 1), self.cfg.landing_root_height, device=self.device
                ),
            ),
            dim=1,
        )
        active = self._cycle_active[:, None]
        self._desired_pos_w[:] = torch.where(active, flight_pos, stance_pos)
        self._planned_velocity_w[:] = torch.where(
            active, flight_vel, torch.zeros_like(flight_vel)
        )

    def _replan(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        p_t, p_t1 = self.commands.lookahead(env_ids)
        current_delta = p_t - self._robot.data.root_pos_w[env_ids, :2]
        next_delta = p_t1 - p_t
        self._cycle_takeoff_direction_w[env_ids] = self._normalize_xy(current_delta)
        self._cycle_next_direction_w[env_ids] = self._normalize_xy(next_delta)
        compensation = self.cfg.planner_landing_compensation_m
        if compensation != 0.0:
            current_direction = current_delta / torch.linalg.norm(
                current_delta, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            next_direction = next_delta / torch.linalg.norm(
                next_delta, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            p_t = p_t + compensation * current_direction
            p_t1 = p_t1 + compensation * next_direction
        target_height, next_target_height = self._height_commands(env_ids)
        self._active_target_height[env_ids] = target_height
        landing_height = torch.full((len(env_ids),), self.cfg.landing_root_height, device=self.device)
        self.planner.replan(
            env_ids,
            self._robot.data.root_pos_w[env_ids],
            self._robot.data.root_lin_vel_w[env_ids],
            p_t,
            p_t1,
            target_height,
            landing_height,
            next_target_height,
            self.cfg.planner_landing_xy_velocity_scale,
            self.cfg.anticipatory_velocity_blend,
            self.cfg.anticipatory_speed_max,
        )
        self._cycle_time[env_ids] = 0.0

    def _height_commands(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            cycle_index = self.commands.cycle_index
        else:
            cycle_index = self.commands.cycle_index[env_ids]
        count = len(cycle_index)
        if not self.cfg.alternate_target_heights:
            height = self.cfg.target_height
            if self.cfg.fixed_height_curriculum:
                # Isaac Lab increments common_step_counter once per vectorized
                # environment step (not once per individual environment).
                # Divide only by rollout length so the schedule is invariant
                # to training with 16/64/256 parallel environments.
                iteration = (
                    self.cfg.height_curriculum_iteration_offset
                    + float(getattr(self, "common_step_counter", 0))
                    / max(self.cfg.curriculum_steps_per_iteration, 1.0)
                )
                # Cosine easing changes height slowly at both the proven 1.30 m
                # start and the final 0.70 m gait instead of creating a sudden
                # command discontinuity.
                height = cosine_height(
                    iteration,
                    self.cfg.height_curriculum_start,
                    self.cfg.height_curriculum_end,
                    self.cfg.height_curriculum_iterations,
                )
            current = torch.full((count,), height, device=self.device)
            return current, current.clone()
        high = torch.full((count,), self.cfg.alternate_height_high, device=self.device)
        low = torch.full((count,), self.cfg.alternate_height_low, device=self.device)
        current_is_high = (cycle_index % 2) == 0
        return torch.where(current_is_high, high, low), torch.where(current_is_high, low, high)

    def _current_target_tolerance(self) -> float:
        start = self.cfg.target_tolerance_curriculum_start
        duration = self.cfg.target_tolerance_curriculum_iterations
        if start <= 0.0 or duration <= 0.0:
            return self.cfg.target_tolerance
        iteration = float(getattr(self, "common_step_counter", 0)) / max(
            self.cfg.curriculum_steps_per_iteration, 1.0
        )
        blend = max(0.0, min(1.0, iteration / duration))
        return start + blend * (self.cfg.target_tolerance - start)

    def _update_cycle_events(self):
        joint_pos = self._robot.data.joint_pos[:, self._spring_joint_id]
        contact = joint_pos > 0.002
        # At reset the articulation buffers can report q=0 before the first
        # physically settled contact frame. Never interpret that initial
        # non-contact sample as liftoff. A flight may start only after this
        # episode has observed a real compressed-spring contact.
        liftoff = self._contact_confirmed & self._previous_contact & ~contact
        touchdown = self._cycle_active & ~self._previous_contact & contact
        self._contact_confirmed |= contact
        self._target_hit_event.zero_()
        self._target_miss_event.zero_()
        self._apex_event.zero_()
        self._touchdown_event.zero_()
        self._prepared_landing_event.zero_()
        self._circle_complete_event.zero_()

        liftoff_ids = liftoff.nonzero(as_tuple=False).flatten()
        if len(liftoff_ids) > 0:
            self._cycle_active[liftoff_ids] = True
            current_z = self._robot.data.root_pos_w[liftoff_ids, 2]
            self._cycle_max_z[liftoff_ids] = current_z
            self._previous_cycle_max_z[liftoff_ids] = current_z
            self._replan(liftoff_ids)

        self._cycle_time = torch.where(
            self._cycle_active,
            self._cycle_time + self.step_dt,
            self._cycle_time,
        )

        root_z = self._robot.data.root_pos_w[:, 2]
        root_vz = self._robot.data.root_lin_vel_w[:, 2]
        self._previous_cycle_max_z.copy_(self._cycle_max_z)
        self._cycle_max_z = torch.where(
            self._cycle_active,
            torch.maximum(self._cycle_max_z, root_z),
            self._cycle_max_z,
        )
        apex_event = self._cycle_active & (self._previous_vz > 0.0) & (root_vz <= 0.0)
        self._apex_event[apex_event] = True
        self._apex_target_height[apex_event] = self._active_target_height[apex_event]
        self._apex_error[apex_event] = torch.abs(
            self._cycle_max_z[apex_event] - self._active_target_height[apex_event]
        )

        touchdown_ids = touchdown.nonzero(as_tuple=False).flatten()
        if len(touchdown_ids) > 0:
            self._touchdown_event[touchdown_ids] = True
            # Always settle apex quality at touchdown. This gives a dense
            # learning signal even when a low hop never produced a clean
            # positive-to-negative vertical-velocity crossing.
            unresolved_apex = ~self._apex_event[touchdown_ids]
            unresolved_ids = touchdown_ids[unresolved_apex]
            self._apex_event[unresolved_ids] = True
            self._apex_target_height[unresolved_ids] = self._active_target_height[unresolved_ids]
            self._apex_error[unresolved_ids] = torch.abs(
                self._cycle_max_z[unresolved_ids] - self._active_target_height[unresolved_ids]
            )
            p_t, _ = self.commands.lookahead(touchdown_ids)
            error = torch.linalg.norm(
                self._robot.data.root_pos_w[touchdown_ids, :2] - p_t, dim=1
            )
            self._landing_error[touchdown_ids] = error
            self._touchdown_error_sum[touchdown_ids] += error
            flight_delta = (
                p_t - self.planner.positions_w[touchdown_ids, 0, :2]
            )
            flight_direction = flight_delta / torch.linalg.norm(
                flight_delta, dim=1, keepdim=True
            ).clamp_min(1.0e-6)
            landing_delta = (
                self._robot.data.root_pos_w[touchdown_ids, :2] - p_t
            )
            along_error = torch.sum(landing_delta * flight_direction, dim=1)
            lateral_error = torch.abs(
                landing_delta[:, 0] * flight_direction[:, 1]
                - landing_delta[:, 1] * flight_direction[:, 0]
            )
            self._touchdown_along_error_sum[touchdown_ids] += along_error
            self._touchdown_lateral_abs_error_sum[touchdown_ids] += lateral_error
            attitude_error = self._attitude_error_to_direction(
                self._cycle_next_direction_w[touchdown_ids], touchdown_ids
            )
            next_velocity_error = torch.linalg.norm(
                self._robot.data.root_lin_vel_w[touchdown_ids, :2]
                - self.planner.velocities_w[touchdown_ids, -1, :2],
                dim=1,
            )
            touchdown_velocity_xy = self._robot.data.root_lin_vel_w[
                touchdown_ids, :2
            ]
            next_direction = self._cycle_next_direction_w[touchdown_ids]
            next_velocity_projection = torch.sum(
                touchdown_velocity_xy * next_direction, dim=1
            )
            next_velocity_lateral_abs = torch.abs(
                touchdown_velocity_xy[:, 0] * next_direction[:, 1]
                - touchdown_velocity_xy[:, 1] * next_direction[:, 0]
            )
            self._touchdown_attitude_error[touchdown_ids] = attitude_error
            self._touchdown_next_velocity_error[touchdown_ids] = next_velocity_error
            self._touchdown_attitude_error_sum[touchdown_ids] += attitude_error
            self._touchdown_next_velocity_error_sum[touchdown_ids] += (
                next_velocity_error
            )
            self._touchdown_next_velocity_projection_sum[touchdown_ids] += (
                next_velocity_projection
            )
            self._touchdown_next_velocity_lateral_abs_sum[touchdown_ids] += (
                next_velocity_lateral_abs
            )
            self._touchdown_count[touchdown_ids] += 1
            self._settled_apex_height[touchdown_ids] = self._cycle_max_z[touchdown_ids]
            self._settled_apex_target[touchdown_ids] = self._active_target_height[touchdown_ids]
            self._settled_apex_error[touchdown_ids] = torch.abs(
                self._cycle_max_z[touchdown_ids] - self._active_target_height[touchdown_ids]
            )
            self._settled_apex_valid[touchdown_ids] = True
            if self.cfg.alternate_target_heights:
                valid_apex_threshold = torch.maximum(
                    self._active_target_height[touchdown_ids] - self.cfg.apex_tolerance,
                    torch.full_like(
                        self._active_target_height[touchdown_ids],
                        self.cfg.landing_root_height + 0.05,
                    ),
                )
            else:
                valid_apex_threshold = torch.full_like(
                    self._active_target_height[touchdown_ids], self.cfg.minimum_valid_apex
                )
            valid_apex = self._cycle_max_z[touchdown_ids] >= valid_apex_threshold
            if self.cfg.require_apex_tolerance_for_hit:
                apex_within_tolerance = (
                    torch.abs(
                        self._cycle_max_z[touchdown_ids]
                        - self._active_target_height[touchdown_ids]
                    )
                    <= self.cfg.apex_tolerance
                )
            else:
                apex_within_tolerance = torch.ones_like(valid_apex)
            hit_mask = (
                (error < self._current_target_tolerance())
                & valid_apex
                & apex_within_tolerance
            )
            prepared_mask = (
                hit_mask
                & (attitude_error <= self.cfg.prepared_attitude_tolerance_rad)
                & (next_velocity_error <= self.cfg.prepared_velocity_tolerance)
            )
            prepared_ids = touchdown_ids[prepared_mask]
            self._prepared_landing_event[prepared_ids] = True
            self._prepared_landing_count[prepared_ids] += 1
            ema_weight = 1.0 if self._touchdown_ema_updates == 0 else 0.05
            self._touchdown_error_ema.lerp_(torch.mean(error), ema_weight)
            self._target_hit_rate_ema.lerp_(
                torch.mean(hit_mask.float()), ema_weight
            )
            self._prepared_landing_rate_ema.lerp_(
                torch.mean(prepared_mask.float()), ema_weight
            )
            self._touchdown_ema_updates += 1
            hit_ids = touchdown_ids[hit_mask]
            miss_ids = touchdown_ids[~hit_mask]
            self._target_hit_event[hit_ids] = True
            self._target_miss_event[miss_ids] = True
            self._target_hit_count[hit_ids] += 1
            if hasattr(self.commands, "current_hop_is_short"):
                short_mask = self.commands.current_hop_is_short(touchdown_ids)
                short_ids = touchdown_ids[short_mask]
                long_ids = touchdown_ids[~short_mask]
                if torch.any(short_mask):
                    short_weight = 1.0 if self._short_ema_updates == 0 else 0.05
                    self._short_touchdown_error_ema.lerp_(
                        torch.mean(error[short_mask]), short_weight
                    )
                    self._short_target_hit_rate_ema.lerp_(
                        torch.mean(hit_mask[short_mask].float()), short_weight
                    )
                    self._short_ema_updates += 1
                if torch.any(~short_mask):
                    long_weight = 1.0 if self._long_ema_updates == 0 else 0.05
                    self._long_touchdown_error_ema.lerp_(
                        torch.mean(error[~short_mask]), long_weight
                    )
                    self._long_target_hit_rate_ema.lerp_(
                        torch.mean(hit_mask[~short_mask].float()), long_weight
                    )
                    self._long_ema_updates += 1
                self._short_touchdown_error_sum[short_ids] += error[short_mask]
                self._short_touchdown_count[short_ids] += 1
                self._long_touchdown_error_sum[long_ids] += error[~short_mask]
                self._long_touchdown_count[long_ids] += 1
                hit_short_mask = self.commands.current_hop_is_short(hit_ids)
                self._short_target_hit_count[hit_ids[hit_short_mask]] += 1
                self._long_target_hit_count[hit_ids[~hit_short_mask]] += 1
            if len(hit_ids) > 0:
                self._advance_route(hit_ids)
                self._successful_cycles[hit_ids] += 1
                self._consecutive_hits[hit_ids] += 1
                self._max_consecutive_hits[hit_ids] = torch.maximum(
                    self._max_consecutive_hits[hit_ids], self._consecutive_hits[hit_ids]
                )
                completed_hits = (
                    self._consecutive_hits[hit_ids]
                    >= self.commands.steps_per_revolution
                )
                self._circle_complete_event[hit_ids[completed_hits]] = True
            if len(miss_ids) > 0:
                # A recovery hop is physically allowed, but it breaks the
                # no-correction full-circle streak and is never counted as
                # successful completion.
                self._consecutive_hits[miss_ids] = 0
                if self.cfg.advance_route_on_miss:
                    self._advance_route(miss_ids)
            self._cycle_active[touchdown_ids] = False
            self._replan(touchdown_ids)

        self._previous_contact = contact
        self._previous_vz = root_vz
        self._update_reference()

    def _advance_route(self, env_ids: torch.Tensor):
        """Consume one waypoint, restarting a two-hop pair when configured."""
        if len(env_ids) == 0:
            return
        if self.cfg.restart_two_hop_pair and hasattr(self.commands, "restart_pair"):
            is_first_hop = self.commands.current_hop_is_short(env_ids)
            first_ids = env_ids[is_first_hop]
            second_ids = env_ids[~is_first_hop]
            self.commands.advance(first_ids)
            if len(second_ids) > 0:
                self.commands.restart_pair(
                    second_ids, self._robot.data.root_pos_w[second_ids, :2]
                )
            return
        self.commands.advance(env_ids)

    def _get_observations(self) -> dict:
        self._update_reference()
        stable_obs = super()._get_observations()["policy"]
        p_t_error_b, p_t1_error_b = self._lookahead_error_b()
        # Both current and next absolute apex commands are observable.  This
        # lets the recurrent policy prepare its touchdown/stance for H_(t+1).
        target_height, next_target_height = self._height_commands()
        height_commands = torch.stack((target_height, next_target_height), dim=1) / 2.0
        planner_obs = torch.cat((p_t_error_b, p_t1_error_b, height_commands), dim=1)
        if self.cfg.observation_noise_std > 0.0:
            planner_obs += torch.randn_like(planner_obs) * self.cfg.observation_noise_std
        return {"policy": torch.cat((stable_obs, planner_obs), dim=1)}

    def _get_rewards(self) -> torch.Tensor:
        self._update_cycle_events()
        reward = super()._get_rewards()
        position_error = torch.linalg.norm(
            self._robot.data.root_pos_w - self._desired_pos_w, dim=1
        )
        planner_xy_error = torch.linalg.norm(
            self._robot.data.root_pos_w[:, :2] - self._desired_pos_w[:, :2], dim=1
        )
        planner_z_error = torch.abs(
            self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2]
        )
        velocity_error = torch.linalg.norm(
            self._robot.data.root_lin_vel_w - self._planned_velocity_w, dim=1
        )
        if self.cfg.symmetric_height_tracking:
            previous_height_error = torch.abs(
                self._active_target_height - self._previous_cycle_max_z
            )
            current_height_error = torch.abs(
                self._active_target_height - self._cycle_max_z
            )
            # Potential difference: approaching the commanded apex is
            # rewarded and continuing above it is penalized by the same rule.
            height_progress = (
                previous_height_error - current_height_error
            ) * self._cycle_active.float()
        else:
            height_progress = torch.relu(
                self._cycle_max_z - self._previous_cycle_max_z
            )
        apex_quality = torch.exp(-torch.square(self._apex_error / self.cfg.apex_tolerance))
        apex_shortfall = torch.relu(self._apex_target_height - self._cycle_max_z)
        airborne_overshoot = (
            torch.relu(self._robot.data.root_pos_w[:, 2] - self._active_target_height)
            * self._cycle_active.float()
        )
        landing_quality = torch.exp(
            -torch.square(self._landing_error / self.cfg.landing_precision_width)
        )
        # During descent, predict the ballistic touchdown point from the
        # measured state. This gives the policy time to brake laterally before
        # contact instead of learning only from a delayed touchdown penalty.
        root_pos_w = self._robot.data.root_pos_w
        root_vel_w = self._robot.data.root_lin_vel_w
        gravity = abs(self.sim.cfg.gravity[2]) if self.sim.cfg.gravity is not None else 9.81
        height_to_land = torch.clamp(root_pos_w[:, 2] - self.cfg.landing_root_height, min=0.0)
        time_to_land = (
            root_vel_w[:, 2]
            + torch.sqrt(torch.square(root_vel_w[:, 2]) + 2.0 * gravity * height_to_land)
        ) / gravity
        projected_xy = root_pos_w[:, :2] + root_vel_w[:, :2] * time_to_land[:, None]
        p_t, _ = self.commands.lookahead()
        projected_error = torch.linalg.norm(projected_xy - p_t, dim=1)
        desired_landing_vxy = self.planner.velocities_w[:, -1, :2]
        descent_velocity_error = torch.linalg.norm(
            root_vel_w[:, :2] - desired_landing_vxy, dim=1
        )
        descent_mask = (
            self._cycle_active
            & (root_vel_w[:, 2] < -0.05)
            & (root_pos_w[:, 2] < self.cfg.descent_guidance_max_height)
        ).float()
        flight_phase = self._cycle_time / self.planner.flight_duration.clamp_min(1.0e-6)
        anticipation_descent = (
            self._cycle_active
            & (root_vel_w[:, 2] < -0.05)
            & (flight_phase >= self.cfg.anticipation_start_phase)
        )
        spring_contact = (
            self._robot.data.joint_pos[:, self._spring_joint_id] > 0.002
        )
        stance_preparation = ~self._cycle_active & spring_contact
        anticipation_mask = (anticipation_descent | stance_preparation).float()
        anticipation_direction = torch.where(
            self._cycle_active[:, None],
            self._cycle_next_direction_w,
            self._cycle_takeoff_direction_w,
        )
        anticipatory_attitude_error = self._attitude_error_to_direction(
            anticipation_direction
        )
        anticipatory_attitude_quality = torch.exp(
            -torch.square(
                anticipatory_attitude_error
                / max(self.cfg.prepared_attitude_tolerance_rad, 1.0e-3)
            )
        )
        anticipatory_velocity_error = torch.linalg.norm(
            root_vel_w[:, :2] - self.planner.velocities_w[:, -1, :2], dim=1
        )
        projected_quality = torch.exp(
            -torch.square(projected_error / self.cfg.projected_landing_width)
        )
        additions = {
            "planner_position": torch.exp(-position_error / 0.25)
            * self.cfg.planner_position_reward_scale
            * self.step_dt,
            "planner_velocity": torch.exp(-velocity_error / 1.0)
            * self.cfg.planner_velocity_reward_scale
            * self.step_dt,
            # Keep horizontal guidance observable even while the inherited
            # high-jump policy still has a large vertical tracking error.
            "planner_xy": torch.exp(-planner_xy_error / 0.10)
            * self.cfg.planner_xy_reward_scale
            * self.step_dt,
            "planner_z": torch.exp(-planner_z_error / 0.12)
            * self.cfg.planner_z_reward_scale
            * self.step_dt,
            "target_hit": self._target_hit_event.float() * self.cfg.target_hit_reward_scale,
            "target_miss": self._target_miss_event.float() * self.cfg.target_miss_penalty_scale,
            "apex_event": self._apex_event.float()
            * apex_quality
            * self.cfg.apex_event_reward_scale,
            "apex_shortfall": self._apex_event.float()
            * apex_shortfall
            * self.cfg.apex_shortfall_penalty_scale,
            "apex_error": self._apex_event.float()
            * self._apex_error
            * self.cfg.apex_error_penalty_scale,
            "airborne_overshoot": airborne_overshoot
            * self.cfg.airborne_overshoot_penalty_scale
            * self.step_dt,
            "height_progress": height_progress * self.cfg.height_progress_reward_scale,
            "landing_precision": self._touchdown_event.float()
            * landing_quality
            * self.cfg.landing_precision_reward_scale,
            "landing_error": self._touchdown_event.float()
            * self._landing_error
            * self.cfg.landing_error_penalty_scale,
            "projected_landing": descent_mask
            * projected_quality
            * self.cfg.projected_landing_reward_scale
            * self.step_dt,
            "projected_landing_error": descent_mask
            * projected_error
            * self.cfg.projected_landing_penalty_scale
            * self.step_dt,
            "descent_velocity_error": descent_mask
            * descent_velocity_error
            * self.cfg.descent_velocity_penalty_scale
            * self.step_dt,
            "anticipatory_attitude": anticipation_mask
            * anticipatory_attitude_quality
            * self.cfg.anticipatory_attitude_reward_scale
            * self.step_dt,
            "anticipatory_attitude_error": anticipation_mask
            * anticipatory_attitude_error
            * self.cfg.anticipatory_attitude_penalty_scale
            * self.step_dt,
            "anticipatory_velocity_error": anticipation_descent.float()
            * anticipatory_velocity_error
            * self.cfg.anticipatory_velocity_penalty_scale
            * self.step_dt,
            "prepared_landing": self._prepared_landing_event.float()
            * self.cfg.prepared_landing_reward_scale,
            # This is deliberately different from a per-waypoint bonus.  It
            # rewards a hit only when the preceding jump also hit, so P_(t+1)
            # must be used while executing the current landing transition.
            "pair_hit": (
                self._target_hit_event & (self._consecutive_hits >= 2)
            ).float()
            * self.cfg.pair_hit_reward_scale,
            "circle_complete": self._circle_complete_event.float()
            * self.cfg.circle_complete_reward_scale,
            "streak_progress": self._target_hit_event.float()
            * (self._consecutive_hits.float() / self.commands.steps_per_revolution)
            * self.cfg.streak_progress_reward_scale,
        }
        for key, value in additions.items():
            self._episode_sums[key] += value
            reward += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died, timeout = super()._get_dones()
        completed = self._consecutive_hits >= self.commands.steps_per_revolution
        missed = self._target_miss_event if self.cfg.terminate_on_target_miss else False
        return died | completed | missed, timeout

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        mean_cycle_apex = (
            torch.mean(self._cycle_max_z[env_ids])
            if hasattr(self, "_cycle_max_z")
            else torch.tensor(0.0, device=self.device)
        )
        mean_landing_error = (
            torch.mean(self._landing_error[env_ids])
            if hasattr(self, "_landing_error")
            else torch.tensor(0.0, device=self.device)
        )
        touchdown_count = (
            torch.sum(self._touchdown_count[env_ids])
            if hasattr(self, "_touchdown_count")
            else torch.tensor(0, device=self.device)
        )
        mean_episode_touchdown_error = (
            torch.sum(self._touchdown_error_sum[env_ids])
            / torch.clamp(touchdown_count.float(), min=1.0)
            if hasattr(self, "_touchdown_error_sum")
            else torch.tensor(0.0, device=self.device)
        )
        target_hit_rate = (
            torch.sum(self._target_hit_count[env_ids]).float()
            / torch.clamp(touchdown_count.float(), min=1.0)
            if hasattr(self, "_target_hit_count")
            else torch.tensor(0.0, device=self.device)
        )
        mean_touchdown_attitude_error = (
            torch.sum(self._touchdown_attitude_error_sum[env_ids])
            / torch.clamp(touchdown_count.float(), min=1.0)
            if hasattr(self, "_touchdown_attitude_error_sum")
            else torch.tensor(0.0, device=self.device)
        )
        mean_touchdown_next_velocity_error = (
            torch.sum(self._touchdown_next_velocity_error_sum[env_ids])
            / torch.clamp(touchdown_count.float(), min=1.0)
            if hasattr(self, "_touchdown_next_velocity_error_sum")
            else torch.tensor(0.0, device=self.device)
        )
        mean_touchdown_next_velocity_projection = (
            torch.sum(self._touchdown_next_velocity_projection_sum[env_ids])
            / torch.clamp(touchdown_count.float(), min=1.0)
        )
        mean_touchdown_next_velocity_lateral_abs = (
            torch.sum(self._touchdown_next_velocity_lateral_abs_sum[env_ids])
            / torch.clamp(touchdown_count.float(), min=1.0)
        )
        prepared_landing_rate = (
            torch.sum(self._prepared_landing_count[env_ids]).float()
            / torch.clamp(touchdown_count.float(), min=1.0)
            if hasattr(self, "_prepared_landing_count")
            else torch.tensor(0.0, device=self.device)
        )
        short_touchdown_count = (
            torch.sum(self._short_touchdown_count[env_ids])
            if hasattr(self, "_short_touchdown_count")
            else torch.tensor(0, device=self.device)
        )
        long_touchdown_count = (
            torch.sum(self._long_touchdown_count[env_ids])
            if hasattr(self, "_long_touchdown_count")
            else torch.tensor(0, device=self.device)
        )
        short_touchdown_error = (
            torch.sum(self._short_touchdown_error_sum[env_ids])
            / torch.clamp(short_touchdown_count.float(), min=1.0)
            if hasattr(self, "_short_touchdown_error_sum")
            else torch.tensor(0.0, device=self.device)
        )
        long_touchdown_error = (
            torch.sum(self._long_touchdown_error_sum[env_ids])
            / torch.clamp(long_touchdown_count.float(), min=1.0)
            if hasattr(self, "_long_touchdown_error_sum")
            else torch.tensor(0.0, device=self.device)
        )
        short_target_hit_rate = (
            torch.sum(self._short_target_hit_count[env_ids]).float()
            / torch.clamp(short_touchdown_count.float(), min=1.0)
            if hasattr(self, "_short_target_hit_count")
            else torch.tensor(0.0, device=self.device)
        )
        long_target_hit_rate = (
            torch.sum(self._long_target_hit_count[env_ids]).float()
            / torch.clamp(long_touchdown_count.float(), min=1.0)
            if hasattr(self, "_long_target_hit_count")
            else torch.tensor(0.0, device=self.device)
        )
        mean_successful_waypoints = (
            torch.mean(self._successful_cycles[env_ids].float())
            if hasattr(self, "_successful_cycles")
            else torch.tensor(0.0, device=self.device)
        )
        circle_completion = (
            torch.mean(
                (
                    self._consecutive_hits[env_ids]
                    >= self.commands.steps_per_revolution
                ).float()
            )
            if hasattr(self, "_consecutive_hits")
            else torch.tensor(0.0, device=self.device)
        )
        mean_max_hit_streak = (
            torch.mean(self._max_consecutive_hits[env_ids].float())
            if hasattr(self, "_max_consecutive_hits")
            else torch.tensor(0.0, device=self.device)
        )
        live_streak = self._consecutive_hits.float()
        live_mean_streak = torch.mean(live_streak)
        live_p90_streak = torch.quantile(live_streak, 0.90)
        live_max_streak = torch.max(live_streak)
        if hasattr(self, "_settled_apex_target"):
            valid_apex = self._settled_apex_valid[env_ids]
            high_mask = torch.isclose(
                self._settled_apex_target[env_ids],
                torch.tensor(self.cfg.alternate_height_high, device=self.device),
                atol=1.0e-4,
            ) & valid_apex
            low_mask = torch.isclose(
                self._settled_apex_target[env_ids],
                torch.tensor(self.cfg.alternate_height_low, device=self.device),
                atol=1.0e-4,
            ) & valid_apex
            high_apex = (
                torch.mean(self._settled_apex_height[env_ids][high_mask])
                if torch.any(high_mask)
                else torch.tensor(0.0, device=self.device)
            )
            low_apex = (
                torch.mean(self._settled_apex_height[env_ids][low_mask])
                if torch.any(low_mask)
                else torch.tensor(0.0, device=self.device)
            )
            high_apex_error = (
                torch.mean(self._settled_apex_error[env_ids][high_mask])
                if torch.any(high_mask)
                else torch.tensor(0.0, device=self.device)
            )
            low_apex_error = (
                torch.mean(self._settled_apex_error[env_ids][low_mask])
                if torch.any(low_mask)
                else torch.tensor(0.0, device=self.device)
            )
        else:
            high_apex = low_apex = torch.tensor(0.0, device=self.device)
            high_apex_error = low_apex_error = torch.tensor(0.0, device=self.device)
        super()._reset_idx(env_ids)
        self.extras["log"]["Metrics/mean_cycle_apex_height_m"] = mean_cycle_apex
        self.extras["log"]["Metrics/mean_touchdown_error_m"] = mean_landing_error
        self.extras["log"]["Metrics/episode_touchdown_error_m"] = (
            mean_episode_touchdown_error
        )
        self.extras["log"]["Metrics/target_hit_rate"] = target_hit_rate
        self.extras["log"]["Metrics/current_target_tolerance_m"] = (
            self._current_target_tolerance()
        )
        self.extras["log"]["Metrics/touchdown_attitude_error_rad"] = (
            mean_touchdown_attitude_error
        )
        self.extras["log"]["Metrics/touchdown_next_velocity_error_mps"] = (
            mean_touchdown_next_velocity_error
        )
        self.extras["log"]["Metrics/touchdown_next_velocity_projection_mps"] = (
            mean_touchdown_next_velocity_projection
        )
        self.extras["log"]["Metrics/touchdown_next_velocity_lateral_abs_mps"] = (
            mean_touchdown_next_velocity_lateral_abs
        )
        self.extras["log"]["Metrics/prepared_landing_rate"] = prepared_landing_rate
        self.extras["log"]["Metrics/short_touchdown_error_m"] = short_touchdown_error
        self.extras["log"]["Metrics/long_touchdown_error_m"] = long_touchdown_error
        self.extras["log"]["Metrics/short_target_hit_rate"] = short_target_hit_rate
        self.extras["log"]["Metrics/long_target_hit_rate"] = long_target_hit_rate
        self.extras["log"]["Metrics/touchdown_error_ema_m"] = self._touchdown_error_ema
        self.extras["log"]["Metrics/target_hit_rate_ema"] = self._target_hit_rate_ema
        self.extras["log"]["Metrics/prepared_landing_rate_ema"] = (
            self._prepared_landing_rate_ema
        )
        self.extras["log"]["Metrics/short_touchdown_error_ema_m"] = (
            self._short_touchdown_error_ema
        )
        self.extras["log"]["Metrics/short_target_hit_rate_ema"] = (
            self._short_target_hit_rate_ema
        )
        self.extras["log"]["Metrics/long_touchdown_error_ema_m"] = (
            self._long_touchdown_error_ema
        )
        self.extras["log"]["Metrics/long_target_hit_rate_ema"] = (
            self._long_target_hit_rate_ema
        )
        self.extras["log"]["Metrics/live_mean_consecutive_hits"] = live_mean_streak
        self.extras["log"]["Metrics/live_p90_consecutive_hits"] = live_p90_streak
        self.extras["log"]["Metrics/live_max_consecutive_hits"] = live_max_streak
        self.extras["log"]["Metrics/successful_waypoints"] = mean_successful_waypoints
        self.extras["log"]["Metrics/circle_completion"] = circle_completion
        # Generic alias used by non-circular waypoint generators that reuse
        # this two-cycle tracking environment.
        self.extras["log"]["Metrics/route_completion"] = circle_completion
        self.extras["log"]["Metrics/max_consecutive_hits"] = mean_max_hit_streak
        current_height, _ = self._height_commands(env_ids[:1])
        self.extras["log"]["Metrics/command_apex_height_m"] = current_height[0]
        if self.cfg.alternate_target_heights:
            self.extras["log"]["Metrics/high_command_apex_m"] = high_apex
            self.extras["log"]["Metrics/low_command_apex_m"] = low_apex
            self.extras["log"]["Metrics/high_command_apex_error_m"] = high_apex_error
            self.extras["log"]["Metrics/low_command_apex_error_m"] = low_apex_error
        self.commands.reset(
            env_ids,
            self._terrain.env_origins,
            random_phase=self.num_envs > 1 and self.cfg.randomize_route_phase,
        )
        if self.cfg.alternate_target_heights and self.num_envs > 1:
            # Desynchronize the two height phases across vectorized training
            # environments so every rollout contains both commands.
            self.commands.cycle_index[env_ids] = torch.randint(
                0, 2, (len(env_ids),), device=self.device
            )

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :2] = self.commands.start_points(env_ids)
        # Train and play must start from the same grounded distribution.
        root_state[:, 2] = self.cfg.landing_root_height
        root_state[:, 3] = 1.0
        root_state[:, 4:7] = 0.0
        root_state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        self._cycle_time[env_ids] = 0.0
        self._cycle_active[env_ids] = False
        self._previous_contact[env_ids] = False
        self._contact_confirmed[env_ids] = False
        self._target_hit_event[env_ids] = False
        self._target_miss_event[env_ids] = False
        self._successful_cycles[env_ids] = 0
        self._consecutive_hits[env_ids] = 0
        self._max_consecutive_hits[env_ids] = 0
        self._cycle_max_z[env_ids] = self.cfg.landing_root_height
        self._previous_cycle_max_z[env_ids] = self.cfg.landing_root_height
        self._previous_vz[env_ids] = 0.0
        self._apex_event[env_ids] = False
        self._apex_error[env_ids] = 0.0
        self._active_target_height[env_ids] = self.cfg.target_height
        self._apex_target_height[env_ids] = self.cfg.target_height
        self._touchdown_event[env_ids] = False
        self._circle_complete_event[env_ids] = False
        self._landing_error[env_ids] = 0.0
        self._touchdown_error_sum[env_ids] = 0.0
        self._touchdown_along_error_sum[env_ids] = 0.0
        self._touchdown_lateral_abs_error_sum[env_ids] = 0.0
        self._touchdown_count[env_ids] = 0
        self._touchdown_attitude_error_sum[env_ids] = 0.0
        self._touchdown_next_velocity_error_sum[env_ids] = 0.0
        self._touchdown_next_velocity_projection_sum[env_ids] = 0.0
        self._touchdown_next_velocity_lateral_abs_sum[env_ids] = 0.0
        self._prepared_landing_count[env_ids] = 0
        self._touchdown_attitude_error[env_ids] = 0.0
        self._touchdown_next_velocity_error[env_ids] = 0.0
        self._prepared_landing_event[env_ids] = False
        self._target_hit_count[env_ids] = 0
        self._short_touchdown_error_sum[env_ids] = 0.0
        self._short_touchdown_count[env_ids] = 0
        self._short_target_hit_count[env_ids] = 0
        self._long_touchdown_error_sum[env_ids] = 0.0
        self._long_touchdown_count[env_ids] = 0
        self._long_target_hit_count[env_ids] = 0
        self._settled_apex_height[env_ids] = 0.0
        self._settled_apex_target[env_ids] = 0.0
        self._settled_apex_error[env_ids] = 0.0
        self._settled_apex_valid[env_ids] = False
        self._replan(env_ids)
        self._update_reference()

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "_p_t_visualizer"):
                p_t_cfg = CUBOID_MARKER_CFG.copy()
                p_t_cfg.prim_path = "/Visuals/PlannerCircular/P_t"
                p_t_cfg.markers["cuboid"].size = (0.07, 0.07, 0.07)
                p_t_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.55, 0.0)
                self._p_t_visualizer = VisualizationMarkers(p_t_cfg)

                p_t1_cfg = CUBOID_MARKER_CFG.copy()
                p_t1_cfg.prim_path = "/Visuals/PlannerCircular/P_t_plus_1"
                p_t1_cfg.markers["cuboid"].size = (0.06, 0.06, 0.06)
                p_t1_cfg.markers["cuboid"].visual_material.diffuse_color = (0.85, 0.1, 1.0)
                self._p_t1_visualizer = VisualizationMarkers(p_t1_cfg)

                path_cfg = CUBOID_MARKER_CFG.copy()
                path_cfg.prim_path = "/Visuals/PlannerCircular/optimized_path"
                path_cfg.markers["cuboid"].size = (0.018, 0.018, 0.018)
                path_cfg.markers["cuboid"].visual_material.diffuse_color = (0.1, 0.55, 1.0)
                self._path_visualizer = VisualizationMarkers(path_cfg)

                next_path_cfg = CUBOID_MARKER_CFG.copy()
                next_path_cfg.prim_path = "/Visuals/PlannerCircular/next_optimized_path"
                next_path_cfg.markers["cuboid"].size = (0.016, 0.016, 0.016)
                next_path_cfg.markers["cuboid"].visual_material.diffuse_color = (0.1, 1.0, 0.75)
                self._next_path_visualizer = VisualizationMarkers(next_path_cfg)

                circle_cfg = CUBOID_MARKER_CFG.copy()
                circle_cfg.prim_path = "/Visuals/PlannerCircular/full_circle"
                circle_cfg.markers["cuboid"].size = (0.025, 0.025, 0.012)
                circle_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.8, 0.1)
                self._circle_visualizer = VisualizationMarkers(circle_cfg)
            self._p_t_visualizer.set_visibility(True)
            self._p_t1_visualizer.set_visibility(True)
            self._path_visualizer.set_visibility(True)
            self._next_path_visualizer.set_visibility(True)
            self._circle_visualizer.set_visibility(True)
        else:
            for name in (
                "_p_t_visualizer",
                "_p_t1_visualizer",
                "_path_visualizer",
                "_next_path_visualizer",
                "_circle_visualizer",
            ):
                if hasattr(self, name):
                    getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if not hasattr(self, "commands") or not hasattr(self, "_p_t_visualizer"):
            return
        p_t, p_t1 = self.commands.lookahead()
        ground_marker_z = torch.full((self.num_envs, 1), 0.03, device=self.device)
        self._p_t_visualizer.visualize(torch.cat((p_t, ground_marker_z), dim=1))
        self._p_t1_visualizer.visualize(torch.cat((p_t1, ground_marker_z), dim=1))
        # Green is the stationary apex command for this jump.  Blue remains
        # the optimized time-parameterized reference trajectory.
        self.goal_pos_visualizer.visualize(self.planner.positions_w[:, self.planner.mid, :])
        self._path_visualizer.visualize(self.planner.positions_w.reshape(-1, 3))
        self._next_path_visualizer.visualize(self.planner.next_positions_w.reshape(-1, 3))
        route_xy = self._route_visualization_xy()
        route_z = torch.full(
            (*route_xy.shape[:-1], 1), 0.012, device=self.device
        )
        self._circle_visualizer.visualize(torch.cat((route_xy, route_z), dim=-1).reshape(-1, 3))

    def _route_visualization_xy(self) -> torch.Tensor:
        """Return per-environment XY samples for the route debug markers."""
        angles = torch.linspace(
            0.0,
            2.0 * torch.pi,
            self.cfg.circle_vis_points + 1,
            device=self.device,
        )[:-1]
        unit_circle = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        return (
            self.commands.center_w[:, None, :]
            + self.cfg.circle_radius * unit_circle[None, :, :]
        )
