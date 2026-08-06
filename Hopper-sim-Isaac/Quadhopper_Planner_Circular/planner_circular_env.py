from __future__ import annotations

import torch

from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.scene import InteractiveSceneCfg

from Quadhopper_Stable.quadhopper_env import QuadhopperEnv, QuadhopperEnvCfg

from .direct_collocation_planner import DirectCollocationHopPlanner
from .waypoint_command import TwoCycleCircularCommand


@configclass
class PlannerCircularEnvCfg(QuadhopperEnvCfg):
    """Stable 37-D policy contract plus five planner look-ahead values."""

    observation_space = 42
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
    target_height = 1.30
    landing_root_height = 0.38
    target_tolerance = 0.10
    cycle_duration = 0.90
    planner_nodes = 25
    circle_vis_points = 128

    apex_tolerance = 0.12
    minimum_valid_apex = 1.15

    # A 2 m radius route needs separation from neighboring tiled worlds.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=6.0)

    planner_position_reward_scale = 12.0
    planner_velocity_reward_scale = 3.0
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
    height_progress_reward_scale = 35.0
    curriculum_static_apex_iterations = 100.0
    curriculum_full_planner_iterations = 400.0
    curriculum_steps_per_iteration = 256.0
    curriculum_iteration_offset = 0.0
    force_full_planner = False


class PlannerCircularEnv(QuadhopperEnv):
    cfg: PlannerCircularEnvCfg

    def __init__(self, cfg: PlannerCircularEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.commands = TwoCycleCircularCommand(
            self.num_envs, self.device, self.cfg.circle_radius, self.cfg.hop_distance
        )
        self.planner = DirectCollocationHopPlanner(
            self.num_envs, self.device, self.cfg.planner_nodes, self.cfg.cycle_duration
        )
        self._cycle_time = torch.zeros(self.num_envs, device=self.device)
        self._cycle_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._previous_contact = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_hit_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_miss_event = torch.zeros_like(self._target_hit_event)
        self._successful_cycles = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._consecutive_hits = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._max_consecutive_hits = torch.zeros_like(self._consecutive_hits)
        self._circle_complete_event = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._planned_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._cycle_max_z = torch.zeros(self.num_envs, device=self.device)
        self._previous_cycle_max_z = torch.zeros(self.num_envs, device=self.device)
        self._previous_vz = torch.zeros(self.num_envs, device=self.device)
        self._apex_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._apex_error = torch.zeros(self.num_envs, device=self.device)
        self._touchdown_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._landing_error = torch.zeros(self.num_envs, device=self.device)
        for key in (
            "planner_position",
            "planner_velocity",
            "target_hit",
            "target_miss",
            "apex_event",
            "apex_shortfall",
            "height_progress",
            "landing_precision",
            "landing_error",
            "projected_landing",
            "projected_landing_error",
            "descent_velocity_error",
            "circle_complete",
            "streak_progress",
        ):
            self._episode_sums[key] = torch.zeros(self.num_envs, device=self.device)

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
        self._desired_pos_w[:] = apex_pos + planner_weight * (reference_pos - apex_pos)
        self._planned_velocity_w[:] = planner_weight * reference_vel

    def _replan(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        p_t, p_t1 = self.commands.lookahead(env_ids)
        target_height = torch.full((len(env_ids),), self.cfg.target_height, device=self.device)
        landing_height = torch.full((len(env_ids),), self.cfg.landing_root_height, device=self.device)
        self.planner.replan(
            env_ids,
            self._robot.data.root_pos_w[env_ids],
            self._robot.data.root_lin_vel_w[env_ids],
            p_t,
            p_t1,
            target_height,
            landing_height,
        )
        self._cycle_time[env_ids] = 0.0

    def _update_cycle_events(self):
        joint_pos = self._robot.data.joint_pos[:, self._spring_joint_id]
        contact = joint_pos > 0.002
        liftoff = self._previous_contact & ~contact
        touchdown = ~self._previous_contact & contact
        self._target_hit_event.zero_()
        self._target_miss_event.zero_()
        self._apex_event.zero_()
        self._touchdown_event.zero_()
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
        ).clamp(max=self.cfg.cycle_duration)

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
        self._apex_error[apex_event] = torch.abs(self._cycle_max_z[apex_event] - self.cfg.target_height)

        touchdown_ids = touchdown.nonzero(as_tuple=False).flatten()
        if len(touchdown_ids) > 0:
            self._touchdown_event[touchdown_ids] = True
            # Always settle apex quality at touchdown. This gives a dense
            # learning signal even when a low hop never produced a clean
            # positive-to-negative vertical-velocity crossing.
            unresolved_apex = ~self._apex_event[touchdown_ids]
            unresolved_ids = touchdown_ids[unresolved_apex]
            self._apex_event[unresolved_ids] = True
            self._apex_error[unresolved_ids] = torch.abs(
                self._cycle_max_z[unresolved_ids] - self.cfg.target_height
            )
            p_t, _ = self.commands.lookahead(touchdown_ids)
            error = torch.linalg.norm(
                self._robot.data.root_pos_w[touchdown_ids, :2] - p_t, dim=1
            )
            self._landing_error[touchdown_ids] = error
            valid_apex = self._cycle_max_z[touchdown_ids] >= self.cfg.minimum_valid_apex
            hit_mask = (error < self.cfg.target_tolerance) & valid_apex
            hit_ids = touchdown_ids[hit_mask]
            miss_ids = touchdown_ids[~hit_mask]
            self._target_hit_event[hit_ids] = True
            self._target_miss_event[miss_ids] = True
            if len(hit_ids) > 0:
                self.commands.advance(hit_ids)
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
            self._cycle_active[touchdown_ids] = False
            self._replan(touchdown_ids)

        self._previous_contact = contact
        self._previous_vz = root_vz
        self._update_reference()

    def _get_observations(self) -> dict:
        self._update_reference()
        stable_obs = super()._get_observations()["policy"]
        p_t_error_b, p_t1_error_b = self._lookahead_error_b()
        # This is the requested apex height command, not another position
        # error.  A fixed scale leaves room for later height randomization.
        target_height_command = torch.full(
            (self.num_envs, 1), self.cfg.target_height / 2.0, device=self.device
        )
        planner_obs = torch.cat((p_t_error_b, p_t1_error_b, target_height_command), dim=1)
        if self.cfg.observation_noise_std > 0.0:
            planner_obs += torch.randn_like(planner_obs) * self.cfg.observation_noise_std
        return {"policy": torch.cat((stable_obs, planner_obs), dim=1)}

    def _get_rewards(self) -> torch.Tensor:
        self._update_cycle_events()
        reward = super()._get_rewards()
        position_error = torch.linalg.norm(
            self._robot.data.root_pos_w - self._desired_pos_w, dim=1
        )
        velocity_error = torch.linalg.norm(
            self._robot.data.root_lin_vel_w - self._planned_velocity_w, dim=1
        )
        height_progress = torch.relu(self._cycle_max_z - self._previous_cycle_max_z)
        apex_quality = torch.exp(-torch.square(self._apex_error / self.cfg.apex_tolerance))
        apex_shortfall = torch.relu(self.cfg.target_height - self._cycle_max_z)
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
            "target_hit": self._target_hit_event.float() * self.cfg.target_hit_reward_scale,
            "target_miss": self._target_miss_event.float() * self.cfg.target_miss_penalty_scale,
            "apex_event": self._apex_event.float()
            * apex_quality
            * self.cfg.apex_event_reward_scale,
            "apex_shortfall": self._apex_event.float()
            * apex_shortfall
            * self.cfg.apex_shortfall_penalty_scale,
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
        return died | completed, timeout

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
        super()._reset_idx(env_ids)
        self.extras["log"]["Metrics/mean_cycle_apex_height_m"] = mean_cycle_apex
        self.extras["log"]["Metrics/mean_touchdown_error_m"] = mean_landing_error
        self.extras["log"]["Metrics/successful_waypoints"] = mean_successful_waypoints
        self.extras["log"]["Metrics/circle_completion"] = circle_completion
        self.extras["log"]["Metrics/max_consecutive_hits"] = mean_max_hit_streak
        self.commands.reset(env_ids, self._terrain.env_origins, random_phase=self.num_envs > 1)

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
        self._previous_contact[env_ids] = True
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
        self._touchdown_event[env_ids] = False
        self._circle_complete_event[env_ids] = False
        self._landing_error[env_ids] = 0.0
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
        angles = torch.linspace(
            0.0,
            2.0 * torch.pi,
            self.cfg.circle_vis_points + 1,
            device=self.device,
        )[:-1]
        unit_circle = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        circle_xy = self.commands.center_w[:, None, :] + self.cfg.circle_radius * unit_circle[None, :, :]
        circle_z = torch.full(
            (self.num_envs, self.cfg.circle_vis_points, 1), 0.012, device=self.device
        )
        self._circle_visualizer.visualize(torch.cat((circle_xy, circle_z), dim=-1).reshape(-1, 3))
