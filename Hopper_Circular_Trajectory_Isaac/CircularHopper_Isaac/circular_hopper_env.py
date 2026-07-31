from __future__ import annotations

from collections import deque

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms

from .ballistic_planner import BallisticHopPlanner
from .my_hopper_cfg import MY_HOPPER_CFG
from .trajectory_commands import CircularWaypointCommand


@configclass
class CircularHopperEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 2

    action_space = 4
    observation_space = 50
    state_space = 0
    debug_vis = False
    trajectory_vis_points = 72
    upcoming_waypoint_vis_count = 8
    trajectory_arc_vis_points = 24
    upcoming_trajectory_vis_count = 4
    planning_horizon_hops = 5

    circle_radius = 2.0
    hop_distance = 0.22
    target_tolerance = 0.14
    max_successful_hops = 24
    fixed_ccw = True

    flight_time_ref = 0.45
    apex_height_ref = 0.30
    min_target_hop_height = 0.20
    max_target_hop_height = 0.45
    min_flight_time_for_hit = 0.28
    max_under_arc_error_for_hit = 0.10

    touchdown_height = 0.002
    liftoff_height = 0.001
    stance_root_height = 0.38
    liftoff_root_clearance = 0.06
    contact_vz_threshold = 0.20
    liftoff_vz_threshold = 0.35
    touchdown_vz_limit = 1.2
    landing_tilt_limit = 0.35
    max_hop_height = 0.75
    max_workspace_error = 1.4
    max_stance_time = 0.65
    max_missed_landings_per_target = 0
    advance_on_touchdown = True

    survival_scale = 0.2
    upright_scale = -4.0
    hop_velocity_scale = 4.0
    stance_relaunch_scale = 8.0
    progress_scale = 0.25
    radial_path_scale = 1.0
    launch_vxy_scale = 14.0
    launch_vz_scale = 18.0
    flight_vxy_scale = 12.0
    flight_attitude_scale = -2.5
    flight_traj_xy_scale = 8.0
    flight_traj_height_scale = 28.0
    low_flight_penalty_scale = -35.0
    apex_height_scale = 70.0
    touchdown_scale = 45.0
    target_hit_scale = 120.0
    landing_precision_scale = 35.0
    landing_error_scale = -45.0
    touchdown_miss_scale = -60.0
    landing_vxy_scale = -4.0
    action_rate_scale = -0.35
    action_smooth_scale = -0.25
    stance_stall_scale = -8.0
    failure_scale = -40.0

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="max",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=4 * 1024 * 1024,
            min_position_iteration_count=4,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.5, restitution=0.0
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=5.0, replicate_physics=False, clone_in_fabric=False
    )

    robot: ArticulationCfg = MY_HOPPER_CFG.replace(prim_path="/World/envs/env_.*/Robot")


class CircularHopperEnv(DirectRLEnv):
    cfg: CircularHopperEnvCfg

    def __init__(self, cfg: CircularHopperEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._prev_prev_actions = torch.zeros_like(self._actions)

        self.history_len = 3
        self._action_history = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self._action_history.append(torch.zeros_like(self._actions))

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._motor_u = torch.zeros(self.num_envs, 4, device=self.device)
        self._dt_policy = self.sim.cfg.dt * self.cfg.decimation

        self.commands = CircularWaypointCommand(
            self.num_envs,
            self.device,
            radius=self.cfg.circle_radius,
            hop_distance=self.cfg.hop_distance,
            fixed_ccw=self.cfg.fixed_ccw,
            segment_horizon=self.cfg.planning_horizon_hops,
        )
        self.planner = BallisticHopPlanner(
            self.num_envs,
            self.device,
            flight_time=self.cfg.flight_time_ref,
            apex_height=self.cfg.apex_height_ref,
        )

        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._touching = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_touching = torch.ones_like(self._touching)
        self._prev_vz = torch.zeros(self.num_envs, device=self.device)
        self._time_since_liftoff = torch.zeros(self.num_envs, device=self.device)
        self._time_since_touchdown = torch.zeros(self.num_envs, device=self.device)
        self._prev_target_dist = torch.zeros(self.num_envs, device=self.device)
        self._last_landing_error = torch.zeros(self.num_envs, device=self.device)
        self._segment_base_z = torch.zeros(self.num_envs, device=self.device)
        self._hop_max_height = torch.zeros(self.num_envs, device=self.device)
        self._hop_max_under_arc_error = torch.zeros(self.num_envs, device=self.device)
        self._missed_landings = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._target_hits = torch.zeros(self.num_envs, device=self.device)
        self._liftoff_events = torch.zeros(self.num_envs, device=self.device)
        self._touchdown_events = torch.zeros(self.num_envs, device=self.device)
        self._last_done_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_done_tilt = torch.zeros_like(self._last_done_completed)
        self._last_done_height = torch.zeros_like(self._last_done_completed)
        self._last_done_workspace = torch.zeros_like(self._last_done_completed)
        self._last_done_stance_stall = torch.zeros_like(self._last_done_completed)
        self._last_done_repeated_miss = torch.zeros_like(self._last_done_completed)
        self._debug_done_completed = torch.zeros_like(self._last_done_completed)
        self._debug_done_tilt = torch.zeros_like(self._last_done_completed)
        self._debug_done_height = torch.zeros_like(self._last_done_completed)
        self._debug_done_workspace = torch.zeros_like(self._last_done_completed)
        self._debug_done_stance_stall = torch.zeros_like(self._last_done_completed)
        self._debug_done_repeated_miss = torch.zeros_like(self._last_done_completed)

        self.dr_tm = torch.ones(self.num_envs, 4, device=self.device) * 0.12
        self.dr_thrust_multi = torch.ones(self.num_envs, 4, device=self.device)
        self.dr_torque_multi = torch.ones(self.num_envs, 1, device=self.device)
        self.dr_mass_multi = torch.ones(self.num_envs, 1, device=self.device)
        self.dr_inertia_multi = torch.ones(self.num_envs, 3, device=self.device)

        body_ids, _ = self._robot.find_bodies("Body")
        self._body_id = body_ids[0:1]
        self._spring_joint_id = self._robot.find_joints("center_spring_joint")[0][0]

        self._episode_sums = {
            k: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for k in [
                "survival",
                "upright",
                "hop_velocity",
                "stance_relaunch",
                "progress",
                "radial_path",
                "launch_vxy",
                "launch_vz",
                "flight_vxy",
                "flight_attitude",
                "flight_traj_xy",
                "flight_traj_height",
                "low_flight_penalty",
                "apex_height",
                "touchdown",
                "target_hit",
                "landing_precision",
                "landing_error",
                "touchdown_miss",
                "landing_vxy",
                "action_rate",
                "action_smooth",
                "stance_stall",
                "failure",
            ]
        }
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_prev_actions = self._prev_actions.clone()
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._action_history.append(self._actions.clone())

        delayed = self._action_history[-2] if len(self._action_history) > 1 else self._actions
        target_u = (delayed * 0.5 + 0.5).clamp(0.0, 1.0)

        alpha = self._dt_policy / (self.dr_tm + self._dt_policy)
        self._motor_u = alpha * target_u + (1.0 - alpha) * self._motor_u
        u = self._motor_u

        force = -0.2371 * u**2 + 0.8130 * u + 0.0113
        force = force * self.dr_thrust_multi / self.dr_mass_multi

        arm = 0.0813
        k_tau = 5.4e-02
        f1, f2, f3, f4 = force[:, 0], force[:, 1], force[:, 2], force[:, 3]
        u1, u2, u3, u4 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

        tau_scale = self.dr_torque_multi.squeeze()
        self._thrust[:, 0, 2] = f1 + f2 + f3 + f4
        self._moment[:, 0, 0] = arm * (f1 + f4 - f2 - f3) * tau_scale / self.dr_inertia_multi[:, 0]
        self._moment[:, 0, 1] = arm * (f1 + f2 - f3 - f4) * tau_scale / self.dr_inertia_multi[:, 1]
        self._moment[:, 0, 2] = k_tau * -(u1**2 + u3**2 - u2**2 - u4**2) * tau_scale / self.dr_inertia_multi[:, 2]

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _target_error_b(self) -> torch.Tensor:
        target_pos_3d = torch.cat(
            [self.commands.target_pos_w, self._robot.data.root_pos_w[:, 2:3]], dim=-1
        )
        err_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, target_pos_3d
        )
        return err_b[:, :2]

    def _vector_xy_to_body(self, vec_xy_w: torch.Tensor) -> torch.Tensor:
        vec_3d_w = torch.cat([vec_xy_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        vec_b, _ = subtract_frame_transforms(
            torch.zeros_like(vec_3d_w), self._robot.data.root_quat_w, vec_3d_w
        )
        return vec_b[:, :2]

    def _horizon_waypoints_w(self, horizon: int) -> torch.Tensor:
        return self.commands.horizon_waypoints(horizon)

    def _horizon_error_b(self, waypoints_xy_w: torch.Tensor) -> torch.Tensor:
        horizon = waypoints_xy_w.shape[1]
        target_pos_w = torch.cat(
            [
                waypoints_xy_w.reshape(-1, 2),
                self._robot.data.root_pos_w[:, 2:3].repeat_interleave(horizon, dim=0),
            ],
            dim=-1,
        )
        root_pos_w = self._robot.data.root_pos_w.repeat_interleave(horizon, dim=0)
        root_quat_w = self._robot.data.root_quat_w.repeat_interleave(horizon, dim=0)
        err_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w)
        return err_b[:, :2].reshape(self.num_envs, horizon * 2)

    def _sync_fixed_planner(self, env_ids: torch.Tensor, base_z_w: torch.Tensor):
        if len(env_ids) == 0:
            return
        self.planner.set_fixed_xy_plan(
            env_ids,
            self.commands.current_start_pos_w[env_ids],
            self.commands.target_pos_w[env_ids],
            base_z_w,
        )

    def _update_events(self):
        root_z = self._robot.data.root_pos_w[:, 2]
        vz = self._robot.data.root_lin_vel_w[:, 2]
        joint_pos = self._robot.data.joint_pos[:, self._spring_joint_id]

        self._prev_touching = self._touching.clone()
        joint_contact = joint_pos > self.cfg.touchdown_height
        low_root_contact = (root_z < self.cfg.stance_root_height) & (vz < self.cfg.contact_vz_threshold)
        low_and_slow = joint_contact | low_root_contact
        clearly_airborne = (
            (
                (root_z > self.cfg.stance_root_height + self.cfg.liftoff_root_clearance)
                | (vz > self.cfg.liftoff_vz_threshold)
            )
            & (joint_pos < self.cfg.liftoff_height)
        )
        self._touching = torch.where(
            clearly_airborne,
            torch.zeros_like(self._touching),
            torch.where(low_and_slow, torch.ones_like(self._touching), self._touching),
        )

        liftoff_event = self._prev_touching & ~self._touching
        touchdown_event = ~self._prev_touching & self._touching
        apex_event = (self._phase == 1) & (self._prev_vz > 0.0) & (vz <= 0.0)

        self._phase = torch.where(liftoff_event, torch.ones_like(self._phase), self._phase)
        self._phase = torch.where(touchdown_event, torch.full_like(self._phase, 2), self._phase)
        settled = (self._phase == 2) & self._touching & (self._time_since_touchdown > 0.05)
        self._phase = torch.where(settled, torch.zeros_like(self._phase), self._phase)

        self._time_since_liftoff = torch.where(
            liftoff_event, torch.zeros_like(self._time_since_liftoff), self._time_since_liftoff + self.step_dt
        )
        self._time_since_touchdown = torch.where(
            touchdown_event,
            torch.zeros_like(self._time_since_touchdown),
            self._time_since_touchdown + self.step_dt,
        )
        self._prev_vz = vz.clone()
        return liftoff_event, touchdown_event, apex_event

    def _get_observations(self) -> dict:
        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b
        quat = self._robot.data.root_quat_w
        z = self._robot.data.root_pos_w[:, 2:3]
        vz = self._robot.data.root_lin_vel_w[:, 2:3]

        target_err_b = self._target_error_b() / max(self.cfg.hop_distance, 1.0e-3)
        horizon_waypoints_w = self._horizon_waypoints_w(self.cfg.planning_horizon_hops)
        horizon_err_b = self._horizon_error_b(horizon_waypoints_w) / max(self.cfg.hop_distance, 1.0e-3)
        vxy_ref_err_w = self.planner.v_xy_ref_w - self._robot.data.root_lin_vel_w[:, :2]
        vxy_ref_err_b = self._vector_xy_to_body(vxy_ref_err_w).clamp(-3.0, 3.0)
        vz_ref_err = (self.planner.vz_ref[:, None] - vz).clamp(-3.0, 3.0)
        radial_error = (self.commands.radial_error(self._robot.data.root_pos_w[:, :2]) / self.cfg.hop_distance).unsqueeze(-1)
        tangent_b = self._vector_xy_to_body(self.commands.tangent_w())
        joint_pos = self._robot.data.joint_pos[:, self._spring_joint_id].unsqueeze(-1)
        joint_vel = self._robot.data.joint_vel[:, self._spring_joint_id].unsqueeze(-1)
        contact = self._touching.float().unsqueeze(-1)
        phase_one_hot = torch.nn.functional.one_hot(self._phase, num_classes=3).float()
        event_times = torch.stack(
            [
                torch.clamp(self._time_since_liftoff / 1.0, 0.0, 1.0),
                torch.clamp(self._time_since_touchdown / 1.0, 0.0, 1.0),
            ],
            dim=-1,
        )

        lin_vel = lin_vel + torch.randn_like(lin_vel) * 0.02
        ang_vel = ang_vel + torch.randn_like(ang_vel) * 0.01
        quat = torch.nn.functional.normalize(quat + torch.randn_like(quat) * 0.005, p=2, dim=-1)
        target_err_b = target_err_b + torch.randn_like(target_err_b) * 0.02
        horizon_err_b = horizon_err_b + torch.randn_like(horizon_err_b) * 0.02

        history = torch.cat(list(self._action_history), dim=-1)
        obs = torch.cat(
            [
                lin_vel,
                ang_vel,
                quat,
                z,
                vz,
                target_err_b.clamp(-5.0, 5.0),
                vxy_ref_err_b,
                vz_ref_err,
                radial_error.clamp(-5.0, 5.0),
                tangent_b,
                joint_pos,
                joint_vel,
                contact,
                phase_one_hot,
                event_times,
                history,
                horizon_err_b.clamp(-5.0, 5.0),
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self._robot.data.root_pos_w
        root_vel_w = self._robot.data.root_lin_vel_w
        q = self._robot.data.root_quat_w
        liftoff_event, touchdown_event, apex_event = self._update_events()
        liftoff_ids = liftoff_event.nonzero(as_tuple=False).flatten()
        if len(liftoff_ids) > 0:
            self._hop_max_height[liftoff_ids] = 0.0
            self._hop_max_under_arc_error[liftoff_ids] = 0.0
            self._prev_target_dist[liftoff_ids] = torch.linalg.norm(
                self.commands.target_pos_w[liftoff_ids] - root_pos[liftoff_ids, :2], dim=1
            )
        self._hop_max_height = torch.maximum(self._hop_max_height, root_pos[:, 2] - self.planner.takeoff_pos_w[:, 2])

        tilt = torch.sum(q[:, 1:3] ** 2, dim=1)
        target_dist = torch.linalg.norm(self.commands.target_pos_w - root_pos[:, :2], dim=1)
        progress = self._prev_target_dist - target_dist
        self._prev_target_dist = target_dist.detach()

        radial = self.commands.radial_error(root_pos[:, :2])
        vxy_err = torch.linalg.norm(root_vel_w[:, :2] - self.planner.v_xy_ref_w, dim=1)
        vz_err = root_vel_w[:, 2] - self.planner.vz_ref
        ang_vel_sq = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        flight_phase = torch.clamp(self._time_since_liftoff / self.planner.flight_time_ref, 0.0, 1.0)
        planned_xy = self.planner.takeoff_pos_w[:, :2] + (
            self.planner.target_pos_w[:, :2] - self.planner.takeoff_pos_w[:, :2]
        ) * flight_phase[:, None]
        xy_traj_err = torch.linalg.norm(root_pos[:, :2] - planned_xy, dim=1)
        planned_rel_height = 4.0 * flight_phase * (1.0 - flight_phase) * self.cfg.apex_height_ref
        planned_z = self.planner.takeoff_pos_w[:, 2] + planned_rel_height
        z_err = root_pos[:, 2] - planned_z
        in_flight = self._phase == 1
        self._hop_max_under_arc_error = torch.where(
            in_flight,
            torch.maximum(self._hop_max_under_arc_error, torch.relu(planned_z - root_pos[:, 2])),
            self._hop_max_under_arc_error,
        )
        landing_error = target_dist
        self._last_landing_error = torch.where(touchdown_event, landing_error, self._last_landing_error)

        valid_hop_height = (
            (self._hop_max_height >= self.cfg.min_target_hop_height)
            & (self._hop_max_height <= self.cfg.max_target_hop_height)
        )
        stable_landing = (
            (tilt < self.cfg.landing_tilt_limit)
            & (torch.abs(root_vel_w[:, 2]) < self.cfg.touchdown_vz_limit)
            & valid_hop_height
            & (self._time_since_liftoff >= self.cfg.min_flight_time_for_hit)
            & (self._hop_max_under_arc_error <= self.cfg.max_under_arc_error_for_hit)
        )
        target_hit = touchdown_event & stable_landing & (landing_error < self.cfg.target_tolerance)
        target_miss = touchdown_event & ~target_hit
        if self.cfg.advance_on_touchdown:
            advance_event = touchdown_event & stable_landing
        else:
            advance_event = target_hit
        hit_ids = target_hit.nonzero(as_tuple=False).flatten()
        stance_stall = self._touching & (self._phase == 0) & (self._time_since_touchdown > 0.25)
        relaunch_window = self._touching & (self._phase == 0) & (self._time_since_touchdown > 0.08)

        rewards = {
            "survival": torch.ones(self.num_envs, device=self.device) * self.cfg.survival_scale * self.step_dt,
            "upright": tilt * self.cfg.upright_scale * self.step_dt,
            "hop_velocity": torch.relu(root_vel_w[:, 2]) * torch.exp(-root_pos[:, 2] * 10.0)
            * self.cfg.hop_velocity_scale
            * self.step_dt,
            "stance_relaunch": relaunch_window.float()
            * torch.relu(root_vel_w[:, 2])
            * self.cfg.stance_relaunch_scale
            * self.step_dt,
            "progress": (~in_flight).float() * progress * self.cfg.progress_scale,
            "radial_path": torch.exp(-6.0 * radial**2) * self.cfg.radial_path_scale * self.step_dt,
            "launch_vxy": liftoff_event.float() * torch.exp(-4.0 * vxy_err**2) * self.cfg.launch_vxy_scale,
            "launch_vz": liftoff_event.float() * torch.exp(-2.0 * vz_err**2) * self.cfg.launch_vz_scale,
            "flight_vxy": in_flight.float()
            * torch.exp(-4.0 * vxy_err**2)
            * self.cfg.flight_vxy_scale
            * self.step_dt,
            "flight_attitude": in_flight.float()
            * (tilt + 0.05 * ang_vel_sq)
            * self.cfg.flight_attitude_scale
            * self.step_dt,
            "flight_traj_xy": in_flight.float()
            * torch.exp(-45.0 * xy_traj_err**2)
            * self.cfg.flight_traj_xy_scale
            * self.step_dt,
            "flight_traj_height": in_flight.float()
            * torch.exp(-55.0 * z_err**2)
            * self.cfg.flight_traj_height_scale
            * self.step_dt,
            "low_flight_penalty": in_flight.float()
            * torch.square(torch.relu(planned_z - root_pos[:, 2]))
            * self.cfg.low_flight_penalty_scale
            * self.step_dt,
            "apex_height": apex_event.float()
            * (
                torch.exp(-40.0 * (self._hop_max_height - self.cfg.apex_height_ref) ** 2)
                - 2.0 * torch.relu(self.cfg.min_target_hop_height - self._hop_max_height)
            )
            * self.cfg.apex_height_scale,
            "touchdown": touchdown_event.float() * torch.exp(-20.0 * landing_error**2) * self.cfg.touchdown_scale,
            "target_hit": target_hit.float() * self.cfg.target_hit_scale,
            "landing_precision": touchdown_event.float()
            * torch.exp(-120.0 * landing_error**2)
            * self.cfg.landing_precision_scale,
            "landing_error": touchdown_event.float()
            * torch.square(landing_error)
            * self.cfg.landing_error_scale,
            "touchdown_miss": target_miss.float()
            * (1.0 + torch.clamp(landing_error - self.cfg.target_tolerance, min=0.0))
            * self.cfg.touchdown_miss_scale,
            "landing_vxy": touchdown_event.float()
            * torch.sum(root_vel_w[:, :2] ** 2, dim=1)
            * self.cfg.landing_vxy_scale,
            "action_rate": torch.sum(torch.square(self._actions - self._prev_actions), dim=1)
            * self.cfg.action_rate_scale
            * self.step_dt,
            "action_smooth": torch.sum(
                torch.square(self._actions - 2.0 * self._prev_actions + self._prev_prev_actions), dim=1
            )
            * self.cfg.action_smooth_scale
            * self.step_dt,
            "stance_stall": stance_stall.float()
            * (self._time_since_touchdown - 0.25).clamp(min=0.0)
            * self.cfg.stance_stall_scale
            * self.step_dt,
            "failure": torch.zeros(self.num_envs, device=self.device),
        }

        self._target_hits += target_hit.float()
        self._liftoff_events += liftoff_event.float()
        self._touchdown_events += touchdown_event.float()
        self._missed_landings += target_miss.long()

        touchdown_ids = touchdown_event.nonzero(as_tuple=False).flatten()
        if len(touchdown_ids) > 0:
            self._hop_max_height[touchdown_ids] = 0.0
            self._hop_max_under_arc_error[touchdown_ids] = 0.0

        advance_ids = advance_event.nonzero(as_tuple=False).flatten()
        if len(advance_ids) > 0:
            self.commands.advance(advance_ids)
            new_segment = self.commands.segment_index[advance_ids] == 0
            if torch.any(new_segment):
                self._segment_base_z[advance_ids[new_segment]] = root_pos[advance_ids[new_segment], 2]
            self._sync_fixed_planner(advance_ids, self._segment_base_z[advance_ids])
            self._missed_landings[hit_ids] = 0
            self._prev_target_dist[hit_ids] = torch.linalg.norm(
                self.commands.target_pos_w[hit_ids] - root_pos[hit_ids, :2], dim=1
            )
            missed_advance_ids = advance_ids[~target_hit[advance_ids]]
            if len(missed_advance_ids) > 0:
                self._prev_target_dist[missed_advance_ids] = torch.linalg.norm(
                    self.commands.target_pos_w[missed_advance_ids] - root_pos[missed_advance_ids, :2], dim=1
                )

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, val in rewards.items():
            self._episode_sums[key] += val
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        q = self._robot.data.root_quat_w
        z = self._robot.data.root_pos_w[:, 2]
        tilt = torch.sum(q[:, 1:3] ** 2, dim=1)
        target_dist = torch.linalg.norm(self.commands.target_pos_w - self._robot.data.root_pos_w[:, :2], dim=1)
        completed = self.commands.successful_hops >= self.cfg.max_successful_hops
        tilt_done = tilt > 0.5
        height_done = z > self.cfg.max_hop_height
        workspace_done = target_dist > self.cfg.max_workspace_error
        stance_stall_done = self._touching & (self._phase == 0) & (self._time_since_touchdown > self.cfg.max_stance_time)
        repeated_miss_done = self._missed_landings > self.cfg.max_missed_landings_per_target
        died = tilt_done | height_done | workspace_done | stance_stall_done | repeated_miss_done
        self._last_done_completed = completed
        self._last_done_tilt = tilt_done
        self._last_done_height = height_done
        self._last_done_workspace = workspace_done
        self._last_done_stance_stall = stance_stall_done
        self._last_done_repeated_miss = repeated_miss_done
        self._debug_done_completed = completed
        self._debug_done_tilt = tilt_done
        self._debug_done_height = height_done
        self._debug_done_workspace = workspace_done
        self._debug_done_stance_stall = stance_stall_done
        self._debug_done_repeated_miss = repeated_miss_done
        return died | completed, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras = {}
        for key in self._episode_sums:
            extras[f"Episode_Reward/{key}"] = (
                torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode/liftoff_rate"] = torch.mean(self._liftoff_events[env_ids])
        extras["Episode/touchdown_rate"] = torch.mean(self._touchdown_events[env_ids])
        extras["Episode/target_hit_rate"] = torch.mean(self._target_hits[env_ids])
        extras["Episode/successful_hops_mean"] = torch.mean(self.commands.successful_hops[env_ids].float())
        extras["Episode/landing_error_mean"] = torch.mean(self._last_landing_error[env_ids])
        extras["Episode/done_completed_rate"] = torch.mean(self._last_done_completed[env_ids].float())
        extras["Episode/done_tilt_rate"] = torch.mean(self._last_done_tilt[env_ids].float())
        extras["Episode/done_height_rate"] = torch.mean(self._last_done_height[env_ids].float())
        extras["Episode/done_workspace_rate"] = torch.mean(self._last_done_workspace[env_ids].float())
        extras["Episode/done_stance_stall_rate"] = torch.mean(self._last_done_stance_stall[env_ids].float())
        extras["Episode/done_repeated_miss_rate"] = torch.mean(self._last_done_repeated_miss[env_ids].float())
        self.extras["log"] = extras

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        self.dr_thrust_multi[env_ids] = torch.empty(n, 4, device=self.device).uniform_(0.92, 1.08)
        self.dr_torque_multi[env_ids] = torch.empty(n, 1, device=self.device).uniform_(0.9, 1.1)
        self.dr_tm[env_ids] = torch.empty(n, 4, device=self.device).uniform_(0.10, 0.14)
        self.dr_mass_multi[env_ids] = torch.empty(n, 1, device=self.device).uniform_(0.9, 1.1)
        self.dr_inertia_multi[env_ids] = torch.empty(n, 3, device=self.device).uniform_(0.9, 1.1)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_prev_actions[env_ids] = 0.0
        self._motor_u[env_ids] = 0.0
        self._phase[env_ids] = 0
        self._touching[env_ids] = True
        self._prev_touching[env_ids] = True
        self._prev_vz[env_ids] = 0.0
        self._time_since_liftoff[env_ids] = 1.0
        self._time_since_touchdown[env_ids] = 0.0
        self._last_landing_error[env_ids] = 0.0
        self._segment_base_z[env_ids] = 0.0
        self._hop_max_height[env_ids] = 0.0
        self._hop_max_under_arc_error[env_ids] = 0.0
        self._target_hits[env_ids] = 0.0
        self._liftoff_events[env_ids] = 0.0
        self._touchdown_events[env_ids] = 0.0
        self._missed_landings[env_ids] = 0
        self._last_done_completed[env_ids] = False
        self._last_done_tilt[env_ids] = False
        self._last_done_height[env_ids] = False
        self._last_done_workspace[env_ids] = False
        self._last_done_stance_stall[env_ids] = False
        self._last_done_repeated_miss[env_ids] = False
        for i in range(len(self._action_history)):
            self._action_history[i][env_ids] = 0.0

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] = self._terrain.env_origins[env_ids, 2] + 0.23

        self.commands.reset(env_ids, self._terrain.env_origins)
        start_xy = self.commands.current_start_pos_w[env_ids]
        default_root_state[:, 0:2] = start_xy + torch.randn(n, 2, device=self.device) * 0.015

        rand_euler = torch.randn(n, 3, device=self.device) * 0.05
        default_root_state[:, 3:7] = quat_from_euler_xyz(rand_euler[:, 0], rand_euler[:, 1], rand_euler[:, 2])

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self._segment_base_z[env_ids] = default_root_state[:, 2]
        self._sync_fixed_planner(env_ids, self._segment_base_z[env_ids])
        self._prev_target_dist[env_ids] = torch.linalg.norm(
            self.commands.target_pos_w[env_ids] - default_root_state[:, :2], dim=1
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_target_visualizer"):
                target_cfg = CUBOID_MARKER_CFG.copy()
                target_cfg.prim_path = "/Visuals/CircularHopper/current_target"
                target_cfg.markers["cuboid"].size = (0.08, 0.08, 0.08)
                target_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                self._target_visualizer = VisualizationMarkers(target_cfg)

            if not hasattr(self, "_circle_visualizer"):
                circle_cfg = CUBOID_MARKER_CFG.copy()
                circle_cfg.prim_path = "/Visuals/CircularHopper/circle_path"
                circle_cfg.markers["cuboid"].size = (0.025, 0.025, 0.025)
                circle_cfg.markers["cuboid"].visual_material.diffuse_color = (0.1, 0.35, 1.0)
                self._circle_visualizer = VisualizationMarkers(circle_cfg)

            if not hasattr(self, "_waypoint_visualizer"):
                waypoint_cfg = CUBOID_MARKER_CFG.copy()
                waypoint_cfg.prim_path = "/Visuals/CircularHopper/upcoming_waypoints"
                waypoint_cfg.markers["cuboid"].size = (0.045, 0.045, 0.045)
                waypoint_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.55, 0.0)
                self._waypoint_visualizer = VisualizationMarkers(waypoint_cfg)

            if not hasattr(self, "_current_arc_visualizer"):
                current_arc_cfg = CUBOID_MARKER_CFG.copy()
                current_arc_cfg.prim_path = "/Visuals/CircularHopper/current_hop_arc"
                current_arc_cfg.markers["cuboid"].size = (0.035, 0.035, 0.035)
                current_arc_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.05, 0.05)
                self._current_arc_visualizer = VisualizationMarkers(current_arc_cfg)

            if not hasattr(self, "_upcoming_arc_visualizer"):
                upcoming_arc_cfg = CUBOID_MARKER_CFG.copy()
                upcoming_arc_cfg.prim_path = "/Visuals/CircularHopper/upcoming_hop_arcs"
                upcoming_arc_cfg.markers["cuboid"].size = (0.028, 0.028, 0.028)
                upcoming_arc_cfg.markers["cuboid"].visual_material.diffuse_color = (0.75, 0.2, 1.0)
                self._upcoming_arc_visualizer = VisualizationMarkers(upcoming_arc_cfg)

            self._target_visualizer.set_visibility(True)
            self._circle_visualizer.set_visibility(True)
            self._waypoint_visualizer.set_visibility(True)
            self._current_arc_visualizer.set_visibility(True)
            self._upcoming_arc_visualizer.set_visibility(True)
        else:
            for name in (
                "_target_visualizer",
                "_circle_visualizer",
                "_waypoint_visualizer",
                "_current_arc_visualizer",
                "_upcoming_arc_visualizer",
            ):
                if hasattr(self, name):
                    getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_target_visualizer"):
            return

        target_pos = torch.cat(
            [
                self.commands.target_pos_w,
                torch.full((self.num_envs, 1), 0.04, device=self.device),
            ],
            dim=-1,
        )
        self._target_visualizer.visualize(target_pos)

        num_points = self.cfg.trajectory_vis_points
        angles = torch.linspace(0.0, 2.0 * torch.pi, num_points + 1, device=self.device)[:-1]
        circle_xy = (
            self.commands.center_w[:, None, :]
            + self.commands.radius[:, None, None]
            * torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)[None, :, :]
        )
        circle_z = torch.full((self.num_envs, num_points, 1), 0.02, device=self.device)
        self._circle_visualizer.visualize(torch.cat([circle_xy, circle_z], dim=-1).reshape(-1, 3))

        vis_hops = min(self.cfg.upcoming_waypoint_vis_count, self.cfg.max_successful_hops)
        waypoint_xy = self.commands.horizon_waypoints(vis_hops).reshape(-1, 2)
        waypoint_z = torch.full((waypoint_xy.shape[0], 1), 0.03, device=self.device)
        self._waypoint_visualizer.visualize(torch.cat([waypoint_xy, waypoint_z], dim=-1))

        arc_t = torch.linspace(0.0, 1.0, self.cfg.trajectory_arc_vis_points, device=self.device)
        current_start = self.planner.takeoff_pos_w[:, None, :]
        current_end = self.planner.target_pos_w[:, None, :]
        current_arc = current_start + (current_end - current_start) * arc_t[None, :, None]
        current_arc[:, :, 2] += 4.0 * arc_t[None, :] * (1.0 - arc_t[None, :]) * self.cfg.apex_height_ref
        self._current_arc_visualizer.visualize(current_arc.reshape(-1, 3))

        vis_arcs = min(self.cfg.upcoming_trajectory_vis_count, self.cfg.max_successful_hops)
        arc_waypoints = self.commands.horizon_waypoints(vis_arcs)
        start_xy_all = torch.cat([self.commands.current_start_pos_w[:, None, :], arc_waypoints[:, :-1, :]], dim=1)
        start_xy = start_xy_all.reshape(-1, 2)
        end_xy = arc_waypoints.reshape(-1, 2)
        arc_env_ids = torch.arange(self.num_envs, device=self.device).repeat_interleave(vis_arcs)
        base_z = self.planner.target_pos_w[arc_env_ids, 2]
        start_3d = torch.cat([start_xy, base_z[:, None]], dim=-1)[:, None, :]
        end_3d = torch.cat([end_xy, base_z[:, None]], dim=-1)[:, None, :]
        upcoming_arc = start_3d + (end_3d - start_3d) * arc_t[None, :, None]
        upcoming_arc[:, :, 2] += 4.0 * arc_t[None, :] * (1.0 - arc_t[None, :]) * self.cfg.apex_height_ref
        self._upcoming_arc_visualizer.visualize(upcoming_arc.reshape(-1, 3))
