from __future__ import annotations

import gymnasium as gym
import torch
import math
from collections import deque

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz
from isaaclab.markers import CUBOID_MARKER_CFG

from .my_drone_cfg import MY_DRONE_CFG


class QuadcopterEnvWindow(BaseEnvWindow):
    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 1  # 100Hz 同频控制

    action_space = 4
    observation_space = 25  # 3(vel) + 3(ang_vel) + 4(quat) + 3(pos_err) + 12(history)
    state_space = 0
    debug_vis = True

    # 奖励函数系数
    survival_reward_scale = 2.0
    distance_to_goal_reward_scale = 5.0
    lin_vel_reward_scale = -0.2
    ang_vel_reward_scale = -0.1
    yaw_error_reward_scale = -4.0
    yaw_vel_reward_scale = -2
    action_rate_reward_scale = -0.8
    action_smoothness_reward_scale = -0.5

    ui_window_class_type = QuadcopterEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=1.5, replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = MY_DRONE_CFG.replace(prim_path="/World/envs/env_.*/Robot")


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._previous_previous_actions = torch.zeros_like(self._actions)

        self.history_len = 3
        self._action_history = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self._action_history.append(torch.zeros_like(self._actions))

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._motor_u = torch.zeros(self.num_envs, 4, device=self.device)

        self.dr_thrust_multi = torch.ones(self.num_envs, 4, device=self.device)
        self.dr_torque_multi = torch.ones(self.num_envs, 1, device=self.device)
        self.dr_mass_multi = torch.ones(self.num_envs, 1, device=self.device)
        self.dr_inertia_multi = torch.ones(self.num_envs, 3, device=self.device)

        # ==============================================================
        # 【核心修复 1】：将虚假的 69ms 延迟降至符合微型无人机物理特性的 15ms
        # ==============================================================
        self.dr_tm = torch.ones(self.num_envs, 4, device=self.device) * 0.015

        self._dt_policy = self.sim.cfg.dt * self.cfg.decimation
        self._body_id = self._robot.find_bodies(".*")[0]

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["survival", "lin_vel", "ang_vel", "distance_to_goal", "yaw_error",
                        "action_rate", "action_smoothness"]
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
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._action_history.append(self._actions.clone())

        # 1. 模拟 10ms 控制延迟
        delayed_actions = self._action_history[-2] if len(self._action_history) > 1 else self._actions
        # 这里的悬停点是 0.45，保持不变，极其完美
        target_u = (delayed_actions * 0.5 + 0.45).clamp(0.0, 1.0)

        # 2. 模拟电机惯性 (alpha 基于我们修正过的 15ms 延迟)
        alpha = self._dt_policy / (self.dr_tm + self._dt_policy)
        self._motor_u = alpha * target_u + (1.0 - alpha) * self._motor_u
        u = self._motor_u

        # 3. 动力模型与质量/惯量等效随机化
        F_dynamic = -0.9715 * (u ** 2) + 1.2578 * u - 0.0577
        F_extrapolate = 0.349 + (u - 0.64) * ((0.57 - 0.349) / (1.0 - 0.64))
        F = torch.where(u <= 0.64, F_dynamic, F_extrapolate)
        F = torch.where(u < 0.05, torch.zeros_like(F), F).clamp(0.0, 0.6)

        F = F * self.dr_thrust_multi / self.dr_mass_multi

        L = 0.0813
        K_tau = 1.985850e-02
        F1, F2, F3, F4 = F[:, 0], F[:, 1], F[:, 2], F[:, 3]
        u1, u2, u3, u4 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

        tau_x = L * (F1 + F4 - F2 - F3) * self.dr_torque_multi.squeeze() / self.dr_inertia_multi[:, 0]
        tau_y = L * (F1 + F2 - F3 - F4) * self.dr_torque_multi.squeeze() / self.dr_inertia_multi[:, 1]
        tau_z = K_tau * (
                    u2 ** 2 + u4 ** 2 - u1 ** 2 - u3 ** 2) * self.dr_torque_multi.squeeze() / self.dr_inertia_multi[:,
                                                                                              2]

        self._thrust[:, 0, 2] = F1 + F2 + F3 + F4
        self._moment[:, 0, 0], self._moment[:, 0, 1], self._moment[:, 0, 2] = tau_x, tau_y, tau_z

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        lin_vel = self._robot.data.root_lin_vel_b.clone()
        ang_vel = self._robot.data.root_ang_vel_b.clone()
        quat = self._robot.data.root_quat_w.clone()
        pos_error = desired_pos_b.clone()

        pos_error += torch.randn_like(pos_error) * 0.01
        quat = torch.nn.functional.normalize(quat + torch.randn_like(quat) * 0.005, p=2, dim=-1)
        lin_vel += torch.randn_like(lin_vel) * 0.02
        ang_vel += torch.randn_like(ang_vel) * 0.01

        dist_xy = torch.linalg.norm(pos_error[:, :2], dim=1).unsqueeze(1)
        mask_weight = torch.clamp(1.5 - dist_xy, 0.0, 1.0)

        masked_pos_error = pos_error * mask_weight
        masked_lin_vel = lin_vel * mask_weight

        history_actions = torch.cat(list(self._action_history), dim=-1)
        obs = torch.cat([masked_lin_vel, ang_vel, quat, masked_pos_error, history_actions], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_sq = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel_sq = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        q = self._robot.data.root_quat_w
        # ==============================================================
        # 【核心修复 2】：修正了四元数转 Yaw 角的致命减号错误，改为加号。
        # 之前这个错误导致飞机为了拿奖励故意扭曲机身疯狂摇摆。
        # ==============================================================
        yaw_angle = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        yaw_error = torch.square(yaw_angle)

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        dist_reward = torch.exp(-distance_to_goal / 0.5)

        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_smoothness = torch.sum(
            torch.square(self._actions - 2.0 * self._previous_actions + self._previous_previous_actions), dim=1)

        rewards = {
            "survival": torch.ones_like(lin_vel_sq) * self.cfg.survival_reward_scale * self.step_dt,
            "lin_vel": lin_vel_sq * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_sq * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": dist_reward * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "yaw_error": yaw_error * self.cfg.yaw_error_reward_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_smoothness": action_smoothness * self.cfg.action_smoothness_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 3.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        self.dr_thrust_multi[env_ids] = torch.empty(num_resets, 4, device=self.device).uniform_(0.92, 1.08)
        self.dr_torque_multi[env_ids] = torch.empty(num_resets, 1, device=self.device).uniform_(0.9, 1.1)

        # ==============================================================
        # 【核心修复 3】：域随机化中，将电机延迟时间常数的随机范围缩小至真实的 [10ms, 20ms]
        # ==============================================================
        self.dr_tm[env_ids] = torch.empty(num_resets, 4, device=self.device).uniform_(0.01, 0.02)

        self.dr_mass_multi[env_ids] = torch.empty(num_resets, 1, device=self.device).uniform_(0.9, 1.1)
        self.dr_inertia_multi[env_ids] = torch.empty(num_resets, 3, device=self.device).uniform_(0.9, 1.1)

        self._actions[env_ids] = 0.0
        self._motor_u[env_ids] = 0.0
        for i in range(len(self._action_history)):
            self._action_history[i][env_ids] = 0.0

        self._desired_pos_w[env_ids, :2] = torch.empty(num_resets, 2, device=self.device).uniform_(-1.5,
                                                                                                   1.5) + self._terrain.env_origins[
                                                                                                          env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.empty(num_resets, device=self.device).uniform_(0.8, 1.2)

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, :3] += torch.randn_like(default_root_state[:, :3]) * 0.12

        rand_euler = torch.randn(num_resets, 3, device=self.device) * 0.22
        default_root_state[:, 3:7] = quat_from_euler_xyz(rand_euler[:, 0], rand_euler[:, 1], rand_euler[:, 2])

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(self._desired_pos_w)