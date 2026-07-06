from __future__ import annotations

import os
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
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat
from isaaclab.markers import CUBOID_MARKER_CFG

from .my_hopper_cfg import MY_HOPPER_CFG


# =========================================================
# 1. UI窗口与可视化目标点
# =========================================================
class QuadhopperEnvWindow(BaseEnvWindow):
    def __init__(self, env: QuadhopperEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadhopperEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 1  # 100Hz 同频

    action_space = 4
    observation_space = 37
    state_space = 0
    debug_vis = True  # 开启目标点可视化

    ui_window_class_type = QuadhopperEnvWindow

    # === 跳跃机专属奖励系数 ===
    survival_reward_scale = 5.0

    # 👑 1. 重振主线权威：大幅提高高度奖励，让它向往天空
    distance_to_xy_reward_scale = 13.0
    distance_to_z_reward_scale = 20.0

    yaw_penalty_scale = -4.0
    attitude_penalty_scale = -15.0

    # 👑 2. 滞地惩罚：赖在地上不跳会被疯狂扣分
    lazy_ground_penalty_scale = -10.0

    # 👑 3. 真实物理功耗体系下的权重 (假设功耗在100W~300W量级)
    # 彻底杜绝它靠小跳来刷协同分
    power_penalty_scale = -0.002  # 基础耗电惩罚 (100W * -0.002 = -0.2分)
    synergy_reward_scale = 0.02  # 触地反弹奖励 (100W * 0.02 = +2.0分)
    anti_synergy_penalty_scale = -0.02  # 压簧耗电惩罚 (100W * -0.02 = -2.0分)

    # 👑 4. Sim-to-Real: 仅抑制横向移动，允许偏头以防止落地摔倒
    lateral_vel_penalty_scale = -4
    action_rate_reward_scale = -1

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.5, restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5, restitution=0.0),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0)
    robot: ArticulationCfg = MY_HOPPER_CFG.replace(prim_path="/World/envs/env_.*/Robot")


class QuadhopperEnv(DirectRLEnv):
    cfg: QuadhopperEnvCfg

    def __init__(self, cfg: QuadhopperEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.history_len = 5
        self._action_history = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self._action_history.append(torch.zeros(self.num_envs, 4, device=self.device))

        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._motor_u = torch.zeros(self.num_envs, 4, device=self.device)

        self.dr_mass_multi = torch.ones(self.num_envs, 1, device=self.device)
        self.dr_inertia_multi = torch.ones(self.num_envs, 3, device=self.device)
        self.dr_tm = torch.ones(self.num_envs, 4, device=self.device) * 0.069

        self._dt_policy = self.sim.cfg.dt * self.cfg.decimation
        self._body_id = self._robot.find_bodies("Body")[0]
        self._spring_joint_id = self._robot.find_joints("center_spring_joint")[0][0]

        # =========================================================
        # 👑 引入神经网络功耗模型及记忆缓冲区
        # =========================================================
        _power_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "quadhopper_memory_power.pt")
        self.power_model = torch.jit.load(_power_model_path).to(self.device)
        self.power_model.eval()

        self.nn_history_len = 25
        self._motor_u_history = torch.zeros(self.num_envs, self.nn_history_len, 4, device=self.device)
        self._soc_tracker = torch.zeros(self.num_envs, 1, device=self.device)

        sum_keys = ["survival", "distance_to_xy", "distance_to_z", "yaw_p", "power_penalty",
                    "attitude_penalty", "synergy", "anti_synergy", "lateral_vel_p", "lazy_ground_p"]
        self._episode_sums = {key: torch.zeros(self.num_envs, device=self.device) for key in sum_keys}

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
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._action_history.append(self._actions.clone())

        delay_idx = torch.randint(2, 5, (1,)).item()
        delayed_actions = self._action_history[-delay_idx]

        target_u = (delayed_actions * 0.5 + 0.5).clamp(0.0, 1.0)
        alpha = self._dt_policy / (self.dr_tm + self._dt_policy)
        self._motor_u = alpha * target_u + (1.0 - alpha) * self._motor_u
        u = self._motor_u

        # =========================================================
        # 👑 神经网络数据同步 (0~1 映射为 0~1000)
        # =========================================================
        mapped_pwm = u * 1000.0

        self._motor_u_history = torch.roll(self._motor_u_history, shifts=-1, dims=1)
        self._motor_u_history[:, -1, :] = mapped_pwm

        self._soc_tracker += torch.sum(mapped_pwm, dim=1, keepdim=True) / 10000.0

        # === 物理推力计算依然使用归一化的 u ===
        F = -0.2371 * u ** 2 + 0.8130 * u + 0.0113
        F = F / self.dr_mass_multi

        L = 0.0813
        K_tau = 5.002588e-02
        F1, F2, F3, F4 = F[:, 0], F[:, 1], F[:, 2], F[:, 3]
        u1, u2, u3, u4 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

        tau_x = L * (F1 + F4 - F2 - F3) / self.dr_inertia_multi[:, 0]
        tau_y = L * (F1 + F2 - F3 - F4) / self.dr_inertia_multi[:, 1]
        tau_z = K_tau * -(u1 ** 2 + u3 ** 2 - u2 ** 2 - u4 ** 2) / self.dr_inertia_multi[:, 2]

        self._thrust[:, 0, 2] = F1 + F2 + F3 + F4
        self._moment[:, 0, 0], self._moment[:, 0, 1], self._moment[:, 0, 2] = tau_x, tau_y, tau_z

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        pos_error, _ = subtract_frame_transforms(self._robot.data.root_pos_w, self._robot.data.root_quat_w,
                                                 self._desired_pos_w)
        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b
        quat = self._robot.data.root_quat_w
        z_pos = self._robot.data.root_pos_w[:, 2].unsqueeze(1)

        joint_pos = self._robot.data.joint_pos[:, self._spring_joint_id].unsqueeze(1)
        joint_vel = self._robot.data.joint_vel[:, self._spring_joint_id].unsqueeze(1)
        is_contact = (joint_pos > 0.002).float()

        history_actions = torch.cat(list(self._action_history), dim=-1)

        obs = torch.cat([lin_vel, ang_vel, quat, pos_error, z_pos, is_contact, joint_pos, joint_vel, history_actions],
                        dim=-1)
        obs += torch.randn_like(obs) * 0.01
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        q = self._robot.data.root_quat_w
        z_pos = self._robot.data.root_pos_w[:, 2]

        joint_pos = self._robot.data.joint_pos[:, self._spring_joint_id]
        joint_vel = self._robot.data.joint_vel[:, self._spring_joint_id]
        is_contact = (joint_pos > 0.002)

        roll_pitch_sq = q[:, 1] ** 2 + q[:, 2] ** 2
        ground_proximity = torch.clamp(1.0 - z_pos, 0.0, 1.0)
        attitude_penalty = roll_pitch_sq * (1.0 + 5.0 * ground_proximity)

        # 抑制横向移动，防止漂移
        lin_vel_b = self._robot.data.root_lin_vel_b
        lateral_vel_penalty = torch.sum(torch.square(lin_vel_b[:, :2]), dim=1)

        # =========================================================
        # 👑 调用神经网络模型推演真实功耗
        # =========================================================
        pwm_short = self._motor_u_history[:, -10:, :].reshape(self.num_envs, 40)
        pwm_mid = torch.mean(self._motor_u_history[:, -15:, :], dim=1)
        pwm_long = torch.mean(self._motor_u_history[:, -25:, :], dim=1)

        nn_inputs = torch.cat([pwm_short, pwm_mid, pwm_long, self._soc_tracker], dim=-1)

        with torch.no_grad():
            predicted_power = self.power_model(nn_inputs).squeeze(-1)

        power_penalty = predicted_power

        # 使用真实功耗作为协同奖励的基底
        anti_synergy = torch.where(is_contact & (joint_vel > 0.5), predicted_power, torch.zeros_like(predicted_power))
        synergy = torch.where(is_contact & (joint_vel < -0.5), predicted_power, torch.zeros_like(predicted_power))

        xy_error = self._desired_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2]
        distance_to_goal_xy = torch.linalg.norm(xy_error, dim=1)
        dist_reward_xy = torch.exp(-distance_to_goal_xy / 0.6)

        z_error = torch.abs(self._desired_pos_w[:, 2] - self._robot.data.root_pos_w[:, 2])
        dist_reward_z = torch.exp(-z_error / 0.4)

        # 👑 滞地惩罚：如果在 0.3 米以下徘徊，狠狠扣分
        lazy_ground_penalty = torch.where(z_pos < 0.3, torch.ones_like(z_pos), torch.zeros_like(z_pos))

        _, _, yaw = euler_xyz_from_quat(q)
        yaw_penalty = yaw ** 2

        rewards = {
            "survival": torch.ones_like(z_pos) * self.cfg.survival_reward_scale * self.step_dt,
            "distance_to_xy": dist_reward_xy * self.cfg.distance_to_xy_reward_scale * self.step_dt,
            "distance_to_z": dist_reward_z * self.cfg.distance_to_z_reward_scale * self.step_dt,
            "yaw_p": yaw_penalty * self.cfg.yaw_penalty_scale * self.step_dt,

            "power_penalty": power_penalty * self.cfg.power_penalty_scale * self.step_dt,
            "attitude_penalty": attitude_penalty * self.cfg.attitude_penalty_scale * self.step_dt,

            "synergy": synergy * self.cfg.synergy_reward_scale * self.step_dt,
            "anti_synergy": anti_synergy * self.cfg.anti_synergy_penalty_scale * self.step_dt,

            "lateral_vel_p": lateral_vel_penalty * self.cfg.lateral_vel_penalty_scale * self.step_dt,
            "lazy_ground_p": lazy_ground_penalty * self.cfg.lazy_ground_penalty_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            if key in self._episode_sums:
                self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        q = self._robot.data.root_quat_w
        roll_pitch_sq = q[:, 1] ** 2 + q[:, 2] ** 2
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.05, roll_pitch_sq > 0.5)
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
        self.dr_mass_multi[env_ids] = torch.empty(num_resets, 1, device=self.device).uniform_(0.95, 1.05)
        self.dr_inertia_multi[env_ids] = torch.empty(num_resets, 3, device=self.device).uniform_(0.9, 1.1)
        self.dr_tm[env_ids] = torch.empty(num_resets, 4, device=self.device).uniform_(0.04, 0.1)

        self._actions[env_ids] = 0.0
        self._motor_u[env_ids] = 0.0
        for i in range(len(self._action_history)):
            self._action_history[i][env_ids] = 0.0

        # =========================================================
        # 👑 重置时清空神经网络历史与电量
        # =========================================================
        self._motor_u_history[env_ids] = 0.0
        self._soc_tracker[env_ids] = 0.0

        self._desired_pos_w[env_ids, :2] = torch.empty(num_resets, 2, device=self.device).uniform_(-1.0,
                                                                                                   1.0) + self._terrain.env_origins[
                                                                                                          env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.empty(num_resets, device=self.device).uniform_(1.0, 1.5)

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :2] += self._terrain.env_origins[env_ids, :2]
        default_root_state[:, 2] = torch.empty(num_resets, device=self.device).uniform_(0.5, 1.2)
        default_root_state[:, 9] = torch.empty(num_resets, device=self.device).uniform_(-2.0, 0.0)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

    # =========================================================
    # 3. 目标点可视化逻辑
    # =========================================================
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