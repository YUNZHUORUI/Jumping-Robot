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

from .my_hopper_cfg import MY_HOPPER_CFG


# =========================================================
# 1. 恢复UI窗口与可视化目标点
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
    episode_length_s = 8.0
    decimation = 1  # 100Hz 同频

    action_space = 4
    observation_space = 37
    state_space = 0
    debug_vis = True  # 开启目标点可视化

    ui_window_class_type = QuadhopperEnvWindow

    # === 跳跃机专属奖励系数 ===
    survival_reward_scale = 3.0
    # XY 居中奖励（不再含 Z；不奖励"悬停在 target_z"这种平凡解）
    # σ 从 0.5 收紧到 0.2 → 漂离 30cm 就明显失分
    distance_to_goal_reward_scale = 10.0
    distance_to_goal_sigma = 0.2

    # 横向速度惩罚：抑制 xy 漂移
    horizontal_vel_penalty_scale = -1.0

    # touchdown 时漂离 spawn 的惩罚因子（按 xy_distance 衰减）
    touchdown_xy_sigma = 0.25

    # 耗电惩罚：只在离地时计 → 不阻碍起跳时的高 thrust 探索
    power_penalty_scale = -0.5

    # 离地奖励：直接奖励"不在地上"，破除趴地局部最优
    airtime_reward_scale = 4.0

    # 姿态惩罚（贴地时更严）
    attitude_penalty_scale = -15.0

    # 角速度惩罚：直接抑制旋转，防"在 0° 附近抖"
    angular_vel_penalty_scale = -1.0

    # 相位协同：压簧时省电，反弹时全开
    synergy_reward_scale = 15.0
    anti_synergy_penalty_scale = -5.0

    action_rate_reward_scale = -6.0

    # 二阶动作平滑：抓"周期性震荡"（action_rate 是一阶差分，对 +0.5/-0.5/+0.5 这种来回抖差分总和不大，但二阶差分极大）
    action_smoothness_scale = -0.5

    # === 定点跳跃高度奖励 ===
    height_target_z = 0.8
    height_target_reward_scale = 8.0
    height_target_sigma = 0.15
    touchdown_bonus_scale = 40.0          # 加倍：完成一次有效跳跃极重要
    min_valid_hop_height = 0.25
    landing_upright_k = 8.0

    # === 反趴地：在地超过 grace 步后每步固定扣分，强制起跳 ===
    ground_time_penalty_scale = -5.0
    ground_time_grace_steps = 15          # 0.15s 触地宽限（足够吸收能量再蹬）

    # === height_target 门控：必须最近落地过才发，杜绝纯悬停 ===
    height_target_max_air_steps = 150     # 离地 1.5s 内有效；超过则视为悬停 → 不发分

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
        # 二阶差分需要 t-2 时刻动作
        self._pre_previous_actions = torch.zeros_like(self._actions)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._motor_u = torch.zeros(self.num_envs, 4, device=self.device)

        self.dr_mass_multi = torch.ones(self.num_envs, 1, device=self.device)
        self.dr_inertia_multi = torch.ones(self.num_envs, 3, device=self.device)
        self.dr_tm = torch.ones(self.num_envs, 4, device=self.device) * 0.065

        self._dt_policy = self.sim.cfg.dt * self.cfg.decimation
        self._body_id = self._robot.find_bodies("Body")[0]

        # 严格获取第一个关节的整数索引，防止张量维度变 3 维
        self._spring_joint_id = self._robot.find_joints("center_spring_joint")[0][0]

        # 弹跳跟踪
        self._peak_z_since_touch = torch.zeros(self.num_envs, device=self.device)
        self._was_on_ground_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 连续在地步数（用于反趴地惩罚）
        self._steps_on_ground = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 自上次触地经过的步数（用于 dense height_target 门控）
        self._steps_since_touchdown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._episode_sums = {key: torch.zeros(self.num_envs, device=self.device) for key in
                              ["survival", "distance_to_goal", "power_penalty", "attitude_penalty", "synergy",
                               "anti_synergy", "action_rate", "action_smoothness", "height_target", "touchdown_bonus",
                               "ground_time_penalty", "airtime", "horizontal_vel", "angular_vel"]}

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)

        # =========================================================
        # 2. 恢复场景穹顶光照，解决全黑问题
        # =========================================================
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._pre_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._action_history.append(self._actions.clone())

        delay_idx = torch.randint(1, 3, (1,)).item()
        delayed_actions = self._action_history[-delay_idx]

        # 【修改】允许输出0推力，不要偏置到0.6，彻底打破悬停舒适区！
        target_u = (delayed_actions * 0.5 + 0.5).clamp(0.0, 1.0)

        alpha = self._dt_policy / (self.dr_tm + self._dt_policy)
        self._motor_u = alpha * target_u + (1.0 - alpha) * self._motor_u
        u = self._motor_u

        F_dynamic = -0.9715 * (u ** 2) + 1.2578 * u - 0.0577
        F_extrapolate = 0.349 + (u - 0.64) * ((0.57 - 0.349) / (1.0 - 0.64))
        F = torch.where(u <= 0.64, F_dynamic, F_extrapolate).clamp(0.0, 0.6)
        F = F / self.dr_mass_multi

        L = 0.0813
        K_tau = 3.0e-02
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

        # 弹簧状态获取
        joint_pos = self._robot.data.joint_pos[:, self._spring_joint_id].unsqueeze(1)
        joint_vel = self._robot.data.joint_vel[:, self._spring_joint_id].unsqueeze(1)

        # joint_pos > 0 = SpringLeg 上移 = 弹簧被压缩 = 触地
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
        # 空中也严管姿态：基础系数从 3.0 提到 6.0
        attitude_penalty = roll_pitch_sq * (6.0 + 4.0 * ground_proximity)

        # 角速度惩罚：sum(ωx² + ωy² + ωz²)
        ang_vel_b = self._robot.data.root_ang_vel_b
        angular_vel_sq = (ang_vel_b ** 2).sum(dim=1)

        # 能耗惩罚仅在离地时计；触地反弹阶段允许全力推
        thrust_sum = torch.sum(self._motor_u, dim=1)
        power_penalty = thrust_sum * (~is_contact).float()

        anti_synergy = torch.where(is_contact & (joint_vel > 0.5), thrust_sum, torch.zeros_like(thrust_sum))
        synergy = torch.where(is_contact & (joint_vel < -0.5), thrust_sum, torch.zeros_like(thrust_sum))

        # 距离奖励只算 XY：不为"待在 target_z 高度"这种悬停行为发分
        xy_distance = torch.linalg.norm(
            self._desired_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=1
        )
        dist_reward = torch.exp(-xy_distance / self.cfg.distance_to_goal_sigma)

        # 横向速度惩罚：抑制 xy 漂移（vz 不算）
        lin_vel_w = self._robot.data.root_lin_vel_w
        horizontal_vel_sq = lin_vel_w[:, 0] ** 2 + lin_vel_w[:, 1] ** 2

        action_rate = torch.sum((self._actions - self._previous_actions) ** 2, dim=1)
        # 二阶差分（jerk-like）：a_t - 2·a_{t-1} + a_{t-2}
        action_smoothness = torch.sum(
            (self._actions - 2.0 * self._previous_actions + self._pre_previous_actions) ** 2, dim=1
        )

        # ── 定点跳跃高度跟踪 ────────────────────────────────
        on_ground_now = is_contact
        just_touched = on_ground_now & (~self._was_on_ground_prev)
        valid_hop = self._peak_z_since_touch > self.cfg.min_valid_hop_height

        sigma = self.cfg.height_target_sigma
        target_z = self.cfg.height_target_z
        peak_match = torch.exp(-((self._peak_z_since_touch - target_z) ** 2) / (2.0 * sigma * sigma))
        upright_factor = torch.exp(-roll_pitch_sq * self.cfg.landing_upright_k)
        # 触地 bonus 还要乘 XY 因子：漂离 spawn 落地不给奖励
        xy_match = torch.exp(-xy_distance / self.cfg.touchdown_xy_sigma)
        touchdown_bonus = torch.where(
            just_touched & valid_hop,
            peak_match * upright_factor * xy_match,
            torch.zeros_like(peak_match),
        )

        # 更新 peak: 触地复位为 0，否则取 max(peak, z)
        self._peak_z_since_touch = torch.where(
            on_ground_now,
            torch.zeros_like(self._peak_z_since_touch),
            torch.maximum(self._peak_z_since_touch, z_pos),
        )
        self._was_on_ground_prev = on_ground_now

        # 自上次触地经过的步数：任何在地步都重置（不只 just_touched），离地才累计
        # 用 on_ground_now 而非 just_touched 是为了：长时间在地后起跳，门控仍然开放
        self._steps_since_touchdown = torch.where(
            on_ground_now,
            torch.zeros_like(self._steps_since_touchdown),
            self._steps_since_touchdown + 1,
        )

        # dense height_target：仅在"最近 1.5s 内触地过"才有效，杜绝纯悬停刷分
        recent_touchdown = self._steps_since_touchdown < self.cfg.height_target_max_air_steps
        height_target = torch.where(
            recent_touchdown,
            torch.exp(-((self._peak_z_since_touch - target_z) ** 2) / (2.0 * sigma * sigma)),
            torch.zeros_like(self._peak_z_since_touch),
        )

        # 反趴地：累计在地步数，超过 grace 后每步固定扣分
        self._steps_on_ground = torch.where(
            on_ground_now,
            self._steps_on_ground + 1,
            torch.zeros_like(self._steps_on_ground),
        )
        ground_penalty = (self._steps_on_ground > self.cfg.ground_time_grace_steps).float()

        # 直接奖励"在空中"：z 越高 + 不触地 = 越多分。打破趴地局部最优。
        airtime = (~on_ground_now).float() * torch.clamp(z_pos, 0.0, self.cfg.height_target_z) / self.cfg.height_target_z

        rewards = {
            "survival": torch.ones_like(z_pos) * self.cfg.survival_reward_scale * self.step_dt,
            "distance_to_goal": dist_reward * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "power_penalty": power_penalty * self.cfg.power_penalty_scale * self.step_dt,
            "attitude_penalty": attitude_penalty * self.cfg.attitude_penalty_scale * self.step_dt,
            "synergy": synergy * self.cfg.synergy_reward_scale * self.step_dt,
            "anti_synergy": anti_synergy * self.cfg.anti_synergy_penalty_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_smoothness": action_smoothness * self.cfg.action_smoothness_scale * self.step_dt,
            "height_target": height_target * self.cfg.height_target_reward_scale * self.step_dt,
            "ground_time_penalty": ground_penalty * self.cfg.ground_time_penalty_scale * self.step_dt,
            "airtime": airtime * self.cfg.airtime_reward_scale * self.step_dt,
            "horizontal_vel": horizontal_vel_sq * self.cfg.horizontal_vel_penalty_scale * self.step_dt,
            "angular_vel": angular_vel_sq * self.cfg.angular_vel_penalty_scale * self.step_dt,
            "touchdown_bonus": touchdown_bonus * self.cfg.touchdown_bonus_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
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
        self.dr_tm[env_ids] = torch.empty(num_resets, 4, device=self.device).uniform_(0.055, 0.075)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._pre_previous_actions[env_ids] = 0.0
        self._motor_u[env_ids] = 0.0
        for i in range(len(self._action_history)):
            self._action_history[i][env_ids] = 0.0

        # 目标 XY 锁死 spawn 点（原地跳跃，不追随 target），Z = 跳跃 apex 目标
        self._desired_pos_w[env_ids, :2] = self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = self.cfg.height_target_z

        # 弹跳跟踪 reset
        self._peak_z_since_touch[env_ids] = 0.0
        self._was_on_ground_prev[env_ids] = False
        self._steps_on_ground[env_ids] = 0
        # 初始化为"超过门控阈值"：必须先完成一次落地，dense height_target 才解锁
        self._steps_since_touchdown[env_ids] = self.cfg.height_target_max_air_steps + 1

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :2] += self._terrain.env_origins[env_ids, :2]
        default_root_state[:, 2] = torch.empty(num_resets, device=self.device).uniform_(0.5, 1.2)
        default_root_state[:, 9] = torch.empty(num_resets, device=self.device).uniform_(-2.0, 0.0)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

    # =========================================================
    # 3. 恢复目标点可视化逻辑
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