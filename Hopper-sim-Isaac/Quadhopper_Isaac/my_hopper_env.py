from __future__ import annotations

import torch
from collections import deque

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms

from .my_drone_cfg import MY_DRONE_CFG


@configclass
class HopperEnvCfg(DirectRLEnvCfg):
    episode_length_s = 8.0
    decimation = 2           # policy 100 Hz；物理 200 Hz（dt=0.005）

    # obs: lin_vel(3) + ang_vel(3) + quat(4) + z(1) + vz(1) + xy_err_b(2) + action_hist(4*3=12)
    action_space = 4
    observation_space = 26
    state_space = 0

    # ── 奖励系数 ─────────────────────────────
    # 起跳速度：在贴地时奖励向上速度（exp衰减保证只在低空计分）
    hop_velocity_scale   =  8.0
    # 在地面附近的额外奖励（鼓励弹回地面）
    ground_bonus_scale   =  2.0
    # 竖直稳定：倾斜角越小越好
    upright_scale        = -6.0
    # 生存奖励
    survival_scale       =  0.5
    # 动作平滑
    action_rate_scale    = -0.3
    action_smooth_scale  = -0.2
    # 定点保持：在 env 原点附近 exp(-d/0.3) 给正奖励
    xy_pos_reward_scale  =  4.0
    # XY 水平速度阻尼：抑制漂移
    xy_vel_penalty_scale = -0.5

    # 超过此高度视为"飞走"而非"跳跃"，episode 终止
    max_hop_height: float = 0.4   # m

    # ── 物理 ─────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,              # 200 Hz 物理（弹簧稳定需要高频）
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="max",   # max模式：取双方最大弹性系数
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.8,   # 高弹性：代替弹簧关节提供弹跳冲量
        ),
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=4 * 1024 * 1024,
            min_position_iteration_count=4,  # 更多求解器迭代，提高接触稳定性
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.8
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # 2-link articulation 启用后，GPU buffer 限制：1024 → 256
        num_envs=256, env_spacing=2.0, replicate_physics=False, clone_in_fabric=False
    )

    robot: ArticulationCfg = MY_DRONE_CFG.replace(prim_path="/World/envs/env_.*/Robot")


class HopperEnv(DirectRLEnv):
    cfg: HopperEnvCfg

    def __init__(self, cfg: HopperEnvCfg, render_mode: str | None = None, **kwargs):
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

        # 定点目标（在 _reset_idx 中绑定到各 env 的原点）
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # 电机时间常数域随机化
        self.dr_tm = torch.ones(self.num_envs, 4, device=self.device) * 0.015
        self.dr_thrust_multi = torch.ones(self.num_envs, 4, device=self.device)
        self.dr_torque_multi = torch.ones(self.num_envs, 1, device=self.device)
        self.dr_mass_multi   = torch.ones(self.num_envs, 1, device=self.device)
        self.dr_inertia_multi = torch.ones(self.num_envs, 3, device=self.device)

        # 新版资产把刚体改成了 /Robot/body（articulation root 上层是 Xform）
        body_ids, _ = self._robot.find_bodies("body")
        self._body_id = body_ids[0:1]

        self._episode_sums = {
            k: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for k in ["survival", "hop_velocity", "ground_bonus", "upright",
                      "action_rate", "action_smooth", "xy_pos", "xy_vel"]
        }

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
        # 偏移0.30（原0.45）：零动作时推力≈0.93N < 重力1.41N(含腿20g)→自然下落
        # 最大动作时推力≈1.55N > 重力→可以起跳；策略必须靠弹跳维持高度
        target_u = (delayed * 0.5 + 0.30).clamp(0.0, 1.0)

        alpha = self._dt_policy / (self.dr_tm + self._dt_policy)
        self._motor_u = alpha * target_u + (1.0 - alpha) * self._motor_u
        u = self._motor_u

        F_dynamic = -0.9715 * (u ** 2) + 1.2578 * u - 0.0577
        F_extrapolate = 0.349 + (u - 0.64) * ((0.57 - 0.349) / (1.0 - 0.64))
        F = torch.where(u <= 0.64, F_dynamic, F_extrapolate)
        F = torch.where(u < 0.05, torch.zeros_like(F), F).clamp(0.0, 0.6)
        F = F * self.dr_thrust_multi / self.dr_mass_multi

        L = 0.0813
        K_tau = 3.0e-02
        F1, F2, F3, F4 = F[:, 0], F[:, 1], F[:, 2], F[:, 3]
        u1, u2, u3, u4 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

        tau_x = L * (F1 + F4 - F2 - F3) * self.dr_torque_multi.squeeze() / self.dr_inertia_multi[:, 0]
        tau_y = L * (F1 + F2 - F3 - F4) * self.dr_torque_multi.squeeze() / self.dr_inertia_multi[:, 1]
        tau_z = K_tau * (u2**2 + u4**2 - u1**2 - u3**2) * self.dr_torque_multi.squeeze() / self.dr_inertia_multi[:, 2]

        self._thrust[:, 0, 2] = F1 + F2 + F3 + F4
        self._moment[:, 0, 0] = tau_x
        self._moment[:, 0, 1] = tau_y
        self._moment[:, 0, 2] = tau_z

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _get_observations(self) -> dict:
        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b
        quat    = self._robot.data.root_quat_w
        z       = self._robot.data.root_pos_w[:, 2:3]   # 高度
        # 用世界系 vz，和 _get_rewards 里的 vz 保持一致（避免倾斜时观测/奖励对不齐）
        vz      = self._robot.data.root_lin_vel_w[:, 2:3]

        # 定点误差（机体系 XY 分量）：让策略知道往哪个方向倾斜把自己拉回原点
        pos_err_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        xy_err_b = pos_err_b[:, :2]

        # 加噪声
        lin_vel = lin_vel + torch.randn_like(lin_vel) * 0.02
        ang_vel = ang_vel + torch.randn_like(ang_vel) * 0.01
        quat    = torch.nn.functional.normalize(quat + torch.randn_like(quat) * 0.005, p=2, dim=-1)
        xy_err_b = xy_err_b + torch.randn_like(xy_err_b) * 0.01

        history = torch.cat(list(self._action_history), dim=-1)
        obs = torch.cat([lin_vel, ang_vel, quat, z, vz, xy_err_b, history], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        z   = self._robot.data.root_pos_w[:, 2]
        # world-frame vertical velocity (not body-frame, to be tilt-independent)
        vz  = self._robot.data.root_lin_vel_w[:, 2]
        q   = self._robot.data.root_quat_w

        # 1. 起跳速度奖励：只在贴近地面时计分，exp(-z*10) 在 0.1m 处衰减到 ~0.37
        #    鼓励策略在低空猛推力起跳，而非持续悬停
        ground_proximity = torch.exp(-z * 10.0)
        hop_velocity = torch.relu(vz) * ground_proximity

        # 2. 落地奖励：在很低位置（弹跳触地阶段）给小额奖励，鼓励弹回地面
        on_ground = (z < 0.08).float()

        # 3. 竖直稳定惩罚
        tilt = torch.sum(q[:, 1:3] ** 2, dim=1)

        # 4. 动作平滑
        action_rate   = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)
        action_smooth = torch.sum(torch.square(self._actions - 2*self._prev_actions + self._prev_prev_actions), dim=1)

        # 5. 定点保持：原点附近 exp 形奖励 + XY 速度阻尼
        xy_err_w = self._desired_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2]
        dist_xy = torch.linalg.norm(xy_err_w, dim=1)
        xy_pos = torch.exp(-dist_xy / 0.3)
        vxy = self._robot.data.root_lin_vel_w[:, :2]
        xy_vel = torch.sum(vxy ** 2, dim=1)

        rewards = {
            "survival":      torch.ones(self.num_envs, device=self.device) * self.cfg.survival_scale * self.step_dt,
            "hop_velocity":  hop_velocity  * self.cfg.hop_velocity_scale  * self.step_dt,
            "ground_bonus":  on_ground     * self.cfg.ground_bonus_scale  * self.step_dt,
            "upright":       tilt          * self.cfg.upright_scale       * self.step_dt,
            "action_rate":   action_rate   * self.cfg.action_rate_scale   * self.step_dt,
            "action_smooth": action_smooth * self.cfg.action_smooth_scale * self.step_dt,
            "xy_pos":        xy_pos        * self.cfg.xy_pos_reward_scale * self.step_dt,
            "xy_vel":        xy_vel        * self.cfg.xy_vel_penalty_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, val in rewards.items():
            self._episode_sums[key] += val
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        q = self._robot.data.root_quat_w
        z = self._robot.data.root_pos_w[:, 2]
        tilt = torch.sum(q[:, 1:3] ** 2, dim=1)
        # 翻倒 或 飞出弹跳区间（超过max_hop_height视为飞走而非弹跳）
        died = torch.logical_or(tilt > 0.5, z > self.cfg.max_hop_height)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras = {}
        for key in self._episode_sums:
            extras[f"Episode_Reward/{key}"] = (
                torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        # 域随机化
        self.dr_thrust_multi[env_ids]  = torch.empty(n, 4, device=self.device).uniform_(0.92, 1.08)
        self.dr_torque_multi[env_ids]  = torch.empty(n, 1, device=self.device).uniform_(0.9,  1.1)
        self.dr_tm[env_ids]            = torch.empty(n, 4, device=self.device).uniform_(0.01, 0.02)
        self.dr_mass_multi[env_ids]    = torch.empty(n, 1, device=self.device).uniform_(0.9,  1.1)
        self.dr_inertia_multi[env_ids] = torch.empty(n, 3, device=self.device).uniform_(0.9,  1.1)

        self._actions[env_ids] = 0.0
        self._motor_u[env_ids] = 0.0
        for i in range(len(self._action_history)):
            self._action_history[i][env_ids] = 0.0

        # 从离地0.15m出发：腿向下延伸约0.12m，需要足够间隙防止初始穿地
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 0] += torch.randn(n, device=self.device) * 0.05
        default_root_state[:, 1] += torch.randn(n, device=self.device) * 0.05
        default_root_state[:, 2]  = self._terrain.env_origins[env_ids, 2] + 0.15

        # 定点目标：每个 env 的原点 XY，Z 用初始高度（subtract_frame_transforms 需 3D）
        self._desired_pos_w[env_ids, :2] = self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2]  = self._terrain.env_origins[env_ids, 2] + 0.15

        # 初始姿态：基本竖直，加小随机倾斜
        rand_euler = torch.randn(n, 3, device=self.device) * 0.05
        default_root_state[:, 3:7] = quat_from_euler_xyz(rand_euler[:, 0], rand_euler[:, 1], rand_euler[:, 2])

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
