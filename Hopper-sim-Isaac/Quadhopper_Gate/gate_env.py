"""Curriculum environment for hybrid jump, gate traversal, and landing.

The policy outputs collective thrust and desired body rates. A small low-level rate
controller mixes these commands to the four preserved Quadhopper motor channels.
Gate collision is evaluated analytically in this first training environment; visual
and rigid gate geometry can be added without changing the policy contract.
"""

from __future__ import annotations

from collections import deque

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz

from .asset_cfg import POWER_MODEL_PATH, QUADHOPPER_CFG


@configclass
class QuadhopperGateEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 1

    action_space = 4
    observation_space = 50
    state_space = 0

    # Curriculum stage: 0 stable hop, 1 landing target, 2..5 shrinking gate, 6 multi-gate scaffold.
    curriculum_stage: int = 0
    gate_distance: float = 0.65
    gate_height: float = 0.30
    robot_span: float = 0.23
    robot_safety_radius: float = 0.12
    max_height: float = 1.25
    max_xy_distance: float = 2.5

    # Hierarchical command limits and low-level rate gains.
    max_body_rate = (7.0, 7.0, 4.0)
    rate_mix_gain = (0.035, 0.035, 0.020)
    train_motor_time_constant_min = 0.10
    train_motor_time_constant_max = 0.14
    play_motor_time_constant = 0.125
    play_action_delay = 3

    # Reward scales; stage-dependent weighting is applied in _get_rewards.
    alive_scale = 0.2
    upright_scale = 1.0
    angular_rate_scale = -0.03
    action_rate_scale = -0.08
    control_effort_scale = -0.01
    power_penalty_scale = -0.08
    spring_release_scale = 0.8
    spring_compression_scale = -0.8
    hop_scale = 2.5
    trajectory_progress_scale = 8.0
    gate_alignment_scale = 1.0
    gate_crossing_scale = 20.0
    landing_scale = 12.0
    collision_scale = -15.0

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=4.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    robot: ArticulationCfg = QUADHOPPER_CFG.replace(prim_path="/World/envs/env_.*/Robot")


class QuadhopperGateEnv(DirectRLEnv):
    cfg: QuadhopperGateEnvCfg

    def __init__(self, cfg: QuadhopperGateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._action_history = deque(maxlen=5)
        for _ in range(5):
            self._action_history.append(torch.zeros_like(self._actions))

        self._motor_u = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._dt_policy = self.sim.cfg.dt * self.cfg.decimation

        self.dr_tm = torch.full_like(self._actions, 0.069)
        self.dr_mass_multi = torch.ones(self.num_envs, 1, device=self.device)
        self.dr_inertia_multi = torch.ones(self.num_envs, 3, device=self.device)

        body_ids, _ = self._robot.find_bodies("Body")
        if not body_ids:
            raise RuntimeError("higherjump HopperAsset.usd must contain a rigid body named 'Body'.")
        self._body_id = body_ids[0:1]

        joint_ids, _ = self._robot.find_joints("center_spring_joint")
        self._spring_joint_id = joint_ids[0] if joint_ids else None

        self.power_model = torch.jit.load(str(POWER_MODEL_PATH), map_location=self.device).to(self.device)
        self.power_model.eval()
        self._motor_u_history = torch.zeros(self.num_envs, 25, 4, device=self.device)
        self._soc_tracker = torch.zeros(self.num_envs, 1, device=self.device)

        self._gate_center_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._gate_normal_w = torch.zeros_like(self._gate_center_w)
        self._gate_radius = torch.zeros(self.num_envs, 1, device=self.device)
        self._landing_pos_w = torch.zeros_like(self._gate_center_w)
        self._previous_gate_side = torch.zeros(self.num_envs, device=self.device)
        self._previous_target_distance = torch.zeros(self.num_envs, device=self.device)
        self._gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._just_crossed = torch.zeros_like(self._gate_passed)
        self._collision = torch.zeros_like(self._gate_passed)
        self._landed = torch.zeros_like(self._gate_passed)
        self._has_taken_off = torch.zeros_like(self._gate_passed)

        reward_names = [
            "alive", "upright", "angular_rate", "action_rate", "control_effort",
            "hop", "trajectory_progress", "gate_alignment", "gate_crossing",
            "landing", "collision", "power", "spring_release", "spring_compression",
        ]
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in reward_names
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
        self._prev_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))
        self._action_history.append(self._actions.clone())

        # higherjump actuator delay: randomly use a command from 20--40 ms ago.
        delay_idx = (
            self.cfg.play_action_delay
            if self.num_envs == 1
            else torch.randint(2, 5, (1,), device=self.device).item()
        )
        delayed_action = self._action_history[-delay_idx]

        # Policy action: collective command plus desired body rates.
        collective = (0.50 + 0.50 * delayed_action[:, 0:1]).clamp(0.0, 1.0)
        max_rate = torch.tensor(self.cfg.max_body_rate, device=self.device)
        desired_rate = delayed_action[:, 1:4] * max_rate
        rate_error = desired_rate - self._robot.data.root_ang_vel_b
        gains = torch.tensor(self.cfg.rate_mix_gain, device=self.device)
        roll_mix, pitch_mix, yaw_mix = (rate_error * gains).unbind(dim=1)

        # Mixer signs match the preserved baseline torque equations.
        target_u = torch.cat(
            [
                collective + roll_mix[:, None] + pitch_mix[:, None] - yaw_mix[:, None],
                collective - roll_mix[:, None] + pitch_mix[:, None] + yaw_mix[:, None],
                collective - roll_mix[:, None] - pitch_mix[:, None] - yaw_mix[:, None],
                collective + roll_mix[:, None] - pitch_mix[:, None] + yaw_mix[:, None],
            ],
            dim=1,
        ).clamp(0.0, 1.0)

        # Preserve the higherjump 69 ms nominal first-order motor response.
        alpha = self._dt_policy / (self.dr_tm + self._dt_policy)
        self._motor_u = alpha * target_u + (1.0 - alpha) * self._motor_u
        u = self._motor_u

        self._motor_u_history = torch.roll(self._motor_u_history, shifts=-1, dims=1)
        self._motor_u_history[:, -1, :] = u
        self._soc_tracker += u.sum(dim=1, keepdim=True) / 10000.0

        motor_force = (-0.2371 * u.square() + 0.8130 * u + 0.0113) / self.dr_mass_multi

        f1, f2, f3, f4 = motor_force.unbind(dim=1)
        u1, u2, u3, u4 = u.unbind(dim=1)
        tau_x = 0.0813 * (f1 + f4 - f2 - f3) / self.dr_inertia_multi[:, 0]
        tau_y = 0.0813 * (f1 + f2 - f3 - f4) / self.dr_inertia_multi[:, 1]
        tau_z = 5.4e-02 * -(u1.square() + u3.square() - u2.square() - u4.square())
        tau_z = tau_z / self.dr_inertia_multi[:, 2]

        self._thrust[:, 0, 2] = motor_force.sum(dim=1)
        self._moment[:, 0, 0] = tau_x
        self._moment[:, 0, 1] = tau_y
        self._moment[:, 0, 2] = tau_z

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )

    def _leg_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._spring_joint_id is None:
            zeros = torch.zeros(self.num_envs, 1, device=self.device)
            joint_pos, joint_vel = zeros, zeros
        else:
            joint_pos = self._robot.data.joint_pos[:, self._spring_joint_id : self._spring_joint_id + 1]
            joint_vel = self._robot.data.joint_vel[:, self._spring_joint_id : self._spring_joint_id + 1]
        contact = (joint_pos > 0.002).float()
        return joint_pos, joint_vel, contact

    def _phase(self, contact: torch.Tensor) -> torch.Tensor:
        vz = self._robot.data.root_lin_vel_w[:, 2:3]
        compression = torch.logical_and(contact > 0.5, vz <= 0.15)
        flight = torch.logical_and(contact <= 0.5, ~self._gate_passed[:, None])
        landing = ~(compression | flight)
        return torch.cat([compression.float(), flight.float(), landing.float()], dim=1)

    def _get_observations(self) -> dict:
        root_pos = self._robot.data.root_pos_w
        root_quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_b
        ang_vel = self._robot.data.root_ang_vel_b

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        gravity_b = quat_apply_inverse(root_quat, gravity_w)
        gate_pos_b = quat_apply_inverse(root_quat, self._gate_center_w - root_pos)
        gate_normal_b = quat_apply_inverse(root_quat, self._gate_normal_w)
        landing_pos_b = quat_apply_inverse(root_quat, self._landing_pos_w - root_pos)
        joint_pos, joint_vel, contact = self._leg_state()
        phase = self._phase(contact)
        clearance = self._gate_radius - self.cfg.robot_safety_radius

        # Sensor-like noise. Commands, phase, and action history remain exact.
        noisy_lin = lin_vel + torch.randn_like(lin_vel) * 0.02
        noisy_ang = ang_vel + torch.randn_like(ang_vel) * 0.01
        noisy_quat = torch.nn.functional.normalize(
            root_quat + torch.randn_like(root_quat) * 0.005, p=2, dim=-1
        )
        noisy_gate_pos = gate_pos_b + torch.randn_like(gate_pos_b) * 0.01
        history = torch.cat(list(self._action_history), dim=1)

        obs = torch.cat(
            [
                noisy_lin, noisy_ang, noisy_quat, gravity_b,
                joint_pos, joint_vel, contact,
                noisy_gate_pos, gate_normal_b, self._gate_radius, clearance,
                landing_pos_b, phase, history,
            ],
            dim=1,
        )
        return {"policy": obs}

    def _update_gate_events(self):
        pos_rel = self._robot.data.root_pos_w - self._gate_center_w
        gate_side = torch.sum(pos_rel * self._gate_normal_w, dim=1)
        radial_vec = pos_rel - gate_side[:, None] * self._gate_normal_w
        radial_distance = torch.linalg.norm(radial_vec, dim=1)
        forward_crossing = torch.logical_and(self._previous_gate_side < 0.0, gate_side >= 0.0)
        inside = radial_distance <= (self._gate_radius[:, 0] - self.cfg.robot_safety_radius)
        gate_enabled = self.cfg.curriculum_stage >= 2
        self._just_crossed = forward_crossing & inside & gate_enabled
        bad_crossing = forward_crossing & ~inside & gate_enabled
        near_rim = (gate_side.abs() < 0.08) & ~inside & gate_enabled
        self._collision = bad_crossing | near_rim
        self._gate_passed |= self._just_crossed
        self._previous_gate_side.copy_(gate_side)

        _, _, contact = self._leg_state()
        relative_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        self._has_taken_off |= (contact[:, 0] < 0.5) | (relative_height > 0.20)
        landing_error = torch.linalg.norm(
            self._robot.data.root_pos_w[:, :2] - self._landing_pos_w[:, :2], dim=1
        )
        slow = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=1) < 0.6
        objective_done = self._gate_passed if gate_enabled else torch.ones_like(self._gate_passed)
        self._landed = (
            objective_done & self._has_taken_off & (contact[:, 0] > 0.5)
            & slow & (landing_error < 0.25)
        )

    def _get_rewards(self) -> torch.Tensor:
        self._update_gate_events()
        pos = self._robot.data.root_pos_w
        vel_w = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        quat = self._robot.data.root_quat_w
        _, _, contact = self._leg_state()

        gate_enabled = self.cfg.curriculum_stage >= 2
        if gate_enabled:
            active_target = torch.where(
                self._gate_passed[:, None], self._landing_pos_w, self._gate_center_w
            )
        else:
            active_target = self._landing_pos_w
        target_distance = torch.linalg.norm(active_target - pos, dim=1)
        progress = (self._previous_target_distance - target_distance).clamp(-0.2, 0.2)
        self._previous_target_distance.copy_(target_distance)

        toward_gate = torch.nn.functional.normalize(self._gate_center_w - pos, dim=1)
        velocity_dir = torch.nn.functional.normalize(vel_w + 1.0e-6, dim=1)
        gate_alignment = torch.sum(toward_gate * velocity_dir, dim=1).clamp(min=0.0)
        tilt = quat[:, 1].square() + quat[:, 2].square()
        hop = torch.relu(vel_w[:, 2]) * contact[:, 0]

        pwm_short = self._motor_u_history[:, -10:, :].reshape(self.num_envs, 40)
        pwm_mid = self._motor_u_history[:, -15:, :].mean(dim=1)
        pwm_long = self._motor_u_history.mean(dim=1)
        power_input = torch.cat([pwm_short, pwm_mid, pwm_long, self._soc_tracker], dim=1)
        with torch.no_grad():
            predicted_power = self.power_model(power_input).squeeze(-1)
        joint_pos, joint_vel, _ = self._leg_state()
        spring_contact = joint_pos[:, 0] > 0.002
        spring_release = torch.where(
            spring_contact & (joint_vel[:, 0] < -0.5), predicted_power, torch.zeros_like(predicted_power)
        )
        spring_compression = torch.where(
            spring_contact & (joint_vel[:, 0] > 0.5), predicted_power, torch.zeros_like(predicted_power)
        )

        stage = self.cfg.curriculum_stage
        agility_weight = 0.0 if stage < 2 else min(1.0, (stage - 1) / 3.0)
        stability_weight = 1.0 - 0.5 * agility_weight
        rewards = {
            "alive": torch.ones(self.num_envs, device=self.device) * self.cfg.alive_scale * self.step_dt,
            "upright": (1.0 - tilt).clamp(min=0.0) * self.cfg.upright_scale * stability_weight * self.step_dt,
            "angular_rate": ang_vel.square().sum(dim=1) * self.cfg.angular_rate_scale * stability_weight * self.step_dt,
            "action_rate": (self._actions - self._prev_actions).square().sum(dim=1) * self.cfg.action_rate_scale * self.step_dt,
            "control_effort": self._motor_u.square().sum(dim=1) * self.cfg.control_effort_scale * self.step_dt,
            "hop": hop * self.cfg.hop_scale * (1.0 if stage == 0 else 0.25) * self.step_dt,
            "trajectory_progress": progress * self.cfg.trajectory_progress_scale * (0.0 if stage == 0 else 1.0),
            "gate_alignment": gate_alignment * self.cfg.gate_alignment_scale * agility_weight * self.step_dt,
            "gate_crossing": self._just_crossed.float() * self.cfg.gate_crossing_scale,
            "landing": self._landed.float() * self.cfg.landing_scale,
            "collision": self._collision.float() * self.cfg.collision_scale,
            "power": predicted_power * self.cfg.power_penalty_scale * self.step_dt,
            "spring_release": spring_release * self.cfg.spring_release_scale * self.step_dt,
            "spring_compression": spring_compression * self.cfg.spring_compression_scale * self.step_dt,
        }
        for name, value in rewards.items():
            self._episode_sums[name] += value
        return torch.stack(list(rewards.values()), dim=0).sum(dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        pos = self._robot.data.root_pos_w
        tilt = self._robot.data.root_quat_w[:, 1:3].square().sum(dim=1)
        origin_xy = self._terrain.env_origins[:, :2]
        out_of_bounds = torch.linalg.norm(pos[:, :2] - origin_xy, dim=1) > self.cfg.max_xy_distance
        unsafe = (tilt > 0.85) | (pos[:, 2] > self.cfg.max_height) | (pos[:, 2] < -0.05)
        failed = unsafe | out_of_bounds | self._collision
        completed = self._landed & (self.episode_length_buf > 20)
        return failed | completed, timeout

    def _stage_gate_radius(self) -> float:
        # Stage 0/1 keeps a virtual generous gate for a stable observation contract.
        return {0: 0.30, 1: 0.30, 2: 0.24, 3: 0.1725, 4: 0.1265, 5: 0.12075, 6: 0.14}.get(
            int(self.cfg.curriculum_stage), 0.24
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        log = {}
        for name, values in self._episode_sums.items():
            log[f"Episode_Reward/{name}"] = values[env_ids].mean() / self.max_episode_length_s
            values[env_ids] = 0.0
        log["Metrics/gate_pass_rate"] = self._gate_passed[env_ids].float().mean()
        log["Metrics/landing_rate"] = self._landed[env_ids].float().mean()
        log["Curriculum/stage"] = float(self.cfg.curriculum_stage)
        self.extras["log"] = log

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)

        if self.num_envs == 1:
            self.dr_tm[env_ids] = self.cfg.play_motor_time_constant
            self.dr_mass_multi[env_ids] = 1.0
            self.dr_inertia_multi[env_ids] = 1.0
        else:
            self.dr_tm[env_ids] = torch.empty(n, 4, device=self.device).uniform_(
                self.cfg.train_motor_time_constant_min,
                self.cfg.train_motor_time_constant_max,
            )
            self.dr_mass_multi[env_ids] = torch.empty(n, 1, device=self.device).uniform_(0.95, 1.05)
            self.dr_inertia_multi[env_ids] = torch.empty(n, 3, device=self.device).uniform_(0.9, 1.1)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._motor_u[env_ids] = 0.0
        for history in self._action_history:
            history[env_ids] = 0.0
        self._motor_u_history[env_ids] = 0.0
        self._soc_tracker[env_ids] = 0.0

        origins = self._terrain.env_origins[env_ids]
        default_state = self._robot.data.default_root_state[env_ids].clone()
        default_state[:, :3] += origins
        default_state[:, 0:2] += torch.randn(n, 2, device=self.device) * 0.03
        default_state[:, 2] = origins[:, 2] + 0.50
        default_state[:, 9] = torch.empty(n, device=self.device).uniform_(-1.0, 0.0)
        attitude_noise = 0.04 if self.cfg.curriculum_stage < 3 else 0.10
        euler = torch.randn(n, 3, device=self.device) * attitude_noise
        default_state[:, 3:7] = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self._robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)

        lateral = 0.0 if self.cfg.curriculum_stage < 3 else 0.12
        gate_offset_y = torch.empty(n, device=self.device).uniform_(-lateral, lateral)
        self._gate_center_w[env_ids] = origins
        self._gate_center_w[env_ids, 0] += self.cfg.gate_distance
        self._gate_center_w[env_ids, 1] += gate_offset_y
        self._gate_center_w[env_ids, 2] += self.cfg.gate_height
        self._gate_normal_w[env_ids] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self._gate_radius[env_ids] = self._stage_gate_radius()
        self._landing_pos_w[env_ids] = origins
        if self.cfg.curriculum_stage == 1:
            self._landing_pos_w[env_ids, 0] += torch.empty(n, device=self.device).uniform_(0.15, 0.35)
            self._landing_pos_w[env_ids, 1] += torch.empty(n, device=self.device).uniform_(-0.15, 0.15)
        elif self.cfg.curriculum_stage >= 2:
            self._landing_pos_w[env_ids, 0] += self.cfg.gate_distance + 0.45
            self._landing_pos_w[env_ids, 1] += gate_offset_y
        self._landing_pos_w[env_ids, 2] += 0.27335430681705475

        initial_rel = default_state[:, :3] - self._gate_center_w[env_ids]
        initial_side = torch.sum(initial_rel * self._gate_normal_w[env_ids], dim=1)
        self._previous_gate_side[env_ids] = initial_side
        stage_target = self._landing_pos_w[env_ids] if self.cfg.curriculum_stage < 2 else self._gate_center_w[env_ids]
        self._previous_target_distance[env_ids] = torch.linalg.norm(stage_target - default_state[:, :3], dim=1)
        self._gate_passed[env_ids] = False
        self._just_crossed[env_ids] = False
        self._collision[env_ids] = False
        self._landed[env_ids] = False
        self._has_taken_off[env_ids] = False
