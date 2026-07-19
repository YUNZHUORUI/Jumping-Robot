"""Gymnasium environment for QuadHopper target jumping."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import ATTITUDE, ENV, PHYSICS, REWARD
from .physics import PhysicsEngine
from .reward import RewardFunction
from .trajectory import TrajectoryPlanner


class QuadhopperTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, physics_cfg=PHYSICS, attitude_cfg=ATTITUDE, env_cfg=ENV, reward_cfg=REWARD):
        super().__init__()

        self.pcfg = physics_cfg
        self.acfg = attitude_cfg
        self.ecfg = env_cfg

        self.physics  = PhysicsEngine(physics_cfg)
        self.planner  = TrajectoryPlanner(physics_cfg, env_cfg)
        self.reward_fn = RewardFunction(reward_cfg)

        # PPO works best with symmetric normalized actions.  These are mapped
        # to physical motor commands in [0, 1] inside step().
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 15-dim observation (see _get_obs for full description)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        self.q  = np.zeros(4, dtype=np.float64)
        self.dq = np.zeros(4, dtype=np.float64)
        self.current_target_idx      = 0
        self.steps                   = 0
        self.prev_touching           = False
        self.consecutive_stance_steps = 0
        self.last_motor_cmd          = np.zeros(2, dtype=np.float32)
        self.best_foot_x             = 0.0
        self.no_progress_counter     = 0
        self.prev_vy                 = 0.0
        self.hop_max_height          = 0.0

    @staticmethod
    def _action_to_motor_cmd(action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        return np.clip(0.5 * (action + 1.0), 0.0, 1.0).astype(np.float32)

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        options = options or {}
        reset_mode = options.get("mode", self.ecfg.reset_mode)

        self.physics.reset()
        self.planner.reset()
        self.current_target_idx       = 0
        self.steps                    = 0
        self.prev_touching            = False
        self.consecutive_stance_steps = 0
        self.best_foot_x              = 0.0
        self.no_progress_counter      = 0
        self.prev_vy                  = 0.0
        self.hop_max_height           = 0.0

        landing_theta = math.radians(self.reward_fn.cfg.phi_td_target_deg)

        if reset_mode == "ground":
            init_theta = rng.uniform(
                math.radians(self.ecfg.ground_init_theta_min_deg),
                math.radians(self.ecfg.ground_init_theta_max_deg),
            )
            init_l = self.pcfg.leg_length - self.ecfg.ground_init_leg_compression
            init_l = float(np.clip(init_l, self.pcfg.leg_min_length, self.pcfg.leg_length))
            self.q = np.array(
                [0.0, self.pcfg.ground_y, init_theta, init_l],
                dtype=np.float64,
            )
            self.dq = np.zeros(4, dtype=np.float64)
            self.physics.start_stance(self.q)
            self.prev_touching = True
            self.consecutive_stance_steps = 1

            com_pos = self.physics.get_com_pos(self.q)
            self.hop_max_height = float(com_pos[1])
            self.planner.plan(
                com_pos[0], com_pos[1],
                self.ecfg.targets[self.current_target_idx],
                rng=rng,
                landing_theta=landing_theta,
            )
            return self._get_obs(), {}

        if reset_mode != "ballistic":
            raise ValueError(f"Unknown reset mode: {reset_mode!r}")

        init_theta = rng.uniform(
            math.radians(self.ecfg.init_theta_min_deg),
            math.radians(self.ecfg.init_theta_max_deg),
        )
        # 起始时足端在地面上方 1 mm，避免立刻触发触地检测
        init_foot_y = self.pcfg.ground_y + 1e-3

        self.q  = np.array([0.0, init_foot_y, init_theta, self.pcfg.leg_length], dtype=np.float64)
        self.dq = np.zeros(4, dtype=np.float64)

        com_pos = self.physics.get_com_pos(self.q)
        self.hop_max_height = float(com_pos[1])
        self.planner.plan(
            com_pos[0], com_pos[1],
            self.ecfg.targets[self.current_target_idx],
            rng=rng,
            landing_theta=landing_theta,
        )

        if self.planner.valid:
            v_x_nom, v_y_nom = self.planner.get_initial_velocity()
            v_x0 = float(np.clip(rng.normal(v_x_nom, self.ecfg.init_vx_std),
                                  0.80 * v_x_nom, 1.20 * v_x_nom))
            v_y0 = float(np.clip(rng.normal(v_y_nom, self.ecfg.init_vy_std),
                                  0.80 * v_y_nom, 1.20 * v_y_nom))
            # 小幅向标称值回归，减少初始随机噪声
            v_x0 = 0.92 * v_x_nom + 0.08 * v_x0
            v_y0 = 0.92 * v_y_nom + 0.08 * v_y0
        else:
            v_x0 = float(rng.uniform(self.ecfg.init_vx_range[0], self.ecfg.init_vx_range[1]))
            v_y0 = float(rng.uniform(self.ecfg.init_vy_range[0], self.ecfg.init_vy_range[1]))

        dtheta0 = float(rng.uniform(
            math.radians(self.ecfg.init_dtheta_min_deg),
            math.radians(self.ecfg.init_dtheta_max_deg),
        ))
        self.dq = np.array([v_x0, v_y0, dtheta0, 0.0], dtype=np.float64)

        return self._get_obs(), {}

    # -------------------------------------------------------- helpers
    def get_foot_pos(self):
        return self.physics.get_foot_pos(self.q)

    def get_trajectory_state(self, x_current: float):
        """供渲染器和测试脚本调用（保留兼容性）。"""
        com_vx = self.physics.get_com_vel(self.q, self.dq)[0]
        return self.planner.get_state(x_current, v_x=com_vx)

    # -------------------------------------------------------- observation
    def _get_obs(self):
        """
        15-dim observation:
          0  theta           体角（相对竖直）
          1  dtheta          角速度
          2  l_norm          腿长归一化偏差 l/l_nom - 1  (0=自然长)
          3  dl              腿伸缩速度
          4  vx_com          质心水平速度
          5  vy_com          质心竖直速度
          6  com_y           质心高度
          7  height_err      目标跳跃高度 - 质心高度
          8  vx_deficit      规划所需 vx - 实际 vx（支撑相有效，飞行相置零）
          9  vy_deficit      规划所需 vy - 实际 vy（支撑相有效，飞行相置零）
         10  dx_target       足端到当前目标的水平距离
         11  is_touching     接触标志 (0/1)
         12  task_phase      保留维度，恒为 0（控制律不依赖绝对跳序号）
         13  theta_err_td    theta 与落地目标角 phi_td 之差
         14  stance_ratio    当前连续支撑步数 / 最大支撑步数 (0~1)
        """
        com_pos = self.physics.get_com_pos(self.q)
        com_vel = self.physics.get_com_vel(self.q, self.dq)
        foot_pos = self.physics.get_foot_pos(self.q)

        target_valid = self.current_target_idx < len(self.ecfg.targets)
        target_x = (self.ecfg.targets[self.current_target_idx]
                    if target_valid else foot_pos[0])
        dx_target = target_x - foot_pos[0]

        is_touching = 1.0 if self.physics.stance_active else 0.0

        # 速度亏量：支撑相才有意义
        if self.planner.valid and is_touching:
            vx_def = self.planner.vx_nom - float(com_vel[0])
            vy_def = self.planner.vy_nom - float(com_vel[1])
        else:
            vx_def = 0.0
            vy_def = 0.0

        phi_td = math.radians(self.reward_fn.cfg.phi_td_target_deg)
        theta_err_td = float(self.q[2]) - phi_td

        stance_ratio = (self.consecutive_stance_steps
                        / max(self.ecfg.max_consecutive_stance_steps, 1))

        obs = np.array([
            float(self.q[2]),                              # theta
            float(self.dq[2]),                             # dtheta
            float(self.q[3]) / self.pcfg.leg_length - 1.0, # l_norm
            float(self.dq[3]),                             # dl
            float(com_vel[0]),                             # vx_com
            float(com_vel[1]),                             # vy_com
            float(com_pos[1]),                             # com_y
            float(self.reward_fn.cfg.target_height - com_pos[1]), # height_err
            vx_def,                                        # vx_deficit
            vy_def,                                        # vy_deficit
            dx_target,                                     # dx_target
            is_touching,                                   # is_touching
            0.0,                                           # task_phase (translation invariant)
            theta_err_td,                                  # theta error vs phi_td
            float(np.clip(stance_ratio, 0.0, 1.0)),        # stance_ratio
        ], dtype=np.float32)

        return np.clip(obs, -self.ecfg.obs_clip, self.ecfg.obs_clip)

    # ---------------------------------------------------------------- step
    def step(self, action):
        motor_cmd = self._action_to_motor_cmd(action)
        self.last_motor_cmd = motor_cmd
        u1 = float(motor_cmd[0])
        u2 = float(motor_cmd[1])

        # 落地目标角 = phi_td；起飞目标角由规划器给出（SLIP 假设下约 90°-alpha）
        landing_theta   = math.radians(self.reward_fn.cfg.phi_td_target_deg)
        takeoff_theta   = self.planner.takeoff_theta_target  # pi/2 - theta_opt
        takeoff_tol     = self.acfg.takeoff_theta_tol

        touching = self.physics.step(
            self.q, self.dq, motor_cmd,
            landing_theta, takeoff_theta, takeoff_tol,
        )

        touchdown_event = touching and (not self.prev_touching)
        liftoff_event   = (not touching) and self.prev_touching

        # 连续支撑步计数
        if touching:
            self.consecutive_stance_steps += 1
        else:
            self.consecutive_stance_steps = 0

        stance_timeout = (
            self.consecutive_stance_steps >= self.ecfg.max_consecutive_stance_steps
        )

        com_pos  = self.physics.get_com_pos(self.q)
        com_vel  = self.physics.get_com_vel(self.q, self.dq)
        self.hop_max_height = max(self.hop_max_height, float(com_pos[1]))
        apex_event = (
            (not touching) and self.prev_vy > 0.0 and float(com_vel[1]) <= 0.0
        )
        foot_pos = self.physics.get_foot_pos(self.q)

        target_valid = self.current_target_idx < len(self.ecfg.targets)
        target_x     = (self.ecfg.targets[self.current_target_idx]
                        if target_valid else foot_pos[0])
        dx_target     = target_x - foot_pos[0]
        dist_to_target = abs(float(foot_pos[0]) - float(target_x)) if target_valid else 0.0

        target_hit    = False
        all_targets_done = False
        terminated    = False
        truncated     = False

        progress_margin = self.ecfg.no_progress_min_delta
        if float(foot_pos[0]) > self.best_foot_x + progress_margin:
            self.best_foot_x = float(foot_pos[0])
            self.no_progress_counter = 0
        else:
            self.no_progress_counter += 1

        no_progress_timeout = (
            target_valid
            and self.no_progress_counter >= self.ecfg.no_progress_steps
            and dist_to_target > self.ecfg.target_tolerance
        )

        # 目标命中检测
        if (
            touching
            and com_vel[1] > -0.5
            and self.hop_max_height >= self.ecfg.min_target_hop_height
            and self.hop_max_height <= self.ecfg.max_target_hop_height
            and target_valid
        ):
            if dist_to_target < self.ecfg.target_tolerance:
                target_hit = True
                if self.ecfg.print_hit_events:
                    print(
                        f"Hit target {self.current_target_idx} "
                        f"(dist={dist_to_target:.2f}, theta={math.degrees(self.q[2]):.1f}°)"
                    )
                self.current_target_idx += 1
                if self.current_target_idx >= len(self.ecfg.targets):
                    all_targets_done = True
                    terminated = True
                else:
                    self.hop_max_height = float(com_pos[1])
                    com_now = self.physics.get_com_pos(self.q)
                    self.planner.plan(
                        com_now[0], com_now[1],
                        self.ecfg.targets[self.current_target_idx],
                        landing_theta=landing_theta,
                    )

        # 异常终止检测
        terminated_bad = (
            float(com_pos[1]) > self.reward_fn.cfg.max_height
            or float(com_pos[1]) < self.pcfg.min_com_height
            or abs(float(self.q[2])) > self.reward_fn.cfg.max_tilt_rad
        )
        out_of_bounds = (
            float(self.q[0]) < -1.0
            or (target_valid and float(foot_pos[0]) > float(target_x) + self.reward_fn.cfg.max_overshoot)
        )
        if terminated_bad or out_of_bounds or stance_timeout or no_progress_timeout:
            terminated = True

        reward, reward_info = self.reward_fn.compute(
            theta=float(self.q[2]),
            dtheta=float(self.dq[2]),
            vx_com=float(com_vel[0]),
            vy_com=float(com_vel[1]),
            l_curr=float(self.q[3]),
            l_nominal=self.pcfg.leg_length,
            com_y=float(com_pos[1]),
            dl=float(self.dq[3]),
            stroke_length=self.pcfg.stroke_length,
            touching=touching,
            touchdown_event=touchdown_event,
            liftoff_event=liftoff_event,
            apex_event=apex_event,
            traj_valid=self.planner.valid,
            vx_nom=self.planner.vx_nom,
            vy_nom=self.planner.vy_nom,
            dx_target=dx_target,
            u1=u1,
            u2=u2,
            # Bug fix: pass real coordinates — old code defaulted both to 0.0,
            # making landing_proximity reward always 30 regardless of foot position.
            foot_x=float(foot_pos[0]),
            target_x=float(target_x),
            target_hit=target_hit,
            all_targets_done=all_targets_done,
            terminated_bad=terminated_bad or no_progress_timeout,
            out_of_bounds=out_of_bounds,
            stance_timeout=stance_timeout,
        )

        self.steps += 1
        self.prev_touching = touching
        self.prev_vy = float(com_vel[1])
        if self.steps >= self.ecfg.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {"reward_info": reward_info}

    def render(self):
        pass
