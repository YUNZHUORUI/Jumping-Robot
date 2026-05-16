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

        self.physics = PhysicsEngine(physics_cfg, attitude_cfg)
        self.planner = TrajectoryPlanner(physics_cfg, env_cfg, attitude_cfg)
        self.reward_fn = RewardFunction(reward_cfg)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.q = np.zeros(4, dtype=np.float64)
        self.dq = np.zeros(4, dtype=np.float64)
        self.current_target_idx = 0
        self.steps = 0
        self.prev_touching = False
        self.stance_steps = 0
        self.airborne_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        init_theta = rng.uniform(
            math.radians(self.ecfg.init_theta_min_deg),
            math.radians(self.ecfg.init_theta_max_deg),
        )
        # q uses foot coordinates, so initialize foot on the ground.
        init_foot_y = self.pcfg.ground_y

        self.q = np.array([0.0, init_foot_y, init_theta, self.pcfg.leg_length], dtype=np.float64)
        self.dq = np.zeros(4, dtype=np.float64)

        self.physics.reset()
        self.planner.reset()
        self.current_target_idx = 0
        self.steps = 0
        self.prev_touching = False
        self.stance_steps = 0
        self.airborne_steps = 0

        com_pos = self.physics.get_com_pos(self.q)
        self.planner.plan(com_pos[0], com_pos[1], self.ecfg.targets[self.current_target_idx], rng=rng)

        if self.planner.valid:
            v_x_nom, v_y_nom = self.planner.get_initial_velocity()
            v_x0 = rng.normal(v_x_nom, self.ecfg.init_vx_std)
            v_y0 = rng.normal(v_y_nom, self.ecfg.init_vy_std)
            vx_min = max(0.2, 0.80 * v_x_nom)
            vx_max = max(vx_min + 1e-3, 1.20 * v_x_nom)
            vy_min = 0.80 * v_y_nom
            vy_max = 1.20 * v_y_nom
            v_x0 = float(np.clip(v_x0, vx_min, vx_max))
            v_y0 = float(np.clip(v_y0, vy_min, vy_max))
            v_x0 = 0.92 * v_x_nom + 0.08 * v_x0
            v_y0 = 0.92 * v_y_nom + 0.08 * v_y0
            dtheta0 = float(
                rng.uniform(
                    math.radians(self.ecfg.init_dtheta_min_deg),
                    math.radians(self.ecfg.init_dtheta_max_deg),
                )
            )
            self.dq = np.array([v_x0, v_y0, dtheta0, 0.0], dtype=np.float64)
        else:
            v_x0 = float(rng.uniform(self.ecfg.init_vx_range[0], self.ecfg.init_vx_range[1]))
            v_y0 = float(rng.uniform(self.ecfg.init_vy_range[0], self.ecfg.init_vy_range[1]))
            dtheta0 = float(
                rng.uniform(
                    math.radians(self.ecfg.init_dtheta_min_deg),
                    math.radians(self.ecfg.init_dtheta_max_deg),
                )
            )
            self.dq = np.array([v_x0, v_y0, dtheta0, 0.0], dtype=np.float64)

        return self._get_obs(), {}

    def get_foot_pos(self):
        return self.physics.get_foot_pos(self.q)

    def get_trajectory_state(self, x_current: float):
        com_vx = self.physics.get_com_vel(self.q, self.dq)[0]
        return self.planner.get_state(x_current, v_x=com_vx)

    def _get_obs(self):
        foot_pos = self.physics.get_foot_pos(self.q)
        if self.current_target_idx < len(self.ecfg.targets):
            target_x = self.ecfg.targets[self.current_target_idx]
        else:
            target_x = foot_pos[0]

        dist_x = target_x - foot_pos[0]
        is_touching = 1.0 if foot_pos[1] <= self.pcfg.ground_y + 0.02 else 0.0
        y_ideal, dy_ideal = self.get_trajectory_state(self.physics.get_com_pos(self.q)[0])
        y_error = self.physics.get_com_pos(self.q)[1] - y_ideal
        dy_error = self.physics.get_com_vel(self.q, self.dq)[1] - dy_ideal

        obs = np.concatenate([
            self.q,
            self.dq,
            [dist_x, float(self.current_target_idx), is_touching, y_error, dy_error],
        ])
        return np.clip(obs, -self.ecfg.obs_clip, self.ecfg.obs_clip).astype(np.float32)

    def step(self, action):
        u1 = float(np.clip(action[0], 0.0, 1.0))
        u2 = float(np.clip(action[1], 0.0, 1.0))

        touching = self.physics.step(
            self.q,
            self.dq,
            action,
            self.acfg.landing_theta,
            self.planner.takeoff_theta_target,
            self.acfg.takeoff_theta_tol,
        )

        touchdown_event = touching and (not self.prev_touching)
        liftoff_event = (not touching) and self.prev_touching

        if touching:
            self.stance_steps += 1
            self.airborne_steps = 0
        else:
            self.airborne_steps += 1
            self.stance_steps = 0

        foot_pos = self.physics.get_foot_pos(self.q)
        target_valid = self.current_target_idx < len(self.ecfg.targets)
        target_x = self.ecfg.targets[self.current_target_idx] if target_valid else foot_pos[0]
        dist_to_target = abs(foot_pos[0] - target_x) if target_valid else 0.0

        target_hit = False
        all_targets_done = False
        terminated = False
        truncated = False
        stance_timeout = False

        if touching and self.physics.get_com_vel(self.q, self.dq)[1] > -0.5 and target_valid:
            if dist_to_target < self.ecfg.target_tolerance:
                target_hit = True
                print(
                    f"Hit target {self.current_target_idx} "
                    f"(dist={dist_to_target:.2f}, theta={math.degrees(self.q[2]):.1f} deg)"
                )
                self.current_target_idx += 1
                if self.current_target_idx >= len(self.ecfg.targets):
                    all_targets_done = True
                    terminated = True
                else:
                    com_pos = self.physics.get_com_pos(self.q)
                    self.planner.plan(com_pos[0], com_pos[1], self.ecfg.targets[self.current_target_idx])

                # Anti-local-optimum guard: terminate if policy keeps collapsing in stance.
                if touching and self.stance_steps >= self.ecfg.max_consecutive_stance_steps:
                    stance_timeout = True
                    terminated = True

        # Remove angle-limit termination to avoid trapping policy in local optimum.
        # Keep only safety-related vertical bounds.
        terminated_bad = (
            self.physics.get_com_pos(self.q)[1] > self.reward_fn.cfg.max_height
            or self.physics.get_com_pos(self.q)[1] < self.pcfg.min_com_height
        )
        out_of_bounds = (
            self.q[0] < -1.0
            or (target_valid and foot_pos[0] > target_x + self.reward_fn.cfg.max_overshoot)
        )
        if terminated_bad or out_of_bounds:
            terminated = True

        y_ideal, dy_ideal = self.get_trajectory_state(self.physics.get_com_pos(self.q)[0])
        y_error = self.physics.get_com_pos(self.q)[1] - y_ideal
        dy_error = self.physics.get_com_vel(self.q, self.dq)[1] - dy_ideal

        reward, reward_info = self.reward_fn.compute(
            theta=self.q[2],
            dtheta=self.dq[2],
            vx=self.physics.get_com_vel(self.q, self.dq)[0],
            touching=touching,
            touchdown_event=touchdown_event,
            liftoff_event=liftoff_event,
            traj_valid=self.planner.valid,
            y_error=y_error,
            dy_error=dy_error,
            traj_vx_nom=self.planner.vx_nom,
            com_y=self.physics.get_com_pos(self.q)[1],
            l_curr=self.q[3],
            l_nominal=self.pcfg.leg_length,
            stroke_length=self.pcfg.stroke_length,
            landing_theta_target=self.acfg.landing_theta,
            takeoff_theta_target=self.planner.takeoff_theta_target,
            u1=u1,
            u2=u2,
            dist_to_target=dist_to_target,
            target_x=target_x,
            foot_x=foot_pos[0],
            target_valid=target_valid,
            current_target_idx=self.current_target_idx,
            target_hit=target_hit,
            all_targets_done=all_targets_done,
            terminated_bad=terminated_bad,
            out_of_bounds=out_of_bounds,
            stance_timeout=stance_timeout,
        )

        self.steps += 1
        self.prev_touching = touching
        if self.steps >= self.ecfg.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {"reward_info": reward_info}

    def render(self):
        pass
