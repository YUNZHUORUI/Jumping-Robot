# quadhopper/env.py
"""
QuadhopperTargetEnv: Gymnasium environment for the QuadHopper robot.

Orchestrates physics, trajectory planning, and reward computation.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import PHYSICS, ATTITUDE, ENV, REWARD
from .physics import PhysicsEngine
from .trajectory import TrajectoryPlanner
from .reward import RewardFunction


class QuadhopperTargetEnv(gym.Env):
    """
    Single-leg hopping robot that must land on a sequence of targets.

    Observation (13-dim):
        [x, y, theta, l, dx, dy, dtheta, dl,
         dist_x, target_idx, contact, y_err, dy_err]

    Action (2-dim):
        [left_thrust, right_thrust] in [0, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        physics_cfg=PHYSICS,
        attitude_cfg=ATTITUDE,
        env_cfg=ENV,
        reward_cfg=REWARD,
    ):
        super().__init__()

        self.pcfg = physics_cfg
        self.acfg = attitude_cfg
        self.ecfg = env_cfg

        # Sub-modules
        self.physics    = PhysicsEngine(physics_cfg, attitude_cfg)
        self.planner    = TrajectoryPlanner(physics_cfg, env_cfg)
        self.reward_fn  = RewardFunction(reward_cfg)

        # Gymnasium spaces
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # Episode state (initialized in reset)
        self.q:  np.ndarray = np.zeros(4, dtype=np.float64)
        self.dq: np.ndarray = np.zeros(4, dtype=np.float64)
        self.current_target_idx: int = 0
        self.steps: int = 0
        self.prev_touching: bool = False

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Initial tilt angle
        init_theta = rng.uniform(
            math.radians(self.ecfg.init_theta_min_deg),
            math.radians(self.ecfg.init_theta_max_deg),
        )
        # COM height so foot touches the ground at t=0
        init_y = self.pcfg.leg_length * math.cos(init_theta)

        # Reset state vectors
        self.q  = np.array([0.0, init_y, init_theta, self.pcfg.leg_length],
                           dtype=np.float64)
        self.dq = np.zeros(4, dtype=np.float64)

        # Reset sub-modules
        self.physics.reset()
        self.planner.reset()
        self.current_target_idx = 0
        self.steps              = 0
        self.prev_touching      = False

        # Plan first trajectory
        self.planner.plan(
            self.q[0], self.q[1],
            self.ecfg.targets[self.current_target_idx],
            rng=rng,
        )

        # Perfect initial velocity following the parabola
        v_x0, v_y0 = self.planner.get_initial_velocity()
        self.dq = np.array([v_x0, v_y0, 0.0, 0.0], dtype=np.float64)

        return self._get_obs(), {}

    # ---------------------------------------------------------------- helpers
    def get_foot_pos(self) -> np.ndarray:
        return self.physics.get_foot_pos(self.q)

    def get_trajectory_state(self, x_current: float):
        return self.planner.get_state(x_current, v_x=self.dq[0])

    # -------------------------------------------------------------------- obs
    def _get_obs(self) -> np.ndarray:
        foot_pos = self.get_foot_pos()

        if self.current_target_idx < len(self.ecfg.targets):
            target_x = self.ecfg.targets[self.current_target_idx]
        else:
            target_x = foot_pos[0]

        dist_x     = target_x - foot_pos[0]
        is_touching = 1.0 if foot_pos[1] <= self.pcfg.ground_y + 0.02 else 0.0

        y_ideal, dy_ideal = self.get_trajectory_state(self.q[0])
        y_error  = self.q[1] - y_ideal
        dy_error = self.dq[1] - dy_ideal

        obs = np.concatenate([
            self.q, self.dq,
            [dist_x, float(self.current_target_idx),
             is_touching, y_error, dy_error],
        ])
        return np.clip(obs, -self.ecfg.obs_clip, self.ecfg.obs_clip).astype(
            np.float32
        )

    # ------------------------------------------------------------------- step
    def step(self, action):
        u1 = float(np.clip(action[0], 0.0, 1.0))
        u2 = float(np.clip(action[1], 0.0, 1.0))

        # Physics update
        touching = self.physics.step(
            self.q, self.dq,
            action,
            self.planner.takeoff_theta_target,
        )

        # Event detection
        touchdown_event = touching and (not self.prev_touching)
        liftoff_event   = (not touching) and self.prev_touching

        # Target geometry
        foot_pos       = self.get_foot_pos()
        target_valid   = self.current_target_idx < len(self.ecfg.targets)
        dist_to_target = (
            abs(foot_pos[0] - self.ecfg.targets[self.current_target_idx])
            if target_valid else 0.0
        )

        # Target hit logic
        target_hit      = False
        all_targets_done = False
        terminated       = False
        truncated        = False

        if touching and self.dq[1] > -0.5 and target_valid:
            if dist_to_target < self.ecfg.target_tolerance:
                target_hit = True
                print(
                    f"Hit target {self.current_target_idx} "
                    f"(dist={dist_to_target:.2f}, "
                    f"theta={math.degrees(self.q[2]):.1f} deg)"
                )
                self.current_target_idx += 1
                if self.current_target_idx >= len(self.ecfg.targets):
                    all_targets_done = True
                    terminated = True
                else:
                    self.planner.plan(
                        self.q[0], self.q[1],
                        self.ecfg.targets[self.current_target_idx],
                    )

        # Termination conditions
        terminated_bad = (
            abs(self.q[2]) > self.reward_fn.cfg.max_tilt_rad
            or self.q[1] > self.reward_fn.cfg.max_height
            or self.q[1] < self.reward_fn.cfg.min_height
        )
        out_of_bounds = (
            self.q[0] < -1.0
            or (
                target_valid
                and foot_pos[0] > self.ecfg.targets[self.current_target_idx] + 3.0
            )
        )

        if terminated_bad or out_of_bounds:
            terminated = True

        # Trajectory tracking errors
        y_ideal, dy_ideal = self.get_trajectory_state(self.q[0])
        y_error = self.q[1] - y_ideal

        # Reward
        reward, reward_info = self.reward_fn.compute(
            theta=self.q[2],
            dtheta=self.dq[2],
            com_y=self.q[1],
            touching=touching,
            touchdown_event=touchdown_event,
            liftoff_event=liftoff_event,
            traj_valid=self.planner.valid,
            y_error=y_error,
            y_ideal=y_ideal,
            dy_ideal=dy_ideal,
            landing_theta_target=self.acfg.landing_theta,
            takeoff_theta_target=self.planner.takeoff_theta_target,
            u1=u1,
            u2=u2,
            dist_to_target=dist_to_target,
            target_valid=target_valid,
            target_hit=target_hit,
            all_targets_done=all_targets_done,
            terminated_bad=terminated_bad,
            out_of_bounds=out_of_bounds,
        )

        self.steps += 1
        self.prev_touching = touching

        if self.steps >= self.ecfg.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {
            "reward_info": reward_info
        }

    def render(self):
        pass
