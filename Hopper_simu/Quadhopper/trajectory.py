# quadhopper/trajectory.py
"""
Parabolic trajectory planner for QuadHopper.

Computes ideal ballistic arc from current COM position to
target foot contact point and provides real-time tracking state.
"""
import math
import numpy as np
from .config import PhysicsConfig, EnvConfig


class TrajectoryPlanner:
    """
    Plans and evaluates a parabolic (ballistic) trajectory.

    Trajectory model:
        y(x) = a*(x - x0)^2 + b*(x - x0) + y0
    """

    def __init__(self, physics_cfg: PhysicsConfig, env_cfg: EnvConfig):
        self.physics = physics_cfg
        self.env = env_cfg

        # Planned trajectory coefficients
        self.a: float = 0.0
        self.b_local: float = 0.0   # slope at launch
        self.x0: float = 0.0
        self.y0: float = 0.0
        self.valid: bool = False
        self.takeoff_theta_target: float = math.radians(
            env_cfg.traj_tilt_min_deg
        )

    def reset(self):
        """Clear current trajectory."""
        self.valid = False
        self.a = 0.0
        self.b_local = 0.0
        self.x0 = 0.0
        self.y0 = 0.0

    def plan(
        self,
        x_curr: float,
        y_curr: float,
        target_x_foot: float,
        rng: np.random.Generator = None,
    ) -> bool:
        """
        Plan a parabolic trajectory from (x_curr, y_curr) to target_x_foot.

        Args:
            x_curr:       Current COM x position.
            y_curr:       Current COM y position.
            target_x_foot: Foot target x position on the ground.
            rng:          Optional numpy RNG for reproducibility.

        Returns:
            True if planning succeeded, False otherwise.
        """
        estimated_land_theta = math.radians(-15.0)
        offset_x = self.physics.leg_length * math.sin(estimated_land_theta)
        x_target_com = target_x_foot - offset_x

        dx = x_target_com - x_curr
        dy = (
            self.physics.leg_length * math.cos(estimated_land_theta) - y_curr
        )

        if dx <= 0.3:
            self.valid = False
            return False

        # Random launch angle
        if rng is not None:
            tilt_deg = rng.uniform(
                self.env.traj_tilt_min_deg, self.env.traj_tilt_max_deg
            )
        else:
            tilt_deg = np.random.uniform(
                self.env.traj_tilt_min_deg, self.env.traj_tilt_max_deg
            )

        launch_alpha = math.radians(90.0 - tilt_deg)
        self.takeoff_theta_target = math.radians(tilt_deg)

        tan_a = math.tan(launch_alpha)
        cos_a = math.cos(launch_alpha)

        denominator = dx * tan_a - dy
        if denominator <= 0.01:
            # Fallback to steeper angle
            launch_alpha = math.radians(75.0)
            tan_a  = math.tan(launch_alpha)
            cos_a  = math.cos(launch_alpha)
            denominator = max(dx * tan_a - dy, 0.001)

        v0_sq = (self.physics.gravity * dx ** 2) / (
            2.0 * (cos_a ** 2) * denominator
        )

        self.a       = -self.physics.gravity / (2.0 * v0_sq * cos_a ** 2)
        self.b_local = tan_a
        self.x0      = x_curr
        self.y0      = y_curr
        self.valid   = True
        return True

    def get_initial_velocity(self):
        """
        Compute the initial COM velocity that follows the planned trajectory.

        Returns:
            (v_x0, v_y0) tuple, or (3.0, 4.0) fallback if not valid.
        """
        if not self.valid:
            return 3.0, 4.0
        v_x0 = math.sqrt(-self.physics.gravity / (2.0 * self.a))
        v_y0 = self.b_local * v_x0
        return v_x0, v_y0

    def get_state(self, x_current: float, v_x: float = 1.0):
        """
        Query the ideal height and vertical velocity at x_current.

        Args:
            x_current: Current x position of COM.
            v_x:       Current horizontal velocity (used to scale dy/dt).

        Returns:
            (y_ideal, dy_ideal)
        """
        if not self.valid:
            return 1.1, 0.0

        dx = max(x_current - self.x0, 0.0)
        y_ideal    = self.a * dx ** 2 + self.b_local * dx + self.y0
        slope      = 2.0 * self.a * dx + self.b_local
        dy_ideal   = slope * max(v_x, 0.1)
        return y_ideal, dy_ideal
