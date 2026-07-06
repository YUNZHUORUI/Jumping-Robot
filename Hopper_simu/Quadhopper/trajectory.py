"""Parabolic trajectory planner for QuadHopper."""

import math
import numpy as np

from .config import EnvConfig, PhysicsConfig

_DEFAULT_TAKEOFF_THETA = math.pi / 4  # placeholder until plan() computes pi/2 - theta_opt


class TrajectoryPlanner:
    """Plans and evaluates a ballistic trajectory from current state to target."""

    def __init__(
        self,
        physics_cfg: PhysicsConfig,
        env_cfg: EnvConfig,
    ):
        self.physics = physics_cfg
        self.env = env_cfg

        self.a: float = 0.0
        self.b_local: float = 0.0
        self.x0: float = 0.0
        self.y0: float = 0.0
        self.vx_nom: float = 0.0
        self.vy_nom: float = 0.0
        self.v0_min: float = 0.0
        self.theta_opt: float = 0.0
        self.valid: bool = False
        self.takeoff_theta_target: float = _DEFAULT_TAKEOFF_THETA

    def reset(self):
        self.valid = False
        self.a = 0.0
        self.b_local = 0.0
        self.x0 = 0.0
        self.y0 = 0.0
        self.vx_nom = 0.0
        self.vy_nom = 0.0
        self.v0_min = 0.0
        self.theta_opt = 0.0
        self.takeoff_theta_target = _DEFAULT_TAKEOFF_THETA

    def plan(
        self,
        x_curr: float,
        y_curr: float,
        target_x_foot: float,
        rng: np.random.Generator = None,
        landing_theta: float = None,
    ) -> bool:
        estimated_land_theta = math.radians(-3.0) if landing_theta is None else float(landing_theta)
        offset_x = self.physics.leg_length * math.sin(estimated_land_theta)
        x_target_com = target_x_foot - offset_x

        dx = x_target_com - x_curr
        dy = self.physics.leg_length * math.cos(estimated_land_theta) - y_curr

        if dx <= 0.3:
            self.valid = False
            return False

        # Minimum-energy ballistic solution (used as baseline):
        # theta_opt = atan((z + sqrt(x^2 + z^2)) / x)
        # v_min^2   = g * (z + sqrt(x^2 + z^2))
        r = math.sqrt(dx * dx + dy * dy)
        theta_min = math.atan2(dy + r, dx)
        v_min_sq = self.physics.gravity * (dy + r)
        if v_min_sq <= 1e-8:
            self.valid = False
            return False

        v_min = math.sqrt(v_min_sq)
        vx_min_energy = max(0.2, v_min * math.cos(theta_min))
        vy_min_energy = v_min * math.sin(theta_min)

        # Raise parabola apex above the minimum-energy trajectory.
        # This reduces aggressive leg compression near touchdown.
        g = self.physics.gravity
        y_target = self.physics.leg_length * math.cos(estimated_land_theta)
        apex_min_energy_rel = (vy_min_energy * vy_min_energy) / (2.0 * g)

        apex_scale = max(1.0, self.env.traj_apex_scale)
        apex_clearance = max(0.02, self.env.traj_apex_clearance)
        apex_required_rel = max(
            apex_min_energy_rel * apex_scale,
            (max(y_curr, y_target) - y_curr) + apex_clearance,
        )

        vy_nom = math.sqrt(max(2.0 * g * apex_required_rel, 1e-8))

        # Solve dy = vy*(dx/vx) - 0.5*g*(dx/vx)^2 for vx.
        # Use the slower valid branch to keep the trajectory higher.
        disc = vy_nom * vy_nom - 2.0 * g * dy
        vx_nom = vx_min_energy
        if disc > 0.0:
            sqrt_disc = math.sqrt(disc)
            u_candidates = [
                (vy_nom + sqrt_disc) / (g * dx),
                (vy_nom - sqrt_disc) / (g * dx),
            ]

            vx_candidates = []
            for u in u_candidates:
                if u > 1e-8:
                    vx_cand = 1.0 / u
                    vx_candidates.append(vx_cand)

            if vx_candidates:
                # Prefer slower forward speed (higher arc / longer flight time).
                vx_nom = min(vx_candidates)

        vx_nom = float(np.clip(vx_nom, self.env.traj_min_vx, self.env.traj_max_vx))

        # Recompute vy so the endpoint still matches exactly after vx clipping.
        t_flight = dx / max(vx_nom, 1e-6)
        vy_nom = (dy + 0.5 * g * t_flight * t_flight) / max(t_flight, 1e-6)
        if vy_nom <= 1e-6:
            vy_nom = vy_min_energy
            vx_nom = vx_min_energy

        self.a = -self.physics.gravity / (2.0 * vx_nom * vx_nom)
        self.b_local = vy_nom / vx_nom
        self.x0 = x_curr
        self.y0 = y_curr
        self.vx_nom = vx_nom
        self.vy_nom = vy_nom
        self.v0_min = v_min
        self.theta_opt = math.atan2(vy_nom, vx_nom)

        # Keep compatibility with stance attitude control convention
        self.takeoff_theta_target = (math.pi / 2.0) - self.theta_opt
        self.valid = True
        return True

    def get_initial_velocity(self):
        if not self.valid:
            return 3.0, 4.0
        return self.vx_nom, self.vy_nom

    def get_state(self, x_current: float, v_x: float = 1.0):
        if not self.valid:
            return 1.1, 0.0

        dx = max(x_current - self.x0, 0.0)
        y_ideal = self.a * dx ** 2 + self.b_local * dx + self.y0
        slope = 2.0 * self.a * dx + self.b_local
        vx_ref = self.vx_nom if self.vx_nom > 0.0 else max(v_x, 0.1)
        dy_ideal = slope * vx_ref
        return y_ideal, dy_ideal
