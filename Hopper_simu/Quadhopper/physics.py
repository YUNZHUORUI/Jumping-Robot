"""Low-level physics simulation for QuadHopper."""

import math
import numpy as np

from .config import PhysicsConfig, AttitudeConfig


class PhysicsEngine:
    """Encapsulates single-step dynamics for q=[x,y,theta,l], dq=[dx,dy,dtheta,dl]."""

    def __init__(self, physics_cfg: PhysicsConfig, attitude_cfg: AttitudeConfig):
        self.cfg = physics_cfg
        self.att = attitude_cfg

        # Runtime stance state (reset each episode)
        self.stance_active: bool = False
        self.stance_foot_anchor: np.ndarray = np.array(
            [0.0, physics_cfg.ground_y], dtype=np.float64
        )

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.stance_active = False
        self.stance_foot_anchor = np.array(
            [0.0, self.cfg.ground_y], dtype=np.float64
        )

    # -------------------------------------------------------- geometry helpers
    @staticmethod
    def wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_foot_pos(self, q: np.ndarray) -> np.ndarray:
        """
        Compute foot tip (x_f, y_f) from body state.
        During stance, returns the locked anchor point.
        """
        if self.stance_active:
            return self.stance_foot_anchor.copy()
        x, y, theta = q[0], q[1], q[2]
        x_f = x + self.cfg.leg_length * math.sin(theta)
        y_f = y - self.cfg.leg_length * math.cos(theta)
        return np.array([x_f, y_f])

    def stance_theta_from_anchor(self, q: np.ndarray):
        """
        Recompute theta and l_curr from COM position relative to foot anchor.
        Keeps body/leg geometry consistent during SLIP stance.
        """
        r_x = q[0] - self.stance_foot_anchor[0]
        r_y = q[1] - self.stance_foot_anchor[1]
        l_curr = max(math.sqrt(r_x * r_x + r_y * r_y), 1e-6)
        theta = math.atan2(-r_x, r_y)
        return theta, l_curr

    # ---------------------------------------------------------------- SLIP step
    def _compute_slip_forces(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        F_total: float,
        landing_theta_target: float,
        takeoff_theta_target: float,
        takeoff_theta_tol: float,
    ):
        """
        Compute contact forces using the SLIP model.
        Returns (F_act_x, F_act_y, F_spring_x, F_spring_y, touching, l_curr, dl).
        """
        x, y = q[0], q[1]
        theta = q[2]
        dx, dy = dq[0], dq[1]
        s, c = math.sin(theta), math.cos(theta)

        foot_y_virt = y - self.cfg.leg_length * c
        foot_x_virt = x + self.cfg.leg_length * s

        # Touchdown detection with landing-attitude clamp
        if (not self.stance_active) and (foot_y_virt <= self.cfg.ground_y):
            theta_td = min(theta, landing_theta_target)
            q[2] = theta_td
            dq[2] = 0.0
            theta = theta_td
            s, c = math.sin(theta), math.cos(theta)
            foot_x_virt = x + self.cfg.leg_length * s

            self.stance_active = True
            self.stance_foot_anchor = np.array(
                [foot_x_virt, self.cfg.ground_y], dtype=np.float64
            )

        F_act_x = -F_total * s
        F_act_y = F_total * c
        F_spring_x, F_spring_y = 0.0, 0.0
        touching = False
        l_curr = self.cfg.leg_length
        dl = 0.0

        if self.stance_active:
            touching = True
            r_x = x - self.stance_foot_anchor[0]
            r_y = y - self.stance_foot_anchor[1]
            l_curr = max(math.sqrt(r_x ** 2 + r_y ** 2), 1e-6)

            e_x, e_y = r_x / l_curr, r_y / l_curr
            dl = dx * e_x + dy * e_y

            # Thrust along leg axis (SLIP actuation)
            F_act_x = F_total * e_x
            F_act_y = F_total * e_y

            compression = float(np.clip(self.cfg.leg_length - l_curr, 0.0, self.cfg.stroke_length))
            if compression > 0.0:
                F_mag = self.cfg.k_slip * compression - self.cfg.c_slip * dl
                F_mag = max(0.0, F_mag)
                F_spring_x = F_mag * e_x
                F_spring_y = F_mag * e_y

            # Liftoff condition
            theta_leg = math.atan2(-r_x, r_y)
            theta_err_takeoff = abs(self.wrap_angle(theta_leg - takeoff_theta_target))
            reached_takeoff = theta_err_takeoff <= takeoff_theta_tol
            over_extended = l_curr >= self.cfg.leg_length + 0.05 * self.cfg.stroke_length
            natural_liftoff = (l_curr >= self.cfg.leg_length) and (dl > 0.0) and reached_takeoff
            if natural_liftoff or over_extended:
                self.stance_active = False

        return F_act_x, F_act_y, F_spring_x, F_spring_y, touching, l_curr, dl

    # --------------------------------------------------- legacy spring-damper
    def _compute_legacy_forces(
        self, q: np.ndarray, dq: np.ndarray, F_total: float
    ):
        """Legacy penetration-based spring-damper ground contact."""
        x, y, theta = q[0], q[1], q[2]
        dx, dy = dq[0], dq[1]
        s, c = math.sin(theta), math.cos(theta)

        foot_y_virt = y - self.cfg.leg_length * c
        F_act_x = -F_total * s
        F_act_y = F_total * c
        F_spring_x, F_spring_y = 0.0, 0.0
        touching = False
        l_curr = self.cfg.leg_length
        dl = 0.0

        if foot_y_virt < self.cfg.ground_y:
            touching = True
            l_curr = y / max(abs(c), 0.01)
            compression = float(np.clip(self.cfg.leg_length - l_curr, 0.0, self.cfg.stroke_length))
            if compression > 0:
                comp_rate = -(dx * s - dy * c)
                F_mag = (
                    self.cfg.k_spring * compression
                    + self.cfg.c_damping * comp_rate
                )
                F_mag = max(0.0, F_mag)
                F_spring_x = -F_mag * s
                F_spring_y = F_mag * c

        return F_act_x, F_act_y, F_spring_x, F_spring_y, touching, l_curr, dl

    # ---------------------------------------------------------------- main step
    def step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        action: np.ndarray,
        landing_theta_target: float,
        takeoff_theta_target: float,
        takeoff_theta_tol: float,
    ):
        """
        Advance physics by one timestep dt using Symplectic Euler integration.

        Args:
            q:  State position vector [x, y, theta, l]  (modified in-place)
            dq: State velocity vector [dx, dy, dtheta, dl] (modified in-place)
            action: [u1, u2] normalized thrust commands (0~1)
            takeoff_theta_target: Current target takeoff angle (rad)

        Returns:
            touching (bool): Whether foot is in contact this step.
        """
        u1 = float(np.clip(action[0], 0.0, 1.0))
        u2 = float(np.clip(action[1], 0.0, 1.0))

        F1 = u1 * self.cfg.max_thrust
        F2 = u2 * self.cfg.max_thrust
        F_total = F1 + F2
        tau = (F2 - F1) * self.cfg.cg_to_motor

        # Compute contact forces
        if self.cfg.use_slip_stance:
            F_act_x, F_act_y, F_sx, F_sy, touching, l_curr, dl = (
                self._compute_slip_forces(
                    q,
                    dq,
                    F_total,
                    landing_theta_target,
                    takeoff_theta_target,
                    takeoff_theta_tol,
                )
            )
        else:
            F_act_x, F_act_y, F_sx, F_sy, touching, l_curr, dl = (
                self._compute_legacy_forces(q, dq, F_total)
            )

        # Accelerations
        ddx = (F_act_x + F_sx) / self.cfg.mass
        ddy = (F_act_y + F_sy) / self.cfg.mass - self.cfg.gravity
        ddtheta = 0.0 if self.stance_active else tau / self.cfg.inertia

        # Symplectic Euler: update velocities first, then positions
        dq[0] += ddx     * self.cfg.dt
        dq[1] += ddy     * self.cfg.dt
        dq[2] += ddtheta * self.cfg.dt
        q[0]  += dq[0]   * self.cfg.dt
        q[1]  += dq[1]   * self.cfg.dt
        q[2]  += dq[2]   * self.cfg.dt

        # Stance post-correction: lock geometry to anchor
        if self.cfg.use_slip_stance and self.stance_active:
            self._apply_stance_correction(q, dq)

        # Update leg state in q/dq
        if self.stance_active:
            q[3] = l_curr
            dq[3] = dl
        else:
            q[3] = self.cfg.leg_length
            dq[3] = 0.0

        return touching

    def _apply_stance_correction(
        self, q: np.ndarray, dq: np.ndarray
    ):
        """
        Enforce geometric consistency of body orientation during SLIP stance.
        Prevents visual/numeric body deformation.
        """
        theta_prev = q[2]
        theta_stance, l_curr = self.stance_theta_from_anchor(q)
        q[2]  = theta_stance
        q[3]  = l_curr
        dq[2] = self.wrap_angle(theta_stance - theta_prev) / self.cfg.dt

        r_x = q[0] - self.stance_foot_anchor[0]
        r_y = q[1] - self.stance_foot_anchor[1]
        l_curr = max(math.sqrt(r_x ** 2 + r_y ** 2), 1e-6)
        e_x, e_y = r_x / l_curr, r_y / l_curr
        dq[3] = dq[0] * e_x + dq[1] * e_y
