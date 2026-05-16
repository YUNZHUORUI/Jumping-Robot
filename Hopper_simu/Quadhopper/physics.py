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

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.stance_active = False

    # -------------------------------------------------------- geometry helpers
    @staticmethod
    def wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_foot_pos(self, q: np.ndarray) -> np.ndarray:
        """
        Compute foot tip (x_f, y_f) from state.
        The state q is already geometrically formulated with (x,y) as foot.
        """
        return np.array([q[0], q[1]])

    def get_com_pos(self, q: np.ndarray) -> np.ndarray:
        """
        Compute COM position from base foot state.
        """
        x_f, y_f, theta, l = q[0], q[1], q[2], q[3]
        return np.array([x_f + l * math.sin(theta), y_f + l * math.cos(theta)])

    def get_com_vel(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Compute COM velocity from base foot point velocity.
        """
        dx_f, dy_f, dtheta, dl = dq[0], dq[1], dq[2], dq[3]
        theta, l = q[2], q[3]
        
        dx_com = dx_f + dl * math.sin(theta) + l * math.cos(theta) * dtheta
        dy_com = dy_f + dl * math.cos(theta) - l * math.sin(theta) * dtheta
        return np.array([dx_com, dy_com])

    def _project_no_slip_velocity(self, q: np.ndarray, dq: np.ndarray):
        """Project COM velocity so foot contact velocity is zero under stance."""
        # For foot coordinates, no-slip means foot velocity is just zero
        dq[0] = 0.0
        dq[1] = 0.0

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
        Compute stance-mode scalar leg force and contact events.
        State convention here is foot coordinates: q=[x_f, y_f, theta, l].
        Returns (F_leg, touching, touchdown_event).
        """
        y_f = float(q[1])
        theta = float(q[2])
        l_curr = float(q[3])
        dl = float(dq[3])

        touchdown_event = False

        # Touchdown in foot coordinates: directly check foot height.
        if (not self.stance_active) and (y_f <= self.cfg.ground_y):
            touchdown_event = True
            theta_td = min(theta, landing_theta_target)
            q[2] = theta_td
            dq[2] = 0.0

            # Pin foot exactly on ground and remove slip velocity.
            q[1] = self.cfg.ground_y
            self._project_no_slip_velocity(q, dq)
            self.stance_active = True
            theta = theta_td

            if self.cfg.touchdown_zero_xy_velocity:
                dq[0] = 0.0
                dq[1] = 0.0

        F_leg = float(F_total)
        touching = False

        if self.stance_active:
            touching = True

            compression = float(
                np.clip(self.cfg.leg_length - l_curr, 0.0, self.cfg.stroke_length)
            )
            if compression > 0.0:
                F_mag = self.cfg.k_slip * compression - self.cfg.c_slip * dl
                F_mag = max(0.0, F_mag)
                F_leg += F_mag

            # Liftoff condition
            theta_err_takeoff = abs(self.wrap_angle(theta - takeoff_theta_target))
            reached_takeoff = theta_err_takeoff <= takeoff_theta_tol
            over_extended = l_curr >= self.cfg.leg_length + 0.05 * self.cfg.stroke_length
            natural_liftoff = (l_curr >= self.cfg.leg_length) and (dl > 0.0) and reached_takeoff
            if natural_liftoff or over_extended:
                self.stance_active = False

        return F_leg, touching, touchdown_event

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

    def _compute_flight_eom_accel(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        F1: float,
        F2: float,
        tau_att: float,
    ) -> np.ndarray:
        """Compute qdd from flight EOM: M(q) qdd + B(q,dq) + G(q) = F(q)."""
        x, y, theta, l = [float(v) for v in q]
        dx, dy, dtheta, dl = [float(v) for v in dq]

        m_total = float(self.cfg.mass)
        m1 = float(np.clip(self.cfg.eom_leg_mass_ratio, 0.05, 0.95)) * m_total
        m2 = m_total - m1
        cth = math.cos(theta)
        sth = math.sin(theta)
        g = self.cfg.gravity
        k = self.cfg.eom_leg_k
        b = self.cfg.eom_leg_damping

        # M, B, G, F constructed according to `jumping_robot_dynamics_v3.py` matrices where (x,y) is foot
        M = np.array([
            [m1 + m2, 0.0, l * m1 * cth, m1 * sth],
            [0.0, m1 + m2, -l * m1 * sth, m1 * cth],
            [l * m1 * cth, -l * m1 * sth, self.cfg.inertia + (l ** 2) * m1, 0.0],
            [m1 * sth, m1 * cth, 0.0, m1],
        ], dtype=np.float64)

        B = np.array([
            - m1 * (dtheta ** 2) * l * sth,
            - m1 * (dtheta ** 2) * l * cth,
            - 2.0 * m1 * l * dtheta * (dy * cth + dx * sth),
            (theta ** 2) * l * m1 + 2 * dtheta * dx * m1 * cth - 2 * dtheta * dy * m1 * sth,
        ], dtype=np.float64)

        G = np.array([
            0.0,
            (m1 + m2) * g,
            - g * l * m1 * sth,
            m1 * g * cth,
        ], dtype=np.float64)

        F = np.array([
            (F1 + F2) * sth,
            (F1 + F2) * cth,
            (F1 - F2) * self.cfg.cg_to_motor + tau_att,
            0.0,
        ], dtype=np.float64)

        rhs = F - B - G
        try:
            qdd = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            qdd = np.linalg.pinv(M) @ rhs

        return np.clip(qdd, -self.cfg.eom_ddq_clip, self.cfg.eom_ddq_clip)

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
            F_leg, touching, touchdown_event = self._compute_slip_forces(
                q,
                dq,
                F_total,
                landing_theta_target,
                takeoff_theta_target,
                takeoff_theta_tol,
            )
        else:
            touchdown_event = False
            # F_act_x, F_act_y, F_sx, F_sy, touching, l_curr, dl = (
            #     self._compute_legacy_forces(q, dq, F_total)
            # )
            # Legacy forces removed for clarity since q is now Foot coordinate.
            F_leg = F_total
            touching = False

        # Flight-phase attitude assist (PD torque) to suppress aerial somersaults
        tau_att = 0.0
        if (not self.stance_active) and self.cfg.flight_attitude_assist:
            theta_err = self.wrap_angle(q[2] - landing_theta_target)
            tau_att = -self.cfg.flight_att_kp * theta_err - self.cfg.flight_att_kd * dq[2]
            tau_att = float(np.clip(tau_att, -self.cfg.flight_att_tau_limit, self.cfg.flight_att_tau_limit))

        # Accelerations
        if (not self.stance_active) and self.cfg.use_flight_eom:
            qdd = self._compute_flight_eom_accel(q, dq, F1, F2, tau_att)
            dq[0] += qdd[0] * self.cfg.dt
            dq[1] += qdd[1] * self.cfg.dt
            dq[2] += qdd[2] * self.cfg.dt
            dq[3] += qdd[3] * self.cfg.dt
            q[0] += dq[0] * self.cfg.dt
            q[1] += dq[1] * self.cfg.dt
            q[2] += dq[2] * self.cfg.dt
            q[3] += dq[3] * self.cfg.dt
            q[3] = float(np.clip(q[3], self.cfg.leg_min_length, self.cfg.leg_length + self.cfg.stroke_length))
        else:
            # Stance integration using polar generalized coordinates relative to foot
            m = self.cfg.mass
            I = self.cfg.inertia
            g = self.cfg.gravity
            theta = q[2]
            dtheta = dq[2]
            l = q[3]
            dl = dq[3]
            s, c = math.sin(theta), math.cos(theta)
            
            ddl = F_leg / m - g * c + l * (dtheta ** 2)
            ddtheta = (tau + tau_att + m * g * l * s - 2 * m * l * dl * dtheta) / (I + m * (l ** 2))
            
            # Foot is mathematically anchored exactly at the contact point.
            dq[0] = 0.0
            dq[1] = 0.0
            dq[2] += ddtheta * self.cfg.dt
            dq[3] += ddl * self.cfg.dt
            q[2] += dq[2] * self.cfg.dt
            q[3] += dq[3] * self.cfg.dt

            # Keep stance foot exactly pinned to ground in foot-coordinate model.
            q[1] = self.cfg.ground_y

            # Prevent unbounded visual/body stretching during stance.
            l_min = self.cfg.leg_min_length
            l_max = self.cfg.leg_length + self.cfg.stroke_length
            q[3] = float(np.clip(q[3], l_min, l_max))
            if (q[3] <= l_min and dq[3] < 0.0) or (q[3] >= l_max and dq[3] > 0.0):
                dq[3] = 0.0

            # Optional touchdown translational velocity reset (emulated on rotational state).
            if touchdown_event and self.cfg.touchdown_zero_xy_velocity:
                dq[2] = 0.0
                dq[3] = 0.0

        # Leg state is managed explicitly above.
        if not self.stance_active and not self.cfg.use_flight_eom:
            q[3] = self.cfg.leg_length
            dq[3] = 0.0

        return touching
