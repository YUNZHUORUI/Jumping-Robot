import sys
import math

with open('/Users/yunzhuorui/Jumping-Robot/Hopper_simu/Quadhopper/physics.py', 'r') as f:
    code = f.read()

from_str = """    def _compute_flight_eom_accel(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        F1: float,
        F2: float,
        tau_att: float,
    ) -> np.ndarray:
        \"\"\"Compute qdd from flight EOM: M(q) qdd + B(q,dq) + G(q) = F(q).\"\"\"
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

        M = np.array([
            [m1 + m2, 0.0, l * m1 * cth, m1 * sth],
            [0.0, m1 + m2, -l * m1 * sth, m1 * cth],
            [l * m1 * cth, -l * m1 * sth, self.cfg.inertia + (l ** 2) * m1, 0.0],
            [m1 * sth, m1 * cth, 0.0, m1],
        ], dtype=np.float64)

        B = np.array([
            dtheta * m1 * (2.0 * dl * cth - dtheta * l * sth),
            -dtheta * m1 * (2.0 * dl * sth + dtheta * l * cth),
            -2.0 * m1 * (dl * dy * sth - dl * dx * cth - dl * dtheta * l + dtheta * dy * l * cth + dtheta * dx * l * sth),
            b * dl + (dtheta ** 2) * l * m1 + 2.0 * dtheta * dx * m1 * cth - 2.0 * dtheta * dy * m1 * sth,
        ], dtype=np.float64)

        G = np.array([
            0.0,
            (m1 + m2) * g,
            -g * l * m1 * sth,
            k * (l - self.cfg.leg_length) + m1 * g * cth,
        ], dtype=np.float64)

        F = np.array([
            (F1 + F2) * sth,
            (F1 + F2) * cth,
            (F1 - F2) * self.cfg.cg_to_motor + tau_att,
            0.0,
        ], dtype=np.float64)"""

to_str = """    def _compute_flight_eom_accel(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        F1: float,
        F2: float,
        tau_att: float,
    ) -> np.ndarray:
        \"\"\"Compute qdd from flight EOM: M(q) qdd + B(q,dq) + G(q) = F(q).\"\"\"
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
        ], dtype=np.float64)"""

code = code.replace(from_str, to_str)

with open('/Users/yunzhuorui/Jumping-Robot/Hopper_simu/Quadhopper/physics.py', 'w') as f:
    f.write(code)

