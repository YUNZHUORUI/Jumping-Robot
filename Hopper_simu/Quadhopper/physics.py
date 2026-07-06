"""Low-level physics simulation for QuadHopper.

Key improvements over previous version
---------------------------------------
1. **5× sub-stepping** (dt_inner = dt/5 = 2 ms) for stable spring-contact dynamics.
   The SLIP spring at k=400 N/m, m=0.18 kg gives ω_n≈47 rad/s.  The Coriolis
   cross-term during early stance (−2 m l dl̇ θ̇) can drive ddθ above 50 rad/s²
   when dl is large from the touchdown impulse, so a 2 ms inner step keeps that
   term well-controlled.

2. **Impact angular restitution** (config.impact_angular_restitution, default 0.72).
   Physical touchdown is never perfectly rigid – partial slip and deformation absorb
   energy.  Without this the instantaneous angular impulse at landing is ~4 rad/s
   (dx_foot≈1.5 m/s × l/J_eff), causing the body to over-rotate during flight.
   The coefficient reduces converted horizontal momentum to a realistic 2-3 rad/s.

3. **Removed artificial angle snap at touchdown.**
   The old line `theta_td = min(theta, landing_theta_target)` silently clamped any
   forward-leaning robot to −3° at every landing, producing discontinuous histories.
   Removed; the robot's actual theta at touchdown is now preserved.

4. **Minimum stance sub-step guard** (config.min_stance_substeps).
   Prevents the liftoff condition from firing before the spring has completed a
   meaningful compression-extension cycle.
"""

import math
import numpy as np

from .config import PhysicsConfig

# Number of inner integration steps per policy step.
_N_SUBSTEPS = 5


class PhysicsEngine:
    """Encapsulates single-step dynamics for q=[x,y,theta,l], dq=[dx,dy,dtheta,dl]."""

    def __init__(self, physics_cfg: PhysicsConfig):
        self.cfg = physics_cfg

        self.stance_active: bool = False
        self.filtered_u: np.ndarray = np.zeros(2)

        # Sub-step count since stance began (for minimum-contact guard).
        self._stance_substep_count: int = 0
        # Minimum leg length seen this stance (for diagnostics / extension check).
        self._stance_min_l: float = physics_cfg.leg_length

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.stance_active = False
        self.filtered_u[:] = 0.0
        self._stance_substep_count = 0
        self._stance_min_l = self.cfg.leg_length

    def start_stance(self, q: np.ndarray):
        """Initialize a reset state with the foot already pinned on the ground."""
        self.stance_active = True
        self._stance_substep_count = 0
        self._stance_min_l = float(q[3])
        q[1] = self.cfg.ground_y

    # -------------------------------------------------------- geometry helpers
    @staticmethod
    def wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_foot_pos(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[0], q[1]])

    def get_com_pos(self, q: np.ndarray) -> np.ndarray:
        x_f, y_f, theta, l = q[0], q[1], q[2], q[3]
        return np.array([x_f + l * math.sin(theta), y_f + l * math.cos(theta)])

    def get_com_vel(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        dx_f, dy_f, dtheta, dl = dq[0], dq[1], dq[2], dq[3]
        theta, l = q[2], q[3]
        dx_com = dx_f + dl * math.sin(theta) + l * math.cos(theta) * dtheta
        dy_com = dy_f + dl * math.cos(theta) - l * math.sin(theta) * dtheta
        return np.array([dx_com, dy_com])

    # -------------------------------------------------------- SLIP contact
    def _handle_touchdown(self, q: np.ndarray, dq: np.ndarray):
        """
        Apply rigid-body contact impulse when foot first hits the ground.

        Angular momentum conservation about the newly-fixed foot:
            J_eff · Δdθ = m · l · (cos θ · dx_f − sin θ · dy_f)

        Impact restitution scales down the impulse to model energy loss
        (partial slip, ground compliance, vibration at impact).
        """
        s_pre  = math.sin(float(q[2]))
        c_pre  = math.cos(float(q[2]))
        l_pre  = float(q[3])
        dx_f   = float(dq[0])
        dy_f   = float(dq[1])
        dl_pre = float(dq[3])

        J_eff = self.cfg.inertia + self.cfg.mass * l_pre ** 2

        # dl impulse: project foot velocity onto leg axis
        dq[3] = dl_pre + s_pre * dx_f + c_pre * dy_f

        # Angular impulse with restitution
        restitution = float(getattr(self.cfg, 'impact_angular_restitution', 1.0))
        raw_delta   = l_pre * self.cfg.mass * (c_pre * dx_f - s_pre * dy_f) / J_eff
        dq[2]       = float(dq[2]) + restitution * raw_delta

        # Pin foot to ground; preserve actual theta (no artificial angle snap).
        q[1]  = self.cfg.ground_y
        dq[0] = 0.0
        dq[1] = 0.0

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
        Compute stance spring force and handle touchdown / liftoff events.
        Returns (F_leg, touching, touchdown_event).
        """
        y_f    = float(q[1])
        l_curr = float(q[3])
        dl     = float(dq[3])

        touchdown_event = False

        # Touchdown detection
        if (not self.stance_active) and (y_f <= self.cfg.ground_y):
            touchdown_event = True
            self._handle_touchdown(q, dq)
            self.stance_active = True
            self._stance_substep_count = 0
            self._stance_min_l = float(q[3])

        F_leg    = float(F_total)
        touching = False

        if self.stance_active:
            touching = True
            self._stance_substep_count += 1

            if l_curr < self._stance_min_l:
                self._stance_min_l = l_curr

            compression = float(
                np.clip(self.cfg.leg_length - l_curr, 0.0, self.cfg.stroke_length)
            )
            preload  = float(getattr(self.cfg, 'spring_preload', 0.0))
            F_spring = self.cfg.k_slip * compression - self.cfg.c_slip * dl + preload
            F_spring = max(0.0, F_spring)
            F_leg   += F_spring

            # Liftoff: leg returned to/past natural length AND is extending AND
            # minimum sub-step count has elapsed (avoids premature liftoff).
            min_ss = int(getattr(self.cfg, 'min_stance_substeps', 0))
            if (l_curr >= self.cfg.leg_length
                    and dl > 0.0
                    and self._stance_substep_count >= min_ss):
                self.stance_active = False

        return F_leg, touching, touchdown_event

    # ----------------------------------------------------------- inner sub-step
    def _substep(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        F1: float,
        F2: float,
        landing_theta_target: float,
        takeoff_theta_target: float,
        takeoff_theta_tol: float,
        dt: float,
    ) -> bool:
        """Integrate one inner sub-step of duration *dt*. Returns touching."""
        F_total = F1 + F2
        tau     = (F2 - F1) * self.cfg.cg_to_motor

        if self.cfg.use_slip_stance:
            F_leg, touching, _ = self._compute_slip_forces(
                q, dq, F_total,
                landing_theta_target, takeoff_theta_target, takeoff_theta_tol,
            )
        else:
            F_leg    = F_total
            touching = False

        # Flight attitude assist — gentle only, PPO policy provides main control
        tau_att = 0.0
        if (not self.stance_active) and self.cfg.flight_attitude_assist:
            theta_err = self.wrap_angle(float(q[2]) - landing_theta_target)
            tau_att   = (-self.cfg.flight_att_kp * theta_err
                         - self.cfg.flight_att_kd * float(dq[2]))
            tau_att   = float(np.clip(
                tau_att,
                -self.cfg.flight_att_tau_limit,
                self.cfg.flight_att_tau_limit,
            ))

        # ── Flight phase (rigid-body COM ballistic + attitude) ──────────────
        if (not self.stance_active) and self.cfg.use_flight_eom:
            theta = float(q[2])
            l_nom = self.cfg.leg_length
            s, c  = math.sin(theta), math.cos(theta)

            # Reconstruct COM velocity from foot-based state
            _dl     = float(dq[3])
            _dtheta = float(dq[2])
            vx_com  = float(dq[0]) + _dl * s + l_nom * c * _dtheta
            vy_com  = float(dq[1]) + _dl * c - l_nom * s * _dtheta
            com_x   = float(q[0]) + l_nom * s
            com_y   = float(q[1]) + l_nom * c

            # Lock leg at natural length
            q[3]  = l_nom
            dq[3] = 0.0

            # Thrust along body axis [sin θ, cos θ] + gravity
            m      = self.cfg.mass
            ax_com = (F1 + F2) * s / m
            ay_com = (F1 + F2) * c / m - self.cfg.gravity

            # Attitude dynamics
            tau_total = (F2 - F1) * self.cfg.cg_to_motor + tau_att
            ddtheta   = tau_total / self.cfg.inertia

            # Symplectic Euler integration
            vx_com += ax_com * dt
            vy_com += ay_com * dt
            dq[2]  += ddtheta * dt
            q[2]   += float(dq[2]) * dt
            com_x  += vx_com * dt
            com_y  += vy_com * dt

            # Back-compute foot position from updated COM + theta
            s_new = math.sin(float(q[2]))
            c_new = math.cos(float(q[2]))
            q[0]  = com_x - l_nom * s_new
            q[1]  = com_y - l_nom * c_new

            # Store foot velocity for touchdown impulse computation
            dq[0] = vx_com - l_nom * c_new * float(dq[2])
            dq[1] = vy_com + l_nom * s_new * float(dq[2])

        else:
            # ── Stance phase: inverted-pendulum polar EOM ───────────────
            m      = self.cfg.mass
            I      = self.cfg.inertia
            g      = self.cfg.gravity
            theta  = float(q[2])
            dtheta = float(dq[2])
            l      = float(q[3])
            dl     = float(dq[3])
            s, c   = math.sin(theta), math.cos(theta)

            # Radial: m·l̈ = F_leg − m·g·cos θ + m·l·θ̇²
            ddl = F_leg / m - g * c + l * (dtheta ** 2)

            # Pitch: (I + m·l²)·θ̈ = τ + m·g·l·sin θ − 2·m·l·l̇·θ̇
            ddtheta = (tau + tau_att + m * g * l * s
                       - 2.0 * m * l * dl * dtheta) / (I + m * l ** 2)

            # Foot pinned — only l and theta evolve
            dq[0] = 0.0
            dq[1] = 0.0
            dq[2] += ddtheta * dt
            dq[3] += ddl * dt
            q[2]  += float(dq[2]) * dt
            q[3]  += float(dq[3]) * dt

            # Enforce ground and stroke limits
            q[1]  = self.cfg.ground_y
            l_min = self.cfg.leg_min_length
            l_max = self.cfg.leg_length + self.cfg.stroke_length
            q[3]  = float(np.clip(q[3], l_min, l_max))
            if (q[3] <= l_min and dq[3] < 0.0) or (q[3] >= l_max and dq[3] > 0.0):
                dq[3] = 0.0

        # Reset leg DOF in flight
        if not self.stance_active:
            q[3]  = self.cfg.leg_length
            dq[3] = 0.0

        return touching

    # ---------------------------------------------------------------- main step
    def step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        action: np.ndarray,
        landing_theta_target: float,
        takeoff_theta_target: float,
        takeoff_theta_tol: float,
    ) -> bool:
        """
        Advance physics by one policy timestep (cfg.dt) using 5 inner sub-steps.

        Motor lag is applied once per policy step so the lag time constant stays
        consistent with the 100 Hz policy rate.

        Returns touching (bool): True if foot is in contact at end of step.
        """
        # Motor lag filter — once per policy step
        alpha = self.cfg.dt / (self.cfg.motor_tau + self.cfg.dt)
        self.filtered_u[0] = (alpha * float(np.clip(action[0], 0.0, 1.0))
                               + (1.0 - alpha) * self.filtered_u[0])
        self.filtered_u[1] = (alpha * float(np.clip(action[1], 0.0, 1.0))
                               + (1.0 - alpha) * self.filtered_u[1])

        F1 = self.cfg.thrust_from_cmd(float(self.filtered_u[0]))
        F2 = self.cfg.thrust_from_cmd(float(self.filtered_u[1]))

        dt_inner = self.cfg.dt / _N_SUBSTEPS
        touching = False
        for _ in range(_N_SUBSTEPS):
            touching = self._substep(
                q, dq, F1, F2,
                landing_theta_target, takeoff_theta_target, takeoff_theta_tol,
                dt_inner,
            )

        return touching
