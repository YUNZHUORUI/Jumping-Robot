import sys

with open('/Users/yunzhuorui/Jumping-Robot/Hopper_simu/Quadhopper/physics.py', 'r') as f:
    code = f.read()

from_str = """        # Runtime stance state (reset each episode)
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
        \"\"\"Wrap angle to [-pi, pi].\"\"\"
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_foot_pos(self, q: np.ndarray) -> np.ndarray:
        \"\"\"
        Compute foot tip (x_f, y_f) from body state.
        During stance, returns the locked anchor point.
        \"\"\"
        if self.stance_active:
            return self.stance_foot_anchor.copy()
        x, y, theta = q[0], q[1], q[2]
        x_f = x + self.cfg.leg_length * math.sin(theta)
        y_f = y - self.cfg.leg_length * math.cos(theta)
        return np.array([x_f, y_f])"""

to_str = """        # Runtime stance state (reset each episode)
        self.stance_active: bool = False

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.stance_active = False

    # -------------------------------------------------------- geometry helpers
    @staticmethod
    def wrap_angle(angle: float) -> float:
        \"\"\"Wrap angle to [-pi, pi].\"\"\"
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_foot_pos(self, q: np.ndarray) -> np.ndarray:
        \"\"\"
        Compute foot tip (x_f, y_f) from state.
        The state q is already geometrically formulated with (x,y) as foot.
        \"\"\"
        return np.array([q[0], q[1]])

    def get_com_pos(self, q: np.ndarray) -> np.ndarray:
        \"\"\"
        Compute COM position from base foot state.
        \"\"\"
        x_f, y_f, theta, l = q[0], q[1], q[2], q[3]
        return np.array([x_f + l * math.sin(theta), y_f + l * math.cos(theta)])"""

code = code.replace(from_str, to_str)

from_str2 = """    # ---------------------------------------------------------------- SLIP step
    def _compute_slip_forces(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        F_total: float,
        landing_theta_target: float,
        takeoff_theta_target: float,
        takeoff_theta_tol: float,
    ):
        \"\"\"
        Compute contact forces using the SLIP model.
        Returns (F_act_x, F_act_y, F_spring_x, F_spring_y, touching, l_curr, dl).
        \"\"\"
        x, y = q[0], q[1]
        theta = q[2]
        dx, dy = dq[0], dq[1]
        s, c = math.sin(theta), math.cos(theta)

        foot_y_virt = y - self.cfg.leg_length * c
        foot_x_virt = x + self.cfg.leg_length * s

        touchdown_event = False

        # Touchdown detection with landing-attitude clamp
        if (not self.stance_active) and (foot_y_virt <= self.cfg.ground_y):
            touchdown_event = True
            theta_td = min(theta, landing_theta_target)
            q[2] = theta_td
            dq[2] = 0.0
            theta = theta_td
            s, c = math.sin(theta), math.cos(theta)
            foot_x_virt = x + self.cfg.leg_length * s
            
            # Enforce no-slip constraint at touchdown to prevent ground sliding
            self._project_no_slip_velocity(q, dq)

            # Touchdown constraint:
            # 1) bottom contact point is exactly on ground (y = ground_y)
            # 2) optional translational velocity reset (configurable)
            q[0] = foot_x_virt - self.cfg.leg_length * s
            q[1] = self.cfg.ground_y + self.cfg.leg_length * c
            if self.cfg.touchdown_zero_xy_velocity:
                dq[0] = 0.0
                dq[1] = 0.0

            # Refresh local state for force computation after projection
            x, y = q[0], q[1]
            dx, dy = dq[0], dq[1]
            foot_x_virt = x + self.cfg.leg_length * s
            foot_y_virt = y - self.cfg.leg_length * c

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
        
        # Hard no-slip constraint: during stance, foot must not slide
        # This is enforced by the stance anchor and contact dynamics
        # but we reinforce it here to prevent numerical drift.
        # (The velocity projection happens in _apply_stance_correction after integration.)

        return F_act_x, F_act_y, F_spring_x, F_spring_y, touching, l_curr, dl, touchdown_event

    def _compute_flight_eom_accel"""

to_str2 = """    # ---------------------------------------------------------------- SLIP step
    def _compute_slip_forces(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        F_total: float,
        landing_theta_target: float,
        takeoff_theta_target: float,
        takeoff_theta_tol: float,
    ):
        \"\"\"
        Compute contact states and SLIP forces.
        Returns (F_leg, touching, touchdown_event).
        \"\"\"
        y_f = q[1]
        theta = q[2]
        l_curr = q[3]
        dl = dq[3]

        touchdown_event = False

        # Touchdown detection
        if (not self.stance_active) and (y_f <= self.cfg.ground_y):
            touchdown_event = True
            theta_td = min(theta, landing_theta_target)
            q[2] = theta_td
            dq[2] = 0.0
            
            # Enforce absolutely zero foot velocity at touchdown
            dq[0] = 0.0
            dq[1] = 0.0

            # Touchdown constraint:
            q[1] = self.cfg.ground_y
            self.stance_active = True
            theta = theta_td

        F_leg = F_total
        touching = False

        if self.stance_active:
            touching = True
            compression = float(np.clip(self.cfg.leg_length - l_curr, 0.0, self.cfg.stroke_length))
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

    def _compute_flight_eom_accel"""

code = code.replace(from_str2, to_str2)

with open('/Users/yunzhuorui/Jumping-Robot/Hopper_simu/Quadhopper/physics.py', 'w') as f:
    f.write(code)

