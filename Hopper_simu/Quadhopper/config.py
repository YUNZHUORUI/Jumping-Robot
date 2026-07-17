"""Central configuration for the modular QuadHopper package."""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


def make_even_targets(count: int = 6, spacing: float = 0.5) -> np.ndarray:
    """Return target foot positions at spacing, 2*spacing, ..."""
    return spacing * np.arange(1, count + 1, dtype=np.float64)


@dataclass
class PhysicsConfig:
    """Physical simulation parameters (real hardware, 2D pitch-plane mapping)."""
    dt: float = 0.01              # s  100 Hz policy
    mass: float = 0.180           # kg  ← Isaac: TOTAL_MASS=0.180 (body only; leg=0.010 ignored in EOM)
    rotor_span: float = 0.1626    # m   ← Isaac: 2 * MOTOR_OFFSET = 2 * 0.0813
    leg_length: float = 0.30      # m   nominal COM-to-foot (estimated from CAD geometry)
    stroke_length: float = 0.08   # m   ← Isaac: LEG_TRAVEL = 0.08
    inertia: float = 7.667e-4     # kg·m²  ← Isaac: I_YY = 7.666776e-4 (pitch axis)
    gravity: float = 9.81
    ground_y: float = 0.0

    # ── Thrust curve  (per motor, piecewise quadratic → linear) ──────────
    # F(u) = thrust_a·u² + thrust_b·u + thrust_c   (u ≤ thrust_breakpoint)
    # F(u) = thrust_offset_high + (u − breakpoint)·thrust_slope_high  (u > breakpoint)
    # Clamped to [0, thrust_max_per_motor] N per motor.
    # 2D mapping: left side = F1+F2, right side = F3+F4  → n_motors_per_side = 2
    thrust_a: float = -0.9715
    thrust_b: float = 1.2578
    thrust_c: float = -0.0577
    thrust_breakpoint: float = 0.64
    thrust_offset_high: float = 0.349
    thrust_slope_high: float = 0.6139
    thrust_max_per_motor: float = 0.6    # N  (single motor ceiling)
    n_motors_per_side: int = 2           # motors combined per action channel

    # ── Motor first-order lag  u_t = α·u_cmd + (1−α)·u_{t-1},  α=dt/(τ+dt) ──
    motor_tau: float = 0.065             # s  nominal
    motor_tau_min: float = 0.055         # s  DR range
    motor_tau_max: float = 0.075         # s  DR range

    # Legacy spring-damper model (unused when use_slip_stance=True)
    k_spring: float = 500.0
    c_damping: float = 20.0
    # SLIP stance model.
    use_slip_stance: bool = True
    k_slip: float = 400.0          # N/m  ← Isaac: LEG_STIFFNESS = 400
    c_slip: float = 4.0            # N·s/m  ↑ from 1.5: ζ≈0.25 for realistic landing absorption
    spring_preload: float = 1.8    # N    ↓ from 8.0: ≈robot weight so leg is neutral at rest.
    #   Old value (8 N = 4.5× weight) put equilibrium ABOVE natural length, causing
    #   ddl≈+42 m/s² even at l=l0, which triggered liftoff in the first stance step.
    #   New value ≈ weight keeps the leg in a physically neutral hover at natural length.

    # Impact angular restitution: fraction of horizontal foot velocity converted to
    # angular momentum at touchdown. < 1.0 models partial slip + energy absorption.
    # Without this the instantaneous angular impulse is 4+ rad/s for typical hops.
    impact_angular_restitution: float = 0.72

    # Minimum number of stance sub-steps before liftoff is checked.
    # Prevents premature liftoff when the spring hasn't yet completed its cycle.
    min_stance_substeps: int = 8    # 8 × (dt/5) = 8 × 2 ms = 16 ms minimum contact

    # Flight attitude assist
    # τ_limit aligned with physical max differential torque: ΔF_max·L ≈ 0.096 N·m
    flight_attitude_assist: bool = True
    flight_att_kp: float = 0.04           # gentle position restore
    flight_att_kd: float = 0.018          # ↓ from 0.06: old value caused 46 rad/s² spin-back
    #   at dtheta=5 rad/s → robot rotated to −8 rad/s by landing.  New value keeps
    #   assist gentle so the PPO policy controls rotation without fighting the assist.
    flight_att_tau_limit: float = 0.020  # N·m  ↓ from 0.035 — softer assist ceiling

    # Flight EOM
    use_flight_eom: bool = True
    eom_leg_mass_ratio: float = 0.28
    eom_leg_damping: float = 0.9
    eom_leg_k: float = 260.0
    eom_ddq_clip: float = 180.0

    touchdown_zero_xy_velocity: bool = False

    @property
    def beam_half_length(self) -> float:
        return 0.5 * self.rotor_span

    @property
    def cg_to_motor(self) -> float:
        return self.beam_half_length

    @property
    def max_thrust(self) -> float:
        """Per-side thrust ceiling (N): n_motors × per-motor max."""
        return self.n_motors_per_side * self.thrust_max_per_motor

    @property
    def leg_min_length(self) -> float:
        return max(0.05, self.leg_length - self.stroke_length)

    @property
    def min_com_height(self) -> float:
        return max(0.04, 0.35 * self.leg_min_length)

    def thrust_from_cmd(self, u: float) -> float:
        """Per-side thrust (N) from normalized command u ∈ [0, 1]."""
        u = float(np.clip(u, 0.0, 1.0))
        if u <= self.thrust_breakpoint:
            f = self.thrust_a * u * u + self.thrust_b * u + self.thrust_c
        else:
            f = self.thrust_offset_high + (u - self.thrust_breakpoint) * self.thrust_slope_high
        return float(np.clip(f, 0.0, self.thrust_max_per_motor)) * self.n_motors_per_side


@dataclass
class AttitudeConfig:
    """Attitude phase tolerances."""
    takeoff_theta_tol_deg: float = 12.0

    @property
    def takeoff_theta_tol(self) -> float:
        return math.radians(self.takeoff_theta_tol_deg)


@dataclass
class EnvConfig:
    """Environment-level settings."""
    target_count: int = 6
    target_spacing: float = 0.5
    targets: np.ndarray = field(
        default_factory=lambda: make_even_targets(count=6, spacing=0.5)
    )
    target_tolerance: float = 0.15
    max_episode_steps: int = 1500     # 1500 × 0.01 s = 15 s
    obs_clip: float = 20.0
    print_hit_events: bool = True

    # Reset modes:
    # - "ground": start in stance from rest, for real target-jump feasibility tests.
    # - "ballistic": legacy curriculum start with planned initial velocity.
    reset_mode: str = "ground"
    ground_init_leg_compression: float = 0.065
    ground_init_theta_min_deg: float = -6.0
    ground_init_theta_max_deg: float = 6.0

    # Init randomization range (degrees).
    # -7° is the critical angle; -4° gives cold-filter liftoff by step 73 < 120-step limit.
    init_theta_min_deg: float = -4.0
    init_theta_max_deg: float = -1.0

    # Initial velocity randomization
    init_vx_range: Tuple[float, float] = (3.2, 4.4)
    init_vy_range: Tuple[float, float] = (3.4, 4.8)
    init_vx_std: float = 0.06
    init_vy_std: float = 0.10
    init_dtheta_min_deg: float = -12.0
    init_dtheta_max_deg: float = 12.0

    # Trajectory planning tilt range (degrees)
    traj_tilt_min_deg: float = 20.0
    traj_tilt_max_deg: float = 40.0

    # Ballistic planning shape controls
    # Larger values -> higher apex, usually less leg compression near touchdown.
    traj_apex_scale: float = 1.22
    traj_apex_clearance: float = 0.22
    traj_apex_height: float = 1.0
    traj_min_vx: float = 0.25
    traj_max_vx: float = 8.0

    # Anti-local-optimum guards
    # If the policy keeps pressing in stance too long, terminate early.
    max_consecutive_stance_steps: int = 180  # 180 × 0.01 s = 1.8 s (allows full pendulum swing + bigger launches)
    # Expected minimum airborne steps after liftoff before returning stance.
    min_airborne_steps: int = 5              # 5 × 0.01 s = 0.05 s
    no_progress_steps: int = 220             # terminate if it stays near the start too long
    no_progress_min_delta: float = 0.04      # metres of new forward foot progress required


@dataclass
class RewardConfig:
    """
    Ballistic-sector liftoff + inverted-pendulum stance reward.

    Design philosophy
    -----------------
    * Liftoff event is the PRIMARY training signal: reward how closely the
      actual (vx, vy) at liftoff matches the planned ballistic requirement.
      Once those conditions are met, passive ballistic flight reaches the target.
    * Stance phase: reward the inverted-pendulum CW swing (dtheta > 0) and
      the spring compression→extension cycle.
    * Touchdown event: reward backward lean angle phi_td so the next stance
      starts with the correct attack angle.
    * Flight: small attitude shaping only (apex adjustment for landing angle).
    """
    # ── Ballistic sector bounds ───────────────────────────────────────────
    alpha_min_deg: float = 5.0           # min liftoff angle — natural SLIP gives 12-26°, must include this
    alpha_max_deg: float = 85.0          # high 0.5 m hops to 1 m need ~80° launch angle

    # ── Liftoff event (main signal) ───────────────────────────────────────
    liftoff_v_weight: float = 80.0       # reward matching planned v0
    liftoff_v_sharpness: float = 10.0   # softer: 4-5 step stance can't perfectly match
    liftoff_angle_weight: float = 20.0  # reduced: short stance limits angle control
    liftoff_angle_sharpness: float = 4.0  # very soft angle reward

    # ── Stance: inverted pendulum + spring ────────────────────────────────
    stance_pendulum_weight: float = 3.0  # reward dtheta > 0 (short stance, modest weight)
    stance_spring_weight: float = 1.0   # spring reward (spring handles itself with preload)
    stance_theta_pos_weight: float = 2.0 # forward lean during short 4-5 step stance
    stance_stall_penalty: float = 0.05
    stance_timeout_penalty: float = 100.0

    # ── Touchdown event ───────────────────────────────────────────────────
    phi_td_target_deg: float = -3.0     # target backward lean at touchdown (°)
    touchdown_weight: float = 40.0
    touchdown_sharpness: float = 8.0
    touchdown_bad_penalty: float = 20.0

    # ── Flight attitude (apex adjustment toward phi_td) ───────────────────
    flight_attitude_weight: float = 2.0   # stronger signal: policy must control theta during flight
    flight_attitude_sharpness: float = 3.0
    flight_thrust_penalty: float = 0.15
    flight_height_weight: float = 8.0
    flight_height_sharpness: float = 6.0
    overheight_penalty_weight: float = 18.0
    target_height: float = 1.0

    # ── Dense progress shaping ────────────────────────────────────────────
    forward_progress_weight: float = 0.35
    backward_progress_penalty: float = 0.20

    # ── Landing proximity (dense shaping at each touchdown) ──────────────
    landing_proximity_weight: float = 30.0  # reward foot landing near next target
    landing_proximity_sharpness: float = 8.0  # exp(-k * dist²), k=8 → 50% at 0.35m

    # ── Target success ────────────────────────────────────────────────────
    target_hit_reward: float = 150.0
    all_targets_bonus: float = 300.0

    # ── Termination ───────────────────────────────────────────────────────
    termination_penalty: float = 60.0
    out_of_bounds_penalty: float = 60.0
    max_tilt_rad: float = 1.1             # ~63°; enough for liftoff lean, stops tumbles
    max_height: float = 1.7
    min_height: float = 0.04
    max_overshoot: float = 0.4


@dataclass
class TrainingConfig:
    """PPO training hyperparameters."""
    model_path: str = "artifacts/models/ppo_quadhopper_v7_height_bounded_single"
    target_count: int = 1
    n_envs: int = 8
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    ent_coef: float = 0.02
    gamma: float = 0.995
    device: str = "auto"


@dataclass
class RenderConfig:
    """Rendering and visualization settings."""
    gif_path: str = "artifacts/renders/quadhopper_v7_height_bounded_single.gif"
    plot_path: str = "artifacts/renders/thrust_analysis_v7_height_bounded_single.png"
    fps: int = 25
    render_every_n: int = 2
    fig_width: int = 12
    fig_height: int = 6
    dpi: int = 80
    view_x_behind: float = 2.0
    view_x_ahead: float = 3.0
    view_y_min: float = -0.2
    view_y_max: float = 2.8
    test_steps: int = 800

    # geometry-focused render scaling
    render_geom_scale: float = 2.5
    render_rotor_size: float = 55.0
    body_linewidth: float = 4.0
    leg_linewidth: float = 2.6
    foot_markersize: float = 9.0


# ── Singleton instances (import these directly) ──────────────────────────────
PHYSICS = PhysicsConfig()
ATTITUDE = AttitudeConfig()
ENV = EnvConfig()
REWARD = RewardConfig()
TRAINING = TrainingConfig()
RENDER = RenderConfig()
