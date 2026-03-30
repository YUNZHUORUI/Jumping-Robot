"""Central configuration for the modular QuadHopper package."""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PhysicsConfig:
    """Physical simulation parameters (aligned with Quadhopper11-3-26.py)."""
    dt: float = 0.002
    mass: float = 0.213
    rotor_span: float = 0.10
    leg_length: float = 0.30
    stroke_length: float = 0.20
    inertia: float = 0.15
    gravity: float = 9.81
    max_thrust: float = 30.0
    ground_y: float = 0.0

    # Legacy spring-damper model
    k_spring: float = 150.0
    c_damping: float = 20.0

    # SLIP stance model
    use_slip_stance: bool = True
    k_slip: float = 2200.0
    c_slip: float = 18.0

    @property
    def beam_half_length(self) -> float:
        return 0.5 * self.rotor_span

    @property
    def cg_to_motor(self) -> float:
        return self.beam_half_length

    @property
    def leg_min_length(self) -> float:
        return max(0.05, self.leg_length - self.stroke_length)

    @property
    def min_com_height(self) -> float:
        return max(0.04, 0.35 * self.leg_min_length)


@dataclass
class AttitudeConfig:
    """Attitude phase targets and tolerances."""
    landing_theta_deg: float = -20.0
    landing_theta_tol_deg: float = 8.0
    takeoff_theta_deg: float = -25.0
    takeoff_theta_tol_deg: float = 6.0

    @property
    def landing_theta(self) -> float:
        return math.radians(self.landing_theta_deg)

    @property
    def landing_theta_tol(self) -> float:
        return math.radians(self.landing_theta_tol_deg)

    @property
    def takeoff_theta(self) -> float:
        return math.radians(self.takeoff_theta_deg)

    @property
    def takeoff_theta_tol(self) -> float:
        return math.radians(self.takeoff_theta_tol_deg)


@dataclass
class EnvConfig:
    """Environment-level settings."""
    targets: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 7.0, 12.0, 18.0])
    )
    target_tolerance: float = 0.4
    max_episode_steps: int = 3000
    obs_clip: float = 20.0

    # Init randomization range (degrees)
    init_theta_min_deg: float = -35.0
    init_theta_max_deg: float = -20.0

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


@dataclass
class RewardConfig:
    """All reward shaping weights."""
    # --- Flight phase ---
    flight_traj_track_weight: float = 1.8
    flight_traj_track_sharpness: float = 8.0
    flight_dy_track_weight: float = 0.8
    flight_dy_track_sharpness: float = 0.25
    flight_target_weight: float = 2.0
    flight_target_sharpness: float = 1.6
    flight_overshoot_penalty: float = 2.5
    flight_thrust_penalty: float = 0.08
    flight_vx_track_penalty: float = 0.12
    flight_attitude_weight: float = 0.7
    flight_attitude_sharpness: float = 6.0
    flight_angular_vel_penalty: float = 0.03
    first_target_weight: float = 2.0
    first_target_sharpness: float = 1.2

    # --- Touchdown event ---
    touchdown_reward: float = 35.0
    touchdown_sharpness: float = 8.0
    touchdown_bad_penalty: float = 30.0
    touchdown_bad_slope: float = 25.0

    # --- Stance phase ---
    stance_attitude_weight: float = 0.45
    stance_attitude_sharpness: float = 4.0

    # --- Liftoff event ---
    liftoff_reward: float = 25.0
    liftoff_sharpness: float = 6.0

    # --- General attitude penalty ---
    attitude_abs_penalty: float = 0.02
    angular_vel_penalty: float = 0.04

    # --- Target hit ---
    target_hit_reward: float = 150.0
    all_targets_bonus: float = 300.0
    near_target_weight: float = 10.0
    near_target_radius: float = 1.5

    # --- Termination penalties ---
    termination_penalty: float = 100.0
    out_of_bounds_penalty: float = 90.0

    # --- Termination thresholds ---
    max_tilt_rad: float = 0.7
    max_height: float = 7.0
    min_height: float = 0.04
    max_overshoot: float = 1.0


@dataclass
class TrainingConfig:
    """PPO training hyperparameters."""
    model_path: str = "ppo_quadhopper_fixed_v1"
    n_envs: int = 8
    total_timesteps: int = 3_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    ent_coef: float = 0.02
    gamma: float = 0.995
    device: str = "auto"


@dataclass
class RenderConfig:
    """Rendering and visualization settings."""
    gif_path: str = "quadhopper_fixed.gif"
    plot_path: str = "thrust_analysis_fixed.png"
    fps: int = 30
    render_every_n: int = 3
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
