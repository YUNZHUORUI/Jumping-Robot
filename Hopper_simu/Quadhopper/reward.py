"""Reward function for QuadHopper."""

import math
from dataclasses import dataclass
from typing import Tuple

from .config import RewardConfig


@dataclass
class RewardInfo:
    """Breakdown of reward components for logging/debugging."""
    flight_track: float = 0.0
    flight_dy_track: float = 0.0
    flight_target: float = 0.0
    flight_overshoot: float = 0.0
    flight_vx_track: float = 0.0
    flight_attitude: float = 0.0
    flight_thrust: float = 0.0
    flight_ang_vel: float = 0.0
    first_target: float = 0.0
    touchdown: float = 0.0
    stance_attitude: float = 0.0
    liftoff: float = 0.0
    attitude_pen: float = 0.0
    target_hit: float = 0.0
    termination: float = 0.0

    @property
    def total(self) -> float:
        return sum(vars(self).values())


class RewardFunction:
    """Stateless reward function with explicit component breakdown."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def compute(
        self,
        *,
        # Kinematic state
        theta: float,
        dtheta: float,
        vx: float,
        # Contact events
        touching: bool,
        touchdown_event: bool,
        liftoff_event: bool,
        # Trajectory tracking
        traj_valid: bool,
        y_error: float,
        dy_error: float,
        traj_vx_nom: float,
        # Attitude targets
        landing_theta_target: float,
        takeoff_theta_target: float,
        # Actions
        u1: float,
        u2: float,
        # Target geometry
        dist_to_target: float,
        target_x: float,
        foot_x: float,
        target_valid: bool,
        current_target_idx: int,
        # Target success flag (set externally)
        target_hit: bool = False,
        all_targets_done: bool = False,
        # Termination flags
        terminated_bad: bool = False,
        out_of_bounds: bool = False,
    ) -> Tuple[float, RewardInfo]:
        """
        Compute the total reward and its breakdown.

        Returns:
            (total_reward, RewardInfo)
        """
        c = self.cfg
        info = RewardInfo()

        # ── 1. Flight phase ────────────────────────────────────────────────
        if (not touching) and traj_valid:
            info.flight_track = c.flight_traj_track_weight * math.exp(
                -c.flight_traj_track_sharpness * abs(y_error)
            )
            info.flight_dy_track = c.flight_dy_track_weight * math.exp(
                -c.flight_dy_track_sharpness * abs(dy_error)
            )

            if target_valid:
                info.flight_target = c.flight_target_weight * math.exp(
                    -c.flight_target_sharpness * dist_to_target
                )
                overshoot = foot_x - target_x
                if overshoot > 0.0:
                    info.flight_overshoot = -c.flight_overshoot_penalty * overshoot

            info.flight_thrust = -c.flight_thrust_penalty * (u1 + u2)

            if traj_vx_nom > 0.0:
                info.flight_vx_track = -c.flight_vx_track_penalty * abs(vx - traj_vx_nom)

            theta_err_land = abs(
                self._wrap_angle(theta - landing_theta_target)
            )
            info.flight_attitude = c.flight_attitude_weight * math.exp(
                -c.flight_attitude_sharpness * theta_err_land
            )
            info.flight_ang_vel = -c.flight_angular_vel_penalty * abs(dtheta)

            if target_valid and current_target_idx == 0:
                info.first_target = c.first_target_weight * math.exp(
                    -c.first_target_sharpness * dist_to_target
                )

        # ── 2. Touchdown event ─────────────────────────────────────────────
        if touchdown_event:
            theta_td_err = abs(self._wrap_angle(theta - landing_theta_target))
            if theta < 0.0:
                info.touchdown = c.touchdown_reward * math.exp(
                    -c.touchdown_sharpness * theta_td_err
                )
            else:
                info.touchdown = -(c.touchdown_bad_penalty
                                   + c.touchdown_bad_slope * theta_td_err)

        # ── 3. Stance / liftoff ────────────────────────────────────────────
        if touching:
            theta_to_to = abs(
                self._wrap_angle(theta - takeoff_theta_target)
            )
            info.stance_attitude = c.stance_attitude_weight * math.exp(
                -c.stance_attitude_sharpness * theta_to_to
            )

        if liftoff_event:
            theta_lo_err = abs(
                self._wrap_angle(theta - takeoff_theta_target)
            )
            info.liftoff = c.liftoff_reward * math.exp(
                -c.liftoff_sharpness * theta_lo_err
            )

        # ── 4. Attitude penalty ────────────────────────────────────────────
        info.attitude_pen = (
            -c.attitude_abs_penalty * abs(theta)
            - c.angular_vel_penalty * abs(dtheta)
        )

        # ── 5. Target reward ───────────────────────────────────────────────
        if target_hit:
            info.target_hit += c.target_hit_reward
        if all_targets_done:
            info.target_hit += c.all_targets_bonus
        elif target_valid and dist_to_target < c.near_target_radius:
            info.target_hit += c.near_target_weight * (
                c.near_target_radius - dist_to_target
            )

        # ── 6. Termination ─────────────────────────────────────────────────
        if terminated_bad:
            info.termination -= c.termination_penalty
        if out_of_bounds:
            info.termination -= c.out_of_bounds_penalty

        return info.total, info

