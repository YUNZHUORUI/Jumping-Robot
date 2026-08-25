from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass

from Quadhopper_Planner_Circular.planner_circular_env import (
    PlannerCircularEnv,
    PlannerCircularEnvCfg,
)

from .waypoint_command import TwoHopRandomCommand


@configclass
class PlannerRandomTwoHopEnvCfg(PlannerCircularEnvCfg):
    """Arbitrary-direction route with a repeating short/long hop pair."""

    # Used only to normalize the two XY command vectors in this task.  The
    # random command generator owns the actual hop-distance distribution.
    hop_distance = 0.75
    short_hop_radius_min = 0.50
    short_hop_radius_max = 0.80
    long_hop_radius_min = 0.80
    long_hop_radius_max = 1.00
    successful_hops_per_episode = 58
    max_turn_angle_deg = 180.0
    # A random route is a sequence of commanded landings, not a collection of
    # goals that may be retried indefinitely.  This makes every measured hit
    # rate a first-attempt hit rate and guarantees P_(t+1) becomes the next
    # command after the current touchdown.
    advance_route_on_miss = True
    restart_two_hop_pair = False
    randomize_route_phase = False
    distance_curriculum_iterations = 0.0
    distance_curriculum_iteration_offset = 0.0
    curriculum_short_radius_min = 0.30
    curriculum_short_radius_max = 0.50
    curriculum_long_radius_min = 0.30
    curriculum_long_radius_max = 0.50
    curriculum_max_turn_angle_deg = 30.0
    # When True, distance_curriculum_iterations counts completed hops per
    # environment instead of environment steps.  This matches the hop-latched
    # Semi-MDP trainer, whose updates consume only a few hundred steps but
    # dozens of hops, so a step-based schedule would finish training before
    # the full 0.50--0.80 / 0.80--1.00 m range was ever commanded.
    curriculum_by_hops = False


class PlannerRandomTwoHopEnv(PlannerCircularEnv):
    cfg: PlannerRandomTwoHopEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._first_hop_hit_for_pair = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._conditional_second_attempts = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._conditional_second_hits = torch.zeros_like(
            self._conditional_second_attempts
        )
        self._pair_attempts = torch.zeros_like(self._conditional_second_attempts)
        self._pair_hits = torch.zeros_like(self._conditional_second_attempts)
        # Monotonic across resets; drives the hop-based distance curriculum.
        self._total_touchdowns = 0

    def _create_commands(self) -> TwoHopRandomCommand:
        return TwoHopRandomCommand(
            self.num_envs,
            self.device,
            self.cfg.short_hop_radius_min,
            self.cfg.short_hop_radius_max,
            self.cfg.long_hop_radius_min,
            self.cfg.long_hop_radius_max,
            self.cfg.successful_hops_per_episode,
            math.radians(self.cfg.max_turn_angle_deg),
        )

    def _sync_distance_curriculum(self):
        if (
            not hasattr(self, "commands")
            or self.cfg.distance_curriculum_iterations <= 0.0
        ):
            return
        if self.cfg.curriculum_by_hops:
            iteration = self.cfg.distance_curriculum_iteration_offset + float(
                getattr(self, "_total_touchdowns", 0)
            ) / max(self.num_envs, 1)
        else:
            iteration = self.cfg.distance_curriculum_iteration_offset + float(
                getattr(self, "common_step_counter", 0)
            ) / max(self.cfg.curriculum_steps_per_iteration, 1.0)
        blend = max(
            0.0,
            min(1.0, iteration / self.cfg.distance_curriculum_iterations),
        )

        def lerp(start: float, end: float) -> float:
            return start + blend * (end - start)

        self.commands.short_radius_min = lerp(
            self.cfg.curriculum_short_radius_min, self.cfg.short_hop_radius_min
        )
        self.commands.short_radius_max = lerp(
            self.cfg.curriculum_short_radius_max, self.cfg.short_hop_radius_max
        )
        self.commands.long_radius_min = lerp(
            self.cfg.curriculum_long_radius_min, self.cfg.long_hop_radius_min
        )
        self.commands.long_radius_max = lerp(
            self.cfg.curriculum_long_radius_max, self.cfg.long_hop_radius_max
        )
        self.commands.max_turn_angle_rad = math.radians(
            lerp(self.cfg.curriculum_max_turn_angle_deg, self.cfg.max_turn_angle_deg)
        )

    def _update_cycle_events(self):
        # Snapshot the geometric phase because the parent consumes the target
        # immediately after touchdown.
        phase_was_short = (self.commands.route_index % 2) == 0
        super()._update_cycle_events()
        touchdown_ids = self._touchdown_event.nonzero(as_tuple=False).flatten()
        self._total_touchdowns += len(touchdown_ids)
        if len(touchdown_ids) == 0:
            return
        was_short = phase_was_short[touchdown_ids]
        short_ids = touchdown_ids[was_short]
        long_ids = touchdown_ids[~was_short]
        if len(short_ids) > 0:
            self._first_hop_hit_for_pair[short_ids] = self._target_hit_event[short_ids]
        if len(long_ids) > 0:
            first_hit = self._first_hop_hit_for_pair[long_ids]
            second_hit = self._target_hit_event[long_ids]
            self._pair_attempts[long_ids] += 1
            self._pair_hits[long_ids] += (first_hit & second_hit).long()
            eligible_ids = long_ids[first_hit]
            self._conditional_second_attempts[eligible_ids] += 1
            self._conditional_second_hits[eligible_ids] += second_hit[first_hit].long()
            self._first_hop_hit_for_pair[long_ids] = False

    def _reset_idx(self, env_ids: torch.Tensor | None):
        self._sync_distance_curriculum()
        super()._reset_idx(env_ids)
        if env_ids is None or not hasattr(self, "_first_hop_hit_for_pair"):
            return
        self._first_hop_hit_for_pair[env_ids] = False
        self._conditional_second_attempts[env_ids] = 0
        self._conditional_second_hits[env_ids] = 0
        self._pair_attempts[env_ids] = 0
        self._pair_hits[env_ids] = 0

    def _advance_route(self, env_ids: torch.Tensor):
        self._sync_distance_curriculum()
        super()._advance_route(env_ids)

    def _lookahead_error_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_error_b, next_absolute_error_b = super()._lookahead_error_b()
        if self.cfg.relative_next_hop_observation:
            return current_error_b, next_absolute_error_b - current_error_b
        return current_error_b, next_absolute_error_b

    def _route_visualization_xy(self) -> torch.Tensor:
        """Draw the two sampled-radius circles that produced P_t/P_(t+1)."""
        angles = torch.linspace(
            0.0,
            2.0 * torch.pi,
            self.cfg.circle_vis_points + 1,
            device=self.device,
        )[:-1]
        unit_circle = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        p_t, p_t1 = self.commands.lookahead()
        first_radius = torch.linalg.norm(p_t - self.commands.anchor_w, dim=1)
        second_radius = torch.linalg.norm(p_t1 - p_t, dim=1)
        first_circle = (
            self.commands.anchor_w[:, None, :]
            + first_radius[:, None, None] * unit_circle[None, :, :]
        )
        second_circle = (
            p_t[:, None, :]
            + second_radius[:, None, None] * unit_circle[None, :, :]
        )
        return torch.cat((first_circle, second_circle), dim=1)
