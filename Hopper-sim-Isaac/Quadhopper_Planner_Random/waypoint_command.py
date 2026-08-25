from __future__ import annotations

import math

import torch


class TwoHopRandomCommand:
    """Rolling two-target command with alternating short and long hops.

    At reset, P_t is sampled 0.5--0.8 m from the start and P_(t+1) is
    sampled 0.8--1.0 m from P_t.  Once a target is hit, the queue advances
    and a new look-ahead point is sampled.  Independent uniform headings make
    every horizontal direction equally likely while retaining the policy's
    two-cycle look-ahead contract.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        short_radius_min: float,
        short_radius_max: float,
        long_radius_min: float,
        long_radius_max: float,
        successful_hops_per_episode: int,
        max_turn_angle_rad: float = math.pi,
    ):
        self._validate_range("short", short_radius_min, short_radius_max)
        self._validate_range("long", long_radius_min, long_radius_max)
        if successful_hops_per_episode <= 0:
            raise ValueError("successful_hops_per_episode must be positive")

        self.num_envs = int(num_envs)
        self.device = device
        self.short_radius_min = float(short_radius_min)
        self.short_radius_max = float(short_radius_max)
        self.long_radius_min = float(long_radius_min)
        self.long_radius_max = float(long_radius_max)
        if not 0.0 < max_turn_angle_rad <= math.pi:
            raise ValueError("max_turn_angle_rad must be in (0, pi]")
        self.max_turn_angle_rad = float(max_turn_angle_rad)
        # Compatibility name consumed by the shared completion/reward logic.
        self.steps_per_revolution = int(successful_hops_per_episode)

        self.anchor_w = torch.zeros(self.num_envs, 2, device=device)
        self.targets_w = torch.zeros(self.num_envs, 2, 2, device=device)
        # cycle_index is the public height-command phase and may be randomized
        # by the shared environment. Keep route_index independent so that the
        # geometric sequence always remains short, long, short, long.
        self.cycle_index = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.route_index = torch.zeros(self.num_envs, dtype=torch.long, device=device)

    @staticmethod
    def _validate_range(name: str, lower: float, upper: float):
        if lower <= 0.0 or upper < lower:
            raise ValueError(
                f"{name} hop radius must satisfy 0 < min <= max, got [{lower}, {upper}]"
            )

    def _sample_offset(
        self, count: int, radius_min: float, radius_max: float
    ) -> torch.Tensor:
        radius = radius_min + (radius_max - radius_min) * torch.rand(
            count, device=self.device
        )
        heading = 2.0 * math.pi * torch.rand(count, device=self.device)
        return radius[:, None] * torch.stack(
            (torch.cos(heading), torch.sin(heading)), dim=-1
        )

    def _sample_phase_offsets(self, is_short: torch.Tensor) -> torch.Tensor:
        """Sample one offset per environment from its assigned route phase."""
        offsets = torch.empty(len(is_short), 2, device=self.device)
        short_ids = is_short.nonzero(as_tuple=False).flatten()
        long_ids = (~is_short).nonzero(as_tuple=False).flatten()
        if len(short_ids) > 0:
            offsets[short_ids] = self._sample_offset(
                len(short_ids), self.short_radius_min, self.short_radius_max
            )
        if len(long_ids) > 0:
            offsets[long_ids] = self._sample_offset(
                len(long_ids), self.long_radius_min, self.long_radius_max
            )
        return offsets

    def _sample_phase_offsets_around(
        self, is_short: torch.Tensor, reference_offsets: torch.Tensor
    ) -> torch.Tensor:
        """Sample phase radii with headings near the preceding hop."""
        offsets = self._sample_phase_offsets(is_short)
        radii = torch.linalg.norm(offsets, dim=1)
        reference_heading = torch.atan2(reference_offsets[:, 1], reference_offsets[:, 0])
        heading_delta = self.max_turn_angle_rad * (
            2.0 * torch.rand(len(is_short), device=self.device) - 1.0
        )
        heading = reference_heading + heading_delta
        return radii[:, None] * torch.stack(
            (torch.cos(heading), torch.sin(heading)), dim=-1
        )

    def reset(self, env_ids: torch.Tensor, env_origins: torch.Tensor, random_phase: bool):
        count = len(env_ids)
        anchor = env_origins[env_ids, :2]
        route_phase = (
            torch.randint(0, 2, (count,), device=self.device)
            if random_phase
            else torch.zeros(count, dtype=torch.long, device=self.device)
        )
        first_is_short = (route_phase % 2) == 0
        first_offset = self._sample_phase_offsets(first_is_short)
        second_offset = self._sample_phase_offsets_around(~first_is_short, first_offset)
        first = anchor + first_offset
        second = first + second_offset
        self.anchor_w[env_ids] = anchor
        self.targets_w[env_ids, 0] = first
        self.targets_w[env_ids, 1] = second
        self.cycle_index[env_ids] = 0
        self.route_index[env_ids] = route_phase

    def start_points(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.anchor_w if env_ids is None else self.anchor_w[env_ids]

    def lookahead(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        targets = self.targets_w if env_ids is None else self.targets_w[env_ids]
        return targets[:, 0], targets[:, 1]

    def current_hop_is_short(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the geometric phase before the corresponding targets advance."""
        return (self.route_index[env_ids] % 2) == 0

    def advance(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self.anchor_w[env_ids] = self.targets_w[env_ids, 0]
        self.targets_w[env_ids, 0] = self.targets_w[env_ids, 1]
        self.cycle_index[env_ids] += 1
        self.route_index[env_ids] += 1

        # After jump 1, append jump 3 (short); after jump 2, append jump 4
        # (long). This keeps the infinite sequence short, long, short, long.
        append_short = (self.route_index[env_ids] % 2) == 1
        upcoming_offset = self.targets_w[env_ids, 0] - self.anchor_w[env_ids]
        offsets = self._sample_phase_offsets_around(append_short, upcoming_offset)
        self.targets_w[env_ids, 1] = self.targets_w[env_ids, 0] + offsets

    def restart_pair(self, env_ids: torch.Tensor, current_xy_w: torch.Tensor):
        """Start a new short/long pair about the measured touchdown position."""
        if len(env_ids) == 0:
            return
        first_is_short = torch.ones(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        first_offset = self._sample_phase_offsets(first_is_short)
        second_offset = self._sample_phase_offsets_around(
            ~first_is_short, first_offset
        )
        self.anchor_w[env_ids] = current_xy_w
        self.targets_w[env_ids, 0] = current_xy_w + first_offset
        self.targets_w[env_ids, 1] = self.targets_w[env_ids, 0] + second_offset
        self.cycle_index[env_ids] += 1
        # The completed hop was the odd (long) phase.  Advancing once returns
        # the route to an even (short) phase without erasing total progress.
        self.route_index[env_ids] += 1
