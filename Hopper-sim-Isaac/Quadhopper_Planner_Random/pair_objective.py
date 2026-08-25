"""Dense touchdown-state objective for two-hop candidate search."""

from __future__ import annotations

import torch


def touchdown_components(
    position_error: torch.Tensor,
    velocity_error: torch.Tensor,
    attitude_error: torch.Tensor,
    angular_velocity: torch.Tensor,
    spring_position: torch.Tensor,
    spring_velocity: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return dimensionless, separately inspectable touchdown costs."""
    return {
        "position": (position_error / 0.10).square(),
        "velocity": (velocity_error / 0.30).square(),
        "attitude": (attitude_error / torch.deg2rad(torch.tensor(6.0, device=attitude_error.device))).square()
        + 0.25 * (angular_velocity / 1.0).square(),
        "spring": ((spring_position - 0.002) / 0.010).square()
        + (spring_velocity / 0.50).square(),
    }


def touchdown_score(components: dict[str, torch.Tensor]) -> torch.Tensor:
    """Higher is better; zero is a nominal prepared touchdown."""
    cost = (
        4.0 * components["position"]
        + 2.0 * components["velocity"]
        + 1.5 * components["attitude"]
        + 0.5 * components["spring"]
    )
    return -cost.clamp_max(100.0)
