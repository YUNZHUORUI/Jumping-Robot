from __future__ import annotations

import math


def cosine_height(iteration: float, start: float, end: float, duration: float) -> float:
    """Return a clamped cosine-eased fixed-height curriculum command."""
    progress = max(0.0, min(1.0, float(iteration) / max(float(duration), 1.0)))
    blend = 0.5 - 0.5 * math.cos(progress * math.pi)
    return float(start) + blend * (float(end) - float(start))
