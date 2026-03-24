# quadhopper/__init__.py
"""
QuadHopper RL Package
=====================
A modular reinforcement learning project for a single-leg hopping robot.

Modules:
    config     - All hyperparameters and configurations
    physics    - SLIP physics engine
    trajectory - Parabolic trajectory planner
    reward     - Reward function with breakdown
    env        - Gymnasium environment
    renderer   - GIF and plot rendering
    train      - Training / testing entry point
"""

from .env import QuadhopperTargetEnv
from .config import PHYSICS, ATTITUDE, ENV, REWARD, TRAINING, RENDER

__all__ = [
    "QuadhopperTargetEnv",
    "PHYSICS", "ATTITUDE", "ENV", "REWARD", "TRAINING", "RENDER",
]
