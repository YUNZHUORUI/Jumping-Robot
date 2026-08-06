"""Shared Quadhopper hardware asset configuration.

This deliberately points at the baseline asset instead of duplicating the robot model.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg


USD_PATH = Path(__file__).resolve().parent.parent / "Quadhopper_Isaac" / "model" / "HopperAsset.usd"
POWER_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "Quadhopper_Isaac" / "model" / "quadhopper_memory_power.pt"
)

QUADHOPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={"center_spring_joint": 0.0},
        joint_vel={"center_spring_joint": 0.0},
    ),
    actuators={},
)
