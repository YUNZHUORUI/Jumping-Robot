import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(CURRENT_DIR, "model", "QuadhopperAsset.usd")

MY_DRONE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            # 陀螺仪效应，你的 USD 脚本里开了，这里保持开启即可
            enable_gyroscopic_forces=True,
        ),
        # 【核心修复】：直接删除了 mass_props 这一块！
        # 因为你的 DroneAsset.usd 本身已经携带了完美的 Mass 和 Inertia 信息
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)
