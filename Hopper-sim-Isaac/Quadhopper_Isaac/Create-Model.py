from isaacsim import SimulationApp
import numpy as np

# 后台静默运行
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, PhysxSchema

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

# 你的专属物理参数
MASS = 0.121
I_XX, I_YY, I_ZZ = 4.210293e-04, 4.452195e-04, 7.934839e-04
DIAGONAL_M = 0.230
MOTOR_OFFSET = 0.0813

# ==========================================
# 核心修复：单节点物理架构 (拒绝刚体套娃)
# ==========================================
root_path = "/Drone"
root_prim = stage.DefinePrim(root_path, "Xform")
stage.SetDefaultPrim(root_prim)

# 1. 物理注入全部直接挂在根节点上！
UsdPhysics.RigidBodyAPI.Apply(root_prim)
# 提前在 USD 里声明这是物理树的根，省去 Isaac Lab 的猜测
UsdPhysics.ArticulationRootAPI.Apply(root_prim)

physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
physx_api.GetLinearDampingAttr().Set(0.05)
physx_api.GetAngularDampingAttr().Set(0.05)
# 在底层直接为你开启陀螺效应（对无人机极其重要）
physx_api.GetEnableGyroscopicForcesAttr().Set(True)

mass_api = UsdPhysics.MassAPI.Apply(root_prim)
mass_api.GetMassAttr().Set(MASS)
mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(I_XX, I_YY, I_ZZ))

# 2. 视觉机身与碰撞 (作为几何体挂在根节点下)
visual_body_path = f"{root_path}/visual_cube"
visual_cube = stage.DefinePrim(visual_body_path, "Cube")
UsdGeom.Cube(visual_cube).GetSizeAttr().Set(1.0)
UsdGeom.XformCommonAPI(visual_cube).SetScale((0.04, 0.04, 0.012))
UsdGeom.Gprim(visual_cube).GetDisplayColorAttr().Set([(0.2, 0.5, 0.8)])
# 碰撞体积绑定到这个具体的方块几何体上
UsdPhysics.CollisionAPI.Apply(visual_cube)

# 3. 交叉臂
for i, angle in enumerate([45, -45]):
    arm_path = f"{root_path}/arm_{i}"
    arm_prim = stage.DefinePrim(arm_path, "Cylinder")
    UsdGeom.Cylinder(arm_prim).GetRadiusAttr().Set(0.003)
    UsdGeom.Cylinder(arm_prim).GetHeightAttr().Set(DIAGONAL_M)
    UsdGeom.Cylinder(arm_prim).GetDisplayColorAttr().Set([(0.1, 0.1, 0.1)])
    UsdGeom.XformCommonAPI(arm_prim).SetRotate((0, 90, angle))

# 4. 电机标记点
motor_pos = [
    (-MOTOR_OFFSET, MOTOR_OFFSET, 0), (-MOTOR_OFFSET, -MOTOR_OFFSET, 0),
    (MOTOR_OFFSET, -MOTOR_OFFSET, 0), (MOTOR_OFFSET, MOTOR_OFFSET, 0)
]

for i, pos in enumerate(motor_pos):
    m_path = f"{root_path}/motor_{i}"
    m_prim = stage.DefinePrim(m_path, "Cylinder")
    UsdGeom.Cylinder(m_prim).GetRadiusAttr().Set(0.015)
    UsdGeom.Cylinder(m_prim).GetHeightAttr().Set(0.005)

    if i ==1 or i == 3:
        color = (0.8, 0.1, 0.1)
    else:
        color = (0.1, 0.1, 0.1)

    UsdGeom.Cylinder(m_prim).GetDisplayColorAttr().Set([color])
    UsdGeom.XformCommonAPI(m_prim).SetTranslate(pos)

# 保存
asset_path = "DroneAsset.usd"
omni.usd.get_context().save_as_stage(asset_path)
print(f">>> 完美修复版 USD 已生成: {asset_path}")

simulation_app.close()