"""
QuadHopper 模型组装脚本
位置：自动对齐（两个STL合并包围盒中心 + 顶面贴无人机底面）
旋转：手动在顶部调整
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os, struct, re
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema, Sdf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STL_BASE = os.path.join(CURRENT_DIR, "Jump+Base.stl")
STL_LEG  = os.path.join(CURRENT_DIR, "Jump+Leg.stl")
USD_OUT  = os.path.join(CURRENT_DIR, "QuadhopperAsset.usd")

# ══════════════════════════════════════════════════════
#  调参区：只需调整这里的旋转角度（度）
#  位置由脚本自动计算，不用改
# ══════════════════════════════════════════════════════

STL_SCALE = 0.001   # mm → m

# Base 旋转（绕自身中心，XYZ 顺序）
BASE_ROT = (0.0, -90.0, 0.0)   # (rx, ry, rz)

# Leg 旋转（绕自身中心，XYZ 顺序）
LEG_ROT  = (0.0, -90.0, 0.0)   # (rx, ry, rz)

# 额外偏移（m）：正/负对应方向
X_OFFSET = -0.01    # 正 = 向前，负 = 向后
Y_OFFSET = 0.0    # 正 = 向左，负 = 向右
Z_OFFSET = -0.054 # 正 = 上移，负 = 下移（机身底面在 -0.006 m）

# ══════════════════════════════════════════════════════
#  物理参数
# ══════════════════════════════════════════════════════

DRONE_MASS    = 0.1756  # 整机（机体+腿）总质量 175.6g（匹配实物）
I_XX, I_YY, I_ZZ = 1.400285e-03, 1.455107e-03, 2.615538e-03   # 实测惯量值
DIAGONAL_M    = 0.230
MOTOR_OFFSET  = 0.0813
DRONE_BODY_HALF_H = 0.006   # 机身方块 Z 半高

MASS_LEG       = 0.020   # 20g：需要足够质量才能压缩弹簧并存储弹性势能
JOINT_AXIS     = "Z"
SPRING_STIFF   = 400.0   # 4 根弹簧并联（每根 100 N/m）= 400 N/m
SPRING_DAMP    = 0.3     # 欠阻尼，实现真正弹跳（临界阻尼=2√(km)=1.55）
JOINT_LOWER    = -0.07   # 最大压缩7cm（对应1m/s落地速度的弹性形变）
JOINT_UPPER    =  0.005

# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════

def parse_stl(filepath):
    with open(filepath, 'rb') as f:
        raw = f.read()
    try:
        text = raw.decode('ascii')
        if text.lstrip().startswith('solid'):
            pts = []
            for m in re.finditer(r'vertex\s+([\S]+)\s+([\S]+)\s+([\S]+)', text):
                pts.append([float(m[1]), float(m[2]), float(m[3])])
            return np.array(pts, dtype=np.float64)
    except UnicodeDecodeError:
        pass
    n = struct.unpack_from('<I', raw, 80)[0]
    v = np.empty((n * 3, 3), dtype=np.float64)
    off = 84
    for i in range(n):
        off += 12
        for j in range(3):
            v[i*3+j] = struct.unpack_from('<3f', raw, off)
            off += 12
        off += 2
    return v


def make_rot(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx


def rotate_around_pivot(verts_mm, rot_deg, pivot):
    """绕指定轴心旋转（mm 空间）"""
    v = verts_mm - pivot
    R = make_rot(*rot_deg)
    v = (R @ v.T).T
    return v + pivot


def to_usd_mesh(stage, path, verts):
    pts = [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in verts]
    m = UsdGeom.Mesh.Define(stage, path)
    m.GetPointsAttr().Set(pts)
    m.GetFaceVertexCountsAttr().Set([3] * (len(pts) // 3))
    m.GetFaceVertexIndicesAttr().Set(list(range(len(pts))))
    m.GetSubdivisionSchemeAttr().Set("none")
    return m


def add_collision(prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set("convexHull")


# ══════════════════════════════════════════════════════
#  Step 1：解析 + 旋转（在 mm 空间）
# ══════════════════════════════════════════════════════
print("=== 解析 STL ===")
raw_base = parse_stl(STL_BASE)
raw_leg  = parse_stl(STL_LEG)

# ── 位置参考：XY 用 Base 自身包围盒中心（对齐无人机中心），Z 用 Base 顶面 ──
lo_b, hi_b = raw_base.min(0), raw_base.max(0)
lo_l, hi_l = raw_leg.min(0),  raw_leg.max(0)
center_x = (lo_b[0] + hi_b[0]) / 2   # base XY 中心 → 对齐无人机 (0,0)
center_y = (lo_b[1] + hi_b[1]) / 2
z_top    = hi_b[2]                    # base 顶面 → 贴无人机底面

print(f"  Base 原始包围盒: X[{lo_b[0]:.1f}, {hi_b[0]:.1f}]  Y[{lo_b[1]:.1f}, {hi_b[1]:.1f}]  Z[{lo_b[2]:.1f}, {hi_b[2]:.1f}] mm")
print(f"  Base XY 中心: ({center_x:.1f}, {center_y:.1f}) mm → 对齐到无人机 (0,0)")
print(f"  Base Z顶: {z_top:.1f} mm → 对齐到 Z={-DRONE_BODY_HALF_H:.3f} m")

# ── 旋转：绕【整体装配中心】旋转，保留两者相对位置 ──
assembly_pivot = np.array([center_x, center_y, z_top])
base_rot = rotate_around_pivot(raw_base, BASE_ROT, assembly_pivot)
leg_rot  = rotate_around_pivot(raw_leg,  LEG_ROT,  assembly_pivot)


def assemble(verts_mm):
    """XY 居中 + Z顶面贴无人机底面 + 缩放 + 额外 Z 偏移"""
    v = verts_mm.copy()
    v[:, 0] -= center_x
    v[:, 1] -= center_y
    v[:, 2] -= z_top
    v *= STL_SCALE
    v[:, 2] -= DRONE_BODY_HALF_H
    v[:, 0] += X_OFFSET
    v[:, 1] += Y_OFFSET
    v[:, 2] += Z_OFFSET
    return v


base_verts = assemble(base_rot)
leg_verts  = assemble(leg_rot)

print(f"  Base Z 范围: [{base_verts[:,2].min():.4f}, {base_verts[:,2].max():.4f}] m")
print(f"  Leg  Z 范围: [{leg_verts[:,2].min():.4f},  {leg_verts[:,2].max():.4f}] m")


# ══════════════════════════════════════════════════════
#  Step 2：创建 USD
# ══════════════════════════════════════════════════════
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
stage.SetMetadata("metersPerUnit", 1.0)
stage.SetMetadata("upAxis", "Z")

ROOT = "/QuadHopper"
root = stage.DefinePrim(ROOT, "Xform")
stage.SetDefaultPrim(root)

# Articulation 根挂在外层 Xform，下面两个子 Prim 各自带 RigidBodyAPI
UsdPhysics.ArticulationRootAPI.Apply(root)

# ──────────────────────────────────────────────────────
# Link 1: body (机身 + 4 电机 + base mesh)
# ──────────────────────────────────────────────────────
BODY = f"{ROOT}/body"
body_prim = stage.DefinePrim(BODY, "Xform")
UsdPhysics.RigidBodyAPI.Apply(body_prim)
pb = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
pb.GetLinearDampingAttr().Set(0.05)
pb.GetAngularDampingAttr().Set(0.05)
pb.GetEnableGyroscopicForcesAttr().Set(True)
mb = UsdPhysics.MassAPI.Apply(body_prim)
mb.GetMassAttr().Set(DRONE_MASS - MASS_LEG)
mb.GetDiagonalInertiaAttr().Set(Gf.Vec3f(I_XX, I_YY, I_ZZ))

# 无人机几何（全部挂到 body 下）
visual_cube = stage.DefinePrim(f"{BODY}/drone_body", "Cube")
UsdGeom.Cube(visual_cube).GetSizeAttr().Set(1.0)
UsdGeom.XformCommonAPI(visual_cube).SetScale((0.04, 0.04, 0.012))
UsdGeom.Gprim(visual_cube).GetDisplayColorAttr().Set([(0.2, 0.5, 0.8)])
UsdPhysics.CollisionAPI.Apply(visual_cube)

for i, angle in enumerate([45, -45]):
    arm = stage.DefinePrim(f"{BODY}/drone_arm_{i}", "Cylinder")
    UsdGeom.Cylinder(arm).GetRadiusAttr().Set(0.003)
    UsdGeom.Cylinder(arm).GetHeightAttr().Set(DIAGONAL_M)
    UsdGeom.Cylinder(arm).GetDisplayColorAttr().Set([(0.1, 0.1, 0.1)])
    UsdGeom.XformCommonAPI(arm).SetRotate((0, 90, angle))

for i, pos in enumerate([(-MOTOR_OFFSET, MOTOR_OFFSET, 0), (-MOTOR_OFFSET, -MOTOR_OFFSET, 0),
                          ( MOTOR_OFFSET,-MOTOR_OFFSET, 0), ( MOTOR_OFFSET,  MOTOR_OFFSET, 0)]):
    mp = stage.DefinePrim(f"{BODY}/drone_motor_{i}", "Cylinder")
    UsdGeom.Cylinder(mp).GetRadiusAttr().Set(0.015)
    UsdGeom.Cylinder(mp).GetHeightAttr().Set(0.005)
    UsdGeom.Cylinder(mp).GetDisplayColorAttr().Set([(0.8, 0.1, 0.1) if i in (1,3) else (0.1, 0.1, 0.1)])
    UsdGeom.XformCommonAPI(mp).SetTranslate(pos)

base_mesh = to_usd_mesh(stage, f"{BODY}/base_mesh", base_verts)
add_collision(base_mesh.GetPrim())

# ──────────────────────────────────────────────────────
# Link 2: leg (单独刚体，质量来自 MASS_LEG)
# ──────────────────────────────────────────────────────
LEG = f"{ROOT}/leg"
leg_prim = stage.DefinePrim(LEG, "Xform")
UsdPhysics.RigidBodyAPI.Apply(leg_prim)
pb_leg = PhysxSchema.PhysxRigidBodyAPI.Apply(leg_prim)
pb_leg.GetLinearDampingAttr().Set(0.05)
pb_leg.GetAngularDampingAttr().Set(0.05)
mb_leg = UsdPhysics.MassAPI.Apply(leg_prim)
mb_leg.GetMassAttr().Set(MASS_LEG)
# 不设 inertia/COM：PhysX 自动从 collision mesh 计算

leg_mesh = to_usd_mesh(stage, f"{LEG}/mesh", leg_verts)
add_collision(leg_mesh.GetPrim())

# ──────────────────────────────────────────────────────
# Joint: 腿对机身 Z 向直动弹簧
# ──────────────────────────────────────────────────────
# 关节锚点：放在 body 底面（base 与 leg 的接触面）
JOINT_Z = -DRONE_BODY_HALF_H + Z_OFFSET   # ≈ -0.060 m

joint_path = f"{ROOT}/spring_joint"
joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
joint.CreateBody0Rel().SetTargets([Sdf.Path(BODY)])
joint.CreateBody1Rel().SetTargets([Sdf.Path(LEG)])
joint.CreateAxisAttr().Set("Z")
joint.CreateLowerLimitAttr().Set(JOINT_LOWER)
joint.CreateUpperLimitAttr().Set(JOINT_UPPER)
joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, JOINT_Z))
joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, JOINT_Z))
joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

# 弹簧驱动：4 根并联 100 N/m = 400 N/m
drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
drive.CreateTypeAttr().Set("force")
drive.CreateDampingAttr().Set(SPRING_DAMP)
drive.CreateStiffnessAttr().Set(SPRING_STIFF)
drive.CreateTargetPositionAttr().Set(0.0)   # 静止长度对应 joint pos = 0
drive.CreateMaxForceAttr().Set(1e6)

omni.usd.get_context().save_as_stage(USD_OUT)
print(f"\n=== 完成：{USD_OUT} ===")
simulation_app.close()
