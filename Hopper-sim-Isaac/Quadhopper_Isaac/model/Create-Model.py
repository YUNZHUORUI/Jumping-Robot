from isaacsim import SimulationApp
import os
import struct
import numpy as np

# Run Isaac Sim in headless mode to generate the USD asset.
simulation_app = SimulationApp({"headless": True})

import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

# =========================================================
# 1. Basic physical parameters
# =========================================================
# Two rigid bodies: Body gets the drone + fixed jump base mass; SpringLeg is nearly massless.
TOTAL_MASS = 0.166     # kg: measured full jump vehicle mass
LEG_MASS = 0.001      # kg: sliding jump leg + foot
BODY_MASS = TOTAL_MASS - LEG_MASS
BODY_CENTER_OF_MASS = (0.0, 0.0, -0.04)  # m: move body CoM 2 cm lower
LEG_CENTER_OF_MASS = (0.0, 0.0, 0)   # m: move leg CoM 2 cm lower

I_XX, I_YY, I_ZZ = 7.522172e-04 , 8.657694e-04, 1.482076e-03
LEG_I_XX, LEG_I_YY, LEG_I_ZZ = 0.0, 0.0, 0.0

DIAGONAL_M = 0.230
MOTOR_OFFSET = 0.0813

# =========================================================
# 2. CAD placement, travel, and spring parameters
# =========================================================
# Jump Base and Jump Leg come from the same CAD assembly, so they must use one
# shared anchor and the same installation transform to keep the bearing hole and
# sliding cylinder coaxial.
CAD_ROTATE_DEG = (0, -90, 0)
CAD_INSTALL_TRANSLATE_M = (0.0, 0.0, -0.075)

# Move only the sliding leg assembly from the shared CAD pose.
# This makes jump_leg_visual start at Z = -0.1 in the property panel.
LEG_INITIAL_EXTENSION_M = 0.025
BASE_VISUAL_TRANSLATE_M = CAD_INSTALL_TRANSLATE_M
LEG_VISUAL_TRANSLATE_M = (
    CAD_INSTALL_TRANSLATE_M[0],
    CAD_INSTALL_TRANSLATE_M[1],
    CAD_INSTALL_TRANSLATE_M[2] - LEG_INITIAL_EXTENSION_M,
)

# Place the prismatic joint on the same center line as the CAD install.
JOINT_LOCAL_POS_M = CAD_INSTALL_TRANSLATE_M

# Prismatic joint coordinate q:
# q = 0 keeps the STL parts in their shared CAD assembly pose.
# q > 0 moves SpringLeg upward along +Z relative to Body, compressing the leg.
LEG_TRAVEL = 0.043       # m, moves jump_leg_visual from Z = -0.1 up to Z = -0.057
LEG_STIFFNESS = 400.0    # N/m, four parallel 100 N/m springs
LEG_SPRING_PRELOAD_N = 20.0
LEG_SPRING_TARGET_M = -LEG_SPRING_PRELOAD_N / LEG_STIFFNESS
LEG_DAMPING = 0.3        # N*s/m, spring drive damping
LEG_MAX_FORCE = 100.0    # N, drive force limit

FOOT_RADIUS = 0.006      # m
FOOT_EXTRA_DOWN = 4e-3   # m, move the contact sphere below the visual foot
LEG_COLLISION_DIAMETER_SCALE = 0.1
LEG_COLLISION_HEIGHT_M = 0.1

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JUMP_BASE_STL = os.path.join(CURRENT_DIR, "Jump+Base.stl")
JUMP_LEG_STL = os.path.join(CURRENT_DIR, "Jump+Leg.stl")


def read_binary_stl(stl_path: str):
    """Read a binary STL file and return duplicated triangle vertices."""
    with open(stl_path, "rb") as f:
        f.read(80)
        tri_count = struct.unpack("<I", f.read(4))[0]
        vertices = []
        indices = []
        counts = []
        for _ in range(tri_count):
            raw = f.read(50)
            if len(raw) < 50:
                raise RuntimeError(f"Broken STL file: {stl_path}")
            vals = struct.unpack("<12fH", raw)
            tri = vals[3:12]
            base = len(vertices)
            vertices.append((tri[0], tri[1], tri[2]))
            vertices.append((tri[3], tri[4], tri[5]))
            vertices.append((tri[6], tri[7], tri[8]))
            indices += [base, base + 1, base + 2]
            counts.append(3)
    return np.asarray(vertices, dtype=np.float32), counts, indices


def compute_common_anchor(stl_paths):
    """Use one shared CAD anchor so Jump Base and Jump Leg keep their CAD relative position."""
    mins, maxs = [], []
    for path in stl_paths:
        pts, _, _ = read_binary_stl(path)
        mins.append(pts.min(axis=0))
        maxs.append(pts.max(axis=0))
    mn = np.vstack(mins).min(axis=0)
    mx = np.vstack(maxs).max(axis=0)
    return 0.5 * (mn + mx), mn, mx


def add_stl_visual(stage, stl_path, prim_path, anchor_mm, translate_m, rotate_deg, color):
    """Add STL as visual mesh only. Do not use STL mesh for RL collision."""
    pts_mm, counts, indices = read_binary_stl(stl_path)
    pts_m = (pts_mm - anchor_mm) * 0.001  # mm -> m

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in pts_m])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Gprim(mesh.GetPrim()).GetDisplayColorAttr().Set([color])

    xf = UsdGeom.XformCommonAPI(mesh.GetPrim())
    xf.SetTranslate(translate_m)
    xf.SetRotate(rotate_deg)
    return mesh.GetPrim()


def rotation_matrix_xyz(rotate_deg):
    rx, ry, rz = np.radians(np.asarray(rotate_deg, dtype=np.float64))

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def transformed_bounds(stl_path, anchor_mm, translate_m, rotate_deg):
    pts_mm, _, _ = read_binary_stl(stl_path)
    pts_m = (pts_mm - anchor_mm) * 0.001
    rot = rotation_matrix_xyz(rotate_deg)
    transformed = pts_m @ rot.T + np.asarray(translate_m, dtype=np.float64)
    return transformed.min(axis=0), transformed.max(axis=0)


def add_hidden_box_collision(stage, prim_path, bounds_min, bounds_max):
    center = 0.5 * (bounds_min + bounds_max)
    size = np.maximum(bounds_max - bounds_min, 0.001)

    prim = stage.DefinePrim(prim_path, "Cube")
    UsdGeom.Cube(prim).GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
    UsdGeom.XformCommonAPI(prim).SetScale(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def add_hidden_vertical_cylinder_collision(
    stage,
    prim_path,
    bounds_min,
    bounds_max,
    diameter_scale=1.0,
    height_override=None,
):
    center = 0.5 * (bounds_min + bounds_max)
    size = np.maximum(bounds_max - bounds_min, 0.001)
    radius = 0.5 * max(size[0], size[1]) * diameter_scale
    height = size[2] if height_override is None else height_override

    prim = stage.DefinePrim(prim_path, "Cylinder")
    UsdGeom.Cylinder(prim).GetRadiusAttr().Set(float(radius))
    UsdGeom.Cylinder(prim).GetHeightAttr().Set(float(height))
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdGeom.Imageable(prim).MakeInvisible()
    return prim


def add_visual_spring(
    stage,
    prim_path,
    center_xy,
    z_top,
    z_bottom,
    coil_radius=0.003,
    wire_radius=0.00045,
    turns=8,
    color=(0.85, 0.05, 0.05),
    samples_per_turn=16,
    ring_segments=8,
):
    """Add a visual-only vertical helical spring mesh with no collision or force."""
    x0, y0 = center_xy
    total_samples = turns * samples_per_turn + 1

    points = []
    face_counts = []
    face_indices = []

    for i in range(total_samples):
        t = 2.0 * np.pi * turns * i / (total_samples - 1)
        alpha = i / (total_samples - 1)
        z = z_top + alpha * (z_bottom - z_top)
        cx = x0 + coil_radius * np.cos(t)
        cy = y0 + coil_radius * np.sin(t)

        radial = np.array([np.cos(t), np.sin(t), 0.0])
        vertical = np.array([0.0, 0.0, 1.0])

        for j in range(ring_segments):
            beta = 2.0 * np.pi * j / ring_segments
            point = (
                np.array([cx, cy, z])
                + wire_radius * np.cos(beta) * radial
                + wire_radius * np.sin(beta) * vertical
            )
            points.append(Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])))

    for i in range(total_samples - 1):
        for j in range(ring_segments):
            a = i * ring_segments + j
            b = i * ring_segments + (j + 1) % ring_segments
            c = (i + 1) * ring_segments + (j + 1) % ring_segments
            d = (i + 1) * ring_segments + j
            face_counts.append(4)
            face_indices.extend([a, b, c, d])

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Gprim(mesh.GetPrim()).GetDisplayColorAttr().Set([color])
    return mesh.GetPrim()


# =========================================================
# 3. Articulation root
# =========================================================
root_path = "/Drone"
root_prim = stage.DefinePrim(root_path, "Xform")
stage.SetDefaultPrim(root_prim)
UsdPhysics.ArticulationRootAPI.Apply(root_prim)

# =========================================================
# 4. Main body, simple body visuals, arms, and motor markers
# =========================================================
body_path = f"{root_path}/Body"
body_prim = stage.DefinePrim(body_path, "Xform")
UsdGeom.XformCommonAPI(body_prim).SetTranslate((0.0, 0.0, 0.0))

UsdPhysics.RigidBodyAPI.Apply(body_prim)
physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
physx_api.GetLinearDampingAttr().Set(0.05)
physx_api.GetAngularDampingAttr().Set(0.05)
physx_api.GetEnableGyroscopicForcesAttr().Set(True)

mass_api = UsdPhysics.MassAPI.Apply(body_prim)
mass_api.GetMassAttr().Set(BODY_MASS)
mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*BODY_CENTER_OF_MASS))
mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(I_XX, I_YY, I_ZZ))

# Simple body visual and collision proxy.
visual_body_path = f"{body_path}/visual_cube"
visual_cube = stage.DefinePrim(visual_body_path, "Cube")
UsdGeom.Cube(visual_cube).GetSizeAttr().Set(1.0)
UsdGeom.XformCommonAPI(visual_cube).SetScale((0.04, 0.04, 0.012))
UsdGeom.Gprim(visual_cube).GetDisplayColorAttr().Set([(0.2, 0.5, 0.8)])
UsdPhysics.CollisionAPI.Apply(visual_cube)

# Cross arms.
for i, angle in enumerate([45, -45]):
    arm_path = f"{body_path}/arm_{i}"
    arm_prim = stage.DefinePrim(arm_path, "Cylinder")
    UsdGeom.Cylinder(arm_prim).GetRadiusAttr().Set(0.003)
    UsdGeom.Cylinder(arm_prim).GetHeightAttr().Set(DIAGONAL_M)
    UsdGeom.Cylinder(arm_prim).GetDisplayColorAttr().Set([(0.1, 0.1, 0.1)])
    UsdGeom.XformCommonAPI(arm_prim).SetRotate((0, 90, angle))

# Motor marker positions in the body frame.
motor_pos = [
    (-MOTOR_OFFSET, MOTOR_OFFSET, 0),
    (-MOTOR_OFFSET, -MOTOR_OFFSET, 0),
    (MOTOR_OFFSET, -MOTOR_OFFSET, 0),
    (MOTOR_OFFSET, MOTOR_OFFSET, 0),
]

for i, pos in enumerate(motor_pos):
    m_path = f"{body_path}/motor_{i}"
    m_prim = stage.DefinePrim(m_path, "Cylinder")
    UsdGeom.Cylinder(m_prim).GetRadiusAttr().Set(0.015)
    UsdGeom.Cylinder(m_prim).GetHeightAttr().Set(0.005)
    color = (0.8, 0.1, 0.1) if i in [1, 3] else (0.1, 0.1, 0.1)
    UsdGeom.Cylinder(m_prim).GetDisplayColorAttr().Set([color])
    UsdGeom.XformCommonAPI(m_prim).SetTranslate(pos)

# =========================================================
# 5. STL visuals: fixed jump base belongs to Body
# =========================================================
anchor_mm, bounds_min, bounds_max = compute_common_anchor([JUMP_BASE_STL, JUMP_LEG_STL])
print(f"STL bounds min(mm): {bounds_min}, max(mm): {bounds_max}, anchor(mm): {anchor_mm}")
base_bounds_min, base_bounds_max = transformed_bounds(
    JUMP_BASE_STL,
    anchor_mm,
    BASE_VISUAL_TRANSLATE_M,
    CAD_ROTATE_DEG,
)
leg_bounds_min, leg_bounds_max = transformed_bounds(
    JUMP_LEG_STL,
    anchor_mm,
    LEG_VISUAL_TRANSLATE_M,
    CAD_ROTATE_DEG,
)

# Jump Base is fixed to the drone body.
add_stl_visual(
    stage,
    JUMP_BASE_STL,
    f"{body_path}/jump_base_visual",
    anchor_mm,
    BASE_VISUAL_TRANSLATE_M,
    CAD_ROTATE_DEG,
    (0.05, 0.05, 0.05),
)
base_collision = add_hidden_box_collision(
    stage,
    f"{body_path}/jump_base_collision",
    base_bounds_min,
    base_bounds_max,
)

# =========================================================
# 6. Sliding spring leg body and foot contact proxy
# =========================================================
leg_path = f"{root_path}/SpringLeg"
leg_prim = stage.DefinePrim(leg_path, "Xform")
UsdGeom.XformCommonAPI(leg_prim).SetTranslate((0.0, 0.0, 0.0))

UsdPhysics.RigidBodyAPI.Apply(leg_prim)
leg_physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(leg_prim)
leg_physx_api.GetLinearDampingAttr().Set(0.01)
leg_physx_api.GetAngularDampingAttr().Set(0.01)

leg_mass_api = UsdPhysics.MassAPI.Apply(leg_prim)
leg_mass_api.GetMassAttr().Set(LEG_MASS)
leg_mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*LEG_CENTER_OF_MASS))
leg_mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(LEG_I_XX, LEG_I_YY, LEG_I_ZZ))

# Jump Leg belongs to SpringLeg. At q = 0, it keeps the same CAD assembly pose
# as Jump Base. LEG_INITIAL_EXTENSION_M is only an optional extra visual offset.
add_stl_visual(
    stage,
    JUMP_LEG_STL,
    f"{leg_path}/jump_leg_visual",
    anchor_mm,
    LEG_VISUAL_TRANSLATE_M,
    CAD_ROTATE_DEG,
    (0.9, 0.75, 0.1),
)
leg_collision = add_hidden_vertical_cylinder_collision(
    stage,
    f"{leg_path}/jump_leg_collision",
    leg_bounds_min,
    leg_bounds_max,
    LEG_COLLISION_DIAMETER_SCALE,
    LEG_COLLISION_HEIGHT_M,
)

# Hidden contact sphere for the physical foot-ground contact.
foot_col = stage.DefinePrim(f"{leg_path}/foot_collision", "Sphere")
UsdGeom.Sphere(foot_col).GetRadiusAttr().Set(FOOT_RADIUS)
foot_center_xy = 0.5 * (leg_bounds_min[:2] + leg_bounds_max[:2])
foot_local_z = float(leg_bounds_min[2] + FOOT_RADIUS - FOOT_EXTRA_DOWN)
UsdGeom.XformCommonAPI(foot_col).SetTranslate((float(foot_center_xy[0]), float(foot_center_xy[1]), foot_local_z))
UsdGeom.Gprim(foot_col).GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
UsdPhysics.CollisionAPI.Apply(foot_col)

# The bearing guide and sliding leg overlap by design, so their internal
# collision is filtered. The prismatic joint is what enforces the fit.
UsdPhysics.FilteredPairsAPI.Apply(base_collision).CreateFilteredPairsRel().SetTargets(
    [leg_collision.GetPath(), foot_col.GetPath()]
)


# =========================================================
# 7. Center prismatic joint and spring drive
# =========================================================
joint_path = f"{root_path}/center_spring_joint"
joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
joint.CreateBody0Rel().SetTargets([body_prim.GetPath()])
joint.CreateBody1Rel().SetTargets([leg_prim.GetPath()])

# Axis is expressed in Body0 coordinates. q > 0 means SpringLeg moves upward.
joint.CreateAxisAttr("Z")
joint.CreateLowerLimitAttr(0.0)
joint.CreateUpperLimitAttr(LEG_TRAVEL)

joint.CreateLocalPos0Attr(Gf.Vec3f(*JOINT_LOCAL_POS_M))
joint.CreateLocalPos1Attr(Gf.Vec3f(*JOINT_LOCAL_POS_M))

# The linear drive pulls the spring leg back toward q = 0.
drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
drive_api.CreateTargetPositionAttr(LEG_SPRING_TARGET_M)
drive_api.CreateStiffnessAttr(LEG_STIFFNESS)
drive_api.CreateDampingAttr(LEG_DAMPING)
drive_api.CreateMaxForceAttr(LEG_MAX_FORCE)

# =========================================================
# 8. Save generated asset
# =========================================================
asset_path = os.path.join(CURRENT_DIR, "DroneAsset.usd")
omni.usd.get_context().save_as_stage(asset_path)
print(f">>> Spring-leg DroneAsset.usd saved to: {asset_path}")
print(">>> Expected bodies: Body, SpringLeg")
print(">>> Expected joint: center_spring_joint")

simulation_app.close()
