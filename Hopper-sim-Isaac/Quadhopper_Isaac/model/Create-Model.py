from isaacsim import SimulationApp
import json
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
# Two rigid bodies: Body carries the measured full-system mass/inertia. The
# lower SpringLeg keeps a small numerical stabilizer mass for PhysX contact and
# joint solving. The measured full-system inertia is still carried by Body.
TOTAL_MASS = 0.183    # kg: measured full jump vehicle mass
LEG_MASS = 0.0100      # kg: numerical stabilizer, not a measured physical mass
BODY_MASS = TOTAL_MASS - LEG_MASS
BODY_CENTER_OF_MASS = (0.0, 0.0, 0)  # m: move body CoM 2 cm lower
LEG_CENTER_OF_MASS = (0.0, 0.0, 0)   # m: move leg CoM 2 cm lower

I_XX, I_YY, I_ZZ = 1.231252e-03, 1.286169e-03, 2.305957e-03
LEG_I_XX, LEG_I_YY, LEG_I_ZZ = 1e-5, 1e-5, 1e-5

DIAGONAL_M = 0.230
MOTOR_OFFSET = 0.0813

# =========================================================
# 2. CAD placement, travel, and spring parameters
# =========================================================
# Jump Base and Jump Leg come from the same CAD assembly, so they must use one
# shared anchor and the same installation transform to keep the bearing hole and
# sliding cylinder coaxial.
CAD_ROTATE_DEG = (0, -90, 90)
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
# q = 0 keeps the STL parts in their shared CAD assembly pose and gives the
# extension springs their minimum 3 cm length. q > 0 stretches the springs.
LEG_TRAVEL = 0.08       # m, maximum extension from the 3 cm spring length
LEG_STIFFNESS = 604.0   # N/m, four extension springs in total
LEG_SPRING_PRELOAD_N = 0.0
LEG_SPRING_TARGET_M = -LEG_SPRING_PRELOAD_N / LEG_STIFFNESS
LEG_DAMPING = 2       # N*s/m, spring drive damping
LEG_MAX_FORCE = 200.0    # N, drive force limit
LEG_MAX_JOINT_VELOCITY = 12.0  # m/s, keep headroom above 1 m drop impact speed

MESH_COLLISION_APPROXIMATION = "convexHull"
CONTACT_OFFSET = 0.002
REST_OFFSET = 0.0
FOOT_CONTACT_OFFSET = 0.010

# The new leg STL does not include the real-world long rod. Model it explicitly
# so the drop test contacts the foot/rod instead of the base.
ROD_LENGTH_M = 0.14
ROD_RADIUS_M = 0.003
ROD_INSERTION_OVERLAP_M = 0.002

FOOT_RADIUS = 0.012

SPRING_MIN_LENGTH_M = 0.03
SPRING_AXIS_INSET_X_M = 0.006
SPRING_AXIS_INSET_Y_M = 0.003
SPRING_TOP_INSET_M = 0.002
SPRING_COIL_RADIUS_M = 0.0025
SPRING_WIRE_RADIUS_M = 0.00035
SPRING_VISUAL_TURNS = 7

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = CURRENT_DIR
JUMP_BASE_STL = os.path.join(ASSET_DIR, "Jump+Base.stl")
JUMP_LEG_STL = os.path.join(ASSET_DIR, "Jump+Leg.stl")


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
    """Add STL as a mesh. Collision can be enabled separately."""
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


def enable_collision(prim):
    """Enable contact on a prim and set small offsets for thin robot parts."""
    UsdPhysics.CollisionAPI.Apply(prim)
    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision_api.GetContactOffsetAttr().Set(CONTACT_OFFSET)
    physx_collision_api.GetRestOffsetAttr().Set(REST_OFFSET)
    return prim


def enable_mesh_collision(mesh_prim, approximation=MESH_COLLISION_APPROXIMATION):
    """Use PhysX mesh cooking for STL parts instead of a coarse bounding box."""
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_collision_api.GetApproximationAttr().Set(approximation)
    return enable_collision(mesh_prim)


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


def add_vertical_cylinder(stage, prim_path, center_xy, z_top, z_bottom, radius, color, collision=True, visible=True):
    """Add a vertical cylinder in the local frame of its parent body."""
    height = abs(z_top - z_bottom)
    z_center = 0.5 * (z_top + z_bottom)
    prim = stage.DefinePrim(prim_path, "Cylinder")
    UsdGeom.Cylinder(prim).GetRadiusAttr().Set(float(radius))
    UsdGeom.Cylinder(prim).GetHeightAttr().Set(float(height))
    UsdGeom.XformCommonAPI(prim).SetTranslate(
        Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(z_center))
    )
    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])
    if collision:
        enable_collision(prim)
    if not visible:
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
physx_api.CreateEnableCCDAttr(True)
physx_api.CreateEnableSpeculativeCCDAttr(True)

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
enable_collision(visual_cube)

# Cross arms.
for i, angle in enumerate([45, -45]):
    arm_path = f"{body_path}/arm_{i}"
    arm_prim = stage.DefinePrim(arm_path, "Cylinder")
    UsdGeom.Cylinder(arm_prim).GetRadiusAttr().Set(0.003)
    UsdGeom.Cylinder(arm_prim).GetHeightAttr().Set(DIAGONAL_M)
    UsdGeom.Cylinder(arm_prim).GetDisplayColorAttr().Set([(0.1, 0.1, 0.1)])
    UsdGeom.XformCommonAPI(arm_prim).SetRotate((0, 90, angle))
    enable_collision(arm_prim)

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
    enable_collision(m_prim)

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

# Jump Base is fixed to the drone body. The STL mesh is also the collision
# source, cooked as a convex hull for stable dynamic rigid-body contact.
base_visual = add_stl_visual(
    stage,
    JUMP_BASE_STL,
    f"{body_path}/jump_base_visual",
    anchor_mm,
    BASE_VISUAL_TRANSLATE_M,
    CAD_ROTATE_DEG,
    (0.05, 0.05, 0.05),
)
base_collision = enable_mesh_collision(base_visual)

# =========================================================
# 6. Sliding spring leg body and foot contact proxy
# =========================================================
rod_center_xy = 0.5 * (leg_bounds_min[:2] + leg_bounds_max[:2])
rod_top_z = float(leg_bounds_min[2] + ROD_INSERTION_OVERLAP_M)
rod_bottom_z = rod_top_z - ROD_LENGTH_M
foot_center_xy = rod_center_xy
foot_local_z = float(rod_bottom_z)
foot_bottom_z = foot_local_z - FOOT_RADIUS
spring_leg_origin_m = np.array(
    [float(foot_center_xy[0]), float(foot_center_xy[1]), foot_local_z],
    dtype=np.float64,
)
leg_visual_translate_rel_m = tuple(
    float(v) for v in (np.asarray(LEG_VISUAL_TRANSLATE_M, dtype=np.float64) - spring_leg_origin_m)
)
rod_center_xy_rel = (
    float(rod_center_xy[0] - spring_leg_origin_m[0]),
    float(rod_center_xy[1] - spring_leg_origin_m[1]),
)

leg_path = f"{root_path}/SpringLeg"
leg_prim = stage.DefinePrim(leg_path, "Sphere")
UsdGeom.Sphere(leg_prim).GetRadiusAttr().Set(FOOT_RADIUS)
UsdGeom.XformCommonAPI(leg_prim).SetTranslate(
    (float(spring_leg_origin_m[0]), float(spring_leg_origin_m[1]), float(spring_leg_origin_m[2]))
)
UsdGeom.Gprim(leg_prim).GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])

UsdPhysics.RigidBodyAPI.Apply(leg_prim)
leg_physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(leg_prim)
leg_physx_api.GetLinearDampingAttr().Set(0.01)
leg_physx_api.GetAngularDampingAttr().Set(0.01)
leg_physx_api.CreateEnableCCDAttr(True)
leg_physx_api.CreateEnableSpeculativeCCDAttr(True)
enable_collision(leg_prim)
PhysxSchema.PhysxCollisionAPI.Apply(leg_prim).GetContactOffsetAttr().Set(FOOT_CONTACT_OFFSET)

leg_mass_api = UsdPhysics.MassAPI.Apply(leg_prim)
leg_mass_api.GetMassAttr().Set(LEG_MASS)
leg_mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*LEG_CENTER_OF_MASS))
leg_mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(LEG_I_XX, LEG_I_YY, LEG_I_ZZ))

# Jump Leg belongs to SpringLeg. At q = 0, it keeps the same CAD assembly pose
# as Jump Base. LEG_INITIAL_EXTENSION_M is only an optional extra visual offset.
leg_visual = add_stl_visual(
    stage,
    JUMP_LEG_STL,
    f"{leg_path}/jump_leg_visual",
    anchor_mm,
    leg_visual_translate_rel_m,
    CAD_ROTATE_DEG,
    (0.9, 0.75, 0.1),
)
leg_collision = enable_mesh_collision(leg_visual)

# The real 14 cm rod is not in the STL, so it is an explicit moving part on
# SpringLeg. It overlaps the printed leg slightly to avoid a contact gap.
rod_collision = add_vertical_cylinder(
    stage,
    f"{leg_path}/real_rod_14cm",
    rod_center_xy_rel,
    rod_top_z - foot_local_z,
    rod_bottom_z - foot_local_z,
    ROD_RADIUS_M,
    (0.62, 0.62, 0.62),
)

# Four visual springs are placed around the guide axes shown in the reference
# photo. The physical spring force remains the center prismatic drive below.
spring_axis_x = max(
    0.001,
    0.5 * min(base_bounds_max[0] - base_bounds_min[0], leg_bounds_max[0] - leg_bounds_min[0])
    - SPRING_AXIS_INSET_X_M,
)
spring_axis_y = max(
    0.001,
    0.5 * min(base_bounds_max[1] - base_bounds_min[1], leg_bounds_max[1] - leg_bounds_min[1])
    - SPRING_AXIS_INSET_Y_M,
)
spring_top_z = float(leg_bounds_max[2] - SPRING_TOP_INSET_M)
spring_bottom_z = float(spring_top_z - SPRING_MIN_LENGTH_M)
spring_visuals = []
for i, (sx, sy) in enumerate(
    [
        (-spring_axis_x, -spring_axis_y),
        (-spring_axis_x, spring_axis_y),
        (spring_axis_x, -spring_axis_y),
        (spring_axis_x, spring_axis_y),
    ]
):
    spring_path = f"{body_path}/visual_spring_{i}"
    add_visual_spring(
        stage,
        spring_path,
        (sx, sy),
        spring_top_z,
        spring_bottom_z,
        coil_radius=SPRING_COIL_RADIUS_M,
        wire_radius=SPRING_WIRE_RADIUS_M,
        turns=SPRING_VISUAL_TURNS,
        color=(0.08, 0.08, 0.08),
    )
    spring_visuals.append(
        {
            "path": spring_path,
            "center_xy_m": [float(sx), float(sy)],
            "top_z_m": float(spring_top_z),
            "bottom_z_m": float(spring_bottom_z),
        }
    )

# The bearing guide and sliding leg overlap by design, so their internal
# collision is filtered. The prismatic joint is what enforces the fit.
UsdPhysics.FilteredPairsAPI.Apply(base_collision).CreateFilteredPairsRel().SetTargets(
    [leg_collision.GetPath(), rod_collision.GetPath(), leg_prim.GetPath()]
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
joint_local_pos1_m = tuple(
    float(v) for v in (np.asarray(JOINT_LOCAL_POS_M, dtype=np.float64) - spring_leg_origin_m)
)
joint.CreateLocalPos1Attr(Gf.Vec3f(*joint_local_pos1_m))

# The linear drive pulls the spring leg back toward q = 0.
drive_api = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
drive_api.CreateTypeAttr("force")
drive_api.CreateTargetPositionAttr(LEG_SPRING_TARGET_M)
drive_api.CreateStiffnessAttr(LEG_STIFFNESS)
drive_api.CreateDampingAttr(LEG_DAMPING)
drive_api.CreateMaxForceAttr(LEG_MAX_FORCE)
physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
physx_joint_api.CreateMaxJointVelocityAttr(LEG_MAX_JOINT_VELOCITY)

# Store key dimensions in the USD so Drop-Test.py can print and visualize them.
static_body_origin_height_m = -foot_bottom_z
static_body_top_height_m = static_body_origin_height_m + 0.006
static_base_bottom_height_m = static_body_origin_height_m + float(base_bounds_min[2])
static_base_top_height_m = static_body_origin_height_m + float(base_bounds_max[2])
static_leg_top_height_m = static_body_origin_height_m + float(leg_bounds_max[2])

measurements = {
    "mesh_files": {
        "base": os.path.basename(JUMP_BASE_STL),
        "leg": os.path.basename(JUMP_LEG_STL),
    },
    "cad_rotate_deg": [float(v) for v in CAD_ROTATE_DEG],
    "base_bounds_m": {
        "min": [float(v) for v in base_bounds_min],
        "max": [float(v) for v in base_bounds_max],
        "size": [float(v) for v in base_bounds_max - base_bounds_min],
    },
    "leg_bounds_m": {
        "min": [float(v) for v in leg_bounds_min],
        "max": [float(v) for v in leg_bounds_max],
        "size": [float(v) for v in leg_bounds_max - leg_bounds_min],
    },
    "rod": {
        "length_m": float(ROD_LENGTH_M),
        "radius_m": float(ROD_RADIUS_M),
        "top_z_m": float(rod_top_z),
        "bottom_z_m": float(rod_bottom_z),
    },
    "foot": {
        "radius_m": float(FOOT_RADIUS),
        "center_m": [float(foot_center_xy[0]), float(foot_center_xy[1]), float(foot_local_z)],
        "bottom_z_m": float(foot_bottom_z),
        "rod_tip_at_sphere_center": True,
        "spring_leg_origin_at_foot_center": True,
    },
    "visual_springs": {
        "minimum_length_m": float(SPRING_MIN_LENGTH_M),
        "count": len(spring_visuals),
        "force_free_at_q0": True,
        "springs": spring_visuals,
    },
    "static_on_ground": {
        "body_origin_height_m": float(static_body_origin_height_m),
        "body_visual_top_height_m": float(static_body_top_height_m),
        "base_bottom_height_m": float(static_base_bottom_height_m),
        "base_top_height_m": float(static_base_top_height_m),
        "leg_top_height_m": float(static_leg_top_height_m),
        "foot_center_height_m": float(FOOT_RADIUS),
        "foot_bottom_height_m": 0.0,
    },
    "joint": {
        "local_pos_m": [float(v) for v in JOINT_LOCAL_POS_M],
        "local_pos0_m": [float(v) for v in JOINT_LOCAL_POS_M],
        "local_pos1_m": [float(v) for v in joint_local_pos1_m],
        "travel_m": float(LEG_TRAVEL),
        "stiffness_n_per_m": float(LEG_STIFFNESS),
        "damping_n_s_per_m": float(LEG_DAMPING),
        "max_force_n": float(LEG_MAX_FORCE),
        "max_velocity_m_per_s": float(LEG_MAX_JOINT_VELOCITY),
        "preload_n": float(LEG_SPRING_PRELOAD_N),
    },
    "mass": {
        "total_kg": float(TOTAL_MASS),
        "body_kg": float(BODY_MASS),
        "spring_leg_kg": float(LEG_MASS),
        "mass_inertia_lumped_on_body": True,
    },
    "collision": {
        "stl_mesh_approximation": MESH_COLLISION_APPROXIMATION,
        "contact_offset_m": float(CONTACT_OFFSET),
        "foot_contact_offset_m": float(FOOT_CONTACT_OFFSET),
        "rest_offset_m": float(REST_OFFSET),
        "colliders": [
            "/Drone/Body/visual_cube",
            "/Drone/Body/arm_0",
            "/Drone/Body/arm_1",
            "/Drone/Body/motor_0",
            "/Drone/Body/motor_1",
            "/Drone/Body/motor_2",
            "/Drone/Body/motor_3",
            "/Drone/Body/jump_base_visual",
            "/Drone/SpringLeg",
            "/Drone/SpringLeg/jump_leg_visual",
            "/Drone/SpringLeg/real_rod_14cm",
        ],
        "filtered_pairs_from_jump_base": [
            "/Drone/SpringLeg",
            "/Drone/SpringLeg/jump_leg_visual",
            "/Drone/SpringLeg/real_rod_14cm",
        ],
    },
}
root_prim.SetCustomDataByKey("hopper_measurements_json", json.dumps(measurements, sort_keys=True))

# =========================================================
# 8. Save generated asset
# =========================================================
asset_path = os.path.join(CURRENT_DIR, "HopperAsset.usd")
omni.usd.get_context().save_as_stage(asset_path)
print(f">>> Spring-leg HopperAsset.usd saved to: {asset_path}")
print(">>> Expected bodies: Body, SpringLeg")
print(">>> Expected joint: center_spring_joint")
print(">>> Main dimensions:")
print(f"    Base STL size (m): {(base_bounds_max - base_bounds_min).round(4)}")
print(f"    Leg STL size  (m): {(leg_bounds_max - leg_bounds_min).round(4)}")
print(f"    Real rod length (m): {ROD_LENGTH_M:.4f}")
print(f"    Rod tip / foot sphere center z (m): {foot_local_z:.4f}")
print(f"    SpringLeg bottom at q=0 (m): {foot_bottom_z:.4f}")
print(f"    Joint travel (m): {LEG_TRAVEL:.4f}")
print(f"    Spring minimum length (m): {SPRING_MIN_LENGTH_M:.4f}")
print(f"    Spring total stiffness (N/m): {LEG_STIFFNESS:.1f}")
print(f"    Spring drive damping (N*s/m): {LEG_DAMPING:.1f}")
print(f"    Joint max velocity (m/s): {LEG_MAX_JOINT_VELOCITY:.2f}")
print(f"    Spring force at q=0 natural length (N): {LEG_SPRING_PRELOAD_N:.4f}")
print(f"    Body mass carrying measured system mass (kg): {BODY_MASS:.4f}")
print(f"    SpringLeg numerical mass only (kg): {LEG_MASS:.6f}")
print(f"    Static body origin height above ground (m): {static_body_origin_height_m:.4f}")
print(f"    Static body visual top height above ground (m): {static_body_top_height_m:.4f}")
print(f"    Static base bottom height above ground (m): {static_base_bottom_height_m:.4f}")
print(f"    Visual springs: {len(spring_visuals)}")

simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
