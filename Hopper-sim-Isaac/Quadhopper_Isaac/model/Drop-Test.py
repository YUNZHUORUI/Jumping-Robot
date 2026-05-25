from isaacsim import SimulationApp

# Open Isaac Sim with a window so you can watch the drop.
simulation_app = SimulationApp({"headless": False})

import os

import omni.timeline
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.join(CURRENT_DIR, "DroneAsset_physics_override.usda")

START_HEIGHT_M = 0.35
GROUND_THICKNESS_M = 0.02
SIM_SECONDS = 8.0
STEPS_PER_SECOND = 120


def set_transform(prim, translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(Gf.Vec3d(*translate))
    xform.SetRotate(Gf.Vec3f(*rotate))
    xform.SetScale(Gf.Vec3f(*scale))


if not os.path.exists(ASSET_PATH):
    raise FileNotFoundError(f"Generate DroneAsset.usd first: {ASSET_PATH}")

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

world = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world)

physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
physics_scene.CreateGravityMagnitudeAttr(9.81)

drone = stage.DefinePrim("/World/Drone", "Xform")
drone.GetReferences().AddReference(ASSET_PATH.replace("\\", "/"))
set_transform(drone, translate=(0.0, 0.0, START_HEIGHT_M))

ground = stage.DefinePrim("/World/Ground", "Cube")
UsdGeom.Cube(ground).GetSizeAttr().Set(1.0)
set_transform(
    ground,
    translate=(0.0, 0.0, -0.5 * GROUND_THICKNESS_M),
    scale=(1.2, 1.2, GROUND_THICKNESS_M),
)
UsdGeom.Gprim(ground).GetDisplayColorAttr().Set([(0.45, 0.45, 0.45)])
UsdPhysics.CollisionAPI.Apply(ground)

light = UsdLux.DistantLight.Define(stage, "/World/Sun")
light.CreateIntensityAttr(7000.0)
set_transform(light.GetPrim(), rotate=(45.0, 0.0, 35.0))

fill_light = UsdLux.SphereLight.Define(stage, "/World/FillLight")
fill_light.CreateIntensityAttr(25000.0)
fill_light.CreateRadiusAttr(0.08)
set_transform(fill_light.GetPrim(), translate=(0.25, -0.25, 0.45))

camera = UsdGeom.Camera.Define(stage, "/World/Camera")
set_transform(camera.GetPrim(), translate=(0.45, -0.55, 0.28), rotate=(62.0, 0.0, 40.0))
camera.CreateFocalLengthAttr(24.0)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 1000.0))
stage.GetRootLayer().defaultPrim = "World"
stage.SetMetadata("customLayerData", {"cameraSettings": {"cameraPrim": "/World/Camera"}})

timeline = omni.timeline.get_timeline_interface()
timeline.set_looping(False)
timeline.play()

for _ in range(int(SIM_SECONDS * STEPS_PER_SECOND)):
    simulation_app.update()

timeline.stop()
print("Drop test finished. Inspect the scene, then close the Isaac Sim window.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
