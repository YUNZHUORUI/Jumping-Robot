"""Minimal machine-level diagnostic: stock Isaac Lab Cartpole on GPU.

If this asserts/hangs, the GPU/driver/PhysX install is broken independent of
the quadhopper project.  If it steps cleanly, the problem is project-specific.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg  # noqa: E402

cfg = CartpoleEnvCfg()
cfg.scene.num_envs = 4
cfg.sim.device = "cuda:0"
env = ManagerBasedRLEnv(cfg)
print("[DIAG] env created", flush=True)
obs, _ = env.reset()
obs_t = obs["policy"] if isinstance(obs, dict) else obs
print(f"[DIAG] reset OK, obs shape {tuple(obs_t.shape)}", flush=True)
for i in range(100):
    obs, rew, term, trunc, extras = env.step(torch.zeros(4, env.action_space.shape[1], device="cuda:0"))
    if i % 25 == 0:
        print(f"[DIAG] step {i} OK rew={rew.mean().item():.4f}", flush=True)
print("[DIAG] ALL OK - machine/GPU/PhysX healthy", flush=True)
env.close()
simulation_app.close()
