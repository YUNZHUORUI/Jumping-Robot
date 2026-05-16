# 训练脚本：MyQuadcopter-v0 (Isaac Lab + RSL-RL PPO)
# 运行: bash /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac/run_train.sh
#    1024

import argparse
print("[DEBUG 1] argparse imported", flush=True)
from isaaclab.app import AppLauncher
print("[DEBUG 2] AppLauncher imported", flush=True)

parser = argparse.ArgumentParser(description="Train MyQuadcopter with RSL-RL PPO")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 训练时强制禁用渲染，提升速度
args_cli.headless = True
args_cli.rendering_mode = "performance"

print("[DEBUG 3] launching AppLauncher...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[DEBUG 4] AppLauncher started", flush=True)

# ---- 以下 import 必须在 SimulationApp 启动之后 ----
import os
import sys
import torch
from datetime import datetime

# 把 Hopper-sim-Isaac 加入路径，让 Quadhopper_Isaac 可以`作为 package 导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 把 isaaclab_rl 源码路径加入，使 `from isaaclab_rl.rsl_rl import ...` 可用
import isaaclab as _il
_ISAACLAB_SOURCE = os.path.join(os.path.dirname(_il.__file__), "source")
_p = os.path.join(_ISAACLAB_SOURCE, "isaaclab_rl")
if _p not in sys.path:
    sys.path.insert(0, _p)

print("[DEBUG 5] importing Quadhopper_Isaac...", flush=True)
import Quadhopper_Isaac  # 触发 __init__.py，完成 gym.register
print("[DEBUG 6] importing gym / rsl_rl wrappers...", flush=True)
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
print("[DEBUG 7] all imports done", flush=True)

from Quadhopper_Isaac.my_hopper_env import HopperEnvCfg
from Quadhopper_Isaac.rsl_rl_ppo_cfg import HopperPPORunnerCfg


def main():
    env_cfg = HopperEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("myhopper", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = HopperPPORunnerCfg()
    log_dir = os.path.join(
        os.path.dirname(__file__),
        "logs", "rsl_rl", "myhopper",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
