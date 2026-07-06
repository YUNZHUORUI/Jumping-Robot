# quadhopper/train.py
"""
Entry point for training and testing the QuadHopper PPO agent.

Usage:
    python -m quadhopper.train --mode train
    python -m quadhopper.train --mode test
    python -m quadhopper.train --mode test --model path/to/model
"""
import math
import os
import argparse
import sys
from pathlib import Path
from multiprocessing import freeze_support

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

try:
    # Preferred: package execution, e.g. `python -m Quadhopper.train`
    from .config import TRAINING, RENDER
    from .env import QuadhopperTargetEnv
except ImportError:
    # Fallback: direct script execution in IDEs (PyCharm runfile)
    this_dir = Path(__file__).resolve().parent
    package_parent = this_dir.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    from Quadhopper.config import TRAINING, RENDER
    from Quadhopper.env import QuadhopperTargetEnv


def train(cfg=TRAINING):
    """Train a PPO agent on QuadhopperTargetEnv."""
    print(f"Starting training for {cfg.total_timesteps:,} timesteps...")

    vec_env = make_vec_env(
        QuadhopperTargetEnv,
        n_envs=cfg.n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        ent_coef=cfg.ent_coef,
        gamma=cfg.gamma,
        device=cfg.device,
    )

    model.learn(total_timesteps=cfg.total_timesteps)
    model.save(cfg.model_path)
    print(f"Model saved: {cfg.model_path}")
    vec_env.close()


def test(model_path=TRAINING.model_path, render_cfg=RENDER):
    """Run one test episode and produce GIF + analysis plots."""
    try:
        from .renderer import QuadhopperRenderer
    except ImportError:
        from Quadhopper.renderer import QuadhopperRenderer

    print(f"Testing model: {model_path}")

    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
    else:
        print("Model not found. Running untrained agent.")
        model = PPO("MlpPolicy", QuadhopperTargetEnv())

    env      = QuadhopperTargetEnv()
    renderer = QuadhopperRenderer(render_cfg)
    obs, _   = env.reset()

    history = {
        'step': [], 'x': [], 'y': [],
        'target_y': [], 'theta': [],
        'thrust_l': [], 'thrust_r': [],
    }

    for i in range(render_cfg.test_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        com_pos = env.physics.get_com_pos(env.q)
        com_x   = float(com_pos[0])
        traj_y, _ = env.get_trajectory_state(com_x)
        history['step'].append(i)
        history['x'].append(com_x)
        history['y'].append(float(com_pos[1]))
        history['theta'].append(math.degrees(float(env.q[2])))
        history['thrust_l'].append(float(action[0]))
        history['thrust_r'].append(float(action[1]))
        history['target_y'].append(traj_y)

        renderer.maybe_render_frame(i, obs, env, action=action)

        if done or truncated:
            print(
                f"Episode ended at step {i} "
                f"(done={done}, truncated={truncated})"
            )
            break

    renderer.save_gif()
    QuadhopperRenderer.save_analysis_plots(history, render_cfg.plot_path)


def ballistic_test(render_cfg=RENDER, n_steps=500, seed=None):
    """
    Open-loop ballistic test: random initial velocity + angle, zero thrust.
    Verifies that the physics produces correct parabolic arcs before adding RL.
    """
    import copy
    import numpy as np
    try:
        from .renderer import QuadhopperRenderer
    except ImportError:
        from Quadhopper.renderer import QuadhopperRenderer

    rng = np.random.default_rng(seed)

    env = QuadhopperTargetEnv()
    env.reset()

    v0        = float(rng.uniform(4.0, 7.0))
    angle_deg = float(rng.uniform(30.0, 60.0))
    angle_rad = math.radians(angle_deg)
    vx        = v0 * math.cos(angle_rad)
    vy        = v0 * math.sin(angle_rad)
    # Positive theta = forward lean (COM ahead of foot), correct for liftoff.
    theta0    = math.radians(float(rng.uniform(15.0, 30.0)))

    env.q[:]  = [0.0, 1e-3, theta0, env.pcfg.leg_length]
    env.dq[:] = [vx, vy, 0.0, 0.0]
    env.physics.stance_active = False

    # Re-plan from the actual liftoff state so the planned arc is correct and
    # the out-of-bounds guard uses the real landing x rather than targets[0]=0.3m.
    import copy as _copy
    com0 = env.physics.get_com_pos(env.q)
    t_land = 2.0 * vy / env.pcfg.gravity          # same-height time of flight
    x_com_land = float(com0[0]) + vx * t_land
    phi_td = math.radians(env.reward_fn.cfg.phi_td_target_deg)
    x_foot_land = x_com_land - env.pcfg.leg_length * math.sin(phi_td)
    env.ecfg = _copy.copy(env.ecfg)               # don't mutate the singleton
    env.ecfg.targets = np.array([x_foot_land])
    env.planner.plan(float(com0[0]), float(com0[1]), x_foot_land)
    env.current_target_idx = 0

    print(
        f"Ballistic launch: v0={v0:.2f} m/s @ {angle_deg:.1f}°  "
        f"theta0={math.degrees(theta0):.1f}°  vx={vx:.2f}  vy={vy:.2f}  "
        f"target_x={x_foot_land:.2f} m"
    )

    bal_cfg          = copy.copy(render_cfg)
    bal_cfg.gif_path = "ballistic_test.gif"
    renderer         = QuadhopperRenderer(bal_cfg)
    obs              = env._get_obs()
    zero_action      = np.zeros(2, dtype=np.float32)

    history = {
        'step': [], 'x': [], 'y': [],
        'target_y': [], 'theta': [],
        'thrust_l': [], 'thrust_r': [],
    }

    for i in range(n_steps):
        obs, _, done, truncated, _ = env.step(zero_action)
        com = env.physics.get_com_pos(env.q)
        com_x = float(com[0])
        traj_y, _ = env.get_trajectory_state(com_x)
        history['step'].append(i)
        history['x'].append(com_x)
        history['y'].append(float(com[1]))
        history['target_y'].append(traj_y)
        history['theta'].append(math.degrees(float(env.q[2])))
        history['thrust_l'].append(0.0)
        history['thrust_r'].append(0.0)
        renderer.maybe_render_frame(i, obs, env, action=zero_action)
        if done or truncated:
            print(f"Episode ended at step {i}")
            break

    renderer.save_gif()
    QuadhopperRenderer.save_analysis_plots(history, "ballistic_analysis.png")


def main():
    parser = argparse.ArgumentParser(description="QuadHopper RL")
    parser.add_argument(
        "--mode", choices=["train", "test", "ballistic"], default="test",
        help="Run mode: 'train', 'test', or 'ballistic' (physics-only arc check)"
    )
    parser.add_argument(
        "--model", type=str, default=TRAINING.model_path,
        help="Path to model (without .zip)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for ballistic test"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "ballistic":
        ballistic_test(seed=args.seed)
    else:
        test(model_path=args.model)


if __name__ == "__main__":
    freeze_support()
    main()

