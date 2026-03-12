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
from multiprocessing import freeze_support

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from .config import TRAINING, RENDER
from .env import QuadhopperTargetEnv
from .renderer import QuadhopperRenderer


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

        history['step'].append(i)
        history['x'].append(float(obs[0]))
        history['y'].append(float(obs[1]))
        history['theta'].append(math.degrees(float(obs[2])))
        history['thrust_l'].append(float(action[0]))
        history['thrust_r'].append(float(action[1]))
        traj_y, _ = env.get_trajectory_state(float(obs[0]))
        history['target_y'].append(traj_y)

        renderer.maybe_render_frame(i, obs, env)

        if done or truncated:
            print(
                f"Episode ended at step {i} "
                f"(done={done}, truncated={truncated})"
            )
            break

    renderer.save_gif()
    QuadhopperRenderer.save_analysis_plots(history, render_cfg.plot_path)


def main():
    parser = argparse.ArgumentParser(description="QuadHopper RL")
    parser.add_argument(
        "--mode", choices=["train", "test"], default="test",
        help="Run mode: 'train' or 'test'"
    )
    parser.add_argument(
        "--model", type=str, default=TRAINING.model_path,
        help="Path to model (without .zip)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        test(model_path=args.model)


if __name__ == "__main__":
    freeze_support()
    main()
