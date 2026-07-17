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
import copy
from pathlib import Path
from multiprocessing import freeze_support

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

try:
    # Preferred: package execution, e.g. `python -m Quadhopper.train`
    from .config import ENV, TRAINING, RENDER, make_even_targets
    from .env import QuadhopperTargetEnv
except ImportError:
    # Fallback: direct script execution in IDEs (PyCharm runfile)
    this_dir = Path(__file__).resolve().parent
    package_parent = this_dir.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    from Quadhopper.config import ENV, TRAINING, RENDER, make_even_targets
    from Quadhopper.env import QuadhopperTargetEnv


def make_target_env(target_count=None):
    env_cfg = copy.copy(ENV)
    if target_count is not None:
        env_cfg.target_count = int(target_count)
        env_cfg.targets = make_even_targets(
            count=env_cfg.target_count,
            spacing=env_cfg.target_spacing,
        )
    return QuadhopperTargetEnv(env_cfg=env_cfg)


def train(cfg=TRAINING, target_count=None, model_path=None, resume_model=None, timesteps=None):
    """Train a PPO agent on QuadhopperTargetEnv."""
    target_count = cfg.target_count if target_count is None else int(target_count)
    total_timesteps = cfg.total_timesteps if timesteps is None else int(timesteps)
    model_path = cfg.model_path if model_path is None else model_path
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"Training target count: {target_count}")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(
        lambda: make_target_env(target_count),
        n_envs=cfg.n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    if resume_model is not None:
        print(f"Resuming from: {resume_model}")
        model = PPO.load(resume_model, env=vec_env, device=cfg.device)
    else:
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

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"Model saved: {model_path}")
    vec_env.close()


def test(model_path=TRAINING.model_path, render_cfg=RENDER, target_count=None):
    """Run one test episode and produce GIF + analysis plots."""
    try:
        from .renderer import QuadhopperRenderer
    except ImportError:
        from Quadhopper.renderer import QuadhopperRenderer

    print(f"Testing model: {model_path}")

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip. "
            "Train first with `python -m Quadhopper.train --mode train`."
        )
    model = PPO.load(model_path)

    target_count = TRAINING.target_count if target_count is None else int(target_count)
    env      = make_target_env(target_count)
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
        motor_cmd = env.last_motor_cmd

        com_pos = env.physics.get_com_pos(env.q)
        com_x   = float(com_pos[0])
        traj_y, _ = env.get_trajectory_state(com_x)
        history['step'].append(i)
        history['x'].append(com_x)
        history['y'].append(float(com_pos[1]))
        history['theta'].append(math.degrees(float(env.q[2])))
        history['thrust_l'].append(float(motor_cmd[0]))
        history['thrust_r'].append(float(motor_cmd[1]))
        history['target_y'].append(traj_y)

        renderer.maybe_render_frame(i, obs, env, action=motor_cmd)

        if done or truncated:
            print(
                f"Episode ended at step {i} "
                f"(done={done}, truncated={truncated})"
            )
            break

    renderer.save_gif()
    QuadhopperRenderer.save_analysis_plots(history, render_cfg.plot_path)


def evaluate(model_path=TRAINING.model_path, episodes=20, seed=0, target_count=None):
    """Evaluate target progression without rendering."""
    import numpy as np

    print(f"Evaluating model: {model_path}")
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip. "
            "Train first with `python -m Quadhopper.train --mode train`."
        )
    model = PPO.load(model_path)
    hits = []
    final_errors = []
    target_count = TRAINING.target_count if target_count is None else int(target_count)

    for ep in range(episodes):
        env = make_target_env(target_count)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)

        target_idx = env.current_target_idx
        foot_x = float(env.physics.get_foot_pos(env.q)[0])
        if target_idx < len(env.ecfg.targets):
            err = abs(foot_x - float(env.ecfg.targets[target_idx]))
        else:
            err = 0.0
        hits.append(target_idx)
        final_errors.append(err)

    print(
        f"episodes={episodes}  "
        f"mean_targets={np.mean(hits):.2f}/{len(env.ecfg.targets)}  "
        f"max_targets={max(hits)}/{len(env.ecfg.targets)}  "
        f"mean_final_error={np.mean(final_errors):.3f} m"
    )


def _motor_to_action(left: float, right: float):
    motor = np.array([left, right], dtype=np.float32)
    return 2.0 * np.clip(motor, 0.0, 1.0) - 1.0


def demo(render_cfg=RENDER, seed=0):
    """Render a hand-tuned open-loop hopping baseline for debugging."""
    import copy
    import numpy as np
    try:
        from .renderer import QuadhopperRenderer
    except ImportError:
        from Quadhopper.renderer import QuadhopperRenderer

    env = QuadhopperTargetEnv()
    obs, _ = env.reset(seed=seed)

    demo_cfg = copy.copy(render_cfg)
    demo_cfg.gif_path = "artifacts/renders/quadhopper_open_loop_demo.gif"
    demo_cfg.plot_path = "artifacts/renders/thrust_analysis_open_loop_demo.png"
    renderer = QuadhopperRenderer(demo_cfg)

    history = {
        'step': [], 'x': [], 'y': [],
        'target_y': [], 'theta': [],
        'thrust_l': [], 'thrust_r': [],
    }

    burn_steps = 50
    stance_clock = 0
    was_touching = env.physics.stance_active

    for i in range(render_cfg.test_steps):
        touching = env.physics.stance_active
        if touching and not was_touching:
            stance_clock = 0
        if touching:
            stance_clock += 1

        if touching and stance_clock <= burn_steps:
            motor_cmd = np.array([0.90, 1.00], dtype=np.float32)
        else:
            motor_cmd = np.zeros(2, dtype=np.float32)

        action = _motor_to_action(float(motor_cmd[0]), float(motor_cmd[1]))
        obs, reward, done, truncated, info = env.step(action)
        motor_cmd = env.last_motor_cmd

        com_pos = env.physics.get_com_pos(env.q)
        com_x = float(com_pos[0])
        traj_y, _ = env.get_trajectory_state(com_x)
        history['step'].append(i)
        history['x'].append(com_x)
        history['y'].append(float(com_pos[1]))
        history['theta'].append(math.degrees(float(env.q[2])))
        history['thrust_l'].append(float(motor_cmd[0]))
        history['thrust_r'].append(float(motor_cmd[1]))
        history['target_y'].append(traj_y)

        renderer.maybe_render_frame(i, obs, env, action=motor_cmd)

        was_touching = touching
        if done or truncated:
            print(
                f"Demo ended at step {i} "
                f"(done={done}, truncated={truncated}, targets={env.current_target_idx}/{len(env.ecfg.targets)})"
            )
            break

    renderer.save_gif()
    QuadhopperRenderer.save_analysis_plots(history, demo_cfg.plot_path)


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
    bal_cfg.gif_path = "artifacts/renders/ballistic_test.gif"
    renderer         = QuadhopperRenderer(bal_cfg)
    obs              = env._get_obs()
    zero_action      = -np.ones(2, dtype=np.float32)

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
        renderer.maybe_render_frame(i, obs, env, action=np.zeros(2, dtype=np.float32))
        if done or truncated:
            print(f"Episode ended at step {i}")
            break

    renderer.save_gif()
    QuadhopperRenderer.save_analysis_plots(history, "artifacts/renders/ballistic_analysis.png")


def main():
    parser = argparse.ArgumentParser(description="QuadHopper RL")
    parser.add_argument(
        "--mode", choices=["train", "test", "eval", "demo", "ballistic"], default="test",
        help="Run mode: train, test, eval, demo, or ballistic (physics-only arc check)"
    )
    parser.add_argument(
        "--model", type=str, default=TRAINING.model_path,
        help="Path to model (without .zip)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for ballistic test"
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of evaluation episodes for --mode eval"
    )
    parser.add_argument(
        "--target-count", type=int, default=TRAINING.target_count,
        help="Number of 0.5 m-spaced targets used for train/test/eval"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total training timesteps for --mode train"
    )
    parser.add_argument(
        "--resume-model", type=str, default=None,
        help="Path to an existing PPO model to continue training from, without .zip"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(
            target_count=args.target_count,
            model_path=args.model,
            resume_model=args.resume_model,
            timesteps=args.timesteps,
        )
    elif args.mode == "ballistic":
        ballistic_test(seed=args.seed)
    elif args.mode == "eval":
        evaluate(
            model_path=args.model,
            episodes=args.episodes,
            seed=args.seed or 0,
            target_count=args.target_count,
        )
    elif args.mode == "demo":
        demo(seed=args.seed or 0)
    else:
        test(model_path=args.model, target_count=args.target_count)


if __name__ == "__main__":
    freeze_support()
    main()
