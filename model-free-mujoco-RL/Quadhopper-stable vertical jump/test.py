import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import PogoDroneEnv

plt.ion()


def main():
    print("🤖 加载 Strict Vertical 模型...")
    env = DummyVecEnv([lambda: PogoDroneEnv()])

    try:
        # 加载归一化参数
        env = VecNormalize.load("vec_normalize_vertical.pkl", env)
        env.training = False
        env.norm_reward = False

        # 加载模型
        model = PPO.load("pogo_vertical_final", env=env)
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请确保已运行 train.py 并生成了模型文件")
        return

    raw_env = env.envs[0]
    obs = env.reset()

    log_z, log_vz = [], []

    print("\n" + "=" * 50)
    print("🚀 仿真开始 - 观察下落时的姿态锁定")
    print("=" * 50 + "\n")

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        viewer.cam.trackbodyid = raw_env.drone_id
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -10

        step_dt = raw_env.dt * raw_env.n_substeps

        while viewer.is_running():
            step_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

            # 记录数据
            z = raw_env.data.xpos[raw_env.drone_id][2]
            vz = raw_env.data.cvel[raw_env.drone_id][5]
            log_z.append(z)
            log_vz.append(vz)

            viewer.sync()

            if done:
                print(f"🔄 Reset | Max Height: {max(log_z[-100:]):.2f}m")
                obs = env.reset()

            # 时间同步
            compute_time = time.time() - step_start
            if compute_time < step_dt:
                time.sleep(step_dt - compute_time)

    # 绘图
    plt.ioff()
    plt.figure(figsize=(6, 6))
    cut = max(0, len(log_z) - 3000)
    plt.plot(log_z[cut:], log_vz[cut:], 'b-', alpha=0.5)
    plt.title('Strict Limit Cycle')
    plt.xlabel('Height Z (m)')
    plt.ylabel('Vertical Velocity Vz (m/s)')
    plt.grid(True)
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('strict_limit_cycle.png')
    plt.show()


if __name__ == "__main__":
    main()