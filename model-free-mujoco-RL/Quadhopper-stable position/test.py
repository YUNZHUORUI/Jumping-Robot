import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib

# ==========================================
# 🛠️ 关键修复：强制使用非交互式后端
# 这行代码必须在 import pyplot 之前
# 它告诉 Matplotlib："只在内存里画画，别弹窗"
# ==========================================
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import PogoDroneEnv

# === ⚙️ 设置 ===
SLOW_MOTION = False
SIM_DURATION = 60.0


def main():
    print("🎬 启动【PPO模型】性能分析...")
    print("👉 请在 Mujoco 窗口中点击【关闭】(X) 按钮来停止仿真并生成图表。")

    # 1. 检查模型
    model_name = "pogo_hop_final"
    vec_norm_name = "vec_normalize_hop.pkl"

    if not os.path.exists(f"{model_name}.zip"):
        print(f"❌ 找不到模型 {model_name}.zip，请先运行 train.py")
        return

    # 2. 加载环境
    env = DummyVecEnv([lambda: PogoDroneEnv()])
    env = VecNormalize.load(vec_norm_name, env)
    env.training = False
    env.norm_reward = False

    # 3. 加载模型
    model = PPO.load(model_name, env=env)

    raw_env = env.envs[0]
    m = raw_env.model
    d = raw_env.data

    drone_id = raw_env.drone_id
    foot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot")

    try:
        thrust_coeff = m.actuator_gear[0, 2]
    except:
        thrust_coeff = 1.0

    obs = env.reset()

    # === 数据记录容器 ===
    log_time = []
    log_z, log_x, log_y = [], [], []
    log_lambda = []
    log_thrust = []

    frame_time = 1.0 / raw_env.control_freq
    if SLOW_MOTION: frame_time *= 2.0

    print(f"🚀 目标 X: {raw_env.target_x:.2f}m")

    # ==========================================
    # 仿真主循环
    # ==========================================
    # launch_passive 在 macOS 上通常表现更好
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance = 18.0
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [raw_env.target_x / 2, 0.0, 2.0]

        start_time = time.time()

        while viewer.is_running() and d.time < SIM_DURATION:
            step_start = time.time()

            # 1. 预测
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)

            # 2. 采集
            pos_x = d.xpos[drone_id][0]
            pos_y = d.xpos[drone_id][1]
            pos_z = d.xpos[drone_id][2]

            foot_force = 0.0
            for i in range(d.ncon):
                contact = d.contact[i]
                if (m.geom_bodyid[contact.geom1] == foot_id or
                        m.geom_bodyid[contact.geom2] == foot_id):
                    c_force = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(m, d, i, c_force)
                    foot_force += c_force[0]

            total_force_N = np.sum(d.ctrl) * thrust_coeff

            log_time.append(d.time)
            log_x.append(pos_x)
            log_y.append(pos_y)
            log_z.append(pos_z)
            log_lambda.append(foot_force)
            log_thrust.append(total_force_N)

            # 3. Viewer更新
            viewer.user_scn.ngeom = 0
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                viewer.user_scn.ngeom += 1
                geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                geom.size[:] = [0.1, 0.1, 0.1]
                geom.pos[:] = [raw_env.target_x, 0, 0]
                geom.mat[:] = np.eye(3)
                geom.rgba[:] = [1, 0, 0, 0.5]

            viewer.sync()

            if d.time % 0.5 < 0.02:
                print(f"\r[{d.time:.1f}s] H:{pos_z:.2f}m | X:{pos_x:.2f}", end="")

            if dones[0]:
                pass

            elapsed = time.time() - step_start
            if frame_time - elapsed > 0:
                time.sleep(frame_time - elapsed)

    # ==========================================
    # 窗口关闭后执行
    # ==========================================

    if len(log_time) == 0:
        print("\n⚠️ 未采集到数据。")
        return

    print("\n\n✅ 仿真结束，正在后台生成图表...")

    targets_for_plot = [np.array([raw_env.target_x, 0.0])]
    plot_stable_style(log_time, log_x, log_y, log_z, log_lambda, log_thrust, targets_for_plot)


def plot_stable_style(log_time, log_x, log_y, log_z, log_lambda, log_thrust, targets):
    """
    后台静默绘图并保存
    """
    plt.figure(figsize=(12, 10))

    # 1. 轨迹
    plt.subplot(2, 2, 1)
    plt.plot(log_x, log_y, 'b-', linewidth=2)
    plt.plot(0, 0, 'go', label='Start')
    for idx, tgt in enumerate(targets):
        plt.plot(tgt[0], tgt[1], 'ro', label=f'Target {idx + 1}')

    last_t = -10
    for i, t in enumerate(log_time):
        if t - last_t >= 2.0:
            plt.text(log_x[i], log_y[i], f"{t:.0f}s", fontsize=9, fontweight='bold')
            plt.plot(log_x[i], log_y[i], 'k.', markersize=5)
            last_t = t

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('2D Trajectory with Time Stamps')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

    # 2. 高度
    plt.subplot(2, 2, 2)
    plt.plot(log_time, log_z, 'k-')
    plt.axhline(1.0, color='r', linestyle='--', label='Ref Height')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.title('Height Stability')
    plt.grid(True)

    # 3. 接触力
    plt.subplot(2, 2, 3)
    plt.plot(log_time, log_lambda, 'r-')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Ground Contact Force')
    plt.grid(True)

    # 4. 推力
    plt.subplot(2, 2, 4)
    plt.plot(log_time, log_thrust, 'orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust (N)')
    plt.title('Total Motor Thrust')
    plt.grid(True)

    plt.tight_layout()

    save_path = 'ppo_final_analysis.png'
    plt.savefig(save_path)

    print(f"✅ 图表已保存至文件: {save_path}")
    print("👉 请在文件管理器中打开该图片查看结果。")

    # ❌ 必须移除 plt.show()，否则会在 macOS 上再次导致 crash
    # plt.show()


if __name__ == "__main__":
    main()