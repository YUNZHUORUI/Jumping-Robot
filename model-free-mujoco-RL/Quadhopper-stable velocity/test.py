import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
from scipy.spatial.transform import Rotation as R  # [新增] 用于计算角度

# 关键：使用 Agg 后端防止与 MuJoCo 窗口冲突导致崩溃
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import PogoDroneEnv


def calculate_instant_power(motors, thrust_coeff=15.0, power_coeff=6.5):
    """
    计算瞬时功率 (Watts)
    模型: P = 6.5 * (15 * u)^1.5
    """
    thrusts = motors * thrust_coeff
    powers = power_coeff * np.power(np.maximum(thrusts, 0), 1.5)
    return np.sum(powers)


def main():
    print("🎬 准备启动实时演示...")

    # 1. 加载环境与模型
    env = DummyVecEnv([lambda: PogoDroneEnv()])
    try:
        env = VecNormalize.load("/Users/yunzhuorui/Jumping-Robot/model-free-mujoco-RL/Quadhopper-stable velocity/vec_normalize_natural.pkl", env)
        env.training = False
        env.norm_reward = False
        model = PPO.load("/Users/yunzhuorui/Jumping-Robot/model-free-mujoco-RL/Quadhopper-stable velocity/pogo_natural_final.zip", env=env)
    except Exception as e:
        print(f"❌ 错误: 无法加载模型 ({e})")
        return

    raw_env = env.envs[0]
    obs = env.reset()

    # 2. 数据容器
    # [新增] 'pitch' 用于第二张图
    data = {
        'time': [], 'x': [], 'y': [], 'z': [],
        'vx': [], 'energy': [], 'pitch': []
    }

    current_energy = 0.0
    sim_time = 0.0
    dt_step = raw_env.dt * raw_env.n_substeps  # 0.02s

    print("\n" + "=" * 50)
    print("🚀 仿真开始！(限时 20秒)")
    print("👀 请在弹出的窗口中观看无人机表演")
    print("=" * 50 + "\n")

    # 3. 启动 MuJoCo Viewer
    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        # 摄像机自动跟随
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = raw_env.drone_id
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 90

        start_real_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            # === 时间限制 ===
            if sim_time > 20.0:
                print("⏰ 20秒时间到，自动结束仿真...")
                break

            # === AI 控制 ===
            action, _ = model.predict(obs, deterministic=True)

            # 还原电机信号 (用于计算能耗)
            hover_throttle = raw_env.hover_throttle
            scaled_action = hover_throttle * 0.9 + 0.6 * action
            motors = np.clip(scaled_action, 0.0, 1.0)

            # 物理步进
            obs, _, done, _ = env.step(action)

            # === 数据采集 ===
            pos = raw_env.data.xpos[raw_env.drone_id]
            vel = raw_env.data.cvel[raw_env.drone_id]

            # [新增] 获取 Pitch 角度
            quat = raw_env.data.xquat[raw_env.drone_id]
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            pitch_deg = np.degrees(r.as_euler('xyz')[1])

            power = calculate_instant_power(motors)
            current_energy += power * dt_step

            data['time'].append(sim_time)
            data['x'].append(pos[0])
            data['y'].append(pos[1])
            data['z'].append(pos[2])
            data['vx'].append(vel[3])  # Index 3 is linear vel x
            data['energy'].append(current_energy)
            data['pitch'].append(pitch_deg)  # [新增]

            sim_time += dt_step

            # === 渲染更新 ===
            viewer.sync()

            if done:
                print(f"💥 重置 | Dist: {pos[0]:.2f}m")
                obs = env.reset()
                # 重置物理时间锁，防止卡顿
                start_real_time = time.time() - sim_time

            # === 真实时间同步 (核心) ===
            # 让仿真速度匹配现实时间，否则你会看到20倍速的鬼畜画面
            compute_time = time.time() - step_start
            if compute_time < dt_step:
                time.sleep(dt_step - compute_time)

    # 4. 绘图
    print(f"📊 正在生成分析图表 (数据点: {len(data['time'])})...")

    try:
        # ==========================================
        # 图表 1: 能量与轨迹分析 (对应 comparison.py 的 PID-energy-analysis.png)
        # ==========================================
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Pogo Drone Flight Analysis (PPO Controller)', fontsize=16)

        # 1. Y vs X 轨迹图 (带时间戳)
        ax1 = axes[0, 0]
        ax1.plot(data['x'], data['y'], 'b-', label='Path', linewidth=1.5)
        ax1.set_title('Trajectory (Top-Down View: Y vs X)')
        ax1.set_xlabel('Distance X (m)')
        ax1.set_ylabel('Lateral Drift Y (m)')
        ax1.grid(True)
        ax1.axis('equal')

        # 标记时间戳 (每2秒)
        timestamp_interval = 2.0
        last_mark_time = -timestamp_interval
        for i, t in enumerate(data['time']):
            if t - last_mark_time >= timestamp_interval:
                ax1.scatter(data['x'][i], data['y'][i], color='red', zorder=5)
                ax1.text(data['x'][i], data['y'][i] + 0.05, f"{t:.0f}s", color='red', fontsize=9, fontweight='bold')
                last_mark_time = t

        # 限制 Y 轴范围
        y_max = max(np.max(np.abs(data['y'])), 0.5)
        ax1.set_ylim(-y_max * 1.5, y_max * 1.5)

        # 2. Z vs Time 高度图
        ax2 = axes[0, 1]
        ax2.plot(data['time'], data['z'], 'm-')
        ax2.set_title('Height over Time (Z)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Height (m)')
        ax2.grid(True)
        ax2.axhline(0, color='k', linewidth=2)  # 地面

        # 3. Vx vs Time 速度图
        ax3 = axes[1, 0]
        ax3.plot(data['time'], data['vx'], 'g-')
        ax3.set_title('Forward Velocity (Vx)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.grid(True)
        ax3.axhline(1.5, color='r', linestyle='--', label='Target (1.5 m/s)')
        ax3.legend()

        # 4. Energy vs Time 能耗图
        ax4 = axes[1, 1]
        ax4.plot(data['time'], data['energy'], 'orange', linewidth=2)
        ax4.fill_between(data['time'], data['energy'], color='orange', alpha=0.2)
        ax4.set_title('Cumulative Energy Consumption')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Energy (Joules)')
        ax4.grid(True)
        ax4.text(data['time'][-1], data['energy'][-1], f"{data['energy'][-1]:.0f} J",
                 ha='right', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('PPO-energy-analysis.png', dpi=300)
        print("✅ 图表 1 已保存: PPO-energy-analysis.png")

        # ==========================================
        # [新增] 图表 2: 极限环与稳定性 (对应 comparison.py 的 periodic_limit_cycle.png)
        # ==========================================
        print("Generating Limit Cycle Plot...")
        plt.figure(figsize=(14, 10))

        # 极限环 (截取后半段数据以展示稳定状态)
        cutoff = len(data['z']) // 2
        plt.subplot(2, 2, 1)
        # 注意：这里使用 list 切片
        plt.plot(data['z'][cutoff:], data['vx'][cutoff:], 'm-')
        plt.xlabel('Height (m)')
        plt.ylabel('Velocity X (m/s)')
        plt.title('Limit Cycle: Height vs Velocity')
        plt.grid(True)

        # 速度稳定性
        plt.subplot(2, 2, 2)
        plt.plot(data['time'], data['vx'], 'b')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity X (m/s)')
        plt.title('Velocity Stabilization')
        plt.grid(True)

        # 高度稳定性
        plt.subplot(2, 2, 3)
        plt.plot(data['time'], data['z'], 'k')
        plt.title('Height Stability')
        plt.xlabel('Time (s)')
        plt.ylabel('Height (m)')
        plt.grid(True)

        # 角度对称性
        plt.subplot(2, 2, 4)
        plt.plot(data['time'], data['pitch'], 'purple')
        plt.axhline(0, color='k', alpha=0.2)
        plt.title('Pitch Angle Symmetry')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle ($\circ$)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('ppo_limit_cycle.png')
        print("✅ 图表 2 已保存: ppo_limit_cycle.png")

    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()