import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os


# ----------------------------------------------------------------------
# 1. 连续跳跃环境 (Continuous Hopping Environment)
# ----------------------------------------------------------------------
class QuadhopperContinuousEnv:
    def __init__(self,
                 dt=0.005,
                 m_body=0.5,  # 机身质量
                 l_leg=0.50,  # 腿长
                 lc=0.25,  # 机臂长度 (电机到中心距离)
                 J_body=0.05,  # 转动惯量
                 g=9.81,
                 max_thrust=15.0,  # 最大推力 (N)
                 target_dx=1.0,  # 目标前进速度 (m/s)
                 ground_y=0.0):  # 地面高度

        self.dt = dt
        self.m = m_body
        self.l0 = l_leg
        self.lc = lc
        self.J = J_body
        self.g = g
        self.max_thrust = max_thrust
        self.target_dx = target_dx
        self.ground_y = ground_y

        self.reset()

    def reset(self):
        # 初始状态: [x, y, theta, l]
        # 从空中开始，稍微给一点初速度
        self.q = np.array([0.0, 1.2, 0.05, self.l0], dtype=np.float64)
        # 初始速度: [dx, dy, dtheta, dl]
        self.dq = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float64)

        self.steps = 0
        self.hops = 0
        return self._obs()

    def _obs(self):
        # 观测: 状态 + 速度
        return np.concatenate([self.q, self.dq])

    def get_foot_pos(self):
        # 计算脚尖坐标
        # 几何定义: theta=0 时直立，腿垂直向下
        x, y, theta = self.q[0], self.q[1], self.q[2]
        # 脚的位置相对于重心:
        # x_foot = x + l * sin(theta)
        # y_foot = y - l * cos(theta)
        x_f = x + self.l0 * math.sin(theta)
        y_f = y - self.l0 * math.cos(theta)
        return np.array([x_f, y_f])

    def step(self, action):
        """
        action: [u1, u2] 范围 [0, 1]，代表左右电机的推力百分比
        """
        # 1. 解析动作 (Thrust Mixing)
        u1 = np.clip(action[0], 0.0, 1.0)  # 左电机
        u2 = np.clip(action[1], 0.0, 1.0)  # 右电机

        F1 = u1 * self.max_thrust
        F2 = u2 * self.max_thrust

        F_total = F1 + F2
        # 力矩: 右侧(F2)产生逆时针(+)还是顺时针(-)?
        # 假设 F2 在 +x 轴，F1 在 -x 轴。
        # F2 向上推 -> 产生正向力矩(逆时针, 抬头)
        # F1 向上推 -> 产生负向力矩(顺时针, 低头)
        tau = (F2 - F1) * self.lc

        # 2. 飞行阶段动力学 (Flight Dynamics)
        theta = self.q[2]
        s, c = math.sin(theta), math.cos(theta)

        # 简单的刚体动力学 (忽略腿部伸缩的耦合质量变化，简化为刚杆)
        # 力的分解 (注意 theta=0 是直立，推力方向始终垂直于机身)
        # 推力向量在机身系下是 [0, 1]
        # 转换到世界系:
        # Fx = -F_total * sin(theta)
        # Fy = F_total * cos(theta)

        ddx = (-F_total * s) / self.m
        ddy = (F_total * c) / self.m - self.g
        ddtheta = tau / self.J

        # 欧拉积分
        self.dq[0] += ddx * self.dt
        self.dq[1] += ddy * self.dt
        self.dq[2] += ddtheta * self.dt

        self.q[0] += self.dq[0] * self.dt
        self.q[1] += self.dq[1] * self.dt
        self.q[2] += self.dq[2] * self.dt

        # 3. 触地检测与连续跳跃逻辑 (Ground Contact & Continuous Hop)
        foot_pos = self.get_foot_pos()
        foot_y = foot_pos[1]

        impact = False
        # 只有当脚低于地面 且 正在向下运动时 触发反弹
        if foot_y <= self.ground_y and self.dq[1] < 0:
            impact = True
            self.hops += 1

            # === Raibert 物理模拟核心 (Simulated Stance Phase) ===

            # A. 垂直能量注入 (Height Control)
            # 模拟弹簧压缩和推力释放，直接反转垂直速度并补充损耗
            # 目标: 每次跳回 1.0m 高度
            h_target = 1.0
            v_launch = math.sqrt(2 * self.g * h_target)
            self.dq[1] = v_launch

            # B. 水平速度耦合 (Speed dynamics)
            # 如果落地时机身倾斜 (theta != 0)，地面反作用力会改变水平速度
            # 简化的 Raibert 动力学:
            # 如果脚在重心前方 (theta > 0), 产生向后的力 -> 减速
            # 如果脚在重心后方 (theta < 0), 产生向前的力 -> 加速
            # dx_new = dx_old - gain * theta_land
            coupling_gain = 2.0
            self.dq[0] = self.dq[0] - coupling_gain * theta

            # C. 角度冲击 (Angular impact)
            # 脚底受力会产生剧烈力矩
            # dtheta_new = dtheta_old - impact_gain * theta
            self.dq[2] = self.dq[2] - 5.0 * theta

            # D. 防止穿模，重置位置
            # y = ground + l * cos(theta)
            self.q[1] = self.ground_y + self.l0 * math.cos(theta) + 0.01

            print(f"🦘 HOP {self.hops}! Vel X: {self.dq[0]:.2f}, Theta: {math.degrees(theta):.1f}°")

        self.steps += 1
        return self._obs(), impact


# ----------------------------------------------------------------------
# 2. Raibert 控制器 (The "Brain")
# ----------------------------------------------------------------------
class RaibertController:
    def __init__(self, env):
        self.env = env
        # PID 参数
        self.kp = 8.0  # 姿态比例增益
        self.kd = 1.0  # 姿态微分增益

        # Raibert 参数
        self.T_stance = 0.2  # 估计触地时间
        self.k_speed = 0.15  # 速度误差反馈增益 (Placement Gain)

    def get_action(self, obs):
        x, y, theta, l = obs[0:4]
        dx, dy, dtheta, dl = obs[4:8]

        # --- 步骤 1: 计算下一跳的理想落足点 (Foot Placement) ---
        # 公式: x_foot_rel = (dx * Ts / 2) + k * (dx - dx_target)
        # 第一项: 中性点 (Neutral Point)，保持当前速度
        # 第二项: 伺服项 (Servo)，消除速度误差

        x_neutral = dx * self.T_stance / 2.0
        x_servo = self.k_speed * (dx - self.env.target_dx)
        x_foot_rel_des = x_neutral + x_servo

        # --- 步骤 2: 将落足点转换为目标姿态 (Target Theta) ---
        # 几何关系: x_foot_rel = l * sin(theta)
        # sin(theta_des) = x_foot_rel_des / l

        sin_val = x_foot_rel_des / self.env.l0
        sin_val = np.clip(sin_val, -0.7, 0.7)  # 限制最大倾角防止摔倒
        theta_des = math.asin(sin_val)

        # --- 步骤 3: 飞行姿态控制 (Flight Attitude Control) ---
        # 使用 PD 控制器追踪 theta_des

        theta_err = theta_des - theta
        dtheta_err = 0.0 - dtheta  # 目标角速度为0

        torque_demand = self.kp * theta_err + self.kd * dtheta_err

        # --- 步骤 4: 混控器 (Mixer) ---
        # 将力矩需求转换为左右电机推力
        # Tau = (F2 - F1) * lc
        # F_diff = Tau / lc
        # 同时保持一个很小的基础推力 (Base Thrust) 维持张力

        base_thrust = 0.1  # 10% 动力
        diff = torque_demand / self.env.lc

        # F2 = base + diff/2
        # F1 = base - diff/2
        u2 = base_thrust + diff / 2.0
        u1 = base_thrust - diff / 2.0

        return np.array([u1, u2])


# ----------------------------------------------------------------------
# 3. 仿真与可视化 (Simulation Loop)
# ----------------------------------------------------------------------
def run_simulation():
    env = QuadhopperContinuousEnv(target_dx=1.5)  # 目标速度 1.5 m/s
    controller = RaibertController(env)

    max_steps = 600
    frames = []

    print("🚀 Starting Raibert Continuous Hopping Simulation...")

    obs = env.reset()
    for step in range(max_steps):
        # 获取 Raibert 控制动作
        action = controller.get_action(obs)

        # 环境步进
        obs, impact = env.step(action)

        # 记录数据用于渲染
        foot_pos = env.get_foot_pos()
        frames.append({
            'q': env.q.copy(),
            'foot': foot_pos,
            'action': action,
            'step': step
        })

    # --- 生成动画 ---
    print("🎥 Generating Animation...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-1, 8)  # 跑得比较远
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Continuous Jumping with Raibert Logic")

    # 绘图元素
    ground, = ax.plot([-10, 20], [0, 0], 'k-', lw=2)
    body_line, = ax.plot([], [], 'k-', lw=3)  # 机臂
    leg_line, = ax.plot([], [], 'b-', lw=2)  # 腿
    foot_dot, = ax.plot([], [], 'ro', markersize=6)
    prop_l, = ax.plot([], [], 'g^', markersize=8)  # 左桨
    prop_r, = ax.plot([], [], 'g^', markersize=8)  # 右桨
    info_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    def init():
        return body_line, leg_line, foot_dot, prop_l, prop_r, info_text

    def update(frame_data):
        x, y, theta, l = frame_data['q']
        u1, u2 = frame_data['action']

        # 计算机身端点 (左右电机)
        # 左电机: (-lc) 旋转 theta
        xl = x - env.lc * math.cos(theta)
        yl = y - env.lc * math.sin(theta)
        # 右电机: (+lc) 旋转 theta
        xr = x + env.lc * math.cos(theta)
        yr = y + env.lc * math.sin(theta)

        # 腿部
        foot_x, foot_y = frame_data['foot']

        # 更新图形
        body_line.set_data([xl, xr], [yl, yr])
        leg_line.set_data([x, foot_x], [y, foot_y])
        foot_dot.set_data([foot_x], [foot_y])

        prop_l.set_data([xl], [yl])
        prop_r.set_data([xr], [yr])

        # 推力可视化 (桨叶颜色深浅)
        prop_l.set_color(plt.cm.autumn(u1))
        prop_r.set_color(plt.cm.autumn(u2))

        info_text.set_text(f"Step: {frame_data['step']}\nVel X: {env.dq[0]:.2f} m/s\nTarget: {env.target_dx} m/s")

        # 相机跟随
        ax.set_xlim(x - 2, x + 6)

        return body_line, leg_line, foot_dot, prop_l, prop_r, info_text

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=30, blit=True)

    # 保存
    ani.save("quadhopper_continuous_raibert.gif", writer=PillowWriter(fps=30))
    print("✅ Animation saved as 'quadhopper_continuous_raibert.gif'")
    plt.show()


if __name__ == "__main__":
    run_simulation()