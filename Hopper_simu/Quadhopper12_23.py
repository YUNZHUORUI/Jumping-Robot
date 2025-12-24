import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.optim as optim
import os


# ==============================================================================
# 1. 多跳物理环境 (Physics Environment with Multi-hop & Ballistic Logic)
# ==============================================================================

class QuadHopperMultiJumpEnv:
    def __init__(self,
                 dt=0.005,
                 m1=0.0363, m2=0.01, Jm=0.02, lc=0.25, l0=0.50, g=9.81, thrust=0.314,
                 max_steps_per_hop=600,
                 targets=[2.0, 4.5, 7.0],  # 连续跳跃的目标点 x 坐标
                 # --- 约束条件 ---
                 r_inner=0.5, R_outer=1.50,
                 launch_theta_min_deg=30, launch_theta_max_deg=60,
                 landing_pitch_target_deg=-20.0):

        self.dt = float(dt)
        self.m1, self.m2, self.Jm = float(m1), float(m2), float(Jm)
        self.lc, self.l0 = float(lc), float(l0)
        self.g, self.thrust = float(g), float(thrust)
        self.max_steps = int(max_steps_per_hop)

        # 多跳目标管理
        self.global_targets = targets
        self.current_target_idx = 0
        self.landing_pitch_target = math.radians(landing_pitch_target_deg)

        # 区域约束 (Ballistic Sector)
        self.r_inner = r_inner
        self.R_outer = R_outer
        self.r_sq = r_inner ** 2
        self.R_sq = R_outer ** 2
        self.launch_theta_min = math.radians(launch_theta_min_deg)
        self.launch_theta_max = math.radians(launch_theta_max_deg)
        self.launch_theta_min_deg = launch_theta_min_deg
        self.launch_theta_max_deg = launch_theta_max_deg

        self.reset()

    def reset(self):
        # 完全重置环境
        self.current_target_idx = 0
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # x, y, theta, l
        self.dq = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # 初始随机扰动
        self.q[2] = np.random.uniform(math.radians(-5), math.radians(5))

        self.steps = 0
        self.flight_state = "PRE_LAUNCH"  # 状态机: PRE_LAUNCH, BALLISTIC, APEX_ADJUST, DESCENDING
        self.current_origin_x = 0.0  # 当前跳跃的参考原点

        return self._obs()

    def _get_current_target_rel(self):
        # 获取相对于当前起点的目标距离
        if self.current_target_idx >= len(self.global_targets):
            return 0.0
        global_target_x = self.global_targets[self.current_target_idx]
        return global_target_x - self.current_origin_x

    def _obs(self):
        # 观测空间: [dx, dy, dtheta, theta, rel_dist_x, rel_dist_y]
        # rel_dist_x 是 "当前位置" 到 "相对当前起点的目标" 的距离
        target_rel_dist = self._get_current_target_rel()
        current_rel_pos = self.q[0] - self.current_origin_x

        rel_x = target_rel_dist - current_rel_pos
        rel_y = 0.0 - self.q[1]

        obs = np.array([self.dq[0], self.dq[1], self.dq[2], self.q[2], rel_x, rel_y], dtype=np.float32)
        return obs

    def _calculate_ballistic_velocity(self, x0, y0, theta0):
        """
        根据公式计算命中目标所需的初速度 v0
        v0 = sqrt( (g * Dx^2) / (2 * cos^2(theta) * (Dx * tan(theta) - Dz)) )
        """
        # 计算相对于当前跳跃起点(Origin)的 Delta_x
        # 比如 Origin=2.0, Target=4.5, 当前 x=2.5 -> Delta_x = 4.5 - 2.5 = 2.0
        global_target = self.global_targets[self.current_target_idx] if self.current_target_idx < len(
            self.global_targets) else self.q[0]
        Delta_x = global_target - x0
        Delta_z = 0.0 - y0  # 目标高度假设为 0

        if Delta_x <= 0.1: return -1.0  # 目标在身后或太近

        cos_t = math.cos(theta0)
        tan_t = math.tan(theta0)

        denom = 2 * (cos_t ** 2) * (Delta_x * tan_t - Delta_z)

        if denom <= 1e-6: return -1.0  # 无解

        v0_sq = (self.g * (Delta_x ** 2)) / denom
        if v0_sq < 0: return -1.0

        return np.sqrt(v0_sq)

    def _compute_dynamics(self, action, torque_only=False):
        x, y, theta, l = self.q

        # --- Action 处理 ---
        F1, F2 = 0.0, 0.0

        if torque_only:
            # APEX 阶段：仅输出微小差动推力产生力矩，不产生主要升力
            torque_mag = 0.1 * self.thrust
            # action[0]=1 -> F1增加 (逆时针力矩), action[1]=1 -> F2增加 (顺时针力矩)
            # 简化模型：F1产生正升力+力矩，F2产生正升力-力矩。
            # 为了纯力矩，我们假设 F1=T, F2=-T (物理上不可行因为桨不能反转)，
            # 或者 F1=T, F2=0, 忽略微小升力。这里采用后者。
            if action[0] > 0.5: F1 += torque_mag
            if action[1] > 0.5: F2 += torque_mag
        else:
            # 正常推力模式
            F1 = self.thrust * float(action[0])
            F2 = self.thrust * float(action[1])

        c, s = math.cos(theta), math.sin(theta)

        # --- 动力学矩阵 ---
        M = np.array([
            [self.m1 + self.m2, 0.0, l * self.m1 * c, self.m1 * s],
            [0.0, self.m1 + self.m2, -l * self.m1 * s, self.m1 * c],
            [l * self.m1 * c, -l * self.m1 * s, self.Jm + (l ** 2) * self.m1, 0],
            [self.m1 * s, self.m1 * c, 0, self.m1]
        ], dtype=np.float64)

        B = np.array([
            - self.m1 * (self.dq[2] ** 2) * l * s,
            - self.m1 * (self.dq[2] ** 2) * l * c,
            - 2.0 * self.m1 * l * self.dq[2] * (self.dq[1] * c + self.dq[0] * s),
            (self.dq[2] ** 2) * l * self.m1 + 2 * self.dq[2] * self.dq[0] * self.m1 * c - 2 * self.dq[2] * self.dq[
                1] * self.m1 * s
        ], dtype=np.float64)

        G = np.array([
            0.0,
            (self.m1 + self.m2) * self.g,
            -self.g * l * self.m1 * s,
            self.m1 * self.g * c
        ], dtype=np.float64)

        F_vec = np.array([
            (F1 + F2) * s,
            (F1 + F2) * c,
            (F1 - F2) * self.lc,
            0
        ], dtype=np.float64)

        # 求解加速度 ddq = M_inv * (F - B - G)
        try:
            ddq = np.linalg.solve(M, F_vec - B - G)
        except np.linalg.LinAlgError:
            ddq = np.zeros_like(self.dq)

        if np.any(np.isnan(ddq)): ddq = np.zeros_like(ddq)
        return ddq

    def step(self, action):
        reward = 0.0
        done = False
        info = {}

        # 计算相对于当前起点的相对位置
        rel_x = self.q[0] - self.current_origin_x
        rel_y = self.q[1]
        rel_pos_sq = rel_x ** 2 + rel_y ** 2
        current_v = np.sqrt(self.dq[0] ** 2 + self.dq[1] ** 2)

        # ----------------------------------------------------------------------
        # 状态机逻辑 (State Machine)
        # ----------------------------------------------------------------------

        # 1. PRE_LAUNCH (寻找发射窗口)
        if self.flight_state == "PRE_LAUNCH":
            # A. 几何约束检查
            in_sector = (self.r_sq < rel_pos_sq < self.R_sq) and \
                        (self.launch_theta_min < self.q[2] < self.launch_theta_max)

            # B. 速度匹配检查
            req_v = self._calculate_ballistic_velocity(self.q[0], self.q[1], self.q[2])
            velocity_match = False
            if req_v > 0 and abs(current_v - req_v) < 0.2:  # 允许 0.2 m/s 的误差
                velocity_match = True

            # 奖励: 引导进入 Sector 并匹配速度
            if not in_sector:
                reward -= 0.05  # 没进区域，轻微惩罚
            else:
                reward += 0.2  # 进区域奖励
                if req_v > 0:
                    v_err = abs(current_v - req_v)
                    reward += 3.0 * np.exp(-5.0 * v_err)  # 速度匹配奖励

            # 触发发射
            if in_sector and velocity_match:
                self.flight_state = "BALLISTIC"
                reward += 20.0  # 巨大的发射奖励
                # print(f"🚀 Launch! TargetIdx={self.current_target_idx}")

        # 2. BALLISTIC (滑翔阶段 - 推力锁死)
        if self.flight_state == "BALLISTIC":
            action = [0, 0]  # 强制关闭推力
            reward -= 0.01

            # 检测 Apex (最高点)
            if abs(self.dq[1]) < 0.3 and self.q[1] > 0.5:
                self.flight_state = "APEX_ADJUST"

        # 3. APEX_ADJUST (最高点姿态修正 - 仅力矩)
        is_torque_mode = False
        if self.flight_state == "APEX_ADJUST":
            is_torque_mode = True
            # 目标：调整 theta 指向 landing_pitch_target (-20 deg)
            ang_err = abs(self.q[2] - self.landing_pitch_target)
            reward += 1.0 * np.exp(-10.0 * ang_err)

            # 如果开始显著下落，转入 Descending
            if self.dq[1] < -0.5:
                self.flight_state = "DESCENDING"

        # 4. DESCENDING (下落 - 推力锁死)
        if self.flight_state == "DESCENDING":
            action = [0, 0]
            is_torque_mode = False

        # ----------------------------------------------------------------------
        # 物理积分
        # ----------------------------------------------------------------------
        ddq = self._compute_dynamics(action, torque_only=is_torque_mode)
        self.dq += ddq * self.dt
        self.q += self.dq * self.dt

        # 角度归一化 (-pi, pi)
        self.q[2] = (self.q[2] + math.pi) % (2 * math.pi) - math.pi
        self.steps += 1

        # ----------------------------------------------------------------------
        # 落地检测与多跳重置
        # ----------------------------------------------------------------------
        if self.q[1] <= 0:
            if self.current_target_idx < len(self.global_targets):
                target_x = self.global_targets[self.current_target_idx]
                dist_err = abs(self.q[0] - target_x)
                pitch_err = abs(self.q[2] - self.landing_pitch_target)

                # 落地判定：位置准 & 姿态准
                if dist_err < 0.4 and pitch_err < math.radians(20):
                    # --- 成功落地 (Landing Success) ---
                    reward += 50.0

                    # 切换到下一个目标
                    self.current_target_idx += 1

                    if self.current_target_idx >= len(self.global_targets):
                        # 任务全部完成
                        done = True
                        reward += 100.0
                    else:
                        # --- 模拟 Stance Phase (原地重置) ---
                        # 将当前落点设为新的 "原点"
                        self.current_origin_x = self.q[0]
                        # 动能归零 (模拟触地吸能)
                        self.dq[:] = 0.0
                        self.q[1] = 0.0
                        # 状态重置为寻找下一次发射
                        self.flight_state = "PRE_LAUNCH"
                else:
                    # --- 失败 (Crash) ---
                    done = True
                    reward -= 50.0  # 坠毁惩罚
            else:
                done = True

        # 超时
        if self.steps >= self.max_steps:
            done = True

        # 翻滚保护
        if abs(self.q[2]) > math.radians(100):
            done = True
            reward -= 10.0

        info['state_phase'] = self.flight_state
        return self._obs(), reward, done, info

    def beam_endpoints(self):
        # 用于绘图辅助
        cx, cy, theta = self.q[0], self.q[1], self.q[2]
        dx = math.cos(theta) * self.lc
        dy = math.sin(theta) * self.lc
        left = (cx - dx + self.l0 * math.sin(theta), cy + dy + self.l0 * math.cos(theta))
        right = (cx + dx + self.l0 * math.sin(theta), cy - dy + self.l0 * math.cos(theta))
        tip = (cx + math.sin(theta) * self.l0, cy + math.cos(theta) * self.l0)
        return left, right, tip


# ==============================================================================
# 2. PPO 算法 (Training & Inference)
# ==============================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=6, hidden=256):  # Obs dim 增加到 6
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)  # Output logits for Bernoulli
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advs = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if t == T - 1 else values[t + 1]
        next_non_terminal = 0.0 if dones[t] else 1.0

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        lastgaelam = delta + gamma * lam * next_non_terminal * lastgaelam
        advs[t] = lastgaelam
    returns = advs + values
    return returns, (advs - advs.mean()) / (advs.std() + 1e-8)


def train_ppo(env, device, model_path="ppo_multihop.pt", num_updates=2000):
    net = ActorCritic(obs_dim=6).to(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)

    print(f"Starting PPO Training on {device}...")
    batch_size = 2048

    for update in range(num_updates):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        # --- Collection Phase ---
        steps = 0
        while steps < batch_size:
            obs = env.reset()
            done = False
            while not done:
                o_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, val = net(o_tensor)

                dist = torch.distributions.Bernoulli(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action).sum().item()
                action_np = action.cpu().numpy()[0]

                next_obs, r, done, _ = env.step(action_np)

                obs_buf.append(obs)
                act_buf.append(action_np)
                logp_buf.append(logp)
                rew_buf.append(r)
                val_buf.append(val.item())
                done_buf.append(done)

                obs = next_obs
                steps += 1
                if done or steps >= batch_size:
                    break

        # --- Update Phase ---
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.float32, device=device)
        logp_t = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device)

        returns, advs = compute_gae(rew_buf, val_buf, done_buf)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_t = torch.tensor(advs, dtype=torch.float32, device=device)

        # Mini-batch update
        idxs = np.arange(len(obs_buf))
        for _ in range(4):  # epochs
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), 512):
                mb = idxs[start:start + 512]

                logits, val = net(obs_t[mb])
                dist = torch.distributions.Bernoulli(logits=logits)
                new_logp = dist.log_prob(act_t[mb]).sum(dim=-1)

                ratio = torch.exp(new_logp - logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[mb]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((ret_t[mb] - val.squeeze(-1)) ** 2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                opt.zero_grad()
                loss.backward()
                opt.step()

        avg_rew = np.sum(rew_buf) / (np.sum(done_buf) + 1e-8)
        if update % 10 == 0:
            print(f"Update {update}: Avg Reward per Ep = {avg_rew:.2f}")
            torch.save(net.state_dict(), model_path)

    return net


# ==============================================================================
# 3. 可视化 (Visualization with Dynamic Wedge)
# ==============================================================================

def rollout_and_render_multihop(env, policy_net, device, max_steps=800, save_gif=True, filename='multihop_mission.gif'):
    frames = []
    obs = env.reset()
    done = False
    steps = 0

    print("Simulating rollout for visualization...")

    while not done and steps < max_steps:
        # PPO Action
        o_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy_net(o_tensor)
        # 确定性选择 (Deterministic for vis)
        action = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0].astype(int)

        next_obs, _, done, info = env.step(action)

        # 记录数据
        left, right, tip = env.beam_endpoints()
        frames.append({
            'q': env.q.copy(),
            'left': left, 'right': right, 'tip': tip,
            'origin_x': env.current_origin_x,
            'flight_state': env.flight_state,
            'target_idx': env.current_target_idx,
            'step': steps
        })

        obs = next_obs
        steps += 1

    # --- Matplotlib Animation ---
    print(f"Generating animation ({len(frames)} frames)...")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    max_x = env.global_targets[-1] + 2.0
    ax.set_xlim(-1.0, max_x)
    ax.set_ylim(-1.0, 3.5)

    # Ground & Targets
    ax.plot([-2, max_x + 2], [0, 0], 'k-', lw=2)
    for i, tx in enumerate(env.global_targets):
        ax.plot(tx, 0, 'rx', ms=10, markeredgewidth=2)
        ax.text(tx, -0.3, f"T{i + 1}", ha='center')

    # Actors
    leg_line, = ax.plot([], [], 'k-', lw=2)
    beam_line, = ax.plot([], [], 'k-', lw=2)
    body_dot, = ax.plot([], [], 'bo', zorder=5)
    traj_line, = ax.plot([], [], 'g--', alpha=0.5, lw=1)

    # --- Dynamic Wedge (Launch Sector) ---
    # Convert physics angles (CW from Y) to Matplotlib (CCW from X)
    vis_theta1 = 90 - env.launch_theta_max_deg
    vis_theta2 = 90 - env.launch_theta_min_deg

    sector_patch = patches.Wedge(
        (0, 0), env.R_outer, vis_theta1, vis_theta2,
        width=env.R_outer - env.r_inner,
        color='orange', alpha=0.2, label='Launch Zone'
    )
    ax.add_patch(sector_patch)

    info_text = ax.text(0.02, 0.9, "", transform=ax.transAxes,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    def init():
        return leg_line, beam_line, body_dot, traj_line, sector_patch, info_text

    def update(i):
        if i >= len(frames): return init()
        f = frames[i]

        # Geometry
        bx, by = f['q'][0], f['q'][1]
        lx, ly = f['left']
        rx, ry = f['right']
        tx, ty = f['tip']

        leg_line.set_data([bx, tx], [by, ty])
        beam_line.set_data([lx, rx], [ly, ry])
        body_dot.set_data([bx], [by])

        # Trajectory
        start = max(0, i - 100)
        xs = [fr['q'][0] for fr in frames[start:i]]
        ys = [fr['q'][1] for fr in frames[start:i]]
        traj_line.set_data(xs, ys)

        # Wedge Update
        sector_patch.set_center((f['origin_x'], 0))

        # Color Coding
        if f['flight_state'] == "PRE_LAUNCH":
            sector_patch.set_facecolor('orange')
            sector_patch.set_alpha(0.3)
        elif f['flight_state'] == "BALLISTIC":
            sector_patch.set_facecolor('green')
            sector_patch.set_alpha(0.2)
        else:
            sector_patch.set_facecolor('gray')
            sector_patch.set_alpha(0.05)

        info_text.set_text(f"Step: {f['step']}\nState: {f['flight_state']}\nTarget: {f['target_idx']}")
        return leg_line, beam_line, body_dot, traj_line, sector_patch, info_text

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, interval=30, blit=True)

    if save_gif:
        ani.save(filename, writer=PillowWriter(fps=30))
        print(f"Saved {filename}")

    plt.show()


# ==============================================================================
# 4. Main Execution
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "ppo_multihop_v2.pt"

    # 定义环境 (2跳任务示例)
    env = QuadHopperMultiJumpEnv(
        targets=[2.5, 5.0],
        r_inner=0.5, R_outer=1.8,
        launch_theta_min_deg=30, launch_theta_max_deg=60
    )

    policy_net = ActorCritic(obs_dim=6).to(device)

    # 模式选择
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        policy_net.load_state_dict(torch.load(model_path, map_location=device))

        # 可选：继续训练一小会儿
        # train_ppo(env, device, model_path, num_updates=200)
    else:
        print("No model found, starting fresh training...")
        train_ppo(env, device, model_path, num_updates=1000)

    # 运行可视化
    policy_net.eval()
    rollout_and_render_multihop(env, policy_net, device, max_steps=600, save_gif=True)