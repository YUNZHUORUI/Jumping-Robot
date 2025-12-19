import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.patches as patches


# Environment (flight phase, l fixed)
class TJumpEnvFlight2DOF:
    def __init__(self,
                 dt=0.005,
                 m1=0.0363, m2=0.01, Jm=0.02, lc=0.25, l0=0.50, g=9.81, thrust=0.314,
                 max_steps=400,
                 target_x=2.0,
                 # --- Raibert Parameters ---
                 stance_duration=0.2,  # 预估触地时间 (s)
                 desired_vx=1.0,  # 下一跳的期望速度 (m/s)
                 k_raibert=0.1,  # 速度修正增益
                 # --- Ballistic Parameters ---
                 r_sq=0.5 ** 2, R_sq=2.0 ** 2,
                 theta_min_deg=30, theta_max_deg=60):

        self.dt = float(dt)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.Jm = float(Jm)
        self.lc = float(lc)
        self.l0 = float(l0)
        self.g = float(g)
        self.thrust = float(thrust)
        self.max_steps = int(max_steps)
        self.target_x = float(target_x)

        # Raibert Params
        self.Ts = stance_duration
        self.v_des = desired_vx
        self.k_r = k_raibert

        # Launch Zone
        self.r_sq = r_sq
        self.R_sq = R_sq
        self.angle_min_rad = math.radians(theta_min_deg)
        self.angle_max_rad = math.radians(theta_max_deg)

        self.reset()

    def reset(self, x=0, y=0, theta=None, l=1.0, dx=None, dy=None, dtheta=0, dl=0):
        # Initial State
        if dx is None: dx = float(np.random.uniform(0.5, 1.5))
        if dy is None: dy = float(np.random.uniform(0.5, 1.5))
        if theta is None: theta = float(np.random.uniform(math.pi / 6, math.pi / 3.0))

        self.q = np.array([x, y, theta, l], dtype=np.float32)
        self.dq = np.array([dx, dy, dtheta, dl], dtype=np.float32)
        self.steps = 0
        self.thrust_on = True
        return self._obs()

    def _obs(self):
        # --- 关键修改：加入 x, y ---
        # 没有位置信息，Agent 无法知道自己是否在扇形区域内，学习曲线会很差。
        obs = np.array([
            self.q[0], self.q[1],  # x, y
            self.dq[0], self.dq[1],  # dx, dy
            self.dq[2], self.q[2]  # dtheta, theta
        ], dtype=np.float32)
        return obs

    def beam_endpoints(self):
        cx, cy, theta = float(self.q[0]), float(self.q[1]), float(self.q[2])
        dx = math.cos(theta) * self.lc
        dy = math.sin(theta) * self.lc
        left = (cx - dx + self.l0 * math.sin(theta), cy + dy + self.l0 * math.cos(theta))
        right = (cx + dx + self.l0 * math.sin(theta), cy - dy + self.l0 * math.cos(theta))
        return left, right

    def _compute_dynamics(self, action):
        x, y, theta, l = [float(v) for v in self.q]

        # --- 动作处理 ---
        # 假设 action 是连续值 [-1, 1] 或类似的范围
        # 我们将其截断到 [0, 1] 范围来模拟推力比例 (0% 到 100%)
        # 或者如果你想要 Bang-Bang 控制，可以做 round。这里使用连续推力。
        a1 = np.clip(float(action[0]), 0.0, 1.0)
        a2 = np.clip(float(action[1]), 0.0, 1.0)

        if self.thrust_on:
            F1 = self.thrust * a1
            F2 = self.thrust * a2
        else:
            F1, F2 = 0.0, 0.0

        c, s = math.cos(theta), math.sin(theta)
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

        G = np.array([0.0, (self.m1 + self.m2) * self.g, - self.g * l * self.m1 * s, self.m1 * self.g * c],
                     dtype=np.float64)
        F = np.array([(F1 + F2) * s, (F1 + F2) * c, (F1 - F2) * self.lc, 0], dtype=np.float64)

        try:
            ddq = np.linalg.solve(M, F - B - G)
        except np.linalg.LinAlgError:
            ddq = np.zeros(4)
        return ddq

    def _compute_time_to_impact(self, y0, dy0):
        if y0 < 0: return 0.0
        disc = dy0 ** 2 + 2 * self.g * y0
        if disc < 0: return 0.0
        t_impact = (dy0 + np.sqrt(disc)) / self.g
        return t_impact

    def step(self, action):
        # 1. 实时计算弹道需求
        x0, y0, theta0 = self.q[0], self.q[1], self.q[2]
        dx0, dy0, dtheta0 = self.dq[0], self.dq[1], self.dq[2]

        dx_target = self.target_x - x0
        dy_target = 0.0 - y0

        cos_th = math.cos(theta0)
        tan_th = math.tan(theta0)
        denom = 2 * (cos_th ** 2) * (dx_target * tan_th - dy_target)
        v0_sq_curr = dx0 ** 2 + dy0 ** 2

        v_req_sq = -1.0
        if denom > 1e-4 and dx_target > 0:
            v_req_sq = (self.g * (dx_target ** 2)) / denom

        # --- Raibert Attitude ---
        t_rem = self._compute_time_to_impact(y0, dy0)

        Ts = 0.25
        k_speed = 0.1
        vx_current = dx0
        vx_desired = 1.0

        foot_offset_x = (vx_current * Ts / 2.0) + k_speed * (vx_current - vx_desired)
        foot_offset_x = np.clip(foot_offset_x, -0.2, 0.2)

        if self.l0 > 1e-3:
            theta_land_raibert = -math.asin(foot_offset_x / self.l0)
        else:
            theta_land_raibert = 0.0

        dtheta_req = 0.0
        if t_rem > 0.05:
            dtheta_req = (theta_land_raibert - theta0) / t_rem
        else:
            dtheta_req = dtheta0

        # --- 2. 锁定逻辑判定 ---
        if self.thrust_on:
            dist_sq = x0 ** 2 + y0 ** 2
            v0_curr = math.sqrt(v0_sq_curr) if v0_sq_curr > 0 else 0
            v_req = math.sqrt(v_req_sq) if v_req_sq > 0 else 0

            cond_pos = (self.r_sq < dist_sq < self.R_sq)
            cond_ang = (self.angle_min_rad < theta0 < self.angle_max_rad)
            cond_vel = (v_req_sq > 0) and (abs(v0_curr - v_req) < 0.5)
            cond_omega = abs(dtheta0 - dtheta_req) < 1.0

            if cond_pos and cond_ang and cond_vel and cond_omega:
                self.thrust_on = False
                print(f"🔒 LOCK [Raibert]: x={x0:.2f}, v_err={abs(v0_sq_curr - v_req_sq):.2f}, "
                      f"Ang_Tgt={math.degrees(theta_land_raibert):.1f}°, w_err={abs(dtheta0 - dtheta_req):.2f}")

        # 如果 thrust 关机，强制 action 为 0
        if not self.thrust_on:
            # 即使网络输出是非零，物理上也强制为0
            # 这里我们传入网络原始输出给 _compute_dynamics，但在那里我们通过 if self.thrust_on 控制力
            pass

            # --- 3. 物理步进 ---
        ddq = self._compute_dynamics(action)
        self.dq += ddq * self.dt
        self.q += self.dq * self.dt
        self.q[2] = (self.q[2] + math.pi) % (2 * math.pi) - math.pi
        self.steps += 1

        # --- 4. 奖励计算 ---
        reward = 0.0
        done = False
        landed = False

        if self.thrust_on:
            reward -= 0.01
            # 动作惩罚 (节省燃料)
            reward -= 0.005 * (np.sum(np.square(action)))

            if v_req_sq > 0:
                v_err = abs(v0_sq_curr - v_req_sq)
                # 使用较平缓的奖励梯度，避免梯度消失
                reward += 2.0 * np.exp(-2.0 * v_err)
            else:
                reward -= 0.1

            if t_rem > 0.05:
                w_err = abs(dtheta0 - dtheta_req)
                reward += 1.0 * np.exp(-2.0 * w_err)

            if not (self.angle_min_rad < theta0 < self.angle_max_rad):
                reward -= 0.1

        else:
            reward -= 0.001

        if self.q[1] <= 0.0:
            done = True
            landed = True
            self.q[1] = 0.0
        elif self.steps >= self.max_steps:
            done = True
        elif abs(self.q[2]) > np.pi:
            done = True
            reward -= 10.0

        if landed:
            dist_error = abs(self.q[0] - self.target_x)
            if dist_error < 0.2:
                reward += 20.0
            elif dist_error < 1.0:
                reward += (1.0 - dist_error) * 5.0
            else:
                reward -= dist_error * 2.0

            final_foot_offset = (self.dq[0] * Ts / 2.0) + k_speed * (self.dq[0] - 1.0)
            final_foot_offset = np.clip(final_foot_offset, -0.2, 0.2)
            ideal_land_theta = -math.asin(final_foot_offset / self.l0)

            angle_error = abs(self.q[2] - ideal_land_theta)
            if angle_error < 0.2:
                reward += 10.0
            else:
                reward -= angle_error * 5.0

        info = {"dq": self.dq, "ddq": ddq, "thrust_on": self.thrust_on}
        return self._obs(), float(reward), done, info


# --- PPO Network (Continuous) ---
if torch is not None:
    class ActorCritic(nn.Module):
        def __init__(self, obs_dim=6, hidden=256):  # Obs dim changed to 6
            super().__init__()
            self.actor_mean = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 2),  # Output 2 actions
                nn.Tanh()  # Tanh to force range [-1, 1], we map to [0, 1] later
            )
            self.log_std = nn.Parameter(torch.zeros(2) - 0.5)  # Initialize with small std
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1)
            )

        def forward(self, x):
            mean = self.actor_mean(x)
            std = torch.exp(self.log_std)
            value = self.critic(x).squeeze(-1)
            return mean, std, value

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)


def compute_gae(rewards, values, dones, gamma=0.995, lam=0.95):
    T = len(rewards)
    advs = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 0.0 if dones[t] else 1.0
            nextvalues = 0.0
        else:
            nextnonterminal = 0.0 if dones[t] else 1.0
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advs[t] = lastgaelam
    returns = advs + values
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    return returns, advs


def train_ppo(env,
              device,
              num_updates=3000,
              batch_size=2000,
              epochs=4,
              minibatch_size=512,
              gamma=0.995,
              lam=0.95,
              clip_eps=0.2,
              lr=3e-4,
              policy_net=None,
              model_path="ppo_jumping_robot_raibert.pt"):
    # 修正: obs_dim=6
    if policy_net is None:
        net = ActorCritic(obs_dim=6).to(device)
        print("创建新的 PPO 策略网络 (Continuous)")
    else:
        net = policy_net
        print("继续训练已有 PPO 策略网络")

    opt = optim.Adam(net.parameters(), lr=lr)
    all_rewards = []
    best_avg = -1e9

    for update in range(num_updates):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        steps = 0

        while steps < batch_size:
            obs = env.reset()
            done = False
            ep_r = 0.0
            while not done and steps < batch_size:
                o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                # --- 修正: 连续动作采样 ---
                # 解包 mean, std, value
                mean, std, v = net(o)
                mean = mean.squeeze(0)
                std = std.squeeze(0)  # or just use std directly since it's parameter broadcast
                v = v.item()

                dist = torch.distributions.Normal(mean, std)
                action_tensor = dist.sample()

                # 计算 log_prob
                logp = dist.log_prob(action_tensor).sum().item()

                # 转为 numpy 传入 env
                action = action_tensor.cpu().numpy()

                # Env Step (Environment handles clamping/interpretation)
                next_obs, r, done, info = env.step(action)

                obs_buf.append(obs.copy())
                act_buf.append(action.copy())
                logp_buf.append(logp)
                rew_buf.append(r)
                val_buf.append(v)
                done_buf.append(done)

                obs = next_obs
                ep_r += r
                steps += 1
                if done:
                    all_rewards.append(ep_r)

        obs_arr = np.array(obs_buf, dtype=np.float32)
        act_arr = np.array(act_buf, dtype=np.float32)
        logp_arr = np.array(logp_buf, dtype=np.float32)
        rew_arr = np.array(rew_buf, dtype=np.float32)
        val_arr = np.array(val_buf, dtype=np.float32)
        done_arr = np.array(done_buf, dtype=np.bool_)

        returns, advs = compute_gae(rew_arr, val_arr, done_arr, gamma, lam)

        idx = np.arange(len(obs_arr))
        for _ in range(epochs):
            np.random.shuffle(idx)
            for st in range(0, len(idx), minibatch_size):
                mb = idx[st:st + minibatch_size]
                o = torch.tensor(obs_arr[mb], dtype=torch.float32, device=device)
                a = torch.tensor(act_arr[mb], dtype=torch.float32, device=device)
                ol = torch.tensor(logp_arr[mb], dtype=torch.float32, device=device)
                ad = torch.tensor(advs[mb], dtype=torch.float32, device=device)
                rt = torch.tensor(returns[mb], dtype=torch.float32, device=device)

                # --- 修正: 连续动作 Update ---
                mean, std, val = net(o)
                dist = torch.distributions.Normal(mean, std)

                new_logp = dist.log_prob(a).sum(dim=-1)
                ratio = torch.exp(new_logp - ol)
                s1 = ratio * ad
                s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * ad
                policy_loss = -torch.mean(torch.min(s1, s2))
                value_loss = torch.mean((rt - val) ** 2)
                entropy = torch.mean(dist.entropy().sum(dim=-1))

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        avg50 = np.mean(all_rewards[-50:]) if len(all_rewards) else 0.0
        print(f"[Update {update + 1}/{num_updates}] steps={len(obs_arr)} avg50={avg50:.2f}")

        if avg50 > best_avg:
            best_avg = avg50
            best_state = net.state_dict()
            torch.save(best_state, model_path)
            print(f"❗️Best Model saving, average reward = {best_avg:.2f}")

    return net, all_rewards, best_avg


def rollout_and_render(env, policy_net=None, device=None, max_steps=400, deterministic=True, save_gif=True):
    frames = []
    px, py = [], []

    obs = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        if policy_net is None:
            action = np.random.uniform(0, 1, size=(2,))
        else:
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = policy_net(o)

            if deterministic:
                # 确定性模式：直接取 mean
                action = mean.squeeze(0).cpu().numpy()
            else:
                # 随机模式：采样
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().squeeze(0).cpu().numpy()

        # Step
        obs, r, done, info = env.step(action)

        x, y, th = env.q[0], env.q[1], env.q[2]
        left, right = env.beam_endpoints()
        tip = (x + math.sin(th) * env.l0, y + math.cos(th) * env.l0)

        frame_data = {
            'x': x, 'y': y, 'th': th,
            'left': left, 'right': right, 'tip': tip,
            'action': action.copy(),
            'dq': info['dq'].copy(),
            'ddq': info['ddq'].copy(),
            'step': steps
        }
        frames.append(frame_data)
        px.append(x)
        py.append(y)
        steps += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.set_title("Jumping Robot - Continuous Control")
    ax.set_xlim(-1.0, 4)
    ax.set_ylim(0.0, 2.0)
    ax.grid(True, alpha=0.3)

    target_circle = plt.Circle((0, 0), np.sqrt(env.R_sq), color='red', fill=False, linestyle='--', linewidth=1)
    ax.add_patch(target_circle)
    inner_circle = plt.Circle((0, 0), np.sqrt(env.r_sq), color='red', fill=False, linestyle='--', linewidth=1)
    ax.add_patch(inner_circle)

    beam_line, = ax.plot([], [], lw=3, label='Beam', color='black')
    leg_line, = ax.plot([], [], lw=3, label='Leg', color='black')
    left_dot, = ax.plot([], [], 'o', markersize=5)
    right_dot, = ax.plot([], [], 'o', markersize=5)
    path_line, = ax.plot([], [], 'g--', lw=1, alpha=0.7)
    ax.plot([-1, 4], [0, 0], 'k-', lw=2)
    ax.plot([env.target_x], [0], 'gX', markersize=12)
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    def init():
        beam_line.set_data([], [])
        leg_line.set_data([], [])
        left_dot.set_data([], [])
        right_dot.set_data([], [])
        path_line.set_data([], [])
        info_text.set_text("")
        return beam_line, leg_line, left_dot, right_dot, path_line, info_text

    def animate(i):
        if i >= len(frames): return init()
        frame = frames[i]
        x, y, th = frame['x'], frame['y'], frame['th']
        left, right, tip = frame['left'], frame['right'], frame['tip']
        action = frame['action']

        # Clip action for color display to 0-1 (even if physics allowed <0)
        a_disp = np.clip(action, 0, 1)

        beam_line.set_data([left[0], right[0]], [left[1], right[1]])
        leg_line.set_data([x, tip[0]], [y, tip[1]])
        path_line.set_data(px[:i + 1], py[:i + 1])

        left_dot.set_data([left[0]], [left[1]])
        left_dot.set_color(plt.cm.coolwarm(a_disp[0]))  # Color by thrust intensity

        right_dot.set_data([right[0]], [right[1]])
        right_dot.set_color(plt.cm.coolwarm(a_disp[1]))

        info_text.set_text(
            f"Step: {frame['step']}\n"
            f"Action: [{action[0]:.2f}, {action[1]:.2f}]\n"
            f"Pos: ({x:.2f}, {y:.2f})\n"
            f"Vel: ({frame['dq'][0]:.2f}, {frame['dq'][1]:.2f})"
        )
        return beam_line, leg_line, left_dot, right_dot, path_line, info_text

    ani = FuncAnimation(fig, animate, frames=len(frames), init_func=init,
                        interval=50, blit=True, repeat=False)

    if save_gif:
        ani.save('jumping_robot_raibert_continuous.gif', writer=PillowWriter(fps=20))

    plt.tight_layout()
    plt.show()
    return ani


# if __name__ == "__main__":
#     FORCE_TRAIN = True
#     CONTINUE_TRAIN = False
#     MODEL_PATH = "ppo_jumping_robot_raibert.pt"
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     env = TJumpEnvFlight2DOF()
#
#     # 训练
#     if FORCE_TRAIN:
#         print("Starting training...")
#         # obs_dim 必须是 6
#         net, rewards, best_avg = train_ppo(env, device, num_updates=3000, model_path=MODEL_PATH)
#
#         # 绘图
#         plt.figure()
#         plt.plot(rewards)
#         plt.title("Learning Curve")
#         plt.savefig("learning_curve.png")
#         plt.show()
#
#     # 演示
#     print("Simulating...")
#     net = ActorCritic(obs_dim=6).to(device)
#     net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
#     net.eval()
#     rollout_and_render(env, policy_net=net, device=device, deterministic=True)


# ----------------------------------------------------------------------
# Main Execution Block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    FORCE_TRAIN = True
    CONTINUE_TRAIN = False
    MODEL_PATH = "ppo_jumping_robot.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TJumpEnvFlight2DOF(
        dt=0.005,
        m1=0.0363, m2=0.01, Jm=0.02, lc=0.25, l0=0.50, g=9.81, thrust=0.314,
        max_steps=400,
        target_x=2.0,
        # --- Raibert Parameters ---
        stance_duration=0.2,  # 预估触地时间 (s)
        desired_vx=1.0,  # 下一跳的期望速度 (m/s)
        k_raibert=0.1,  # 速度修正增益
        # --- Ballistic Parameters ---
        r_sq=0.5 ** 2, R_sq=2.0 ** 2,
        theta_min_deg=30, theta_max_deg=60
    )

    policy_net = None
    rewards = []  # 保存训练过程中的奖励

    if FORCE_TRAIN or not os.path.exists(MODEL_PATH):
        print("Start a new training...")
        policy_net, rewards, best_avg = train_ppo(
            env, device,
            num_updates=3000,
            batch_size=2000,
            epochs=4,
            minibatch_size=1000,
            gamma=0.995, lam=0.96,
            clip_eps=0.2,
            lr=3e-4,
            model_path=MODEL_PATH
        )
        print(f"Training completed, best average reward = {best_avg:.2f}")

    elif CONTINUE_TRAIN:
        print(f"Load an existing model and continue training: {MODEL_PATH}")
        policy_net = ActorCritic(obs_dim=4).to(device)
        policy_net.load_state_dict(torch.load(
            MODEL_PATH, map_location=device, weights_only=True
        ))
        policy_net.train()
        policy_net, rewards, best_avg = train_ppo(
            env, device,
            num_updates=2000,
            batch_size=1000,
            epochs=4,
            minibatch_size=512,
            gamma=0.995, lam=0.95,
            clip_eps=0.2,
            lr=1e-4,
            policy_net=policy_net,
            model_path=MODEL_PATH
        )
        print(f"Continued training finish, best average reward = {best_avg:.2f}")

    else:
        print(f"Load the best existing model for inference: {MODEL_PATH}")
        policy_net = ActorCritic(obs_dim=4).to(device)
        policy_net.load_state_dict(torch.load(
            MODEL_PATH, map_location=device, weights_only=True
        ))
        policy_net.eval()


    # 绘制学习曲线
    def plot_learning_curve(rewards, model_path="ppo_jumping_robot.pt"):
        if not rewards:
            print("No rewards data to plot.")
            return

        # 计算移动平均（例如最近100个episode）
        window = 100
        if len(rewards) < window:
            window = len(rewards)

        # 移动平均
        rewards_np = np.array(rewards)
        moving_avg = np.convolve(rewards_np, np.ones(window) / window, mode='valid')

        plt.figure(figsize=(12, 6))

        # 绘制原始奖励
        plt.subplot(1, 2, 1)
        plt.plot(rewards_np, alpha=0.3, color='blue', label='Episode Reward')
        plt.plot(np.arange(window - 1, len(rewards_np)), moving_avg, color='red', lw=2,
                 label=f'Moving Average ({window} episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('PPO Learning Curve - Raw Rewards')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 绘制平滑后的奖励
        plt.subplot(1, 2, 2)
        # 使用更大的窗口进行平滑
        smooth_window = min(50, len(rewards) // 10)
        if smooth_window > 1:
            smooth_rewards = np.convolve(rewards_np, np.ones(smooth_window) / smooth_window, mode='valid')
            plt.plot(np.arange(smooth_window - 1, len(rewards_np)), smooth_rewards, color='green', lw=2,
                     label=f'Smoothed ({smooth_window} episodes)')
        plt.plot(rewards_np, alpha=0.2, color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.title('PPO Learning Curve - Smoothed')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        # 保存图像
        plot_filename = os.path.splitext(model_path)[0] + "_learning_curve.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved as {plot_filename}")

        # 显示统计信息
        print(f"\nTraining Statistics:")
        print(f"Total episodes: {len(rewards)}")
        print(f"Final reward: {rewards[-1]:.2f}")
        print(f"Best reward: {np.max(rewards):.2f}")
        print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Last {window} episodes average: {np.mean(rewards[-window:]):.2f}")

        plt.show()


    # 调用学习曲线绘制函数
    if rewards:  # 只有在有训练数据时才绘制
        plot_learning_curve(rewards, MODEL_PATH)
    else:
        print("No training rewards data available for plotting.")

    # 进行演示
    rollout_and_render(env, policy_net=policy_net, device=device,
                       deterministic=True, save_gif=True)