import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.patches as patches

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
        # 观察中加入目标落地角度可能是个好主意，但为了简单，我们先保持原样，让PPO通过Reward去学
        obs = np.array([self.dq[0], self.dq[1], self.dq[2], self.q[2]], dtype=np.float32)
        return obs

    # --- 新增: Raibert 启发式计算 ---
    def _get_raibert_theta_target(self, current_vx):
        # Raibert 足端落点公式: x_f = (vx * Ts)/2 + k*(vx - v_des)
        # 我们希望 vx 保持在 v_des 附近
        xf = (current_vx * self.Ts) / 2.0 + self.k_r * (current_vx - self.v_des)

        # 几何关系: sin(theta) = xf / l0
        # 限制 xf 不超过腿长
        ratio = np.clip(xf / self.l0, -0.9, 0.9)

        # 关键: 我们需要“后仰”(Leaning Back)，意味着脚要在重心前面 (x_foot > x_com)
        # 在这里的坐标系中，如果 theta < 0，腿向右下偏，脚在右边（前面）。
        # 所以 theta 应该是负的 arcsin
        theta_target = -math.asin(ratio)
        return theta_target

    def beam_endpoints(self):
        cx, cy, theta = float(self.q[0]), float(self.q[1]), float(self.q[2])
        dx = math.cos(theta) * self.lc
        dy = math.sin(theta) * self.lc
        left = (cx - dx + self.l0 * math.sin(theta), cy + dy + self.l0 * math.cos(theta))
        right = (cx + dx + self.l0 * math.sin(theta), cy - dy + self.l0 * math.cos(theta))
        return left, right

    def _compute_dynamics(self, action):
        x, y, theta, l = [float(v) for v in self.q]
        if self.thrust_on:
            F1 = self.thrust * float(action[0])
            F2 = self.thrust * float(action[1])
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

    def step(self, action):
        x0, y0, theta0 = self.q[0], self.q[1], self.q[2]
        vx, vy, dtheta = self.dq[0], self.dq[1], self.dq[2]
        v0_sq_curr = vx ** 2 + vy ** 2

        # 1. 弹道计算 (Ballistics)
        dx_target = self.target_x - x0
        dy_target = 0.0 - y0
        cos_th = math.cos(theta0)
        tan_th = math.tan(theta0)
        denom = 2 * (cos_th ** 2) * (dx_target * tan_th - dy_target)

        v_req_sq = -1.0
        if denom > 1e-4 and dx_target > 0:
            v_req_sq = (self.g * (dx_target ** 2)) / denom

        # 2. Raibert 姿态计算
        # 计算理想的落地角度 (Leaning Back)
        raibert_theta_target = self._get_raibert_theta_target(vx)

        # 3. 预测落地时的实际角度
        # 估算剩余飞行时间 t_rem (只考虑重力影响的简单抛物线)
        # y(t) = y0 + vy*t - 0.5*g*t^2 = 0
        delta = vy ** 2 + 2 * self.g * y0
        t_rem = 0.0
        if delta >= 0:
            t_rem = (vy + math.sqrt(delta)) / self.g

        # 预测落地角度 (假设角速度不变，因为关机后力矩为0)
        pred_theta_land = theta0 + dtheta * t_rem

        # 4. 锁定逻辑 (Lock Logic)
        if self.thrust_on:
            dist_sq = x0 ** 2 + y0 ** 2
            cond_pos = (self.r_sq < dist_sq < self.R_sq)
            cond_ang = (self.angle_min_rad < theta0 < self.angle_max_rad)
            cond_vel = (v_req_sq > 0) and (abs(v0_sq_curr - v_req_sq) < 0.5)

            # [新增] 姿态锁定条件: 预测的落地角度必须接近 Raibert 目标
            # 我们稍微放宽一点容忍度，让 PPO 容易学到 (例如 +/- 15度)
            cond_att = abs(pred_theta_land - raibert_theta_target) < math.radians(15)

            if cond_pos and cond_ang and cond_vel and cond_att:
                self.thrust_on = False
                print(
                    f"🔒 LOCKED! v={math.sqrt(v0_sq_curr):.2f}, θ_land_pred={math.degrees(pred_theta_land):.1f}°, Target={math.degrees(raibert_theta_target):.1f}°")

        if not self.thrust_on:
            action = [0, 0]

        # 5. 物理积分
        ddq = self._compute_dynamics(action)
        self.dq += ddq * self.dt
        self.q += self.dq * self.dt
        self.q[2] = (self.q[2] + math.pi) % (2 * math.pi) - math.pi
        self.steps += 1

        # 6. Reward Calculation
        reward = 0.0
        done = False
        landed = False

        if self.thrust_on:
            reward -= 0.01
            # A. 速度引导
            if v_req_sq > 0:
                reward += 2.0 * np.exp(-5.0 * abs(v0_sq_curr - v_req_sq))
            else:
                reward -= 0.1

            # B. [新增] 姿态引导 (Attitude Guidance)
            # 鼓励现在的状态能导致正确的落地姿态
            att_err = abs(pred_theta_land - raibert_theta_target)
            reward += 1.5 * np.exp(-5.0 * att_err)

        else:
            reward -= 0.001

        # Check Landing
        if self.q[1] <= 0.0:
            done = True
            landed = True
            self.q[1] = 0.0
        elif self.steps >= self.max_steps or abs(self.q[2]) > np.pi / 2:
            done = True
            reward -= 10.0

        if landed:
            # 落地精度奖励
            dist_error = abs(self.q[0] - self.target_x)
            if dist_error < 0.2:
                reward += 20.0
            else:
                reward -= dist_error * 2.0

            # [新增] 落地姿态最终奖励
            final_att_err = abs(self.q[2] - raibert_theta_target)
            if final_att_err < math.radians(10):
                reward += 10.0  # 完美姿态落地
            else:
                reward -= 5.0 * final_att_err

        info = {"dq": self.dq, "ddq": ddq, "thrust_on": self.thrust_on}
        return self._obs(), float(reward), done, info

if torch is not None:
    class ActorCritic(nn.Module):
        def __init__(self, obs_dim=4, hidden=512):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 2)
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1)
            )
            # Initialize weights to prevent large initial outputs
            self._initialize_weights()

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

        def forward(self, x):
            logits = self.actor(x)
            value = self.critic(x).squeeze(-1)
            return logits, value


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
              model_path="ppo_jumping_robot.pt"):
    if policy_net is None:
        net = ActorCritic(obs_dim=4).to(device)
        print("创建新的 PPO 策略网络")
    else:
        net = policy_net
        print("继续训练已有 PPO 策略网络")

    opt = optim.Adam(net.parameters(), lr=lr)
    all_rewards = []
    best_avg = -1e9
    best_state = None

    for update in range(num_updates):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        steps = 0

        while steps < batch_size:
            obs = env.reset()
            done = False
            ep_r = 0.0
            while not done and steps < batch_size:
                o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if torch.any(torch.isnan(o)) or torch.any(torch.isinf(o)):
                    print("Warning: NaN or inf in observation input to network:", o)
                    break
                logits, v = net(o)
                if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                    print("Warning: NaN or inf in logits:", logits)
                    break
                logits = logits.squeeze(0)
                v = v.item()
                dist = torch.distributions.Bernoulli(logits=logits)
                action = dist.sample().cpu().numpy().astype(np.float32)
                logp = dist.log_prob(torch.tensor(action, dtype=torch.float32, device=device)).sum().item()

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

                logits, val = net(o)
                dist = torch.distributions.Bernoulli(logits=logits)
                new_logp = dist.log_prob(a).sum(dim=-1)
                ratio = torch.exp(new_logp - ol)
                s1 = ratio * ad
                s2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * ad
                policy_loss = -torch.mean(torch.min(s1, s2))
                value_loss = torch.mean((rt - val) ** 2)
                entropy = torch.mean(dist.entropy().sum(dim=-1))

                loss = policy_loss + 0.5 * value_loss - 0.05 * entropy
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


def rollout_and_render(env, policy_net=None, device=None, max_steps=400, deterministic=False, save_gif=True):
    frames = []
    px, py = [], []

    obs = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        if policy_net is None:
            action = np.random.binomial(1, 0.5, size=(2,))
        else:
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy_net(o)
            p = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            action = (p > 0.5).astype(np.int32) if deterministic else np.random.binomial(1, p).astype(np.int32)

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
    ax.set_title("Jumping Robot 2D Simulation")
    ax.set_xlim(-1.0, 4)
    ax.set_ylim(0.0, 2.0)
    ax.grid(True, alpha=0.3)

    # Target Box (representing the constraint r < |x0, y0| < R)
    target_circle = plt.Circle((0, 0), np.sqrt(env.R_sq), color='red', fill=False, linestyle='--', linewidth=1,
                               alpha=0.3, label='$R$ boundary')
    ax.add_patch(target_circle)
    inner_circle = plt.Circle((0, 0), np.sqrt(env.r_sq), color='red', fill=False, linestyle='--', linewidth=1,
                              alpha=0.3, label='$r$ boundary')
    ax.add_patch(inner_circle)

    beam_line, = ax.plot([], [], lw=3, label='Beam', color='black')
    leg_line, = ax.plot([], [], lw=3, label='Leg', color='black')
    com_point_m1, = ax.plot([], [], 'ko', markersize=4, label='Body COM')
    com_point_m2, = ax.plot([], [], 'ko', markersize=4, label='Leg COM')
    left_dot, = ax.plot([], [], 'o', markersize=5)   # beam 左端点
    right_dot, = ax.plot([], [], 'o', markersize=5)  # beam 右端点
    path_line, = ax.plot([], [], 'g--', lw=1, alpha=0.7, label='Trajectory')
    ground_line, = ax.plot([-1, 4], [0, 0], 'k-', lw=2, label='Ground')
    target_point, = ax.plot([env.target_x], [0], 'gX', markersize=12, label='Target')
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(loc='upper right')

    def init():
        beam_line.set_data([], [])
        leg_line.set_data([], [])
        com_point_m2.set_data([], [])
        com_point_m1.set_data([], [])
        left_dot.set_data([], [])
        right_dot.set_data([], [])
        path_line.set_data([], [])
        info_text.set_text("")
        return beam_line, leg_line, com_point_m2, com_point_m1, left_dot, right_dot, path_line, info_text

    def animate(i):
        if i >= len(frames):
            return init()
        frame = frames[i]
        x, y, th = frame['x'], frame['y'], frame['th']
        left, right, tip = frame['left'], frame['right'], frame['tip']
        action = frame['action']

        # beam & leg
        beam_line.set_data([left[0], right[0]], [left[1], right[1]])
        leg_line.set_data([x, tip[0]], [y, tip[1]])
        com_point_m2.set_data([x], [y])
        com_point_m1.set_data([tip[0]], [tip[1]])
        path_line.set_data(px[:i + 1], py[:i + 1])

        # 左端点：action[0]
        left_dot.set_data([left[0]], [left[1]])
        left_dot.set_color('blue' if action[0] == 0 else 'red')

        # 右端点：action[1]
        right_dot.set_data([right[0]], [right[1]])
        right_dot.set_color('blue' if action[1] == 0 else 'red')

        # info text
        info_text.set_text(
            f"Step: {frame['step']}\n"
            f"Action: {frame['action']}\n"
            f"Position: ({x:.2f}, {y:.2f})\n"
            f"Theta: {math.degrees(th):.1f}°\n"
            f"Velocity: ({frame['dq'][0]:.2f}, {frame['dq'][1]:.2f})"
        )
        return beam_line, leg_line, com_point_m2, com_point_m1, left_dot, right_dot, path_line, info_text

    ani = FuncAnimation(fig, animate, frames=len(frames), init_func=init,
                        interval=50, blit=True, repeat=False)

    if save_gif:
        print("Saving animation as GIF...")
        ani.save('jumping_robot_simulation_ground.gif', writer=PillowWriter(fps=20))
        print("GIF saved as 'jumping_robot_simulation_ground.gif'")

    plt.tight_layout()
    plt.show()
    return ani


# PPO (Bernoulli heads) and helper functions (ActorCritic, compute_gae, train_ppo)
# ... (These sections remain the same as in your original code) ...

# ----------------------------------------------------------------------
# Main Execution Block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    FORCE_TRAIN = True
    CONTINUE_TRAIN = False
    MODEL_PATH = "ppo_jumping_robot.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the Environment with new parameters
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

    # --- Training/Loading Logic ---
    if FORCE_TRAIN or not os.path.exists(MODEL_PATH):
        # Start a new training
        policy_net = ActorCritic(obs_dim=4).to(device)
        print("Start a new training...")
        train_ppo(env, device, num_updates=3000, lr=3e-4, model_path=MODEL_PATH)

    elif CONTINUE_TRAIN:
        # Load and continue training
        print(f"Load an existing model and continue training: {MODEL_PATH}")
        policy_net = ActorCritic(obs_dim=4).to(device)
        policy_net.load_state_dict(torch.load(
            MODEL_PATH, map_location=device, weights_only=True
        ))
        policy_net.train()
        train_ppo(env, device, num_updates=2000, lr=1e-4, policy_net=policy_net, model_path=MODEL_PATH)

    else:
        # Load for inference
        print(f"Load the best existing model for inference: {MODEL_PATH}")
        policy_net = ActorCritic(obs_dim=4).to(device)
        policy_net.load_state_dict(torch.load(
            MODEL_PATH, map_location=device, weights_only=True
        ))
        policy_net.eval()

    # --- Run Simulation ---
    rollout_and_render(env, policy_net=policy_net, device=device,
                       deterministic=True, save_gif=True)