import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # 添加动画支持
import torch
import torch.nn as nn
import torch.optim as optim
import os
# Environment (flight phase, l fixed)
class TJumpEnvFlight2DOF:
    def __init__(self,
                 dt=0.005,
                 m1=0.2,  #COM of beam
                 m2=0.1,  #COM of leg
                 Jm=0.02,  # body inertia about COM
                 lc=0.5,  # half-span (COM to thruster arm)
                 l0=1.0,  # fixed leg length during flight
                 g=9.81,
                 thrust=1.0,  # each thruster when ON = 1 N
                 max_steps=3000,
                 target_x=4.0,
                 target_theta=0):
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
        self.target_theta = float(target_theta)

        self.reset()

    def reset(self, x=None, y=None, theta=np.pi/5,l = 1.0, dx=4.0, dy=4.0, dtheta= 0,dl = 0):
        # Sample initial state: x,y~U(0,1), theta~U(0,pi/2)
        if x is None: x = float(np.random.uniform(0.0, 1.0))
        if y is None: y = float(np.random.uniform(0.0, 1.0))
        if theta is None: theta = float(np.random.uniform(0.0, math.pi / 2.0))

        self.q = np.array([x, y, theta,l], dtype=np.float32)
        self.dq = np.array([dx, dy, dtheta,dl], dtype=np.float32)
        self.steps = 0
        self.prev_action = np.array([0, 0], dtype=np.int32)
        return self._obs()

    def _obs(self):
        return np.concatenate([self.q, self.dq]).astype(np.float32)

    def beam_endpoints(self):
        cx, cy, theta = float(self.q[0]), float(self.q[1]), float(self.q[2])
        # Beam is perpendicular to leg; here we visualize a beam of length 2*lc (end-to-end)
        dx = math.cos(theta) * self.lc
        dy = math.sin(theta) * self.lc
        left = (cx - dx + self.l0 * math.sin(theta), cy + dy + self.l0 * math.cos(theta))
        right = (cx + dx + self.l0 * math.sin(theta), cy - dy + self.l0 * math.cos(theta))
        return left, right

    # ------------- dynamics -------------
    def _compute_M_B_G_F(self, action):
        """
        Build reduced 3x3 dynamics (l fixed = l0, dl=0)
        Returns ddq plus intermediates.
        """
        x, y, theta, l = [float(v) for v in self.q]
        dx, dy, dtheta, dl = [float(v) for v in self.dq]
        m1, m2, Jm, g, lc = self.m1, self.m2, self.Jm, self.g, self.lc

        a = np.array(action, dtype=np.int32)
        F1 = self.thrust * float(a[0])
        F2 = self.thrust * float(a[1])

        c, s = math.cos(theta), math.sin(theta)

        # Inertia matrix M (3x3)
        # Inertia matrix M (3x3) for flight phase (l fixed)
        M = np.array([
            [m1 + m2, 0.0, l * m1 * c, m1 * s],
            [0.0, m1 + m2, -l * m1 * s, m1 * c],
            [l * m1 * c, -l * m1 * s, Jm + (l ** 2) * m1, 0],
            [m1 * s, m1 * c, 0, m1]
        ], dtype=np.float64)

        B = np.array([
            - m1 * (dtheta ** 2) * l * s,
            - m1 * (dtheta ** 2) * l * c,
            - 2.0 * m1 * l * dtheta * (dy * c + dx * s),
            (theta ** 2) * l * m1 + 2 * dtheta * dx * m1 * c - 2 * dtheta * dy * m1 * s
        ], dtype=np.float64)

        G = np.array([
            0.0,
            (m1 + m2) * g,
            - g * l * m1 * s,
            m1 * g * c
        ], dtype=np.float64)

        F = np.array([
            (F1 + F2) * s,
            (F1 + F2) * c,
            (F1 - F2) * lc,
            0
        ], dtype=np.float64)

        # Solve M ddq = F - B - G
        if y <= 0:  # 接触地面
            # 法向雅可比 W (1x4)
            W = np.array([[0, 1, 0, 0]], dtype=np.float64)
            W_dot = np.zeros_like(W)

            A = W @ np.linalg.inv(M) @ W.T
            rhs = W @ np.linalg.inv(M) @ (B + G - F) - W_dot @ self.dq
            lam = np.linalg.solve(A, rhs)

            # 加速度方程
            ddq = np.linalg.inv(M) @ (F - B - G + W.T @ lam)
        else:
            # 飞行阶段
            ddq = np.linalg.solve(M, F - B - G)

        return ddq, (M, B, G, F)

    def step(self, action):
        ddq, dyn = self._compute_M_B_G_F(action)

        # 更新速度和位置
        self.dq = (self.dq.astype(np.float64) + ddq * self.dt).astype(np.float64)
        self.q = (self.q.astype(np.float64) + self.dq * self.dt).astype(np.float64)

        # normalize theta to (-pi, pi]
        self.q[2] = (self.q[2] + math.pi) % (2 * math.pi) - math.pi

        self.steps += 1

        # 判断是否着陆或超过步数
        done = False
        landed = False
        if self.q[1] <= 0.0 or self.steps >= self.max_steps:
            done = True
            if self.q[1] <= 0.0:
                landed = True
                self.q[1] = 0.0
                self.dq[:] = 0.0

        # ---------------- Reward 设计 ----------------
        reward = 0.0
        max_angle_rad = math.radians(40)
        angle = self.q[2]
        if abs(angle) > max_angle_rad:
            # 大惩罚：角度超限
            reward -= 50.0 * (abs(angle) - max_angle_rad)

        # 1. 每步奖励：靠近目标 x，保持角度小，惩罚旋转速度
        pos_err = abs(self.q[0] - self.target_x)
        ang_err = abs(self.q[2] - self.target_theta )
        # 水平位置奖励 - 使用平方误差以强化接近目标
        reward -= 0.5 * (pos_err ** 2)

        # 角度奖励 - 使用平方误差
        reward -= 8.0 * (ang_err ** 2)

        # 角速度惩罚
        reward -= 1 * abs(self.dq[2])

        # 水平速度惩罚 - 但不要太严厉，需要水平移动
        reward -= 0.05 * abs(self.dq[0])


        # 2. 着陆时强化奖励
        if done and landed:
            # 着陆时的位置和角度误差
            pos_err = abs(self.q[0] - self.target_x)
            ang_err = abs(((self.q[2] - self.target_theta + math.pi) % (2 * math.pi)) - math.pi)

            # 着陆奖励 - 基于精度
            landing_reward = 100.0 * max(0, 2.5 - 5.0 * pos_err) * max(0, 3.0 - 10.0 * ang_err)
            reward += landing_reward

            # 成功着陆额外奖励
            if pos_err > 1.0 or ang_err > math.radians(20):
                reward = -1.0
            elif pos_err > 1.5 and ang_err > math.radians(20):
                reward = -250.0

        elif done and not landed:
            # 未着陆的惩罚
            reward -= 50.0

        obs = self._obs()
        info = {
            "ddq": ddq.astype(np.float32),
            "dq": self.dq.astype(np.float32),
            "landed": landed,
            "dyn": dyn,
            "action": np.array(action, dtype=np.int32)
        }
        self.prev_action = np.array(action, dtype=np.int32)
        return obs, float(reward), bool(done), info


# PPO (Bernoulli heads)
if torch is not None:
    class ActorCritic(nn.Module):
        def __init__(self, obs_dim=8, hidden=128):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 2)  # logits for two Bernoulli actions
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1)
            )

        def forward(self, x):
            logits = self.actor(x)
            value = self.critic(x).squeeze(-1)
            return logits, value


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    advs = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    # assume values is length T (values[t]) and we bootstrap with next_val = 0 when done
    for t in reversed(range(T)):
        if t == T-1:
            nextnonterminal = 0.0 if dones[t] else 1.0
            nextvalues = 0.0  # bootstrap 0 at end-of-batch; alternatively pass full trajectory value
        else:
            nextnonterminal = 0.0 if dones[t] else 1.0
            nextvalues = values[t+1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advs[t] = lastgaelam
    returns = advs + values
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    return returns, advs


def train_ppo(env,
              device,
              num_updates=3000,
              batch_size=4096,
              epochs=8,
              minibatch_size=512,
              gamma=0.99, lam=0.95,
              clip_eps=0.2,
              lr=3e-4):
    net = ActorCritic(obs_dim=8).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    all_rewards = []
    for update in range(num_updates):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        steps = 0
        while steps < batch_size:
            obs = env.reset()
            done = False
            ep_r = 0.0
            while not done and steps < batch_size:
                o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, v = net(o)
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

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                opt.zero_grad();
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        avg50 = np.mean(all_rewards[-50:]) if len(all_rewards) else 0.0
        print(f"[Update {update + 1}/{num_updates}] steps={len(obs_arr)} avg50={avg50:.2f}")

    return net, all_rewards



# Realtime rendering with animation and GIF saving

def rollout_and_render(env, policy_net=None, device=None, max_steps=2000, deterministic=False, save_gif=True):
    # 收集所有帧数据
    frames = []
    px, py = [], []

    obs = env.reset()
    done = False
    steps = 0

    # 先运行一次收集所有数据
    while not done and steps < max_steps:
        # ---- 选择动作 ----
        if policy_net is None:
            action = np.random.binomial(1, 0.5, size=(2,))
        else:
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy_net(o)
            p = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            action = (p > 0.5).astype(np.int32) if deterministic else np.random.binomial(1, p).astype(np.int32)

        # stata update
        obs, r, done, info = env.step(action)
        x, y, th = env.q[0], env.q[1], env.q[2]
        left, right = env.beam_endpoints()
        tip = (x + math.sin(th) * env.l0, y + math.cos(th) * env.l0)

        # 保存当前帧数据
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

    # 创建动画
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.set_title("Jumping Robot 2D Simulation")
    ax.set_xlim(-1.0, 5)
    ax.set_ylim(0.0, 3.0)
    ax.grid(True, alpha=0.3)
    beam_line, = ax.plot([], [], lw=3, label='Beam',color = 'black')
    leg_line, = ax.plot([], [], lw=3, label='Leg',color = 'black')
    com_point_m1, = ax.plot([], [], 'ro', markersize=8, label='Body COM')
    com_point_m2, = ax.plot([], [], 'bo', markersize=6, label='Leg COM')
    path_line, = ax.plot([], [], 'g--', lw=1, alpha=0.7, label='Trajectory')

    ground_line, = ax.plot([-1, 5], [0, 0], 'k-', lw=2, label='Ground')
    target_point, = ax.plot([env.target_x], [0], 'gX', markersize=12, label='Target')

    #information text
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.legend(loc='upper right')

    def init():
        beam_line.set_data([], [])
        leg_line.set_data([], [])
        com_point_m2.set_data([], [])
        com_point_m1.set_data([], [])
        path_line.set_data([], [])
        info_text.set_text("")
        return beam_line, leg_line, com_point_m2, com_point_m1, path_line, info_text

    def animate(i):
        if i >= len(frames):
            return init()

        frame = frames[i]
        x, y, th = frame['x'], frame['y'], frame['th']
        left, right, tip = frame['left'], frame['right'], frame['tip']

        #update icon
        beam_line.set_data([left[0], right[0]], [left[1], right[1]])
        leg_line.set_data([x, tip[0]], [y, tip[1]])
        com_point_m2.set_data([x], [y])
        com_point_m1.set_data([tip[0]], [tip[1]])
        path_line.set_data(px[:i + 1], py[:i + 1])

        # update text
        info_text.set_text(
            f"Step: {frame['step']}\n"
            f"Action: {frame['action']}\n"
            f"Position: ({x:.2f}, {y:.2f})\n"
            f"Theta: {math.degrees(th):.1f}°\n"
            f"Velocity: ({frame['dq'][0]:.2f}, {frame['dq'][1]:.2f})"
        )

        return beam_line, leg_line, com_point_m2, com_point_m1, path_line, info_text

    # 创建动画
    ani = FuncAnimation(fig, animate, frames=len(frames), init_func=init,
                        interval=50, blit=True, repeat=False)

    # 保存为GIF
    if save_gif:
        print("Saving animation as GIF...")
        ani.save('jumping_robot_simulation.gif', writer=PillowWriter(fps=20))
        print("GIF saved as 'jumping_robot_simulation.gif'")

    plt.tight_layout()
    plt.show()

    return ani


 # Main
if __name__ == "__main__":
    # ======== 参数配置 ========
    FORCE_TRAIN = True   # 是否强制重新训练
    MODEL_PATH = "ppo_jumping_robot.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TJumpEnvFlight2DOF(
        dt=0.005,
        m1=0.2, m2=0.1, Jm=0.02,
        lc=0.5, l0=1.0,
        g=9.81,
        thrust=1.0,  # F1,F2 ∈ {0,1} N
        max_steps=3000,
        target_x=4.0,
        target_theta=0
    )

    # ======== 判断是否加载或训练模型 ========
    if os.path.exists(MODEL_PATH) and not FORCE_TRAIN:
        # 如果模型存在且不强制训练 → 加载模型
        print(f"检测到已有模型，直接加载: {MODEL_PATH}")
        policy_net = ActorCritic(obs_dim=6).to(device)
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        policy_net.eval()

    else:
        # 如果模型不存在 或 强制训练 → 重新训练
        print("开始训练 PPO 策略网络...")
        policy_net, rewards = train_ppo(
            env,
            device=device,
            num_updates=3000,         # 训练轮数
            batch_size=4096,
            epochs=10,
            minibatch_size=512,
            gamma=0.99,
            lam=0.95,
            lr=3e-4
        )
        # 保存训练好的模型
        torch.save(policy_net.state_dict(), MODEL_PATH)
        print(f"模型训练完成并保存到: {MODEL_PATH}")
        policy_net.eval()

    # ======== 使用模型进行渲染 ========
    rollout_and_render(env, policy_net=policy_net, device=device,
                       deterministic=True, save_gif=True)

