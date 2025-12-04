import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
import torch.optim as optim
import os


# Environment (flight phase, l fixed)
class TJumpEnvFlight2DOF:
    def __init__(self,
                 dt=0.005,
                 m1=0.3,  # COM of beam mass 300g
                 m2=0.03,  # COM of leg mass 30g
                 Jm=0.02,  # body inertia about COM
                 lc=0.5,  # half-span (COM to thruster arm)
                 l0=1.0,  # fixed leg length during flight
                 g=9.81,
                 thrust=1.0,  # each thruster when ON = 1 N
                 max_steps=350,
                 target_x=2,
                 target_theta=-np.pi / 7):
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
        #new
        self.thrust_on = True
        self.target_reached_state = None  # Store state when thrust is locked

    def reset(self, x=0.5, y=0.5, theta=None, l=1.0, dx=None, dy=None, dtheta=0, dl=0):
        if dx is None: dx = float(np.random.uniform(1.0, 2.0))
        if dy is None: dy = float(np.random.uniform(1.0, 2.0))
        if theta is None: theta = float(np.random.uniform(math.pi/6, math.pi / 3.0))
        if dtheta is None: dtheta = float(np.random.uniform(math.pi/6, math.pi / 3.0))
        self.thrust_on = True
        self.target_reached_state = None

        self.q = np.array([x, y, theta, l], dtype=np.float32)
        self.dq = np.array([dx, dy, dtheta, dl], dtype=np.float32)
        self.steps = 0
        self.prev_action = np.array([0, 0], dtype=np.int32)

        # Check for invalid values
        if np.any(np.isnan(self.q)) or np.any(np.isinf(self.q)):
            print("Warning: NaN or inf in reset q:", self.q)
        if np.any(np.isnan(self.dq)) or np.any(np.isinf(self.dq)):
            print("Warning: NaN or inf in reset dq:", self.dq)

        return self._obs()

    def _obs(self):
        # Return [dx, dy, dtheta, theta]
        obs = np.array([self.dq[0], self.dq[1], self.dq[2], self.q[2]], dtype=np.float32)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print("Warning: NaN or inf in observation:", obs)
        return obs

    def beam_endpoints(self):
        cx, cy, theta = float(self.q[0]), float(self.q[1]), float(self.q[2])
        dx = math.cos(theta) * self.lc
        dy = math.sin(theta) * self.lc
        left = (cx - dx + self.l0 * math.sin(theta), cy + dy + self.l0 * math.cos(theta))
        right = (cx + dx + self.l0 * math.sin(theta), cy - dy + self.l0 * math.cos(theta))
        return left, right

    def _compute_M_B_G_F(self, action):
        x, y, theta, l = [float(v) for v in self.q]
        dx, dy, dtheta, dl = [float(v) for v in self.dq]
        m1, m2, Jm, g, lc = self.m1, self.m2, self.Jm, self.g, self.lc

        a = np.array(action, dtype=np.int32)
        F1 = self.thrust * float(a[0])
        F2 = self.thrust * float(a[1])

        c, s = math.cos(theta), math.sin(theta)

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

        if y <= 0:
            W = np.array([[0, 1, 0, 0]], dtype=np.float64)
            W_dot = np.zeros_like(W)
            A = W @ np.linalg.inv(M) @ W.T
            rhs = W @ np.linalg.inv(M) @ (B + G - F) - W_dot @ self.dq
            lam = np.linalg.solve(A, rhs)
            ddq = np.linalg.inv(M) @ (F - B - G + W.T @ lam)
        else:
            ddq = np.linalg.solve(M, F - B - G)

        if np.any(np.isnan(ddq)) or np.any(np.isinf(ddq)):
            print("Warning: NaN or inf in ddq:", ddq)
            print("M:", M, "F:", F, "B:", B, "G:", G)
            ddq = np.zeros_like(ddq)  # Fallback to zero acceleration

        return ddq, (M, B, G, F)

    def step(self, action):
        if self.thrust_on:
            # Calculate current state variables
            x0, y0, theta0 = self.q[0], self.q[1], self.q[2]
            dx0, dy0 = self.dq[0], self.dq[1]
            v0_sq = dx0 ** 2 + dy0 ** 2
            v0 = np.sqrt(v0_sq)

            Delta_x = self.target_x - x0
            Delta_z = 0.0 - y0  # Target y is 0.0 (ground)
            
        ddq, dyn = self._compute_M_B_G_F(action)

        # Update velocity and position
        self.dq = (self.dq.astype(np.float64) + ddq * self.dt).astype(np.float64)
        self.q = (self.q.astype(np.float64) + self.dq * self.dt).astype(np.float64)

        # Clamp state to prevent numerical issues
        self.q = np.clip(self.q, -1e5, 1e5)
        self.dq = np.clip(self.dq, -1e3, 1e3)

        # Normalize theta to (-pi, pi]
        self.q[2] = (self.q[2] + math.pi) % (2 * math.pi) - math.pi

        self.steps += 1

        # Check for invalid values
        if np.any(np.isnan(self.q)) or np.any(np.isinf(self.q)):
            print("Warning: NaN or inf in q after step:", self.q)
        if np.any(np.isnan(self.dq)) or np.any(np.isinf(self.dq)):
            print("Warning: NaN or inf in dq after step:", self.dq)

        done = False
        landed = False
        if self.q[1] <= 0.0 or self.steps >= self.max_steps:
            done = True
            if self.q[1] <= 0.0:
                landed = True
                self.q[1] = 0.0
                self.dq[:] = 0.0

        # Reward design
        reward = 0.0

        # 1. Penalty for dtheta > 0
        if self.dq[2] > 0:
            reward -= 3.0

        # 2 & 3. Landing reward and step penalty
        if done and landed:
            pos_err = abs(self.q[0] - self.target_x)
            ang_err = abs(self.q[2] - self.target_theta)
            if pos_err < 0.5 and ang_err < math.radians(10) and self.dq[2]< math.pi/90:
                reward += 4.0
            else:
                reward -= 4.0 * (pos_err + ang_err)
            reward -= abs(self.steps) * 0.01 # Encourage fewer steps
        elif done and not landed:
            reward = -3000.0

        # 4. 角度限制惩罚
        max_angle_rad = math.radians(60)
        angle = self.q[2]
        if abs(angle) > max_angle_rad:
            reward -= 2 * (abs(angle) - max_angle_rad) ** 2

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
    ax.set_xlim(-1.0, 7)
    ax.set_ylim(0.0, 4.0)
    ax.grid(True, alpha=0.3)
    beam_line, = ax.plot([], [], lw=3, label='Beam', color='black')
    leg_line, = ax.plot([], [], lw=3, label='Leg', color='black')
    com_point_m1, = ax.plot([], [], 'ro', markersize=8, label='Body COM')
    com_point_m2, = ax.plot([], [], 'bo', markersize=6, label='Leg COM')
    path_line, = ax.plot([], [], 'g--', lw=1, alpha=0.7, label='Trajectory')
    ground_line, = ax.plot([-1, 7], [0, 0], 'k-', lw=2, label='Ground')
    target_point, = ax.plot([env.target_x], [0], 'gX', markersize=12, label='Target')
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
        beam_line.set_data([left[0], right[0]], [left[1], right[1]])
        leg_line.set_data([x, tip[0]], [y, tip[1]])
        com_point_m2.set_data([x], [y])
        com_point_m1.set_data([tip[0]], [tip[1]])
        path_line.set_data(px[:i + 1], py[:i + 1])
        info_text.set_text(
            f"Step: {frame['step']}\n"
            f"Action: {frame['action']}\n"
            f"Position: ({x:.2f}, {y:.2f})\n"
            f"Theta: {math.degrees(th):.1f}°\n"
            f"Velocity: ({frame['dq'][0]:.2f}, {frame['dq'][1]:.2f})"
        )
        return beam_line, leg_line, com_point_m2, com_point_m1, path_line, info_text

    ani = FuncAnimation(fig, animate, frames=len(frames), init_func=init,
                        interval=50, blit=True, repeat=False)

    if save_gif:
        print("Saving animation as GIF...")
        ani.save('jumping_robot_simulation.gif', writer=PillowWriter(fps=20))
        print("GIF saved as 'jumping_robot_simulation.gif'")

    plt.tight_layout()
    plt.show()
    return ani


if __name__ == "__main__":
    FORCE_TRAIN = False
    CONTINUE_TRAIN = True
    MODEL_PATH = "ppo_jumping_robot.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TJumpEnvFlight2DOF(
        dt=0.005, m1=0.2, m2=0.1, Jm=0.02,
        lc=0.5, l0=1.0, g=9.81, thrust=1.0,
        max_steps=350, target_x=2, target_theta=-np.pi / 7
    )

    policy_net = None

    if FORCE_TRAIN or not os.path.exists(MODEL_PATH):
        print("Start a new training...")
        policy_net, rewards, best_avg = train_ppo(
            env, device,
            num_updates=3000,
            batch_size=2000,
            epochs=4,
            minibatch_size=1000,
            gamma=0.995, lam=0.95,
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

    rollout_and_render(env, policy_net=policy_net, device=device,
                       deterministic=True, save_gif=True)