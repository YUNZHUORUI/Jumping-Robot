import math
import numpy as np
from numpy import cos, sin, pi, tan
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.patches as patches


# --- Derived Analytical Targets from PDF (Example: Part 1 & 2) ---
# Assuming a flight from (x0, z0) = (0.0, 0.0) to (xt1, zt1) = (2.0, 0.0)
# and an initial angle theta0 = 45 deg (pi/4 rad).
g = 9.81
delta_x = 2.0
delta_z = 0.0
theta_0 = pi / 4.0

# 1. Required Initial Speed (v0) for Flight 1 (from PDF eq 37):
# v0^2 = (g * Delta_x^2) / (2 * cos^2(theta0) * (Delta_x * tan(theta0) - Delta_z))
V0_SQ = (g * delta_x ** 2) / (2 * (cos(theta_0) ** 2) * (delta_x * tan(theta_0) - delta_z))
V0_MAG = np.sqrt(V0_SQ) # ~3.617 m/s

# 2. Time-to-Impact (t*) for Flight 1 (from PDF eq 52):
# t* = Delta_x / (v0 * cos(theta0))
TARGET_TIME_TO_IMPACT = delta_x / (V0_MAG * math.cos(theta_0)) # ~0.782 seconds

# 3. Required Attack Angle (phi) at Impact (from Part 2, simplified)
# We assume the required launch angle for the next hop (xt1 -> xt2) is:
TARGET_ATTACK_ANGLE_RAD = -math.radians(40) # Target attack angle phi = -40 degrees (arbitrary but fixed)
# -------------------------------------------------------------------


# Environment (Rotation Control Phase based on Part 3)
class TJumpEnvRotationControl:
    def __init__(self,
                 dt=0.005,
                 m1=0.0363, m2=0.01, Jm=0.02, lc=0.25, l0=0.50, g = 9.81, thrust=0.314,
                 target_time=TARGET_TIME_TO_IMPACT,
                 target_phi=TARGET_ATTACK_ANGLE_RAD,
                 max_steps=int(TARGET_TIME_TO_IMPACT / 0.005 * 1.5)): # Run 50% longer than target time

        self.dt = float(dt)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.Jm = float(Jm)
        self.lc = float(lc)
        self.l0 = float(l0)
        self.g = float(g)
        self.thrust = float(thrust)
        self.max_steps = int(max_steps)

        # Target parameters derived from analytical solution (Parts 1 & 2 of PDF)
        self.target_time = float(target_time)
        self.target_phi = float(target_phi)

        self.reset()

    def reset(self, x=0, y=0.0, theta=None, l=0.50, dx=None, dy=None, dtheta=0, dl=0):
        # Initial state fixed to the launch condition for the analytically solved flight
        if theta is None: theta = float(np.random.uniform(pi / 6, pi / 3.0)) # Initial body angle phi_0
        # Set launch velocity (v0) components from the analytical solution (Part 1)
        dx = V0_MAG * cos(theta_0)
        dy = V0_MAG * sin(theta_0)

        # Note: The code's theta (q[2]) represents the body angle (phi in PDF), not the flight angle (theta in PDF).
        # We start at the launch position (0, 1.0) with the required launch velocity (dx, dy).
        self.q = np.array([x, y, theta, l], dtype=np.float32)
        self.dq = np.array([dx, dy, dtheta, dl], dtype=np.float32)
        self.steps = 0
        self.prev_action = np.array([0, 0], dtype=np.int32)
        self.current_time = 0.0 # New tracker for time

        return self._obs()

    def _obs(self):
        # State observation: [dx, dy, dtheta, theta] - same as original
        obs = np.array([self.dq[0], self.dq[1], self.dq[2], self.q[2]], dtype=np.float32)
        return obs

    def beam_endpoints(self):
        # Unchanged
        cx, cy, theta = float(self.q[0]), float(self.q[1]), float(self.q[2])
        dx = cos(theta) * self.lc
        dy = sin(theta) * self.lc
        left = (cx - dx + self.l0 * sin(theta), cy + dy + self.l0 * cos(theta))
        right = (cx + dx + self.l0 * sin(theta), cy - dy + self.l0 * cos(theta))
        return left, right

    def _compute_M_B_G_F(self, action):
        # Unchanged 4DOF dynamics, but collision logic is removed as we focus on the flight phase.
        x, y, theta, l = [float(v) for v in self.q]
        a = np.array(action, dtype=np.int32)

        F1 = self.thrust * float(a[0])
        F2 = self.thrust * float(a[1])

        c, s = cos(theta), sin(theta)

        # M, B, G matrices calculation remains the same
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
            - self.g * l * self.m1 * s,
            self.m1 * self.g * c
        ], dtype=np.float64)

        F = np.array([
            (F1 + F2) * s,
            (F1 + F2) * c,
            (F1 - F2) * self.lc, # Torque: tau = (F1-F2)*lc. This is the control input for rotation.
            0
        ], dtype=np.float64)

        # No collision logic for the flight phase
        ddq = np.linalg.solve(M, F - B - G)

        if np.any(np.isnan(ddq)) or np.any(np.isinf(ddq)):
            ddq = np.zeros_like(ddq)

        return ddq, (M, B, G, F)

    def step(self, action):
        # 1. Dynamics Integration
        ddq, dyn = self._compute_M_B_G_F(action)

        self.dq = (self.dq.astype(np.float64) + ddq * self.dt).astype(np.float64)
        self.q = (self.q.astype(np.float64) + self.dq * self.dt).astype(np.float64)

        # State cleaning
        self.q = np.clip(self.q, -1e5, 1e5)
        self.dq = np.clip(self.dq, -1e3, 1e3)
        self.q[2] = (self.q[2] + pi) % (2 * pi) - pi # Normalize theta

        self.steps += 1
        self.current_time += self.dt

        # 2. Termination conditions: Fixed time for impact, or going too high/low/far
        done = False
        landed = False

        if self.current_time >= self.target_time or self.steps >= self.max_steps:
            done = True
            if self.q[1] <= 0.0:
                landed = True
                self.q[1] = 0.0
                # We do NOT stop movement on ground contact here, only check the state AT target time.
                # self.dq[:] = 0.0 # Don't stop movement if only checking state AT time t*

        # 3. Reward Calculation (Focus on Rotational Target)
        reward = 0.0
        angle = self.q[2]
        angular_velocity = self.dq[2]
        current_time_error = abs(self.current_time - self.target_time)

        # R1: Time penalty
        reward -= 0.01 * self.dt

        # R2: Angular Guiding/Control Efficiency
        # Penalize excessive angular acceleration (torque)
        angular_accel = ddq[2]
        reward -= 0.01 * (angular_accel ** 2)

        # R3: Terminal Reward (Maximize alignment at target time)
        if done:
            # We want the angle to be target_phi AT the target time
            # Since the state is not exactly at target_time, we use the closest step.
            ang_err = abs(angle - self.target_phi)

            # Max reward 10.0 for perfect angle match at target time
            reward += 10.0 * np.exp(-1.0 * ang_err)

            # Check translational condition (optional, but good for stability)
            # Penalize missing the landing position (xt1=2.0, zt1=0.0)
            target_x_landing = delta_x
            pos_err_x = abs(self.q[0] - target_x_landing)
            pos_err_y = abs(self.q[1] - 0.0)
            reward += 5.0 * np.exp(-10.0 * pos_err_x)
            reward += 5.0 * np.exp(-10.0 * pos_err_y)

            # Encouraging lower steps/better efficiency
            reward -= self.steps * 0.01


        obs = self._obs()
        info = {
            "ddq": ddq.astype(np.float32), "dq": self.dq.astype(np.float32),
            "landed": landed, "dyn": dyn, "action": np.array(action, dtype=np.int32)
        }
        self.prev_action = np.array(action, dtype=np.int32)
        return obs, float(reward), bool(done), info

# The rest of the code (PPO, ActorCritic, compute_gae, train_ppo, rollout_and_render)
# remains the same, as the PPO structure is general and the visualizer is still useful.
# ... (omitting the PPO/render boilerplate for conciseness) ...

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
              model_path="ppo_jumping_robot_rotation.pt"):
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


def rollout_and_render(env, policy_net=None, device=None, max_steps=1200, deterministic=True, save_gif=True):
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
        tip = (x + sin(th) * env.l0, y + cos(th) * env.l0)

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
    ax.set_title("Jumping Robot Rotational Control Simulation")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.0, 1.5)
    ax.grid(True, alpha=0.3)

    # Target: Landing point and target angle line
    target_x = delta_x
    target_y = 0.0
    ax.plot([target_x], [target_y], 'gX', markersize=12, label='Target Landing (xt1, zt1)')
    target_angle_x = target_x + 0.5 * cos(env.target_phi)
    target_angle_y = target_y + 0.5 * sin(env.target_phi)
    ax.plot([target_x, target_angle_x], [target_y, target_angle_y], 'r--', lw=2, alpha=0.7, label='Target Attack Angle $\\varphi$')


    beam_line, = ax.plot([], [], lw=3, label='Beam', color='black')
    leg_line, = ax.plot([], [], lw=3, label='Leg', color='black')
    com_point_m1, = ax.plot([], [], 'ko', markersize=4, label='Body COM')
    com_point_m2, = ax.plot([], [], 'ko', markersize=4, label='Leg COM')
    left_dot, = ax.plot([], [], 'o', markersize=5)
    right_dot, = ax.plot([], [], 'o', markersize=5)
    path_line, = ax.plot([], [], 'g--', lw=1, alpha=0.7, label='Trajectory')
    ground_line, = ax.plot([-1, 4], [0, 0], 'k-', lw=2, label='Ground')
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

        # Left Thruster: action[0]
        left_dot.set_data([left[0]], [left[1]])
        left_dot.set_color('blue' if action[0] == 0 else 'red')

        # Right Thruster: action[1]
        right_dot.set_data([right[0]], [right[1]])
        right_dot.set_color('blue' if action[1] == 0 else 'red')

        # info text
        info_text.set_text(
            f"Step: {frame['step']} (Time: {frame['step']*env.dt:.3f} s)\n"
            f"Target Time: {env.target_time:.3f} s\n"
            f"Target Angle $\\varphi$: {math.degrees(env.target_phi):.1f}°\n"
            f"Current Angle $\\theta$: {math.degrees(th):.1f}°\n"
            f"Action (Torque): {frame['action']}\n"
            f"Position: ({x:.2f}, {y:.2f})"
        )
        return beam_line, leg_line, com_point_m2, com_point_m1, left_dot, right_dot, path_line, info_text

    ani = FuncAnimation(fig, animate, frames=len(frames), init_func=init,
                        interval=50, blit=True, repeat=False)

    if save_gif:
        print("Saving animation as GIF...")
        ani.save('jumping_robot_rotation_control.gif', writer=PillowWriter(fps=20))
        print("GIF saved as 'jumping_robot_rotation_control.gif'")

    plt.tight_layout()
    plt.show()
    return ani


# ----------------------------------------------------------------------
# Main Execution Block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    FORCE_TRAIN = True
    CONTINUE_TRAIN = False
    MODEL_PATH = "ppo_jumping_robot_rotation.pt" # New model path for rotation task

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the Environment
    env = TJumpEnvRotationControl(
        dt=0.005, m1=0.0363, m2=0.01, Jm=0.02,
        lc=0.25, l0=0.50, thrust=0.314,
        target_time=TARGET_TIME_TO_IMPACT,
        target_phi=TARGET_ATTACK_ANGLE_RAD
    )
    print(f"Environment configured with T*={env.target_time:.3f}s and Phi={math.degrees(env.target_phi):.1f}°")

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