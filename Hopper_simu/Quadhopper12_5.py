import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
import torch.optim as optim
import os


# --- Environment (Simplified 2DOF COM Dynamics) ---
class TJumpEnvFlight2DOF:
    def __init__(self,
                 dt=0.005,
                 m=0.04,  # Total mass (m1+m2)
                 lc=0.5, Jm=0.02,  # Kept for torque calc (simplified)
                 g=9.81, thrust=0.314,
                 max_steps=350,
                 target_x=3.0, target_y=0.0, target_theta=0.0,
                 # Ballistic Constraint Parameters
                 r_sq=0.5 ** 2, R_sq=2.0 ** 2,
                 theta_min_deg=30, theta_max_deg=60,
                 V_TOL=0.05,  # Tolerance for V_actual ≈ V_req (m/s)
                 MAX_FLIGHT_X=7.0, MAX_FLIGHT_Y=4.0):

        self.dt = float(dt)
        self.m = float(m)
        self.lc = float(lc)
        self.Jm = float(Jm)
        self.g = float(g)
        self.thrust = float(thrust)
        self.max_steps = int(max_steps)
        self.target_x = float(target_x)
        self.target_y = float(target_y)  # Should be 0.0 (ground)
        self.target_theta = float(target_theta)

        # Constraints
        self.r_sq = r_sq
        self.R_sq = R_sq
        self.angle_min_rad = math.radians(theta_min_deg)
        self.angle_max_rad = math.radians(theta_max_deg)
        self.V_TOL_sq = V_TOL ** 2
        self.MAX_FLIGHT_X = MAX_FLIGHT_X
        self.MAX_FLIGHT_Y = MAX_FLIGHT_Y

        self.reset()

    def reset(self, x=0, y=0, theta=None, dx=None, dy=None, dtheta=0):
        # Initial state randomization
        if dx is None: dx = float(np.random.uniform(0.0, 2.0))
        if dy is None: dy = float(np.random.uniform(0.0, 2.0))
        if theta is None: theta = float(np.random.uniform(math.pi / 6, math.pi / 3.0))
        if dtheta is None: dtheta = float(np.random.uniform(-math.pi / 6, -math.pi / 3.0))

        # State: [x, y, theta]
        self.q = np.array([x, y, theta], dtype=np.float32)
        # Velocity: [dx, dy, dtheta]
        self.dq = np.array([dx, dy, dtheta], dtype=np.float32)
        self.steps = 0
        self.thrust_on = True
        self.launch_success = False
        self.target_reached_state = None
        self.required_v0 = 0.0  # Store for observation/debugging

        return self._obs()

    def _get_required_v0_sq(self, x0, y0, theta0):
        """Calculates v_req^2 based on the projectile motion formula."""
        Delta_x = self.target_x - x0
        Delta_z = self.target_y - y0  # Delta_z will be negative (launching up)

        cos_theta0 = math.cos(theta0)
        tan_theta0 = math.tan(theta0)

        # Denominator: 2 * cos^2(theta) * (Delta_x * tan(theta) - Delta_z)
        # Note: If Delta_x * tan_theta0 - Delta_z is negative, the target is
        # above the launch parabola, meaning it is unreachable/infeasible.
        den = 2 * (cos_theta0 ** 2) * (Delta_x * tan_theta0 - Delta_z)

        if den > 1e-6 and abs(cos_theta0) > 1e-6 and Delta_x != 0.0:
            required_v0_sq = (self.g * Delta_x ** 2) / den
            # Ensure velocity is real and positive (required_v0_sq > 0)
            return required_v0_sq if required_v0_sq > 0.0 else -1.0
        return -1.0  # Infeasible

    def _obs(self):
        """Observation space: [dx, dy, dtheta, theta, Δx_err, Δv_err]"""
        x0, y0, theta0 = self.q[0], self.q[1], self.q[2]
        dx0, dy0, dtheta0 = self.dq[0], self.dq[1], self.dq[2]
        v0_sq = dx0 ** 2 + dy0 ** 2

        # 1. Calculate required velocity
        required_v0_sq = self._get_required_v0_sq(x0, y0, theta0)
        self.required_v0 = np.sqrt(required_v0_sq) if required_v0_sq > 0 else 0.0

        # 2. Calculate observed errors
        Delta_x_err = self.target_x - x0
        Delta_v_sq_err = v0_sq - required_v0_sq

        # Observation vector: (6 dimensions)
        obs = np.array([dx0, dy0, dtheta0, theta0, Delta_x_err, Delta_v_sq_err], dtype=np.float32)
        return obs

    def step(self, action):

        # --- 1. Forces and Torques ---
        x, y, theta = self.q[0], self.q[1], self.q[2]
        dx, dy, dtheta = self.dq[0], self.dq[1], self.dq[2]

        a = np.array(action, dtype=np.int32)

        F1 = 0.0
        F2 = 0.0
        T = 0.0  # Total Thrust (F1+F2)
        Tau = 0.0  # Torque (F1-F2)*lc

        if self.thrust_on:
            # Thrust force components from agent's binary action
            F1 = self.thrust * float(a[0])
            F2 = self.thrust * float(a[1])
            T = F1 + F2
            Tau = (F1 - F2) * self.lc

            # Check for Ballistic Lock Condition
            required_v0_sq = self._get_required_v0_sq(x, y, theta)
            v0_sq = dx ** 2 + dy ** 2

            pos_sq = x ** 2 + y ** 2

            C1_pos = (self.r_sq < pos_sq < self.R_sq)
            C2_angle = (self.angle_min_rad < theta < self.angle_max_rad)
            C3_speed = (required_v0_sq > 0.0) and (np.abs(v0_sq - required_v0_sq) < self.V_TOL_sq)

            if C1_pos and C2_angle and C3_speed:
                self.thrust_on = False
                self.launch_success = True
                self.target_reached_state = self.q.copy()
                print(f"✅ Thrust Locked OFF at step {self.steps}! V_act={np.sqrt(v0_sq):.2f}, V_req={np.sqrt(required_v0_sq):.2f}")

                # IMPORTANT: Set thrust to zero for the rest of this step/future steps
                T = 0.0
                Tau = 0.0

                # --- 2. Dynamics (Simplified Point Mass) ---
        c, s = math.cos(theta), math.sin(theta)

        # Acceleration components (ddq = [ddx, ddy, ddtheta])

        # Thrust is directed along the y-axis of the body frame, but F1 and F2
        # are calculated as thrust magnitude. The thrust vector is: F_thrust = T * [s, c]
        ddx = (T * s) / self.m
        ddy = (T * c - self.m * self.g) / self.m  # Gravity acts along world Y
        ddtheta = Tau / self.Jm  # Torque

        ddq = np.array([ddx, ddy, ddtheta], dtype=np.float64)

        # --- 3. Integration ---
        self.dq = (self.dq.astype(np.float64) + ddq * self.dt).astype(np.float64)
        self.q = (self.q.astype(np.float64) + self.dq * self.dt).astype(np.float64)

        # State cleaning
        self.q[2] = (self.q[2] + math.pi) % (2 * math.pi) - math.pi
        self.steps += 1

        # --- 4. Termination ---
        done = False
        landed = False

        # Hard Boundaries (preventing flying out)
        if (x < -1.0 or x > self.MAX_FLIGHT_X or y > self.MAX_FLIGHT_Y):
            done = True

        # Angle constraint
        if abs(theta) > self.angle_max_rad:
            done = True

        # Landing and Max Steps
        if self.q[1] <= self.target_y or self.steps >= self.max_steps:
            done = True
            if self.q[1] <= self.target_y:
                landed = True
                self.q[1] = self.target_y
                self.dq[:] = 0.0

        # --- 5. Reward Calculation ---
        reward = 0.0
        current_obs = self._obs()
        v0_sq = dx ** 2 + dy ** 2
        required_v0_sq = self._get_required_v0_sq(x, y, theta)
        Delta_v_sq_err = v0_sq - required_v0_sq

        if self.thrust_on:
            # R1: Time penalty
            reward -= 0.01

            # R2: Velocity matching incentive (Massively boosted)
            if required_v0_sq > 0.0:
                v_mismatch = np.abs(Delta_v_sq_err)  # Use squared velocity mismatch
                # Max reward = 20.0 for perfect match
                reward += 20.0 * np.exp(-500.0 * v_mismatch)

                # R3: Angular guiding
            angle_target_rad = (self.angle_min_rad + self.angle_max_rad) / 2
            ang_err = np.abs(self.q[2] - angle_target_rad)
            reward += 2.0 * np.exp(-10.0 * ang_err)

            # R4: Punish large actions (thrust efficiency)
            reward -= 0.01 * (F1 + F2)  # Discourage overuse of thrust

        else:
            # Ballistic Phase Reward: Minor time penalty
            reward -= 0.005

        # R5: Terminal Reward/Penalty (CRITICAL: Only high reward for SUCCESSFUL lock-off)
        if done and landed:
            pos_err_x = abs(self.q[0] - self.target_x)
            ang_err = abs(self.q[2] - self.target_theta)

            if self.launch_success:
                # HIGH REWARD: Only given if ballistic launch was successful
                if pos_err_x < 0.2:
                    reward += 10.0 - 5.0 * pos_err_x  # MASSIVE BASE REWARD
                else:
                    reward -= 15.0 * pos_err_x  # Heavy penalty for missing, even if locked
            else:
                # LOW PENALTY: Failure to lock, minimal reward, heavy penalty for miss
                reward -= 10.0 * pos_err_x

            # Angular landing reward (applies to both)
            reward += 10.0 * np.exp(-10.0 * ang_err)

            # Discourage long flight time
            reward -= abs(self.steps) * 0.01

        elif done and not landed:
            # R6: CRASH PENALTY (Timeout, Out-of-Bounds, or Extreme Angle)
            reward = -500.0

        info = {
            "ddq": ddq.astype(np.float32),
            "dq": self.dq.astype(np.float32),
            "landed": landed,
            "action": np.array(action, dtype=np.int32)
        }
        return current_obs, float(reward), bool(done), info


# --- PPO Network (Modified for 6-dim observation) ---
if torch is not None:
    class ActorCritic(nn.Module):
        def __init__(self, obs_dim=6, hidden=512):  # obs_dim changed to 6
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 2)  # Output for 2 thrusters
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1)
            )
            self._initialize_weights()

        # ... (rest of ActorCritic remains the same) ...
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
        net = ActorCritic(obs_dim=6).to(device)
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


# --- Rollout and Render (Modified for 3DOF state) ---
def rollout_and_render(env, policy_net=None, device=None, max_steps=400, deterministic=False, save_gif=True):
    frames = []
    px, py = [], []

    obs = env.reset()
    done = False
    steps = 0

    # Simple COM visualization (since we simplified dynamics)
    def beam_endpoints(x, y, theta, lc):
        dx = math.cos(theta) * lc
        dy = math.sin(theta) * lc
        left = (x - dx, y + dy)
        right = (x + dx, y - dy)
        return left, right

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
        left, right = beam_endpoints(x, y, th, env.lc)

        frame_data = {
            'x': x, 'y': y, 'th': th,
            'left': left, 'right': right,
            'action': action.copy(),
            'dq': info['dq'].copy(),
            'ddq': info['ddq'].copy(),
            'step': steps,
            'thrust_on': env.thrust_on
        }
        frames.append(frame_data)
        px.append(x)
        py.append(y)
        steps += 1

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.set_title("Jumping Robot 2D Simulation (Hybrid Control)")
    ax.set_xlim(-1.0, env.MAX_FLIGHT_X + 1.0)
    ax.set_ylim(-0.2, env.MAX_FLIGHT_Y + 0.5)
    ax.grid(True, alpha=0.3)

    # Target Box (representing the constraint r < |x0, y0| < R)
    target_circle = plt.Circle((0, 0), np.sqrt(env.R_sq), color='red', fill=False, linestyle='--', linewidth=1,
                               alpha=0.3, label='$R$ boundary')
    ax.add_patch(target_circle)
    inner_circle = plt.Circle((0, 0), np.sqrt(env.r_sq), color='red', fill=False, linestyle='--', linewidth=1,
                              alpha=0.3, label='$r$ boundary')
    ax.add_patch(inner_circle)

    # Visualization elements
    beam_line, = ax.plot([], [], lw=5, label='Body', color='black')
    com_point, = ax.plot([], [], 'ko', markersize=8, label='COM')
    left_dot, = ax.plot([], [], 'o', markersize=8)
    right_dot, = ax.plot([], [], 'o', markersize=8)
    path_line, = ax.plot([], [], 'g--', lw=1, alpha=0.7, label='Trajectory')
    ground_line, = ax.plot([-1, env.MAX_FLIGHT_X + 1.0], [env.target_y, env.target_y], 'k-', lw=2, label='Ground')
    target_point, = ax.plot([env.target_x], [env.target_y], 'gX', markersize=12, label='Target')
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(loc='upper right')

    def init():
        beam_line.set_data([], [])
        com_point.set_data([], [])
        left_dot.set_data([], [])
        right_dot.set_data([], [])
        path_line.set_data([], [])
        info_text.set_text("")
        return beam_line, com_point, left_dot, right_dot, path_line, info_text

    def animate(i):
        if i >= len(frames):
            return init()
        frame = frames[i]
        x, y, th = frame['x'], frame['y'], frame['th']
        left, right = frame['left'], frame['right']
        action = frame['action']

        # Beam visualization
        beam_line.set_data([left[0], right[0]], [left[1], right[1]])
        com_point.set_data([x], [y])
        path_line.set_data(px[:i + 1], py[:i + 1])

        # Thruster visualization (blue=ON, red=OFF)
        left_dot.set_data([left[0]], [left[1]])
        left_dot.set_color('blue' if action[0] == 1 else 'red')
        right_dot.set_data([right[0]], [right[1]])
        right_dot.set_color('blue' if action[1] == 1 else 'red')

        # Color the beam based on control state
        beam_line.set_color('black' if frame['thrust_on'] else 'green')

        # info text
        info_text.set_text(
            f"Step: {frame['step']}\n"
            f"Control: {'ON (PPO)' if frame['thrust_on'] else 'OFF (Ballistic)'}\n"
            f"Action: {frame['action']}\n"
            f"Position: ({x:.2f}, {y:.2f})\n"
            f"Theta: {math.degrees(th):.1f}°\n"
            f"Velocity: ({frame['dq'][0]:.2f}, {frame['dq'][1]:.2f})"
        )
        return beam_line, com_point, left_dot, right_dot, path_line, info_text

    ani = FuncAnimation(fig, animate, frames=len(frames), init_func=init,
                        interval=50, blit=True, repeat=False)

    if save_gif:
        print("Saving animation as GIF...")
        ani.save('jumping_robot_hybrid_control.gif', writer=PillowWriter(fps=20))
        print("GIF saved as 'jumping_robot_hybrid_control.gif'")

    plt.tight_layout()
    plt.show()
    return ani


if __name__ == "__main__":
    FORCE_TRAIN = True
    CONTINUE_TRAIN = False
    MODEL_PATH = "ppo_jumping_robot_hybrid.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the Environment (6 DOF Observation)
    env = TJumpEnvFlight2DOF(
        dt=0.005, m=0.04, lc=0.5, Jm=0.02, g=9.81, thrust=0.314,
        max_steps=400, target_x=3.0, target_y=0.0, target_theta=0.0,
        r_sq=0.5 ** 2, R_sq=2.0 ** 2,
        theta_min_deg=30, theta_max_deg=60,
        V_TOL=0.05,
        MAX_FLIGHT_X=7.0, MAX_FLIGHT_Y=4.0
    )

    policy_net = None

    if FORCE_TRAIN or not os.path.exists(MODEL_PATH):
        print("Start a new training...")
        policy_net = ActorCritic(obs_dim=6).to(device)
        train_ppo(env, device, num_updates=3000, batch_size=2500, lr=3e-4, model_path=MODEL_PATH)

    elif CONTINUE_TRAIN:
        print(f"Load an existing model and continue training: {MODEL_PATH}")
        policy_net = ActorCritic(obs_dim=6).to(device)
        policy_net.load_state_dict(torch.load(
            MODEL_PATH, map_location=device, weights_only=True
        ))
        policy_net.train()
        train_ppo(env, device, num_updates=2000, batch_size=2500, lr=1e-4, policy_net=policy_net, model_path=MODEL_PATH)

    else:
        print(f"Load the best existing model for inference: {MODEL_PATH}")
        policy_net = ActorCritic(obs_dim=6).to(device)
        policy_net.load_state_dict(torch.load(
            MODEL_PATH, map_location=device, weights_only=True
        ))
        policy_net.eval()

    rollout_and_render(env, policy_net=policy_net, device=device,
                       deterministic=True, save_gif=True)