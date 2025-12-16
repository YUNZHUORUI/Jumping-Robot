import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
import torch.optim as optim
import os
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


# ==========================================
# 1. PHYSICS & PLANNING UTILITIES
# ==========================================

class SLIPDynamics:
    def __init__(self, m=1.0, k=2000.0, l0=0.5, J=0.01, g=9.81):
        self.m = m
        self.k = k
        self.l0 = l0
        self.J = J
        self.g = g

    def flight_dynamics(self, state, t, torque):
        # state: [x, z, theta, vx, vz, omega]
        # x, z: CM position
        # theta: body angle
        return [state[3], state[4], state[5], 0, -self.g, torque / self.J]

    def stance_dynamics(self, t, y):
        # Lagrangian Dynamics for SLIP Stance
        # y = [r, theta, r_dot, theta_dot]
        # Contact point is fixed at origin (0,0) relative to the stance frame
        r, th, dr, dth = y

        # Prevent division by zero or negative radii
        r = max(r, 0.01)

        # Equations of Motion
        # 1. Radial: m*r_ddot - m*r*th_dot^2 + m*g*sin(th) + k(r - l0) = 0
        # 2. Tangential: (m*r^2 + J)*th_ddot + 2*m*r*dr*dth + m*g*r*cos(th) = 0

        # Radial Acceleration
        r_ddot = r * dth ** 2 - self.g * np.sin(th) - (self.k / self.m) * (r - self.l0)

        # Angular Acceleration
        numerator = -2 * self.m * r * dr * dth - self.m * self.g * r * np.cos(th)
        denominator = self.m * r ** 2 + self.J
        th_ddot = numerator / denominator

        return [dr, dth, r_ddot, th_ddot]

    def solve_stance(self, impact_state, dt_max=0.5):
        # impact_state: [r, theta, r_dot, theta_dot]
        # Events: Liftoff when spring extends back to l0
        def liftoff_event(t, y):
            return y[0] - self.l0

        liftoff_event.terminal = True
        liftoff_event.direction = 1.0  # Crossing from negative to positive (extension)

        sol = solve_ivp(
            self.stance_dynamics,
            [0, dt_max],
            impact_state,
            events=liftoff_event,
            rtol=1e-6, atol=1e-9,
            max_step=0.001
        )
        return sol


# ==========================================
# 2. GYM-STYLE ENVIRONMENT
# ==========================================

class SLIPHopperEnv:
    def __init__(self):
        # System Constants
        self.m = 1.0
        self.g = 9.81
        self.l0 = 0.5
        self.k = 2000.0
        self.J = 0.01
        self.dt = 0.005
        self.max_torque = 5.0  # N-m
        self.max_steps = 300

        # Targets
        self.target1_x = 2.0
        self.target1_z = 0.0  # Ground
        self.target2_x = 4.0
        self.target2_z = 1.5

        # Feasibility Zone
        self.R_min = 0.5
        self.R_max = 2.0
        self.Theta_launch_min = np.radians(30)
        self.Theta_launch_max = np.radians(60)

        self.physics = SLIPDynamics(self.m, self.k, self.l0, self.J, self.g)

        # State: [x, z, theta, vx, vz, omega]
        self.state = np.zeros(6)
        self.steps = 0
        self.planned_attack_angle = 0.0
        self.stance_data = None  # Storage for plotting

    def reset(self):
        self.steps = 0
        self.stance_data = None

        # --- 1. Generate Valid Start State in Feasibility Zone ---
        r = np.sqrt(np.random.uniform(self.R_min ** 2, self.R_max ** 2))
        theta_launch = np.random.uniform(self.Theta_launch_min, self.Theta_launch_max)

        # We start to the left of Target 1
        dx_launch = r * np.cos(theta_launch)
        dz_launch = r * np.sin(theta_launch)

        # Start Position
        start_x = self.target1_x - dx_launch
        start_z = dz_launch

        # Calculate Ballistic Velocity required to hit Target 1 (2,0) from Start
        # Standard Projectile Motion: z = z0 + tan(th)*x - g*x^2 / (2*v^2*cos^2(th))
        # We know Start and End, need V0.
        # However, the prompt says "Launch angle" is theta_0.
        # Velocity v0 is determined by the "Ballistic Constraint" formula in prompt.

        Delta_x = self.target1_x - start_x
        Delta_z = self.target1_z - start_z  # Target z is 0, start z is positive

        num = self.g * Delta_x ** 2
        den = 2 * (np.cos(theta_launch) ** 2) * (Delta_x * np.tan(theta_launch) - Delta_z)

        if den <= 0: return self.reset()  # Invalid config

        v0 = np.sqrt(num / den)
        vx0 = v0 * np.cos(theta_launch)
        vz0 = v0 * np.sin(theta_launch)  # Usually negative if we are aiming down? No, aiming up.

        # Actually, if we launch at angle theta, vz is positive.
        # But we are falling TOWARDS T1.
        # Let's assume the "launch" was in the past and we are at (x0, z0) with velocity vector pointing at T1?
        # The prompt implies (x0, z0) is the PEAK or start of a ballistic arc.
        # Let's assume we simply launch FROM (x0, z0) with calculated velocity to hit T1.
        # Since T1 is at z=0 and we are at z>0, and launch angle is 30-60 (up), it's a parabolic arc.

        # Random initial body angle
        initial_theta = np.random.uniform(-np.pi / 2, np.pi / 2)

        self.state = np.array([start_x, start_z, initial_theta, vx0, vz0, 0.0])

        # --- 2. THE PLANNER: Calculate Required Attack Angle at T1 ---
        # Goal: Find phi (attack angle) such that Stance Phase throws us to Target 2
        self.planned_attack_angle = self._plan_stance_phase()

        return self._get_obs()

    def _plan_stance_phase(self):
        """
        Solves the Inverse Problem:
        What angle (phi) must the leg be at Touchdown (Target 1)
        so that the passive spring rebound sends the drone to Target 2?
        """
        # Incoming Velocity at T1 (Approximate from conservation of energy/projectile)
        # v_impact_sq = v0^2 + 2*g*(z_start - 0)
        v0_sq = self.state[3] ** 2 + self.state[4] ** 2
        v_impact_mag = np.sqrt(v0_sq + 2 * self.g * (self.state[1]))

        # Incoming angle (purely kinematic estimate)
        dx = self.target1_x - self.state[0]
        t_flight = dx / self.state[3]
        vz_impact = self.state[4] - self.g * t_flight
        gamma_impact = np.arctan2(vz_impact, self.state[3])  # Flight path angle

        # Optimization: Minimize distance to T2 based on attack angle phi
        def objective(phi_guess):
            # Convert Cartesian Impact to Polar Stance Coordinates
            # Stance frame: Origin is at the foot (fixed at T1)
            # Impact State: r=l0, theta = phi_guess
            # Velocity must be projected.

            # Rotation matrix from World to Polar
            # But simpler:
            # r_dot = vx*cos(phi) + vz*sin(phi) (Projection on leg)
            # r*th_dot = -vx*sin(phi) + vz*cos(phi) (Projection tangent)

            # Using actual impact velocity vector
            vx_imp = self.state[3]  # Approximation: horizontal v constant
            vz_imp = vz_impact  # Calculated above

            r_dot = vx_imp * np.cos(phi_guess) + vz_imp * np.sin(phi_guess)
            th_dot = (-vx_imp * np.sin(phi_guess) + vz_imp * np.cos(phi_guess)) / self.l0

            # Condition: Compression required (r_dot < 0)
            if r_dot > 0: return 100.0  # Penalty for non-compression

            # Run Stance Sim
            sol = self.physics.solve_stance([self.l0, phi_guess, r_dot, th_dot])

            if not sol.success or len(sol.t) < 2: return 100.0

            # Liftoff State
            end_y = sol.y[:, -1]
            theta_lo = end_y[1]
            dr_lo = end_y[2]
            dth_lo = end_y[3]

            # Convert back to Cartesian for Flight 2
            # V_lift_x = dr*cos(th) - r*dth*sin(th)
            # V_lift_z = dr*sin(th) + r*dth*cos(th)
            vx_lo = dr_lo * np.cos(theta_lo) - self.l0 * dth_lo * np.sin(theta_lo)
            vz_lo = dr_lo * np.sin(theta_lo) + self.l0 * dth_lo * np.cos(theta_lo)

            # Ballistic check to T2 (4, 1.5)
            # Starting at T1 (2,0)
            dx_t2 = self.target2_x - self.target1_x
            dz_t2 = self.target2_z - self.target1_z

            # Projectile error
            # t_peak = vz_lo / g
            # Check if we can reach x=4
            t_reach = dx_t2 / vx_lo if vx_lo > 0.1 else 0
            z_at_reach = 0 + vz_lo * t_reach - 0.5 * self.g * t_reach ** 2

            return abs(z_at_reach - dz_t2)

        # Search for optimal attack angle in range [60 deg, 120 deg] (leaning back to throw forward)
        res = minimize_scalar(objective, bounds=(np.radians(60), np.radians(130)), method='bounded')
        return res.x

    def step(self, action):
        # Action: Torque [-1, 1] scaled
        torque = np.clip(action[0], -1.0, 1.0) * self.max_torque

        # 1. Integration (Flight Phase)
        # Simple Euler for control loop (physics is verified robustly later)
        x, z, th, vx, vz, omega = self.state

        # Apply Gravity
        vz -= self.g * self.dt
        x += vx * self.dt
        z += vz * self.dt

        # Apply Torque
        alpha = torque / self.J
        omega += alpha * self.dt
        th += omega * self.dt

        # Normalize angle
        th = (th + np.pi) % (2 * np.pi) - np.pi

        self.state = np.array([x, z, th, vx, vz, omega])
        self.steps += 1

        # 2. Check Touchdown Condition
        # Leg tip position
        tip_z = z - self.l0 * np.cos(np.pi / 2 - th)  # Assuming theta 0 is horizontal right?
        # Let's align with standard polar: theta is angle from horizontal.
        # Leg is fixed to body?
        # Simplification: Body angle Theta IS the leg angle in this Monopod model for the "Active" phase.
        # The prompt says "adjust body attitude so it lands with target attack angle".

        foot_z = z - self.l0 * np.sin(th)  # If theta=90 (vertical), z - l0.

        reward = 0
        done = False
        info = {}

        # Check ground impact
        if foot_z <= 0:
            done = True

            # --- 3. TRANSITION TO STANCE & EVALUATE ---
            # Calculate error between current angle and planned angle
            angle_error = abs(th - self.planned_attack_angle)

            # Run the Stance Physics (High Fidelity)
            # Transform to polar velocity
            r_dot = vx * np.cos(th) + vz * np.sin(th)
            th_dot = (-vx * np.sin(th) + vz * np.cos(th)) / self.l0

            # Solve Stance
            sol = self.physics.solve_stance([self.l0, th, r_dot, th_dot])
            self.stance_data = sol

            # Did we crash in stance? (r < 0.1)
            if np.min(sol.y[0]) < 0.1:
                reward = -50.0  # Crash
            else:
                # Liftoff State
                end_y = sol.y[:, -1]
                th_lo = end_y[1]
                dr_lo = end_y[2]
                dth_lo = end_y[3]

                # Flight 2 Projectile
                vx_lo = dr_lo * np.cos(th_lo) - self.l0 * dth_lo * np.sin(th_lo)
                vz_lo = dr_lo * np.sin(th_lo) + self.l0 * dth_lo * np.cos(th_lo)

                # Check closeness to T2
                dx_t2 = self.target2_x - x  # Approximate x is T1
                if vx_lo > 0.1:
                    t_reach = dx_t2 / vx_lo
                    z_reach = vz_lo * t_reach - 0.5 * self.g * t_reach ** 2

                    dist_error = np.sqrt((z_reach - self.target2_z) ** 2)

                    # Reward: High for matching T2, High for matching planned angle
                    r_dist = 10.0 * np.exp(-2.0 * dist_error)
                    r_angle = 5.0 * np.exp(-10.0 * angle_error)
                    reward = r_dist + r_angle

                    info['z_reach'] = z_reach
                    info['dist_error'] = dist_error
                else:
                    reward = -10.0  # Backward or stalled

            # Print analysis as requested
            print(f"Impact: Planned Phi={np.degrees(self.planned_attack_angle):.1f}°, Actual={np.degrees(th):.1f}°")
            print(f"Reward: {reward:.2f}")

        else:
            # Flight Phase Shaping Reward: Minimize angle error progressively
            err = abs(th - self.planned_attack_angle)
            reward = -0.1 * err  # Dense penalty to guide PPO

        # Bounds check
        if x > self.target1_x + 0.5 or z < -0.1 or self.steps > self.max_steps:
            done = True
            reward -= 10.0

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # Observation: [z, theta, vz, omega, target_theta_error]
        err = self.state[2] - self.planned_attack_angle
        return np.array([self.state[1], self.state[2], self.state[4], self.state[5], err], dtype=np.float32)


# ==========================================
# 3. PPO AGENT (Standard Implementation)
# ==========================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 256), nn.Tanh(),
            nn.Linear(256, 64), nn.Tanh(),
            nn.Linear(64, act_dim), nn.Tanh()  # Output -1 to 1
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_action(self, x, action=None):
        mu, v = self.forward(x)
        log_std = -0.5 * torch.ones_like(mu)  # Fixed std dev for exploration
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob, v


def train_ppo(env, num_episodes=500):
    device = torch.device("cpu")
    model = ActorCritic(obs_dim=5, act_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    gamma = 0.99
    clip_ratio = 0.2

    rewards_history = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False

        states, actions, rewards, log_probs, values = [], [], [], [], []

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, val = model.get_action(obs_t)

            act_np = action.cpu().numpy()[0]
            next_obs, r, done, info = env.step(act_np)

            states.append(obs_t)
            actions.append(action)
            rewards.append(r)
            log_probs.append(log_prob)
            values.append(val)

            obs = next_obs

        # PPO Update (Simplified one-step batch per episode for brevity)
        rewards_history.append(sum(rewards))

        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Optimize
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.stack(log_probs).squeeze()
        values = torch.cat(values).squeeze()

        adv = returns - values.detach()

        # Re-evaluate
        _, new_log_probs, _ = model.get_action(states, actions)
        ratio = torch.exp(new_log_probs - log_probs)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss = -torch.min(surr1, surr2).mean() + 0.5 * ((returns - values) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 50 == 0:
            print(f"Episode {ep}, Reward: {sum(rewards):.2f}")

    return model, rewards_history


# ==========================================
# 4. VISUALIZATION
# ==========================================

def visualize(env, model):
    obs = env.reset()
    done = False

    # Storage for Flight 1
    x_traj, z_traj = [], []

    print("\nSimulating Final Trajectory...")
    while not done:
        x_traj.append(env.state[0])
        z_traj.append(env.state[1])

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = model.get_action(obs_t)

        obs, r, done, info = env.step(action.numpy()[0])

    # Prepare Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Monopod Hopping Drone: Flight 1 -> SLIP Stance -> Flight 2")

    # 1. Plot Flight 1
    ax.plot(x_traj, z_traj, 'b--', label='Flight 1 (Active Control)')

    # 2. Plot Stance (if available)
    if env.stance_data:
        sol = env.stance_data
        # Convert polar back to cartesian relative to T1
        r = sol.y[0]
        th = sol.y[1]
        stance_x = env.target1_x + r * np.cos(th) * -1  # Foot is at T1, body is r away
        # Correction: In polar, r is dist from origin. If origin is foot (T1):
        # x_body = x_foot + r cos(theta)
        # z_body = z_foot + r sin(theta)
        sx = env.target1_x + r * np.cos(th)
        sz = env.target1_z + r * np.sin(th)
        ax.plot(sx, sz, 'r-', linewidth=2, label='Stance (Passive SLIP)')

        # 3. Plot Flight 2 (Analytic)
        end_y = sol.y[:, -1]
        th_lo = end_y[1]
        dr_lo = end_y[2]
        dth_lo = end_y[3]
        vx_lo = dr_lo * np.cos(th_lo) - env.l0 * dth_lo * np.sin(th_lo)
        vz_lo = dr_lo * np.sin(th_lo) + env.l0 * dth_lo * np.cos(th_lo)

        t = np.linspace(0, 1.0, 50)
        fx = sx[-1] + vx_lo * t
        fz = sz[-1] + vz_lo * t - 0.5 * 9.81 * t ** 2
        ax.plot(fx, fz, 'g--', label='Flight 2 (Ballistic)')

    # Targets
    ax.plot(env.target1_x, env.target1_z, 'ko', markersize=10, label='Target 1 (Bounce)')
    ax.plot(env.target2_x, env.target2_z, 'kx', markersize=10, label='Target 2 (Goal)')
    ax.plot(x_traj[0], z_traj[0], 'mo', label='Start')

    # Feasibility Zone (Approx visualization)
    theta = np.linspace(env.Theta_launch_min, env.Theta_launch_max, 20)
    for r in [env.R_min, env.R_max]:
        zx = env.target1_x - r * np.cos(theta)
        zz = r * np.sin(theta)
        ax.plot(zx, zz, 'k:', alpha=0.3)

    ax.legend()
    plt.show()


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    env = SLIPHopperEnv()

    print("Training PPO Agent for Attitude Control...")
    model, history = train_ppo(env, num_episodes=600)

    plt.figure()
    plt.plot(history)
    plt.title("Learning Curve: Attitude Control Accuracy")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    visualize(env, model)