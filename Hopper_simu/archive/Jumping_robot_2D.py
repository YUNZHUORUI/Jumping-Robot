# t_tdrone_ppo.py
# 2D T-shape jumping robot (simplified) + PPO training + animation
# Author: assistant (example)
# Requirements: torch, numpy, matplotlib

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Environment (2D T-model)
# -------------------------
class TDrone2DEnv:
    """
    State: [x, y, theta, xdot, ydot, thetadot]
    T-bar: length L, mass m. Thrusts F1 (left), F2 (right) act upward in body frame (aligned with body 'up' direction).
    Leg: a spring attached at center bottom with rest_length and stiffness k; ground at y=0.
    Actions: two bernoulli (p1, p2) -> forces F1 = a1 * F_MAG, F2 = a2 * F_MAG
    Dynamics integrated with dt.
    """
    def __init__(self,
                 dt=0.005,
                 mass=1.0,
                 L=1,
                 I=0.02,
                 g=9.81,
                 F_MAG=6.0,
                 k_leg=1000.0,
                 leg_rest=0.5,
                 max_episode_time=3.0,
                 target_pos=(10.0, 0.0),    # target landing x,y (y is ground 0)
                 target_theta=0.0):
        self.dt = dt
        self.m = mass
        self.L = L
        self.I = I  # moment of inertia about center (approx)
        self.g = g
        self.F_MAG = F_MAG
        self.k_leg = k_leg
        self.leg_rest = leg_rest
        self.max_steps = int(max_episode_time / dt)
        self.target_pos = target_pos
        self.target_theta = target_theta
        self.reset()

    def reset(self, x=0.0, y=1.0, theta=math.radians(20.0)):
        # initial state
        self.state = np.array([x, y, theta, 0.0, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        return self.state.copy()

    def _compute_forces(self, action):
        # action: two values 0/1
        a1, a2 = action
        F1 = float(a1) * self.F_MAG
        F2 = float(a2) * self.F_MAG
        # total thrust in body up direction (in world frame)
        T = F1 + F2
        # torque around center: (F2 - F1) * (L/2)
        tau = (F2 - F1) * (self.L / 2.0)
        return T, tau

    def step(self, action):
        """
        action: array-like [0/1, 0/1]
        returns: obs, reward, done, info
        """
        s = self.state
        x, y, theta, xdot, ydot, thetad = s
        T, tau = self._compute_forces(action)

        # thrust vector in world frame (assume body 'up' axis rotated by theta)
        # Body up vector: [ -sin(theta), cos(theta) ] (since theta rotates CW? we assume small-angle linearization later)
        fu = np.array([-math.sin(theta), math.cos(theta)]) * T  # upward along body
        # gravity
        fg = np.array([0.0, -self.m * self.g])

        # leg spring force if leg penetrates ground: leg attaches at center bottom: vertical position = y - leg_rest
        # leg length measured from bar center downward to ground
        leg_length = y - 0.0  # center height above ground
        spring_force = 0.0
        spring_force_point = 0.0
        if leg_length < self.leg_rest:
            # compression = rest - length
            compress = self.leg_rest - leg_length
            spring_force = self.k_leg * compress
            # acts upward at center: produces no torque (attached at center)
            fg_leg = np.array([0.0, spring_force])
        else:
            fg_leg = np.array([0.0, 0.0])

        # sum forces
        total_force = fu + fg + fg_leg
        # accelerations
        ax = total_force[0] / self.m
        ay = total_force[1] / self.m

        # angular acceleration
        alpha = tau / self.I

        # integrate (semi-implicit Euler)
        xdot += ax * self.dt
        ydot += ay * self.dt
        thetad += alpha * self.dt

        x += xdot * self.dt
        y += ydot * self.dt
        theta += thetad * self.dt

        # simple ground contact: prevent y < 0 (penetration)
        if y < 0.0:
            y = 0.0
            if ydot < 0:
                ydot = -0.2 * ydot  # restitution small
            # Could add friction/impulse but simplified

        self.state = np.array([x, y, theta, xdot, ydot, thetad], dtype=np.float32)
        self.step_count += 1

        # reward design:
        # encourage final landing near target position and angle, penalize large tilt during flight
        # shaping: small penalty on deviation to target (progressive)
        pos_err = np.hypot(x - self.target_pos[0], y - self.target_pos[1])
        ang_err = abs((theta - self.target_theta + math.pi) % (2 * math.pi) - math.pi)
        # running reward (we'll encourage being near target x horizontally and near vertical at landing)
        # also penalize high angular velocity
        r = -0.01 * abs(thetad) - 0.001 * pos_err

        # termination conditions: if step limit or if rested on ground and near zero velocities after some time
        done = False
        info = {}
        if self.step_count >= self.max_steps:
            done = True
            # final dense reward: landing closeness
            r += -pos_err * 1.0 - ang_err * 2.0
            info['final'] = True
        # optional: if crashed (tilt too big while low)
        if y <= 0.01 and abs(theta) > math.radians(80):
            done = True
            r -= 50.0
            info['crash'] = True

        self.done = done
        return self._get_obs(), r, done, info

# -------------------------
# PPO Agent (PyTorch)
# -------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        # Bernoulli logits for 2 independent binary actions
        self.logits = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.net(x)
        logits = self.logits(h)
        return logits

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# helper to compute discounted returns
def discount_cumsum(x, gamma):
    out = np.zeros_like(x)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + gamma * running
        out[t] = running
    return out

# -------------------------
# Training loop
# -------------------------
def train_ppo(env, total_updates=200, steps_per_update=2048, epochs=10, gamma=0.99, lam=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, lr=3e-4, batch_size=64):
    obs_dim = env.reset().shape[0]
    policy = PolicyNet(obs_dim).to(device)
    value = ValueNet(obs_dim).to(device)
    optim_policy = torch.optim.Adam(policy.parameters(), lr=lr)
    optim_value = torch.optim.Adam(value.parameters(), lr=lr)

    F_MAG = env.F_MAG

    for update in range(total_updates):
        # collect trajectories
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        obs = env.reset()
        for step in range(steps_per_update):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(obs_tensor)
            probs = torch.sigmoid(logits)  # for Bernoulli
            # sample two bernoulli
            dist = Bernoulli(probs=probs)
            action_sample = dist.sample()
            logp = dist.log_prob(action_sample).sum(dim=-1).item()
            action = action_sample.cpu().numpy().astype(int).flatten().tolist()
            val = value(obs_tensor).item()

            obs_buf.append(obs.copy())
            act_buf.append(action)
            logp_buf.append(logp)
            val_buf.append(val)

            obs, rew, done, info = env.step(action)
            rew_buf.append(rew)
            done_buf.append(done)

            if done:
                obs = env.reset()

        # convert buffers to arrays
        obs_buf = np.array(obs_buf)
        act_buf = np.array(act_buf)
        logp_buf = np.array(logp_buf)
        val_buf = np.array(val_buf)
        rew_buf = np.array(rew_buf)
        done_buf = np.array(done_buf, dtype=np.bool_)

        # compute advantages (GAE)
        vals = np.append(val_buf, 0)
        adv_buf = np.zeros_like(rew_buf)
        lastgaelam = 0
        for t in reversed(range(len(rew_buf))):
            nonterminal = 1.0 - float(done_buf[t])
            delta = rew_buf[t] + gamma * vals[t + 1] * nonterminal - vals[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv_buf[t] = lastgaelam
        ret_buf = adv_buf + val_buf

        # normalize advantages
        adv_mean, adv_std = adv_buf.mean(), adv_buf.std() + 1e-8
        adv_buf = (adv_buf - adv_mean) / adv_std

        # convert to torch
        obs_t = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.float32, device=device)  # actions are 0/1
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_buf, dtype=torch.float32, device=device)

        # PPO epochs
        N = len(obs_buf)
        inds = np.arange(N)
        for epoch in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, N, batch_size):
                mb_inds = inds[start:start+batch_size]
                mb_obs = obs_t[mb_inds]
                mb_act = act_t[mb_inds]
                mb_old_logp = old_logp_t[mb_inds]
                mb_ret = ret_t[mb_inds]
                mb_adv = adv_t[mb_inds]

                logits = policy(mb_obs)
                probs = torch.sigmoid(logits)
                dist = Bernoulli(probs=probs)
                # log prob of multi-binary
                logp = dist.log_prob(mb_act).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1).mean()

                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy

                # value loss
                v = value(mb_obs)
                value_loss = ((v - mb_ret) ** 2).mean() * vf_coef

                optim_policy.zero_grad()
                policy_loss.backward()
                optim_policy.step()

                optim_value.zero_grad()
                value_loss.backward()
                optim_value.step()

        # diagnostics
        avg_return = ret_buf.mean()
        avg_reward = rew_buf.mean()
        print(f"Update {update+1}/{total_updates}  avg_reward={avg_reward:.4f} avg_return={avg_return:.4f}")

    return policy, value

# -------------------------
# Rollout & animate
# -------------------------
def rollout_and_animate(env, policy, filename="rollout.mp4", max_steps=1000, save=True):
    obs = env.reset()
    states = []
    actions = []
    for _ in range(max_steps):
        states.append(obs.copy())
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = policy(obs_tensor)
        probs = torch.sigmoid(logits).cpu().detach().numpy().flatten()
        # pick deterministic action as p>0.5
        act = (probs > 0.5).astype(int).tolist()
        actions.append(act)
        obs, r, done, info = env.step(act)
        if done:
            states.append(obs.copy())
            break

    states = np.array(states)

    # set up animation
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(-0.2, 1.6)
    line, = ax.plot([], [], lw=4, marker='o')

    def draw_frame(i):
        ax.clear()
        ax.set_xlim(-1.0, 2.0)
        ax.set_ylim(-0.2, 1.6)
        s = states[min(i, len(states)-1)]
        x, y, theta = s[0], s[1], s[2]
        # endpoints of bar
        half = env.L/2.0
        p_left = np.array([x, y]) + np.array([ math.cos(theta+math.pi/2)*(-half), math.sin(theta+math.pi/2)*(-half)])
        p_right = np.array([x, y]) + np.array([ math.cos(theta+math.pi/2)*(half), math.sin(theta+math.pi/2)*(half)])
        # draw bar
        ax.plot([p_left[0], p_right[0]], [p_left[1], p_right[1]], lw=6)
        # draw center
        ax.plot(x, y, 'ko')
        # draw leg
        ax.plot([x, x], [y, 0.0], lw=3)
        # ground
        ax.axhline(0.0, color='k')
        ax.set_title(f"t={i*env.dt:.3f}s x={x:.2f} y={y:.2f} theta={math.degrees(theta):.1f}")
        ax.set_aspect('equal')

    # create animation
    ani = animation.FuncAnimation(fig, draw_frame, frames=len(states), interval=env.dt*1000)
    if save:
        # try to save
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=int(1/env.dt), metadata=dict(artist='me'), bitrate=1800)
            ani.save(filename, writer=writer)
            print(f"Saved animation to {filename}")
        except Exception as e:
            print("Could not save mp4 (ffmpeg missing?). Showing inline instead.")
            plt.show()
    else:
        plt.show()

    return states, actions

# -------------------------
# Main run
# -------------------------
if __name__ == "__main__":
    # env parameters and target
    env = TDrone2DEnv(dt=0.005, mass=1.0, L=0.4, I=0.02, F_MAG=6.0,
                      k_leg=2000.0, leg_rest=0.5,
                      max_episode_time=3.0,
                      target_pos=(1.0, 0.0),
                      target_theta=0.0)

    # Train PPO (short demo)
    start = time.time()
    policy, value = train_ppo(env, total_updates=60, steps_per_update=1024, epochs=5, lr=3e-4)
    print("Training done in {:.2f}s".format(time.time() - start))

    # Rollout + animation
    rollout_and_animate(env, policy, filename="t_drone_rollout.mp4", max_steps=1000, save=True)
