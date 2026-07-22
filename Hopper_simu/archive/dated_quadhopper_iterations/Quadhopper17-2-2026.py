import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import freeze_support
import os
import matplotlib
# matplotlib.use('Agg')  # Commented out for local testing to see plots, uncomment for server usage
import matplotlib.pyplot as plt
import imageio


class QuadhopperTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # ==================== 1. Physics Parameters (From Hybrid Dynamics.pdf) ====================
        self.dt = 0.002
        self.m1 = 1.0  # Upper body mass
        self.m2 = 0.21  # Lower body mass
        self.l0 = 1.0  # Original length of the spring
        self.lc = 0.177  # Length between motor and upper body
        self.J = 0.15  # Main body rotation inertia
        self.g = 9.81
        self.max_thrust = 30.0
        self.ground_y = 0.0

        # Spring & Damping for the leg internal mechanics
        self.k_spring = 150.0
        self.c_damping = 20.0
        # Impact restitution (fraction of vertical velocity preserved on impact)
        self.restitution = 0.35

        # ==================== 2. Targets and Space ====================
        self.targets = np.array([3.0, 7.0, 12.0, 18.0])
        self.target_tolerance = 0.4

        # Action: [Left Thrust, Right Thrust]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: [x_foot, y_foot, theta, l, dx_foot, dy_foot, dtheta, dl, dist_x, target_idx, contact, y_err, dy_err]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.current_target_idx = 0
        self.steps = 0
        self.max_episode_steps = 3000

        # Trajectory parameters
        self.traj_a = 0.0
        self.traj_b_local = 0.0
        self.traj_x0 = 0.0
        self.traj_y0 = 0.0
        self.traj_valid = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        init_theta = np.random.uniform(math.radians(-35), math.radians(-20))

        # IMPORTANT: q[0], q[1] is now the FOOT position. So initial height is ground_y (0.0).
        self.q = np.array([0.0, 0.0, init_theta, self.l0], dtype=np.float64)
        self.current_target_idx = 0
        self.steps = 0

        self._plan_trajectory()

        if self.traj_valid:
            v_x0 = math.sqrt(-self.g / (2 * self.traj_a))
            v_y0 = self.traj_b_local * v_x0
            # Starting on ground, matching flight trajectory velocities
            self.dq = np.array([v_x0, v_y0, 0.0, 0.0], dtype=np.float64)
        else:
            self.dq = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float64)

        return self._get_obs(), {}

    def get_foot_pos(self):
        """Coordinate system origin is now the lower body (foot)"""
        return np.array([self.q[0], self.q[1]])

    def get_com_pos(self):
        """Calculate upper body (COM) position based on kinematics
        NOTE: change makes the leg perpendicular to the fuselage used for rendering.
        Leg vector (from foot to COM) = l * [-sin(theta), cos(theta)] so the body axis
        (cos(theta), sin(theta)) is perpendicular to the leg.
        """
        x, y, theta, l = self.q
        # Body axis is (cos(theta), sin(theta)); leg should be perpendicular => use -sin for x
        x_com = x - l * math.sin(theta)
        y_com = y + l * math.cos(theta)
        return np.array([x_com, y_com])

    def _plan_trajectory(self):
        if self.current_target_idx >= len(self.targets):
            self.traj_valid = False
            return

        com_pos = self.get_com_pos()
        x_curr_com, y_curr_com = com_pos[0], com_pos[1]
        x_target_foot = self.targets[self.current_target_idx]

        estimated_land_theta = math.radians(-15)
        offset_x = self.l0 * math.sin(estimated_land_theta)
        x_target_com = x_target_foot + offset_x
        y_target_com = self.l0 * math.cos(estimated_land_theta)

        dx = x_target_com - x_curr_com
        dy = y_target_com - y_curr_com

        if dx <= 0.3:
            self.traj_valid = False
            return

        tilt_deg = np.random.uniform(20, 40)
        launch_alpha = math.radians(90 - tilt_deg)
        tan_a = math.tan(launch_alpha)
        cos_a = math.cos(launch_alpha)

        denominator = dx * tan_a - dy
        if denominator <= 0.01:
            launch_alpha = math.radians(75)
            tan_a = math.tan(launch_alpha)
            cos_a = math.cos(launch_alpha)
            denominator = max(0.001, dx * tan_a - dy)

        v0_sq = (self.g * dx ** 2) / (2 * (cos_a ** 2) * denominator)
        self.traj_a = -self.g / (2 * v0_sq * cos_a ** 2)
        self.traj_b_local = tan_a
        self.traj_x0 = x_curr_com
        self.traj_y0 = y_curr_com
        self.traj_valid = True

    def get_trajectory_state(self, x_current_com):
        if not self.traj_valid:
            return 1.1, 0.0

        dx = x_current_com - self.traj_x0
        if dx < 0: dx = 0

        y_ideal = self.traj_a * (dx ** 2) + self.traj_b_local * dx + self.traj_y0
        slope_ideal = 2 * self.traj_a * dx + self.traj_b_local

        # Estimate COM dx velocity
        vx_com = self.dq[0] + self.dq[3] * math.sin(self.q[2]) + self.q[3] * self.dq[2] * math.cos(self.q[2])
        current_vx = max(0.1, vx_com)
        dy_ideal = slope_ideal * current_vx

        return y_ideal, dy_ideal

    def _get_obs(self):
        foot_pos = self.get_foot_pos()
        com_pos = self.get_com_pos()
        target_x = self.targets[self.current_target_idx] if self.current_target_idx < len(self.targets) else foot_pos[0]
        dist_x = target_x - foot_pos[0]

        is_touching = 1.0 if foot_pos[1] <= self.ground_y and self.dq[1] <= 0 else 0.0

        y_ideal, dy_ideal = self.get_trajectory_state(com_pos[0])
        # Track errors relative to COM
        y_error = com_pos[1] - y_ideal
        dy_com = self.dq[1] + self.dq[3] * math.cos(self.q[2]) - self.q[3] * self.dq[2] * math.sin(self.q[2])
        dy_error = dy_com - dy_ideal

        obs = np.concatenate([
            self.q,
            self.dq,
            [dist_x, float(self.current_target_idx), is_touching, y_error, dy_error]
        ])
        return np.clip(obs, -20.0, 20.0).astype(np.float32)

    def _compute_dynamics(self, action):
        """Rigid Body Dynamics from Hybrid Dynamics.pdf"""
        u1, u2 = action[0], action[1]
        F1 = u1 * self.max_thrust
        F2 = u2 * self.max_thrust

        x, y, theta, l = self.q
        dx, dy, dtheta, dl = self.dq

        m1, m2, Im = self.m1, self.m2, self.J
        g, k, c, l0, lc = self.g, self.k_spring, self.c_damping, self.l0, self.lc

        s = math.sin(theta)
        c_th = math.cos(theta)

        # M(q): Matrix of inertia
        M = np.array([
            [m1 + m2, 0, l * m1 * c_th, m1 * s],
            [0, m1 + m2, -l * m1 * s, m1 * c_th],
            [l * m1 * c_th, -l * m1 * s, Im + (l ** 2) * m1, 0],
            [m1 * s, m1 * c_th, 0, m1]
        ], dtype=np.float64)

        # B(q, dq): Quadratic in velocities (with fixes for typos in the PDF)
        B = np.array([
            -m1 * dtheta * (2 * dl * c_th - dtheta * l * s),
            -m1 * dtheta * (2 * dl * s + dtheta * l * c_th),
            -2 * m1 * (dl * dy * s - dl * dx * c_th + dtheta * dy * l * c_th + dtheta * dx * l * s),
            c * dl + (dtheta ** 2) * l * m1 + 2 * dtheta * dx * m1 * c_th - 2 * dtheta * dy * m1 * s
        ], dtype=np.float64)

        # G(q): Gravity and spring potential matrix
        G = np.array([
            0.0,
            (m1 + m2) * g,
            -g * l * m1 * s,
            k * (l - l0) + m1 * g * c_th
        ], dtype=np.float64)

        # F(q): Generalized forces
        # Project total thrust along the body axis (cos,sin) -> correct mapping
        F_vec = np.array([
            (F1 + F2) * c_th,
            (F1 + F2) * s,
            (F1 - F2) * lc,
            0.0
        ], dtype=np.float64)

        M_inv = np.linalg.inv(M)
        RHS = F_vec - B - G

        # Hybrid Contact Dynamics (Lagrange Multipliers)
        is_contact = (y <= self.ground_y) and (dy <= 0.0)

        if is_contact:
            # Contact Jacobian: only constrain the normal (y) direction so tangential motion
            # (sliding) is allowed. This avoids forcibly zeroing horizontal acceleration/velocity
            # which prevents continuous forward motion.
            W = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float64)  # normal only

            W_Minv = W @ M_inv
            W_Minv_WT_inv = np.linalg.inv(W_Minv @ W.T)  # scalar inverse (1x1)

            # Contact force (normal only)
            contact_lambda = W_Minv_WT_inv @ (W_Minv @ (-RHS))
            normal_force = contact_lambda[0]

            # If normal force > 0, ground pushes back
            if normal_force > 0:
                # W.T has shape (4,1). Convert to 1D (4,) before adding to RHS (4,)
                RHS += (W.T.ravel() * normal_force)
                ddq = M_inv @ RHS
            else:
                # No compressive normal force -> treat as liftoff
                ddq = M_inv @ RHS
                is_contact = False
        else:
            ddq = M_inv @ RHS

        return ddq, is_contact

    def step(self, action):
        u1, u2 = np.clip(action[0], 0.0, 1.0), np.clip(action[1], 0.0, 1.0)

        # 1. Physics integration using Hybrid EOM
        ddq, touching = self._compute_dynamics([u1, u2])

        self.dq += ddq * self.dt
        self.q += self.dq * self.dt

        # Post-contact correction: keep the foot on the ground but do NOT forcibly zero
        # horizontal velocity (allow sliding). Only prevent penetration / negative vertical
        # velocity into the ground.
        if touching:
            self.q[1] = self.ground_y
            if self.dq[1] < 0.0:
                # Reflect vertical velocity using restitution instead of zeroing to
                # preserve some kinetic energy after impact which aids continuous hopping.
                self.dq[1] = -self.restitution * self.dq[1]
            # small tangential damping to model friction (soft), optional:
            self.dq[0] *= 0.995

        # ==================== Reward Function ====================
        reward = 0.0
        terminated = False
        truncated = False

        foot_pos = self.get_foot_pos()
        com_pos = self.get_com_pos()
        dist_to_target = abs(foot_pos[0] - self.targets[self.current_target_idx])

        if not touching and self.traj_valid:
            y_ideal, dy_ideal = self.get_trajectory_state(com_pos[0])
            error_y = abs(com_pos[1] - y_ideal)
            reward += 1.0 * math.exp(-5.0 * error_y)
            reward -= 0.05 * (u1 + u2)

        reward -= 0.3 * abs(self.q[2])
        reward -= 0.1 * abs(self.dq[2])

        if touching and self.dq[1] > -0.5:
            if dist_to_target < self.target_tolerance:
                reward += 150.0
                print(
                    f"🎯 命中目标 {self.current_target_idx} (距离: {dist_to_target:.2f}, 角度: {math.degrees(self.q[2]):.1f}°)")
                self.current_target_idx += 1
                if self.current_target_idx >= len(self.targets):
                    terminated = True
                    reward += 300.0
                else:
                    self._plan_trajectory()
            elif dist_to_target < 1.5:
                reward += 5.0 * (1.5 - dist_to_target)

        if abs(self.q[2]) > 0.7 or com_pos[1] > 7.0 or com_pos[1] < 0.2:
            terminated = True
            reward -= 100.0

        if self.q[0] < -1.0 or (foot_pos[0] > self.targets[self.current_target_idx] + 3.0):
            terminated = True
            reward -= 50.0

        self.steps += 1
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass


# ==================== Main Entrypoint ====================
if __name__ == '__main__':
    freeze_support()

    MODEL_PATH = "ppo_quadhopper_eom_v1"
    MODE = "test"

    if MODE == "train":
        print("🚀 Starting training with Hybrid EOM...")
        env = make_vec_env(QuadhopperTargetEnv, n_envs=8, vec_env_cls=SubprocVecEnv)
        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            ent_coef=0.02,
            gamma=0.995,
            device="auto"
        )
        model.learn(total_timesteps=3_000_000)
        model.save(MODEL_PATH)
        print(f"✅ Model saved to {MODEL_PATH}")
        env.close()

    elif MODE == "test":
        print(f"🎬 Testing model: {MODEL_PATH}")
        if os.path.exists(MODEL_PATH + ".zip"):
            model = PPO.load(MODEL_PATH)
        else:
            print("⚠️ Model not found. Running untrained agent.")
            model = PPO("MlpPolicy", QuadhopperTargetEnv())

        env = QuadhopperTargetEnv()
        obs, _ = env.reset()

        history = {'x': [], 'y': [], 'target_y': [], 'thrust_l': [], 'thrust_r': [], 'step': [], 'theta': []}
        frames = []

        for i in range(800):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # Record COM history for plotting
            com_pos = env.get_com_pos()
            history['step'].append(i)
            history['x'].append(com_pos[0])
            history['y'].append(com_pos[1])
            history['theta'].append(math.degrees(obs[2]))
            history['thrust_l'].append(action[0])
            history['thrust_r'].append(action[1])
            traj_y, _ = env.get_trajectory_state(com_pos[0])
            history['target_y'].append(traj_y)

            if i % 3 == 0:
                fig = plt.figure(figsize=(10, 5), dpi=80)
                ax = fig.add_subplot(111)

                cx = com_pos[0]
                ax.set_xlim(cx - 4, cx + 6)
                ax.set_ylim(-1, 5)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

                ax.axhline(0, color='k', lw=2)
                for tid, tx in enumerate(env.targets):
                    color = 'g' if tid == env.current_target_idx else 'gray'
                    ax.plot(tx, 0, 'x', color=color, markersize=10, markeredgewidth=3)

                if env.traj_valid:
                    tx = np.linspace(env.traj_x0, env.targets[env.current_target_idx], 50)
                    dx_arr = tx - env.traj_x0
                    ty = env.traj_a * dx_arr ** 2 + env.traj_b_local * dx_arr + env.traj_y0
                    valid_mask = ty > -0.2
                    ax.plot(tx[valid_mask], ty[valid_mask], 'r--', alpha=0.5)

                # Rendering adapted to the new coordinate system
                x_foot, y_foot, theta, l = env.q
                x_com, y_com = com_pos

                # Body
                body_x = [x_com - 0.2 * math.cos(theta), x_com + 0.2 * math.cos(theta)]
                body_y = [y_com - 0.2 * math.sin(theta), y_com + 0.2 * math.sin(theta)]
                ax.plot(body_x, body_y, 'k-', lw=6)

                # Leg (from foot to COM)
                ax.plot([x_foot, x_com], [y_foot, y_com], 'b-', lw=3)

                # Foot
                ax.plot(x_foot, y_foot, 'ro', markersize=12)

                ax.text(cx - 3.5, 4.5, f"Step: {i}\nTheta: {math.degrees(theta):.1f}°", fontsize=12)

                fig.canvas.draw()
                # Robust capture compatible with different matplotlib backends
                canvas = fig.canvas
                w, h = canvas.get_width_height()
                image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
                frames.append(image)
                plt.close(fig)

            if done or truncated:
                print(f"Episode finished at step {i}. Reason: Done={done}, Trunc={truncated}")
                break

        imageio.mimsave('quadhopper_eom.gif', frames, fps=30)
        print("✅ GIF saved: quadhopper_eom.gif")

        print("📊 Generating analysis plots...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        ax1.plot(history['step'], history['y'], 'k-', label='COM Y')
        ax1.plot(history['step'], history['target_y'], 'r--', alpha=0.6, label='Ideal Trajectory Y')
        ax1.set_ylabel("Height (m)")
        ax1.set_title("Trajectory Tracking")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history['step'], history['theta'], 'g-', label='Tilt Angle (deg)')
        ax2.axhline(40, color='r', ls='--', alpha=0.3)
        ax2.axhline(-40, color='r', ls='--', alpha=0.3)
        ax2.set_ylabel("Theta (degrees)")
        ax2.set_title("Body Tilt Angle (Target < 40°)")
        ax2.legend()
        ax2.grid(True)

        ax3.plot(history['step'], history['thrust_l'], 'b-', alpha=0.6, label='Left Thrust')
        ax3.plot(history['step'], history['thrust_r'], 'm-', alpha=0.6, label='Right Thrust')
        ax3.set_ylabel("Thrust (0-1)")
        ax3.set_xlabel("Step")
        ax3.set_title("Control Inputs")
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig('thrust_analysis_eom.png')
        print("✅ Analysis saved: thrust_analysis_eom.png")
