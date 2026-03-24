import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from multiprocessing import freeze_support
import torch as th
import os
import matplotlib
# matplotlib.use('Agg')  # Commented out for local testing to see plots, uncomment for server usage
import matplotlib.pyplot as plt
import imageio


class QuadhopperTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        # ==================== 1. Physics Parameters (FIXED) ====================
        self.dt = 0.002
        self.m = 1.21
        self.rotor_span = 0.10  # Distance between two rotors (m)
        self.beam_half_length = 0.5 * self.rotor_span
        self.lc = self.beam_half_length  # CG to each rotor (torque arm)

        self.l0 = 0.30  # Leg nominal/max extension length (m)
        self.stroke_length = 0.20  # Max compression/extension stroke (m)
        self.leg_min_length = max(0.05, self.l0 - self.stroke_length)
        self.min_com_height = max(0.04, 0.35 * self.leg_min_length)
        self.max_tilt_rad = math.radians(80.0)
        self.J = 0.15  # Inertia
        self.g = 9.81
        self.max_thrust = 30.0
        self.ground_y = 0.0

        # Rendering config for small-size model
        self.render_frame_stride = 2
        self.render_dpi = 120
        self.render_body_scale = 3.2
        self.render_view_w = max(1.8, 6.0 * self.l0)
        self.render_view_h = max(1.1, 3.6 * self.l0)

        # --- FIX 2: Ground Penetration ---
        # Increased significantly to make the ground stiffer and reduce penetration depth
        self.k_spring = 150.0  # Was 150.0
        # Increased damping to stabilize the stiffer spring
        self.c_damping = 20.0  # Was 1.0

        # SLIP stance model (recommended for contact phase)
        self.use_slip_stance = True
        self.k_slip = 2200.0
        self.c_slip = 18.0
        self.stance_active = False
        self.stance_foot_anchor = np.array([0.0, self.ground_y], dtype=np.float64)

        # Attitude phase targets: flight -> negative touchdown, stance -> next takeoff angle
        self.landing_theta_target = math.radians(-20.0)
        self.landing_theta_tol = math.radians(8.0)
        self.takeoff_theta_target = math.radians(-25.0)
        self.takeoff_theta_tol = math.radians(6.0)
        self.prev_touching = False

        # ==================== 2. Targets and Space ====================
        self.targets = np.array([3.0, 7.0, 12.0, 18.0])
        self.target_tolerance = 0.4  # Slightly relaxed tolerance

        # Action: [Left Thrust, Right Thrust]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: [x, y, theta, l, dx, dy, dtheta, dl, dist_x, target_idx, contact, y_err, dy_err]
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

        # Randomized launch state for robustness (instead of perfect ballistic init)
        self.init_vx_range = (3.2, 4.4)
        self.init_vy_range = (3.4, 4.8)
        self.init_vx_std = 0.12
        self.init_vy_std = 0.18
        self.init_dtheta_range = (math.radians(-12.0), math.radians(12.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # 1. 设定初始倾角 (-35 到 -20 度)
        init_theta = np.random.uniform(math.radians(-35), math.radians(-20))

        # 2. 关键修复：计算质心高度，使得脚尖初始时刻刚好接触地面 (y_foot = 0)
        # foot_y = COM_y - l0 * cos(theta) => 如果 foot_y = 0，则 COM_y = l0 * cos(theta)
        init_y = self.l0 * math.cos(init_theta)

        # 初始化状态
        self.q = np.array([0.0, init_y, init_theta, self.l0], dtype=np.float64)
        self.current_target_idx = 0
        self.steps = 0
        self.stance_active = False
        self.stance_foot_anchor = np.array([0.0, self.ground_y], dtype=np.float64)
        self.prev_touching = False

        # 3. 规划第一跳的抛物线
        self._plan_trajectory()

        # 4. 初速度随机化：不再使用“完美抛物线初速度”，提升鲁棒性
        if self.traj_valid:
            # 用目标弹道速度作为中心，再加入随机扰动
            v_x_nom = math.sqrt(max(0.5, -self.g / (2 * self.traj_a)))
            v_y_nom = self.traj_b_local * v_x_nom

            v_x0 = np.random.normal(v_x_nom, self.init_vx_std)
            v_y0 = np.random.normal(v_y_nom, self.init_vy_std)

            v_x0 = float(np.clip(v_x0, self.init_vx_range[0], self.init_vx_range[1]))
            v_y0 = float(np.clip(v_y0, self.init_vy_range[0], self.init_vy_range[1]))

            # Keep robustness while ensuring first target is still learnable.
            v_x0 = 0.85 * v_x_nom + 0.15 * v_x0
            v_y0 = 0.85 * v_y_nom + 0.15 * v_y0

            dtheta0 = float(np.random.uniform(self.init_dtheta_range[0], self.init_dtheta_range[1]))

            self.dq = np.array([v_x0, v_y0, dtheta0, 0.0], dtype=np.float64)
        else:
            # 兜底：纯随机可飞行初速
            v_x0 = float(np.random.uniform(self.init_vx_range[0], self.init_vx_range[1]))
            v_y0 = float(np.random.uniform(self.init_vy_range[0], self.init_vy_range[1]))
            dtheta0 = float(np.random.uniform(self.init_dtheta_range[0], self.init_dtheta_range[1]))
            self.dq = np.array([v_x0, v_y0, dtheta0, 0.0], dtype=np.float64)

        return self._get_obs(), {}

    def get_foot_pos(self):
        """Calculate foot tip coordinates based on current state."""
        if self.stance_active:
            return self.stance_foot_anchor.copy()

        x, y, theta = self.q[0], self.q[1], self.q[2]
        # Assuming leg is a rigid rod of length l0 defined by theta
        x_f = x + self.l0 * math.sin(theta)
        y_f = y - self.l0 * math.cos(theta)
        return np.array([x_f, y_f])

    def _wrap_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _stance_theta_from_anchor(self):
        """Keep body/leg geometry consistent during SLIP stance."""
        r_x = self.q[0] - self.stance_foot_anchor[0]
        r_y = self.q[1] - self.stance_foot_anchor[1]
        l_curr = max(math.sqrt(r_x * r_x + r_y * r_y), 1e-6)
        theta = math.atan2(-r_x, r_y)
        return theta, l_curr

    def _plan_trajectory(self):
        if self.current_target_idx >= len(self.targets):
            self.traj_valid = False
            return

        x_curr, y_curr = self.q[0], self.q[1]
        x_target_foot = self.targets[self.current_target_idx]

        estimated_land_theta = math.radians(-15)
        offset_x = self.l0 * math.sin(estimated_land_theta)
        x_target_com = x_target_foot - offset_x

        dx = x_target_com - x_curr
        # 修改：因为起点在地面，目标点也在地面，假设落地时质心高度约等于初始的 l0 * cos(theta)
        dy = (self.l0 * math.cos(estimated_land_theta)) - y_curr

        if dx <= 0.3:
            self.traj_valid = False
            return

        tilt_deg = np.random.uniform(20, 40)
        launch_alpha = math.radians(90 - tilt_deg)
        self.takeoff_theta_target = math.radians(tilt_deg)

        tan_a = math.tan(launch_alpha)
        cos_a = math.cos(launch_alpha)

        denominator = dx * tan_a - dy
        if denominator <= 0.01:
            launch_alpha = math.radians(75)
            tan_a = math.tan(launch_alpha)
            cos_a = math.cos(launch_alpha)
            denominator = dx * tan_a - dy
            if denominator <= 0.001: denominator = 0.001

        v0_sq = (self.g * dx ** 2) / (2 * (cos_a ** 2) * denominator)

        self.traj_a = -self.g / (2 * v0_sq * cos_a ** 2)
        self.traj_b_local = tan_a
        self.traj_x0 = x_curr
        self.traj_y0 = y_curr
        self.traj_valid = True

    def get_trajectory_state(self, x_current):
        """Calculate ideal height (y) and vertical velocity (dy) for current x."""
        if not self.traj_valid:
            # If no trajectory active, aim for a standby height
            return 1.1, 0.0

        dx = x_current - self.traj_x0
        if dx < 0: dx = 0  # Don't track backwards

        # 1. Ideal height
        y_ideal = self.traj_a * (dx ** 2) + self.traj_b_local * dx + self.traj_y0

        # 2. Ideal slope (dy/dx)
        slope_ideal = 2 * self.traj_a * dx + self.traj_b_local

        # 3. Ideal vertical velocity (dy/dt) = (dy/dx) * (dx/dt)
        # Use current horizontal velocity to scale vertical velocity requirement
        current_vx = max(0.1, self.dq[0])
        dy_ideal = slope_ideal * current_vx

        return y_ideal, dy_ideal

    def _get_obs(self):
        foot_pos = self.get_foot_pos()
        target_x = self.targets[self.current_target_idx] if self.current_target_idx < len(self.targets) else foot_pos[0]
        dist_x = target_x - foot_pos[0]

        # Check contact based on physical position
        is_touching = 1.0 if foot_pos[1] <= self.ground_y + 0.02 else 0.0

        # Trajectory errors
        y_ideal, dy_ideal = self.get_trajectory_state(self.q[0])
        y_error = self.q[1] - y_ideal
        dy_error = self.dq[1] - dy_ideal

        # Clip observations to prevent numerical issues
        obs = np.concatenate([
            self.q,  # [x, y, theta, l]
            self.dq,  # [dx, dy, dtheta, dl]
            [dist_x, float(self.current_target_idx), is_touching, y_error, dy_error]
        ])
        return np.clip(obs, -20.0, 20.0).astype(np.float32)

    def step(self, action):
        # Action: [Left Thrust, Right Thrust] (0~1)
        u1, u2 = np.clip(action[0], 0.0, 1.0), np.clip(action[1], 0.0, 1.0)

        # Physics forces
        F1 = u1 * self.max_thrust
        F2 = u2 * self.max_thrust
        F_total = F1 + F2
        tau = (F2 - F1) * self.lc

        x, y, theta = self.q[0], self.q[1], self.q[2]
        dx, dy, dtheta = self.dq[0], self.dq[1], self.dq[2]
        s, c = math.sin(theta), math.cos(theta)

        # Ground interaction (SLIP model)
        foot_x_virt = x + self.l0 * s
        foot_y_virt = y - self.l0 * c
        F_act_x = -F_total * s
        F_act_y = F_total * c
        F_spring_x, F_spring_y = 0.0, 0.0
        touching = False

        if self.use_slip_stance:
            # Touchdown: lock foot contact point on the ground
            if (not self.stance_active) and (foot_y_virt <= self.ground_y):
                # Hard landing-attitude constraint:
                # enforce backward (negative) tilt at contact for both physics and rendering.
                theta_td = min(theta, self.landing_theta_target)
                self.q[2] = theta_td
                self.dq[2] = 0.0
                theta = theta_td
                s, c = math.sin(theta), math.cos(theta)
                foot_x_virt = x + self.l0 * s

                self.stance_active = True
                self.stance_foot_anchor = np.array([foot_x_virt, self.ground_y], dtype=np.float64)

            if self.stance_active:
                touching = True
                # Leg from foot anchor to COM
                r_x = x - self.stance_foot_anchor[0]
                r_y = y - self.stance_foot_anchor[1]
                l_curr = math.sqrt(r_x * r_x + r_y * r_y)
                l_curr = max(l_curr, 1e-6)

                e_x = r_x / l_curr
                e_y = r_y / l_curr

                # Leg length rate dl = v_com · e_leg
                dl = dx * e_x + dy * e_y

                # In stance, thrust is applied along the leg axis (standard SLIP-style actuation)
                F_act_x = F_total * e_x
                F_act_y = F_total * e_y

                compression = float(np.clip(self.l0 - l_curr, 0.0, self.stroke_length))
                if compression > 0.0:
                    # SLIP: F = k * compression - c * dl (damping resists length change)
                    F_mag = self.k_slip * compression - self.c_slip * dl
                    F_mag = max(0.0, F_mag)
                    F_spring_x = F_mag * e_x
                    F_spring_y = F_mag * e_y

                # Liftoff condition:
                # 1) spring has re-extended and moving outward
                # 2) leg/body has rotated to desired takeoff direction
                # 3) fail-safe release when over-extended
                theta_leg = math.atan2(-r_x, r_y)
                theta_err_takeoff = abs(self._wrap_angle(theta_leg - self.takeoff_theta_target))
                reached_takeoff_att = theta_err_takeoff <= self.takeoff_theta_tol
                if ((l_curr >= self.l0) and (dl > 0.0) and reached_takeoff_att) or (l_curr >= self.l0 + 0.05 * self.stroke_length):
                    self.stance_active = False

            # Keep observation leg states meaningful
            if self.stance_active:
                self.q[3] = l_curr
                self.dq[3] = dl
            else:
                self.q[3] = self.l0
                self.dq[3] = 0.0
        else:
            # Legacy penetration spring-damper model
            if foot_y_virt < self.ground_y:
                touching = True
                l_curr = y / max(abs(c), 0.01)
                compression = float(np.clip(self.l0 - l_curr, 0.0, self.stroke_length))
                if compression > 0:
                    comp_rate = -(dx * s - dy * c)
                    F_mag = self.k_spring * compression + self.c_damping * comp_rate
                    F_mag = max(0.0, F_mag)
                    F_spring_x = -F_mag * s
                    F_spring_y = F_mag * c

        # Dynamics Integration (Symplectic Euler)
        ddx = (F_act_x + F_spring_x) / self.m
        ddy = (F_act_y + F_spring_y) / self.m - self.g
        ddtheta = 0.0 if self.stance_active else tau / self.J

        # Update Velocities
        self.dq[0] += ddx * self.dt
        self.dq[1] += ddy * self.dt
        self.dq[2] += ddtheta * self.dt

        # Update Positions
        self.q[0] += self.dq[0] * self.dt
        self.q[1] += self.dq[1] * self.dt
        self.q[2] += self.dq[2] * self.dt

        # During stance, lock body orientation to the leg direction to avoid visual/body deformation.
        if self.use_slip_stance and self.stance_active:
            theta_prev = self.q[2]
            theta_stance, l_curr = self._stance_theta_from_anchor()
            self.q[2] = theta_stance
            self.q[3] = l_curr
            self.dq[2] = self._wrap_angle(theta_stance - theta_prev) / self.dt
            r_x = self.q[0] - self.stance_foot_anchor[0]
            r_y = self.q[1] - self.stance_foot_anchor[1]
            l_curr = max(math.sqrt(r_x * r_x + r_y * r_y), 1e-6)
            e_x = r_x / l_curr
            e_y = r_y / l_curr
            self.dq[3] = self.dq[0] * e_x + self.dq[1] * e_y

        # No rigid constraint applied here. We rely on the stiff spring to push it back up.
        # ==================== Reward Function ====================
        reward = 0.0
        terminated = False
        truncated = False

        foot_pos = self.get_foot_pos()
        target_valid = self.current_target_idx < len(self.targets)
        dist_to_target = abs(foot_pos[0] - self.targets[self.current_target_idx]) if target_valid else 0.0
        touchdown_event = touching and (not self.prev_touching)
        liftoff_event = (not touching) and self.prev_touching

        # 1. 轨迹与飞行奖励
        if not touching and self.traj_valid:
            y_ideal, dy_ideal = self.get_trajectory_state(self.q[0])
            error_y = abs(self.q[1] - y_ideal)

            # 跟踪奖励：放宽指数惩罚，让它更容易学
            reward += 1.0 * math.exp(-5.0 * error_y)

            # 关键修复：空中推力惩罚！
            # 抛物线是重力主导的，在空中开电机会改变弹道并带来力矩。鼓励空中“滑翔”。
            reward -= 0.05 * (u1 + u2)

            # Flight attitude tracking for negative-angle touchdown
            theta_err_land = abs(self._wrap_angle(self.q[2] - self.landing_theta_target))
            reward += 1.2 * math.exp(-6.0 * theta_err_land)
            reward -= 0.03 * abs(self.dq[2])

            # First target shaping: reduce large miss distance at the first jump.
            if target_valid and self.current_target_idx == 0:
                reward += 1.0 * math.exp(-0.9 * dist_to_target)

        # Touchdown attitude constraint (must land with negative tilt)
        if touchdown_event:
            theta_td = self.q[2]
            theta_td_err = abs(self._wrap_angle(theta_td - self.landing_theta_target))
            if theta_td < 0.0:
                reward += 35.0 * math.exp(-8.0 * theta_td_err)
            else:
                reward -= 30.0 + 25.0 * theta_td_err

        # Stance attitude shaping: rotate via SLIP toward next takeoff angle
        if touching:
            theta_to_to = abs(self._wrap_angle(self.q[2] - self.takeoff_theta_target))
            reward += 0.8 * math.exp(-4.0 * theta_to_to)

        if liftoff_event:
            theta_lo_err = abs(self._wrap_angle(self.q[2] - self.takeoff_theta_target))
            reward += 25.0 * math.exp(-6.0 * theta_lo_err)

        # 2. 严厉的姿态惩罚 (保持身体直立)
        # 改为温和通用项，主导由相位目标决定
        reward -= 0.08 * abs(self.q[2])
        reward -= 0.1 * abs(self.dq[2])  # 旋转角速度惩罚加大，防止一直转

        # 3. 落地与判定逻辑
        if touching and self.dq[1] > -0.5 and target_valid:
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

        # 4. 终止条件 (更严的倾角限制)
        if abs(self.q[2]) > self.max_tilt_rad or self.q[1] > 7.0 or self.q[1] < self.min_com_height:
            terminated = True
            reward -= 100.0

        if self.q[0] < -1.0 or (
            self.current_target_idx < len(self.targets)
            and foot_pos[0] > self.targets[self.current_target_idx] + 3.0
        ):
            terminated = True
            reward -= 50.0

        self.steps += 1
        self.prev_touching = touching
        if self.steps >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass


# ==================== Main Entrypoint ====================
if __name__ == '__main__':
    # Necessary for Windows multiprocessing
    freeze_support()

    MODEL_PATH = "ppo_quadhopper_fixed_v1"
    MODE = "test"  # Set to "train" to retrain with fixes, "test" to view

    if MODE == "train":
        print("🚀 Starting training with physics fixes...")
        # Use multiprocessing for faster training
        env = make_vec_env(QuadhopperTargetEnv, n_envs=8, vec_env_cls=SubprocVecEnv)

        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            # Increased entropy coef slightly to encourage exploration with tighter constraints
            ent_coef=0.02,
            gamma=0.995,
            device="auto"
        )

        # Increased training timesteps needed for stiffer physics
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

            history['step'].append(i)
            history['x'].append(obs[0])
            history['y'].append(obs[1])
            history['theta'].append(math.degrees(obs[2]))
            history['thrust_l'].append(action[0])
            history['thrust_r'].append(action[1])
            traj_y, _ = env.get_trajectory_state(obs[0])
            history['target_y'].append(traj_y)

            # --- Rendering GIF ---
            if i % env.render_frame_stride == 0:
                fig = plt.figure(figsize=(9, 5.5), dpi=env.render_dpi)
                ax = fig.add_subplot(111)

                cx = obs[0]
                view_w = env.render_view_w
                view_h = env.render_view_h
                x_left = cx - 0.65 * view_w
                x_right = cx + 1.35 * view_w

                ax.set_xlim(x_left, x_right)
                ax.set_ylim(-0.12, view_h)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

                # Ground and Targets
                ax.axhline(0, color='k', lw=2)
                for tid, tx in enumerate(env.targets):
                    color = 'g' if tid == env.current_target_idx else 'gray'
                    ax.plot(tx, 0, 'x', color=color, markersize=8, markeredgewidth=2)

                # Ideal Trajectory
                if env.traj_valid:
                    tx = np.linspace(env.traj_x0, env.targets[env.current_target_idx], 50)
                    dx_arr = tx - env.traj_x0
                    ty = env.traj_a * dx_arr ** 2 + env.traj_b_local * dx_arr + env.traj_y0
                    valid_mask = ty > -0.2
                    ax.plot(tx[valid_mask], ty[valid_mask], 'r--', alpha=0.5)

                # Robot
                x, y, theta = obs[0], obs[1], obs[2]
                foot_pos = env.get_foot_pos()

                # --- FIX 2b: Cosmetic Render Clamp ---
                # Visually ensure the ball doesn't look like it's underground,
                # even if physics penetration is slightly non-zero.
                render_foot_y = max(0.0, foot_pos[1])

                # Body
                beam_half_vis = env.beam_half_length * env.render_body_scale
                body_x = [
                    x - beam_half_vis * math.cos(theta),
                    x + beam_half_vis * math.cos(theta)
                ]
                body_y = [
                    y - beam_half_vis * math.sin(theta),
                    y + beam_half_vis * math.sin(theta)
                ]
                ax.plot(body_x, body_y, 'k-', lw=7)

                # Rotors (thrust visualization)
                thrust_l = float(np.clip(action[0], 0.0, 1.0))
                thrust_r = float(np.clip(action[1], 0.0, 1.0))
                rotor_l_color = plt.cm.coolwarm(thrust_l)
                rotor_r_color = plt.cm.coolwarm(thrust_r)
                ax.scatter(body_x[0], body_y[0], s=170, c=[rotor_l_color], edgecolors='k', linewidths=1.0, zorder=5)
                ax.scatter(body_x[1], body_y[1], s=170, c=[rotor_r_color], edgecolors='k', linewidths=1.0, zorder=5)

                # Leg
                ax.plot([x, foot_pos[0]], [y, render_foot_y], 'b-', lw=4)
                # Foot (Red Ball)
                ax.plot(foot_pos[0], render_foot_y, 'ro', markersize=14)

                # Info text
                ax.text(
                    x_left + 0.04 * view_w,
                    view_h - 0.08 * view_h,
                    f"Step: {i}\nTheta: {math.degrees(theta):.1f}°\nL thrust: {thrust_l:.2f}\nR thrust: {thrust_r:.2f}",
                    fontsize=11,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
                )

                fig.canvas.draw()
                image = np.array(fig.canvas.buffer_rgba())[:, :, :3]
                frames.append(image)
                plt.close(fig)

            if done or truncated:
                print(f"Episode finished at step {i}. Reason: Done={done}, Trunc={truncated}")
                break

        imageio.mimsave('quadhopper_fixed.gif', frames, fps=25, loop=0)
        print("✅ GIF saved: quadhopper_fixed.gif")

        # ==================== Analysis Plots ====================
        print("📊 Generating analysis plots...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot 1: Height Tracking
        ax1.plot(history['step'], history['y'], 'k-', label='COM Y')
        ax1.plot(history['step'], history['target_y'], 'r--', alpha=0.6, label='Ideal Trajectory Y')
        ax1.set_ylabel("Height (m)")
        ax1.set_title("Trajectory Tracking")
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Tilt Angle (Checking Fix 1)
        ax2.plot(history['step'], history['theta'], 'g-', label='Tilt Angle (deg)')
        ax2.axhline(40, color='r', ls='--', alpha=0.3)
        ax2.axhline(-40, color='r', ls='--', alpha=0.3)
        ax2.set_ylabel("Theta (degrees)")
        ax2.set_title("Body Tilt Angle (Target < 40°)")
        ax2.legend()
        ax2.grid(True)

        # Plot 3: Thrust Inputs
        ax3.plot(history['step'], history['thrust_l'], 'b-', alpha=0.6, label='Left Thrust')
        ax3.plot(history['step'], history['thrust_r'], 'm-', alpha=0.6, label='Right Thrust')
        ax3.set_ylabel("Thrust (0-1)")
        ax3.set_xlabel("Step")
        ax3.set_title("Control Inputs")
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig('thrust_analysis_fixed.png')
        print("✅ Analysis saved: thrust_analysis_fixed.png")