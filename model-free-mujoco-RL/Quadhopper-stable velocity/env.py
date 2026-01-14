import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
import os


class PogoDroneEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        if not os.path.exists("/Users/terry/Jumping-Robot/model-free-mujoco-RL/Quadhopper-stable velocity/scene.xml"):
            raise FileNotFoundError("❌ 找不到 scene.xml")

        self.model = mujoco.MjModel.from_xml_path("/Users/terry/Jumping-Robot/model-free-mujoco-RL/Quadhopper-stable velocity/scene.xml")
        self.data = mujoco.MjData(self.model)

        self.drone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadcopter")
        self.foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot")

        self.dt = 0.001
        self.control_freq = 50
        self.n_substeps = int(1 / (self.dt * self.control_freq))
        self.l0 = 1.0

        # 移除了所有相位/频率变量，让物理自己决定周期

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # 回归 18维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        self.last_action = np.zeros(4)
        self.last_spring_len = 1.0
        self.target_vx = 1.5

        # 计时器
        self.air_time = 0.0
        self.ground_time = 0.0

        # 悬停油门
        mass = self.model.body_subtreemass[self.drone_id]
        gravity = 9.81
        thrust_coeff = 15.0
        self.hover_throttle = np.clip((mass * gravity) / (4 * thrust_coeff), 0.0, 1.0)

    def _get_obs(self):
        pos = self.data.xpos[self.drone_id]
        quat = self.data.xquat[self.drone_id]
        vel = self.data.cvel[self.drone_id][3:6]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        ang_vel_body = r.inv().apply(self.data.cvel[self.drone_id][0:3])

        spring_vec = self.data.xpos[self.foot_id] - pos
        spring_len = np.linalg.norm(spring_vec)

        dt_real = self.dt * self.n_substeps
        spring_vel = (spring_len - self.last_spring_len) / dt_real
        self.last_spring_len = spring_len

        # 18维：去掉了相位信号，因为我们不需要它了
        obs = np.concatenate([
            [pos[2]], quat, vel, ang_vel_body, spring_vec, self.last_action
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 随机初始化：空中
        self.data.qpos[2] = 2.0 + np.random.uniform(-0.2, 0.2)
        self.data.qvel[0] = 1.0 + np.random.uniform(-0.2, 0.2)  # Vx
        self.data.qvel[2] = -1.0  # 初始向下投掷

        # 姿态
        r = R.from_euler('xyz', [0, -0.1, 0])
        q = r.as_quat()
        self.data.xquat[self.drone_id] = [q[3], q[0], q[1], q[2]]

        mujoco.mj_step(self.model, self.data)
        self.last_action = np.zeros(4)
        self.air_time = 0.0
        self.ground_time = 0.0
        return self._get_obs(), {}

    def step(self, action):
        scaled_action = self.hover_throttle * 0.9 + 0.6 * action
        motors = np.clip(scaled_action, 0.0, 1.0)

        self.data.ctrl[:] = motors
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        pos_z = obs[0]
        vel_x, vel_y, vel_z = obs[5], obs[6], obs[7]
        pos_y = self.data.xpos[self.drone_id][1]

        quat = obs[1:5]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = r.as_euler('xyz')
        roll, pitch = euler[0], euler[1]

        spring_len = np.linalg.norm(obs[11:14])
        is_contact = spring_len < (self.l0 - 0.02)

        dt = self.dt * self.n_substeps
        if is_contact:
            self.ground_time += dt
            self.air_time = 0.0
        else:
            self.air_time += dt
            self.ground_time = 0.0

        # ================================
        #      V9 奖励：动态能源管控
        # ================================
        reward = 0.0

        # 计算当前的平均推力输出 (0~1)
        mean_thrust = np.mean(motors)
        power_usage = np.sum(np.square(motors))

        # 1. 核心：空中油贵，地面油便宜
        if is_contact:
            # 地面：奖励注入能量！
            # 只有在地面推油门，才能获得动能补充阻尼损耗
            reward -= 0.01 * power_usage  # 象征性收点费
        else:
            # 空中：油价暴涨
            # 迫使它收油门 (Idle)，只做姿态调整
            reward -= 0.5 * power_usage

            # 【双重保险】下落阶段严禁推油
            # 如果正在掉下来 (Vz < -0.5) 且还在推油门 (> 0.2) -> 重罚
            if vel_z < -0.5 and mean_thrust > 0.2:
                reward -= 2.0 * mean_thrust

        # 2. 速度保持
        reward += 2.0 * np.exp(-2.0 * abs(vel_x - self.target_vx))

        # 3. 轨迹约束
        reward -= 2.0 * abs(pos_y)
        reward -= 1.0 * abs(roll)

        # 4. 高度/弹簧逻辑
        # 我们不再强制 Vz 符号，而是奖励 "Apex Height" (跳跃顶点)
        # 只要能跳到 1.5m ~ 2.5m 之间，就给分。
        # 但这个很难算，不如直接奖励接触时的弹簧压缩
        if is_contact:
            reward += 10.0 * (self.l0 - spring_len)

        # 5. Raibert 姿态辅助 (保留)
        if vel_x < 0.9 and pitch > -0.05: reward -= 0.5
        if vel_x > 1.1 and pitch < -0.3: reward -= 0.5

        # 存活
        reward += 0.5

        # ================================
        #      终止条件
        # ================================
        terminated = False

        # 允许稍微长一点的滞空（因为自然跳跃可能很高），但不能无限飞
        if self.air_time > 1.5: terminated = True; reward -= 10.0
        if self.ground_time > 0.6: terminated = True; reward -= 10.0

        # 翻车
        if pos_z < 0.2 or abs(roll) > 0.8: terminated = True; reward -= 10.0
        if pos_z > 4.0: terminated = True; reward -= 5.0

        return obs, reward, terminated, False, {}