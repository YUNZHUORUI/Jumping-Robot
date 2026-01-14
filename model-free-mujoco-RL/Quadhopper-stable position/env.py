import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
import os


class PogoDroneEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        # === 1. 物理环境 ===
        if not os.path.exists("scene.xml"):
            raise FileNotFoundError("❌ 找不到 scene.xml！")

        self.model = mujoco.MjModel.from_xml_path("scene.xml")
        self.data = mujoco.MjData(self.model)

        self.drone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadcopter")

        # 弹簧原长
        self.l0 = 1.0

        # === 2. 任务参数 ===
        self.target_x = 5.0
        self.max_speed = 1.0

        self.dt = 0.001
        self.control_freq = 50
        self.n_substeps = int(1 / (self.dt * self.control_freq))

        # Action: 4电机
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: 18维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 初始高度 1.2m，自由落体
        self.data.qpos[2] = 1.2

        # 随机目标
        self.target_x = np.random.uniform(2.0, 5.0)

        mujoco.mj_step(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. 执行动作 (映射到 0 ~ 4.0N)
        # 即使这里推力够飞，我们也会在奖励函数里制止它
        real_ctrl = (action + 1.0) / 2.0 * 4.0
        self.data.ctrl[:] = real_ctrl

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        # 2. 获取状态
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        dist_x = self.target_x - pos[0]

        # 判定是否触地 (脚的高度 < 0.05 或 弹簧被压缩)
        # 获取弹簧长度 (通常在 qpos 的某个位置，这里简化用高度判断)
        # 更准确的方法是看 contact array，但高度法足够训练
        is_contact = pos[2] < (self.l0 - 0.05)

        # 计算弹簧压缩量 (近似)
        spring_compression = max(0, self.l0 - pos[2])

        # === 3. 奖励函数 (空中禁油令) ===
        reward = 0.0
        reward += 0.5  # 生存

        # 计算平均推力使用率 (0~1)
        thrust_usage = np.mean((action + 1.0) / 2.0)

        if is_contact:
            # 【地面阶段】：鼓励推力！鼓励压缩！
            # 只有在地面推力，才能给弹簧充能
            reward += 1.0 * thrust_usage
            reward += 5.0 * spring_compression  # 压得越深分越高
        else:
            # 【空中阶段】：严厉惩罚推力！(逼它关火)
            # 如果在空中还敢大脚给油，重罚
            reward -= 5.0 * thrust_usage

            # 额外惩罚：如果正在上升还给油，罚得更重 (防止火箭起飞)
            if vel[2] > 0:
                reward -= 5.0 * thrust_usage

        # 速度追踪 (仅在 X 轴)
        # 离得远全速，离得近减速
        target_vx = np.clip(dist_x * 1.5, -self.max_speed, self.max_speed)
        if abs(dist_x) < 0.2: target_vx = 0.0

        v_error = abs(vel[0] - target_vx)
        reward += 2.0 * np.exp(-2.0 * v_error)

        # 定点奖励
        if abs(dist_x) < 0.3:
            reward += 1.0
            # 到了终点，必须保持跳跃 (Z轴速度不能为0)
            # 如果它停在地上不动 (z < 0.5 且 vz=0)，扣分！
            if pos[2] < 0.5 and abs(vel[2]) < 0.1:
                reward -= 1.0

        # 姿态惩罚
        r = R.from_quat([self.data.qpos[4], self.data.qpos[5], self.data.qpos[6], self.data.qpos[3]])
        roll, pitch, yaw = r.as_euler('xyz')
        if abs(roll) > 0.5 or abs(pitch) > 0.5:
            reward -= 2.0

        # 5. 终止条件
        terminated = False
        truncated = False

        # 摔倒
        if pos[2] < 0.05 or abs(roll) > 1.0 or abs(pitch) > 1.0:
            terminated = True
            reward -= 10.0

        # 飞太高 (超过 2.5米 判负，防止它硬飞)
        if pos[2] > 2.5:
            terminated = True
            reward -= 10.0

        if self.step_count > 1200:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        quat = self.data.qpos[3:7]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = r.as_euler('xyz')
        ang_vel = self.data.qvel[3:6]

        dist_x = self.target_x - pos[0]

        obs = np.array([
            pos[2],  # 1
            vel[0], vel[1], vel[2],  # 3
            roll, pitch, yaw,  # 3
            ang_vel[0], ang_vel[1], ang_vel[2],  # 3
            dist_x,  # 1
            pos[1],  # 1
            0, 0, 0, 0, 0, 0  # 6
        ], dtype=np.float32)

        return np.clip(obs, -10, 10)