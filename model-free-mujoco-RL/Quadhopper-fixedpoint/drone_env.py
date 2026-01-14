import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import os


class DroneHoverEnv(gym.Env):
    def __init__(self, xml_file='appearance model/scene.xml', render_mode=None):
        super().__init__()

        # 路径处理
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(xml_file):
            xml_path = os.path.join(current_dir, xml_file)
        else:
            xml_path = xml_file

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # 动作空间: [推力, 滚转, 俯仰, 偏航]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # 观察空间:
        # [Pos_Err(3), Vel(3), Omega(3), Body_Z(3), Body_X(3)] = 15维
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self.dt = 0.02
        self.physics_steps_per_step = int(self.dt / self.model.opt.timestep)

        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.hover_input = 0.52

        # 增加时间限制，防止训练卡死
        self.max_steps = 500
        self.current_step = 0

    def _get_obs(self):
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadcopter")
        jnt_adr = self.model.body_jntadr[idx]

        pos = self.data.qpos[jnt_adr: jnt_adr + 3]
        quat = self.data.qpos[jnt_adr + 3: jnt_adr + 7]
        vel = self.data.qvel[jnt_adr: jnt_adr + 6]  # [vx, vy, vz, wx, wy, wz]

        # 1. 位置误差
        pos_err = pos - self.target_pos

        # 2. 姿态向量提取
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        rot_mat = r.as_matrix()

        # 提取机体坐标轴在世界系下的投影
        body_x = rot_mat[:, 0]  # 机头朝向
        body_z = rot_mat[:, 2]  # 机顶朝向

        # 拼装
        lin_vel = vel[0:3]
        ang_vel = vel[3:6]
        obs = np.concatenate([pos_err, lin_vel, ang_vel, body_z, body_x]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0  # 重置步数

        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadcopter")
        jnt_adr = self.model.body_jntadr[idx]

        # === 混合难度初始化 (课程学习) ===
        # 80% 简单模式：让它学会精细的悬停和锁定机头
        # 20% 困难模式：让它学会被猛拉后怎么救回来
        difficulty = np.random.rand()

        if difficulty < 0.8:
            # [简单模式]
            init_pos = self.target_pos + np.random.uniform(-0.1, 0.1, size=3)
            rand_euler = np.random.uniform(-0.1, 0.1, size=3)  # 稍微歪一点点
            init_vel = np.random.uniform(-0.1, 0.1, size=6)
        else:
            # [困难模式]
            init_pos = self.target_pos + np.random.uniform(-1.0, 1.0, size=3)
            init_pos[2] = np.random.uniform(0.8, 1.5)

            # 姿态随机：特别是偏航角(Yaw)要大范围随机，强迫它学会转回去
            rand_euler = np.random.uniform(-0.5, 0.5, size=3)
            rand_euler[2] = np.random.uniform(-3.14, 3.14)  # 偏航 360度随机

            init_vel = np.random.uniform(-1.5, 1.5, size=6)

        init_quat_scipy = R.from_euler('xyz', rand_euler).as_quat()
        mujoco_quat = np.array([init_quat_scipy[3], init_quat_scipy[0], init_quat_scipy[1], init_quat_scipy[2]])

        self.data.qpos[jnt_adr: jnt_adr + 3] = init_pos
        self.data.qpos[jnt_adr + 3: jnt_adr + 7] = mujoco_quat
        self.data.qvel[jnt_adr: jnt_adr + 6] = init_vel

        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None: self.viewer.sync()
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # === 增强动作权限 ===
        # 提升力矩系数，让它更有力气抵抗干扰
        thrust_cmd = self.hover_input + action[0] * 0.35
        roll_cmd = action[1] * 0.15  # 提升 50%
        pitch_cmd = action[2] * 0.15  # 提升 50%
        yaw_cmd = action[3] * 0.2  # 提升 100% (重点加强偏航能力)

        m1 = thrust_cmd + roll_cmd - pitch_cmd + yaw_cmd
        m2 = thrust_cmd - roll_cmd - pitch_cmd - yaw_cmd
        m3 = thrust_cmd - roll_cmd + pitch_cmd + yaw_cmd
        m4 = thrust_cmd + roll_cmd + pitch_cmd - yaw_cmd

        self.data.ctrl[:] = np.clip(np.array([m1, m2, m3, m4]), 0.0, 1.0)

        for _ in range(self.physics_steps_per_step):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # === 奖励函数 ===
        # 解包
        # obs: [pos_err(3), vel(3), omega(3), body_z(3), body_x(3)]
        pos_err = obs[0:3]
        lin_vel = obs[3:6]
        ang_vel = obs[6:9]
        body_z = obs[9:12]
        body_x = obs[12:15]

        dist = np.linalg.norm(pos_err)

        # 1. 距离奖励 (指数型，非常强烈的定点要求)
        # 距离越近，分越高。
        r_dist = 2.0 * np.exp(-3.0 * dist)

        # 2. 姿态稳定奖励 (直立)
        # Body Z 应该指向上方 [0,0,1]
        upright = body_z[2]
        r_att = np.clip(upright, -1.0, 1.0)

        # 3. === 核心：航向锁定奖励 (Heading Lock) ===
        # Body X (机头) 应该指向 World X [1,0,0]
        # 使用点积：越接近 1 越好。
        # 加大这个权重，强迫它转回来！
        heading_alignment = body_x[0]
        r_heading = 1.5 * heading_alignment

        # 4. 惩罚项
        # 惩罚旋转速度 (特别是偏航旋转)，防止它在那转圈圈
        # 但在误差很大时(heading_alignment小)，我们允许它旋转来修正
        # 所以这个惩罚是常驻的，它会学会在对齐后停止旋转
        r_spin = -0.1 * np.linalg.norm(ang_vel)
        r_vel = -0.1 * np.linalg.norm(lin_vel)  # 惩罚漂移速度
        r_act = -0.02 * np.linalg.norm(action)

        reward = r_dist + r_att + r_heading + r_spin + r_vel + r_act + 1.0

        # === 终止条件 ===
        terminated = False
        truncated = False

        # 撞地
        if self.data.qpos[2] < 0.05:
            terminated = True
            reward = -10.0

        # 飞太远
        if dist > 3.0:
            terminated = True
            reward = -5.0

        # 超时 (不算失败，算截断)
        if self.current_step >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None