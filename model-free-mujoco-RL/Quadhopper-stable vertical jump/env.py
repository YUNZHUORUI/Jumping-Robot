import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
import os


class PogoDroneEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        if not os.path.exists("scene.xml"):
            raise FileNotFoundError("❌ 找不到 scene.xml，请检查路径")

        self.model = mujoco.MjModel.from_xml_path("scene.xml")
        self.data = mujoco.MjData(self.model)

        self.drone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadcopter")
        self.foot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "foot")

        # === 物理参数 ===
        self.dt = 0.001
        self.control_freq = 100  # 100Hz
        self.n_substeps = int(1 / (self.dt * self.control_freq))
        self.l0 = 1.0  # 弹簧原长

        # 动作: [M1, M2, M3, M4]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # 观察: 增加了一些有助于判断状态的变量，共 17 维
        # [Z, Quat(4), LinVel(3), AngVel(3), SpringLen, SpringVel, LastAction(4)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self.last_action = np.zeros(4)
        self.last_spring_len = 1.0
        self.was_contact = False

        # 计时器
        self.air_time = 0.0  # 连续滞空时间
        self.ground_time = 0.0  # 连续触地时间
        self.cycle_timer = 0.0  # 存活时间

        # 计算悬停油门
        mass = self.model.body_subtreemass[self.drone_id]
        gravity = 9.81
        thrust_coeff = 15.0  # 请确保这与XML一致
        # 基础油门设为 悬停油门，这样 action=0 时刚好悬停
        self.hover_throttle = np.clip((mass * gravity) / (4 * thrust_coeff), 0.0, 1.0)

    def _get_obs(self):
        pos = self.data.xpos[self.drone_id]
        quat = self.data.xquat[self.drone_id]
        vel = self.data.cvel[self.drone_id][3:6]

        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        # 获取机体坐标系下的角速度 (对姿态控制更友好)
        ang_vel_body = r.inv().apply(self.data.cvel[self.drone_id][0:3])

        pos_foot = self.data.xpos[self.foot_id]
        spring_len = np.linalg.norm(pos - pos_foot)

        dt_step = self.dt * self.n_substeps
        spring_vel = (spring_len - self.last_spring_len) / dt_step
        self.last_spring_len = spring_len

        obs = np.concatenate([
            [pos[2]],  # [0] Height
            quat,  # [1-4] Attitude
            vel,  # [5-7] World Lin Vel
            ang_vel_body,  # [8-10] Body Ang Vel
            [spring_len],  # [11]
            [spring_vel],  # [12]
            self.last_action  # [13-16]
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # === 随机初始化，增强鲁棒性 ===
        # 在 1.5m ~ 2.5m 之间随机高度扔下
        self.data.qpos[2] = 2.0 + np.random.uniform(-0.3, 0.3)

        # 稍微带一点水平速度噪音，逼它学会修正漂移
        self.data.qvel[0] = np.random.uniform(-0.1, 0.1)
        self.data.qvel[1] = np.random.uniform(-0.1, 0.1)
        self.data.qvel[2] = -0.5  # 初始向下速度

        # 稍微带一点姿态噪音
        self.data.xquat[self.drone_id] = [1, np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 0]

        mujoco.mj_step(self.model, self.data)

        self.last_action = np.zeros(4)
        self.was_contact = False
        self.air_time = 0.0
        self.ground_time = 0.0
        self.cycle_timer = 0.0

        pos = self.data.xpos[self.drone_id]
        pos_foot = self.data.xpos[self.foot_id]
        self.last_spring_len = np.linalg.norm(pos - pos_foot)

        return self._get_obs(), {}

    def step(self, action):
        # 动作映射: 0.0 -> hover, -1.0 -> idle, +1.0 -> max
        # 这样网络更容易学，因为 0 就是平衡点
        scaled_action = self.hover_throttle + 0.5 * action
        motors = np.clip(scaled_action, 0.0, 1.0)

        self.data.ctrl[:] = motors
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # 获取状态
        obs = self._get_obs()
        pos_z = obs[0]
        # quat = obs[1:5]
        vel_x, vel_y, vel_z = obs[5], obs[6], obs[7]
        ang_vel = obs[8:11]
        spring_len = obs[11]

        # 状态判定
        is_contact = spring_len < (self.l0 - 0.01)  # 弹簧只要压缩就算触地
        is_falling = vel_z < -0.1

        dt = self.dt * self.n_substeps
        self.cycle_timer += dt

        if is_contact:
            self.ground_time += dt
            self.air_time = 0.0
        else:
            self.air_time += dt
            self.ground_time = 0.0

        # ================================
        #      REWARD FUNCTION (核心)
        # ================================
        reward = 0.0

        # 1. 计算姿态惩罚 (Uprightness)
        # 这是一个 [0, 1] 的值，1表示完全垂直，0表示倒扣
        # 我们用 Z 轴在世界坐标系的投影来算
        rot_mat = self.data.xmat[self.drone_id].reshape(3, 3)
        z_axis_world = rot_mat[:, 2]  # 局部Z轴在世界的方向
        upright = z_axis_world[2]  # 既然世界Z是[0,0,1], 点积就是 z_axis_world[2]

        # 姿态惩罚基础值
        r_att = -2.0 * np.arccos(np.clip(upright, -1.0, 1.0))

        # 【关键修改】下降阶段的姿态“严打”
        # 如果正在快速下落，姿态惩罚翻倍！逼迫它在触地前摆正
        if is_falling:
            r_att *= 3.0
            # 额外惩罚角速度 (防止下落时旋转)
            reward -= 0.5 * np.sum(np.square(ang_vel))

        reward += r_att

        # 2. 水平漂移惩罚 (钉在 X=0, Y=0)
        pos_x = self.data.xpos[self.drone_id][0]
        pos_y = self.data.xpos[self.drone_id][1]
        r_drift = -2.0 * (pos_x ** 2 + pos_y ** 2) - 0.2 * (vel_x ** 2 + vel_y ** 2)
        reward += r_drift

        # 3. 弹簧压缩奖励 (强制去跳)
        # 鼓励压缩弹簧：压缩量越大，分越高
        compression = max(0, self.l0 - spring_len)
        if is_contact:
            reward += 10.0 * compression

            # 4. 高度限制与循环引导
        target_height = 2.0

        if pos_z > target_height:
            # 飞太高了：惩罚！逼它关油门下来
            # 距离越远罚越重
            reward -= 5.0 * (pos_z - target_height)
            # 如果飞太高还在往上冲，重罚
            if vel_z > 0:
                reward -= 2.0 * vel_z

        elif pos_z < target_height and not is_contact:
            # 在空中的正常高度：鼓励往上飞 (仅当向上移动时)
            if vel_z > 0:
                reward += 1.0 * vel_z

        # 5. 存活奖励 (微小，防止急于自杀)
        reward += 0.1

        # 6. 动作平滑 (减少电机高频抖动)
        reward -= 0.1 * np.linalg.norm(motors - self.last_action)
        self.last_action = motors

        # ================================
        #      TERMINATION (终止条件)
        # ================================
        terminated = False

        # 1. 严重倾斜 (翻车)
        # upright < 0.7 大约是倾斜 45度
        if upright < 0.7:
            terminated = True
            reward -= 20.0  # 翻车重罚

        # 2. 触地姿态保护 (防止用桨叶锄地)
        # 如果高度很低(Z<0.3) 且 没压弹簧(侧翻了)，杀
        if pos_z < 0.3 and not is_contact:
            terminated = True
            reward -= 20.0

        # 3. 飞丢了 (太高或太偏)
        if pos_z > 4.0 or (pos_x ** 2 + pos_y ** 2) > 1.0:
            terminated = True
            reward -= 10.0

        # 4. 节奏死亡 (Timeouts)
        # 在空中停留太久 (>2.0s) -> 判为“悬停偷懒” -> 杀
        if self.air_time > 2.0:
            terminated = True
            reward -= 10.0  # 惩罚悬停

        # 在地上趴太久 (>1.0s) -> 判为“起不来” -> 杀
        if self.ground_time > 1.0:
            terminated = True
            reward -= 10.0

        return obs, reward, terminated, False, {}