import serial
import struct
import time
import math
import msvcrt
import threading
import numpy as np
import csv
from datetime import datetime
from collections import deque
from vicon_dssdk import ViconDataStream
from scipy.spatial.transform import Rotation as R
import onnxruntime as ort

COM_PORT = 'COM9'
BAUD_RATE = 115200
VICON_HOST = 'localhost:801'
POLICY_PATH = 'policy.onnx'

PAYLOAD_FORMAT = "<BBI6hff"
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FORMAT)
PACKET_SIZE = 2 + PAYLOAD_SIZE

# 跳跃机的目标定点设置
TARGET_X = 0.0
TARGET_Y = 0.0
TARGET_Z = 0.8  # 匹配仿真训练的目标跳跃高度（0.8m）

MAX_PWM = 1000
MIN_PWM = 0

# 【硬件参数对齐】跳跃机完全静止在地面时，Vicon测得的机身CoM高度(单位: 米)
# 如果你的起落架更换或者动捕球位置变动，请微调此数值
REST_LEG_LENGTH = 0.115


class PT1Filter:
    def __init__(self, cutoff_freq, dt):
        rc = 1.0 / (2.0 * math.pi * cutoff_freq)
        self.alpha = dt / (rc + dt)
        self.state = None

    def apply(self, sample):
        if self.state is None:
            self.state = np.copy(sample)
        else:
            self.state = self.state + self.alpha * (sample - self.state)
        return self.state


class PosVelKF:
    def __init__(self, init_pos):
        self.X = np.array([[init_pos], [0.0]])
        self.P = np.eye(2) * 0.1
        self.Q = np.array([[1e-4, 0.0], [0.0, 5e-2]])

    def predict(self, dt, accel):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([[0.5 * dt ** 2], [dt]])
        self.X = F @ self.X + B * accel
        self.P = F @ self.P @ F.T + self.Q
        return self.X[0, 0], self.X[1, 0]

    def update(self, pos_meas, vel_meas=None):
        if vel_meas is not None:
            H = np.array([[1.0, 0.0], [0.0, 1.0]])
            y = np.array([[pos_meas - self.X[0, 0]], [vel_meas - self.X[1, 0]]])
            R_noise = np.array([[1e-4, 0.0], [0.0, 2e-2]])
        else:
            H = np.array([[1.0, 0.0]])
            y = np.array([[pos_meas - self.X[0, 0]]])
            R_noise = np.array([[1e-4]])
        S = H @ self.P @ H.T + R_noise
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P


class ViconReceiver:
    def __init__(self, vicon_host):
        self.running = True
        self.connected = False
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.pos = np.array([0.0, 0.0, 0.0])
        self.new_frame_event = threading.Event()
        self.client = ViconDataStream.Client()
        self.vicon_host = vicon_host
        self._init_vicon()
        if self.connected:
            self.thread = threading.Thread(target=self._update_loop)
            self.thread.daemon = True
            self.thread.start()

    def _init_vicon(self):
        try:
            self.client.Connect(self.vicon_host)
            self.client.EnableSegmentData()
            self.client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)
            self.client.SetBufferSize(1)
            self.connected = True
        except:
            pass

    def _update_loop(self):
        first_frame = True
        while self.running and self.connected:
            try:
                if self.client.GetFrame():
                    subj = self.client.GetSubjectNames()[0]
                    seg = self.client.GetSegmentNames(subj)[0]
                    rot_data, rot_occ = self.client.GetSegmentGlobalRotationQuaternion(subj, seg)
                    trans_data, trans_occ = self.client.GetSegmentGlobalTranslation(subj, seg)
                    if not trans_occ and not rot_occ:
                        rq = np.array(rot_data)
                        rp = np.array([trans_data[0] / 1000.0, trans_data[1] / 1000.0, trans_data[2] / 1000.0])
                        if first_frame:
                            self.quat = rq
                            self.pos = rp
                            first_frame = False
                        else:
                            if np.linalg.norm(rp - self.pos) < 0.2:
                                if np.dot(self.quat, rq) < 0.0: rq = -rq
                                self.quat = rq / np.linalg.norm(rq)
                                self.pos = rp
                                self.new_frame_event.set()
            except Exception as e:
                pass
            time.sleep(0.001)

    def close(self):
        self.running = False


def run_autonomous_flight():
    try:
        ort_session = ort.InferenceSession(POLICY_PATH)
        # LSTM policy: inputs = [obs, h_in, c_in], outputs = [actions, h_out, c_out]
        # 自动适配 MLP (1 输入) 或 LSTM (3 输入) ONNX
        input_names = [i.name for i in ort_session.get_inputs()]
        is_lstm = ("h_in" in input_names) and ("c_in" in input_names)
        if is_lstm:
            lstm_hidden = ort_session.get_inputs()[1].shape  # e.g. [1, 1, 256]
            h_state = np.zeros(lstm_hidden, dtype=np.float32)
            c_state = np.zeros(lstm_hidden, dtype=np.float32)
            print(f"✅ 加载 LSTM PPO 模型: {POLICY_PATH}, hidden={lstm_hidden}")
        else:
            h_state = None
            c_state = None
            print(f"✅ 加载 MLP PPO 模型: {POLICY_PATH}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    vicon = ViconReceiver(VICON_HOST)
    if not vicon.connected: print("Waiting for Vicon..."); return
    while np.all(vicon.pos == 0.0): time.sleep(0.1)

    kf_px = PosVelKF(vicon.pos[0])
    kf_py = PosVelKF(vicon.pos[1])
    kf_pz = PosVelKF(vicon.pos[2])

    prev_vicon_pos = np.copy(vicon.pos)

    log_data = []
    is_calibrating = True
    calib_count = 0
    CALIB_SAMPLES = 400
    gyro_sum = np.zeros(3)
    accel_sum = np.zeros(3)
    gyro_bias = np.zeros(3)
    accel_bias = np.zeros(3)

    accel_lpf = PT1Filter(cutoff_freq=40.0, dt=0.005)
    gyro_lpf = PT1Filter(cutoff_freq=40.0, dt=0.005)
    vicon_vel_lpf = PT1Filter(cutoff_freq=40.0, dt=0.005)

    flight_mode = 'CALIBRATE'
    current_target_x, current_target_y = TARGET_X, TARGET_Y

    # 👑 【核心修改】：将动作记录拓展到 5 帧，完美对应仿真中 37 维状态空间
    action_history = deque([np.zeros(4, dtype=np.float32) for _ in range(5)], maxlen=5)
    last_obs_log = [0.0] * 37

    m1, m2, m3, m4 = 0, 0, 0, 0
    latest_filt_acc = np.zeros(3)
    latest_filt_gyro = np.zeros(3)
    latest_raw_g = [0.0, 0.0, 0.0]
    kf_vx, kf_vy, kf_vz = 0.0, 0.0, 0.0

    last_calc_time = time.perf_counter()
    vicon_tick = 0

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0)
        ser.reset_input_buffer()
        print(f"🚀 跳跃机系统就绪: 等待传感器校准后，按下 'H' 键开始跳跃...")

        buffer = bytearray()
        start_time = None

        while True:
            # ==== 1. 键盘监听 ====
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b'\x00', b'\xe0'):
                    arrow = msvcrt.getch()
                    if flight_mode == 'KEYBOARD':
                        if arrow == b'H':
                            current_target_x += 0.05
                        elif arrow == b'P':
                            current_target_x -= 0.05
                        elif arrow == b'K':
                            current_target_y += 0.05
                        elif arrow == b'M':
                            current_target_y -= 0.05
                else:
                    key_lower = key.lower()
                    if key_lower == b'e':
                        break
                    elif key_lower == b'h':
                        flight_mode = 'HOVER'  # 激活跳跃策略
                        current_target_x, current_target_y = vicon.pos[0], vicon.pos[1]
                        print("==== 👑 跳跃机 PPO 控制模式已激活 ====")
                    elif key_lower == b'k':
                        flight_mode = 'KEYBOARD'
                        current_target_x, current_target_y = vicon.pos[0], vicon.pos[1]

            # ==== 2. 串口数据解析 ====
            waiting = ser.in_waiting
            if waiting > 1024:
                ser.reset_input_buffer()
                buffer.clear()
            elif waiting > 0:
                buffer.extend(ser.read(waiting))

            while len(buffer) >= PACKET_SIZE:
                if buffer[0] == 0xAA and buffer[1] == 0xBB:
                    payload = buffer[2:PACKET_SIZE]
                    del buffer[:PACKET_SIZE]
                    try:
                        data = struct.unpack(PAYLOAD_FORMAT, payload)
                        if data[0] == 0xAA:
                            raw_ax_o, raw_ay_o, raw_az_o = data[3] / 16384 * 9.81, data[4] / 16384 * 9.81, data[
                                5] / 16384 * 9.81
                            raw_gx_o, raw_gy_o, raw_gz_o = math.radians(data[6] / 16.4), math.radians(
                                data[7] / 16.4), math.radians(data[8] / 16.4)

                            mapped_accel = np.array([-raw_ay_o, raw_ax_o, raw_az_o])
                            mapped_gyro = np.array([-raw_gy_o, raw_gx_o, raw_gz_o])

                            if is_calibrating:
                                gyro_sum += mapped_gyro
                                accel_sum += mapped_accel
                                calib_count += 1
                                if calib_count >= CALIB_SAMPLES:
                                    gyro_bias = gyro_sum / CALIB_SAMPLES
                                    init_r = R.from_quat(vicon.quat)
                                    g_body = init_r.inv().apply([0, 0, 9.81])
                                    accel_bias = (accel_sum / CALIB_SAMPLES) - g_body
                                    is_calibrating = False
                                    print("✅ 传感器校准成功。请将机器置于平地，按下 'H' 起飞跳跃！")
                                continue

                            clean_acc = mapped_accel - accel_bias
                            clean_gyro = mapped_gyro - gyro_bias
                            latest_filt_acc = accel_lpf.apply(clean_acc)
                            latest_filt_gyro = gyro_lpf.apply(clean_gyro)
                            latest_raw_g = [raw_gx_o, raw_gy_o, raw_gz_o]
                    except Exception:
                        pass
                else:
                    del buffer[0]

            # ==== 3. Vicon 心跳触发控制循环 (200Hz) ====
            if not is_calibrating and vicon.new_frame_event.wait(timeout=0.001):
                vicon.new_frame_event.clear()
                vicon_tick += 1

                curr_calc_time = time.perf_counter()
                dt = curr_calc_time - last_calc_time
                if dt <= 0.001 or dt > 0.05: dt = 0.005
                last_calc_time = curr_calc_time

                raw_vicon_vel = (vicon.pos - prev_vicon_pos) / dt
                if np.linalg.norm(raw_vicon_vel) > 10.0: raw_vicon_vel = np.zeros(3)
                vicon_vel_meas = vicon_vel_lpf.apply(raw_vicon_vel)
                prev_vicon_pos = np.copy(vicon.pos)

                pos_w = np.copy(vicon.pos)
                latest_pure_q = vicon.quat
                r_mat = R.from_quat(latest_pure_q)
                a_world = r_mat.apply(latest_filt_acc) - np.array([0, 0, 9.81])

                kf_px.predict(dt, a_world[0])
                kf_py.predict(dt, a_world[1])
                kf_pz.predict(dt, a_world[2])

                kf_px.update(pos_w[0], vicon_vel_meas[0])
                kf_py.update(pos_w[1], vicon_vel_meas[1])
                kf_pz.update(pos_w[2], vicon_vel_meas[2])

                kf_px_pos, kf_vx = kf_px.X[0, 0], kf_px.X[1, 0]
                kf_py_pos, kf_vy = kf_py.X[0, 0], kf_py.X[1, 0]
                kf_pz_pos, kf_vz = kf_pz.X[0, 0], kf_pz.X[1, 0]

                if flight_mode == 'CALIBRATE':
                    m1, m2, m3, m4 = 0, 0, 0, 0
                else:
                    # 100Hz 策略降频执行
                    if vicon_tick % 2 == 0:
                        target_w = np.array([current_target_x, current_target_y, TARGET_Z])
                        vel_w = np.array([kf_vx, kf_vy, kf_vz])

                        R_w2b = r_mat.inv()
                        lin_vel_b = R_w2b.apply(vel_w)
                        ang_vel_b = latest_filt_gyro
                        quat_w_isaac = np.array(
                            [latest_pure_q[3], latest_pure_q[0], latest_pure_q[1], latest_pure_q[2]])

                        pos_error_w = target_w - pos_w
                        pos_error_b = R_w2b.apply(pos_error_w)

                        # 👑 【核心修改】：通过动捕高度精确盲估弹簧运动相位
                        z_pos = np.array([pos_w[2]])

                        # 判定触地状态（当高度低于完全伸展高度加5mm滑移空间时，判定触地）
                        is_contact = 1.0 if pos_w[2] < (REST_LEG_LENGTH + 0.005) else 0.0
                        is_contact_arr = np.array([is_contact])

                        if is_contact > 0.5:
                            joint_pos = np.array([max(0.0, REST_LEG_LENGTH - pos_w[2])])
                            joint_vel = np.array([-kf_vz])  # 质心向下运动对应着弹簧阻尼压缩
                        else:
                            joint_pos = np.array([0.0])
                            joint_vel = np.array([0.0])

                        # 拼接 5 帧动作历史（共 20 维）
                        history_actions = np.concatenate(action_history).astype(np.float32)

                        # 👑 【严格对齐】：拼装成与仿真完全一致的 37 维 Observation
                        obs = np.concatenate([
                            lin_vel_b, ang_vel_b, quat_w_isaac, pos_error_b,
                            z_pos, is_contact_arr, joint_pos, joint_vel, history_actions
                        ]).astype(np.float32)

                        last_obs_log = obs.tolist()
                        obs_input = obs.reshape(1, -1)

                        # PPO 网络推理（LSTM: 传 obs+h+c，输出 action+新 h+新 c；MLP: 只传 obs）
                        if is_lstm:
                            outs = ort_session.run(None, {"obs": obs_input, "h_in": h_state, "c_in": c_state})
                            action = outs[0][0]
                            h_state = outs[1]
                            c_state = outs[2]
                        else:
                            action = ort_session.run(None, {ort_session.get_inputs()[0].name: obs_input})[0][0]
                        action = np.clip(action, -1.0, 1.0)

                        action_history.append(action)

                        # 👑 【核心修改】：去除幽灵悬停偏置，与仿真完全对齐
                        target_u = action * 0.5 + 0.5
                        target_u = np.clip(target_u, 0.0, 1.0)

                        pwm_cmd = target_u * MAX_PWM
                        m1, m2, m3, m4 = int(pwm_cmd[0]), int(pwm_cmd[1]), int(pwm_cmd[2]), int(pwm_cmd[3])

                ser.write(struct.pack('<BBHHHH', 0x55, 0xAA, m1, m2, m3, m4))

                if start_time is None: start_time = time.perf_counter()

                log_row = [
                              time.perf_counter() - start_time, dt,
                              pos_w[0], pos_w[1], pos_w[2],
                              kf_vx, kf_vy, kf_vz,
                              current_target_x, current_target_y, TARGET_Z,
                              m1, m2, m3, m4,
                          ] + last_obs_log + [math.degrees(latest_raw_g[0]), math.degrees(latest_raw_g[1]),
                                              math.degrees(latest_raw_g[2])]
                log_data.append(log_row)

    except KeyboardInterrupt:
        pass
    finally:
        if 'ser' in locals() and ser.is_open:
            for _ in range(5): ser.write(struct.pack('<BBHHHH', 0x55, 0xAA, 0, 0, 0, 0)); time.sleep(0.02)
            ser.close()
        vicon.close()

    if log_data:
        csv_filename = f"quadhopper_deployed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        obs_headers = [
            'Obs_LinVel_bx', 'Obs_LinVel_by', 'Obs_LinVel_bz',
            'Obs_AngVel_bx', 'Obs_AngVel_by', 'Obs_AngVel_bz',
            'Obs_Quat_w', 'Obs_Quat_x', 'Obs_Quat_y', 'Obs_Quat_z',
            'Obs_PosErr_bx', 'Obs_PosErr_by', 'Obs_PosErr_bz',
            'Obs_Z_Height', 'Obs_IsContact', 'Obs_JointPos', 'Obs_JointVel'
        ]
        history_headers = [f'Obs_ActHist_tminus{5 - i // 4}_m{i % 4 + 1}' for i in range(20)]

        headers = [
                      'Time_s', 'loop_dt', 'X', 'Y', 'Z', 'Vel_X', 'Vel_Y', 'Vel_Z',
                      'Target_X', 'Target_Y', 'Target_Z', 'M1', 'M2', 'M3', 'M4',
                  ] + obs_headers + history_headers + [
                      'Raw_IMU_Gx_deg', 'Raw_IMU_Gy_deg', 'Raw_IMU_Gz_deg'
                  ]

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(log_data)
        print(f"✅ 实飞日志已完美保存: {csv_filename}")


if __name__ == '__main__':
    run_autonomous_flight()