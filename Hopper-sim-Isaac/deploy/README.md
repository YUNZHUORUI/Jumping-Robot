# Quadhopper 实机部署包

## 文件清单
- `policy.onnx` — 训练好的 LSTM PPO 策略（37 维 obs → 4 维 action）
- `run.py` — Windows 实机推理脚本（Vicon + 串口 + ONNX 推理）
- `requirements.txt` — Python 依赖

## 依赖安装
```cmd
pip install -r requirements.txt
```

**vicon_dssdk**：pip 装不了，需要从 Vicon 官网下载 **DataStream SDK** 后跑安装包里的 Python wheel：
```cmd
pip install vicon_dssdk-*.whl
```

## 硬件配置（在 run.py 顶部）
| 常量 | 含义 | 当前值 | 调整 |
|---|---|---|---|
| `COM_PORT` | 飞控串口 | `COM9` | 改成实际端口 |
| `BAUD_RATE` | 波特率 | `115200` | 匹配飞控 |
| `VICON_HOST` | Vicon Tracker 地址 | `localhost:801` | Tracker 在另一台机就改 IP |
| `TARGET_X/Y/Z` | 目标位置 | `(0, 0, 0.8)` | 跳跃 apex 目标 |
| `MAX_PWM` | 最大 PWM | `1000` | 匹配 ESC 范围 |
| `REST_LEG_LENGTH` | 静止时 Vicon 测得 body 高度 | `0.115` m | **必须校准**：手动按住 hopper 在地面，读 vicon body z |

## 运行
```cmd
python run.py
```

启动流程：
1. 连接 Vicon → 等候 hopper 出现在动捕区
2. 400 帧 IMU 校准（约 2 秒，hopper 必须完全静止）
3. 进入 100 Hz 控制循环 → policy 自动起跳

## ONNX 接口（LSTM）
- **Inputs**: `obs (1,37)`, `h_in (1,1,256)`, `c_in (1,1,256)`
- **Outputs**: `actions (1,4)`, `h_out (1,1,256)`, `c_out (1,1,256)`

`run.py` 已自动检测 LSTM 并维护 hidden state。如果将来切换成 MLP policy 重训，也能自动适配单输入接口。

## obs 拼装（37 维，需严格对齐仿真）
```
[0:3]   lin_vel_b         # body 坐标系线速度（vicon 速度 → R_w2b 转换）
[3:6]   ang_vel_b         # body 坐标系角速度（IMU gyro，已校准）
[6:10]  quat_w_isaac      # [w, x, y, z]（注意 Isaac 顺序，vicon 通常给 [x,y,z,w]）
[10:13] pos_error_b       # body 坐标系目标位置误差
[13]    z_pos             # body 高度（vicon）
[14]    is_contact        # 1=触地, 0=空中，按 z < REST_LEG_LENGTH+0.005 判定
[15]    joint_pos         # 弹簧压缩量（盲估：max(0, REST_LEG_LENGTH - z)）
[16]    joint_vel         # 弹簧速度（盲估：-kf_vz）
[17:37] history_actions   # 最近 5 帧 action × 4 motors
```

## Sim2Real Gap 检查清单
- [ ] `REST_LEG_LENGTH` 已用 vicon 校准
- [ ] 电机推力曲线匹配仿真 `F = -0.9715u² + 1.2578u - 0.0577` (u≤0.64)
- [ ] 电池电压充足（TWR > 1.3 才能跳起来）
- [ ] 控制循环稳定在 100 Hz（用 time.perf_counter 验证）
- [ ] Vicon 帧率 ≥ 100 Hz
- [ ] IMU 校准时 hopper 完全静止（手扶住，不要放在桌上）

## 如果起飞效果不好
| 症状 | 检查 |
|---|---|
| 跳不起来 | 电压低 / `REST_LEG_LENGTH` 错（导致 obs 假装一直在空中）/ 电机推力小于仿真 |
| 起飞但晃 | IMU 噪声大 → 调低 `accel_lpf` / `gyro_lpf` cutoff（当前 40 Hz）|
| 漂移很厉害 | Vicon 标定不准 / pos_error 计算错（检查坐标系） |
| 第一秒乱飞 | LSTM 还在"找感觉"。等 1-2 秒后看是否稳定。如果一直乱 → obs 拼装顺序错了 |
