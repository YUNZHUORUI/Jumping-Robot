# v54 phase-conditioned residual 当前上下文

更新时间: 2026-08-24 (Asia/Shanghai)

## 目标

连续随机两跳成功率太低的主因已经从 v50/v53 收敛为:

- second-hop best-of-8 覆盖率很高,说明物理可控性存在;
- first-hop scorer 离线 oracle 高、在线选择弱,说明单步延迟 pair 预测不稳;
- 当前 state planner 只在下降段施加很小 motor correction,无法显著改变支撑期起跳冲量和早期飞行轨迹。

v54 的结构目标是:保留冻结 teacher 作为稳定内核,给 residual policy
小幅、相位条件化的整跳 motor authority,让它能修正 stance/ascent/descent,
同时零初始化保证初始行为等价于 teacher。

## 已修改文件

- `Quadhopper_Planner_Random/phase_residual_wrapper.py`
  - 新增 `TeacherTwoHopPhaseResidualVecEnv`
  - observation = 67-D = 43-D teacher obs + 24-D 辅助状态
  - action = 4-D: collective / roll / pitch / yaw high-level residual
  - phase-specific limits:
    - collective `(0.060, 0.040, 0.025, 0.030)`
    - attitude `(0.050, 0.040, 0.030, 0.040)`
    - yaw `(0.020, 0.015, 0.012, 0.015)`
    - phase 顺序: contact, takeoff transition, ascent, descent
  - 支撑期保留完整 residual authority;飞行期小落点误差时用 safety gate 收小 residual。
  - 输出日志:
    - `Metrics/phase_residual_motor_abs_mean`
    - `Metrics/phase_residual_motor_abs_max`
    - `Metrics/phase_residual_slew_abs_mean`
    - `Metrics/phase_residual_regularization`
    - `Metrics/phase_residual_slew_regularization`
    - `Metrics/phase_residual_stance_duty`
    - `Metrics/phase_residual_clip_fraction`
    - pair/conditional metrics

- `train_two_hop_residual.py`
  - 新增 `--phase_residual`
  - 新实验名:
    `quadhopper_planner_random_two_hop_v54_phase_residual_ppo`
  - v54 使用 hop-based distance curriculum
  - v54 PPO 默认:
    - `init_noise_std = --init_noise_std` (默认 0.02)
    - learning rate `1e-4`
    - entropy coef `2e-4`
  - v54 与 `--state_planner` 互斥。

- `tests/test_two_hop_residual_wrapper.py`
  - 新增 4 个 phase residual 单元测试:
    - 67-D observation width
    - 零 action 不改变 motor
    - yaw residual collective sum 为 0
    - safety gate 不削弱 stance,但会收小飞行期小误差 residual

## 已完成验证

```text
python -m compileall ... 通过
git diff --check ... 通过
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  -m unittest discover -s Hopper-sim-Isaac/tests -p 'test_*.py'
Ran 11 tests in 0.025s
OK
```

注意:系统 conda Python 没有 `pytest`/`torch`,因此测试使用 Isaac Sim
自带 `python.sh` 跑 `unittest discover`。

## 当前阻塞

已解除。Codex 沙盒内看不到 GPU,但非沙盒执行可正常使用 GPU:

```text
NVIDIA-SMI 570.211.01 / CUDA 12.8
```

注意:后续 Isaac 命令需要在非沙盒/已授权环境中运行,否则会出现
`No CUDA GPUs are available`。

## v54 首轮训练结果

训练命令:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  train_two_hop_residual.py \
  --headless \
  --phase_residual \
  --num_envs 256 \
  --iterations 80 \
  --curriculum_iterations 50 \
  --save_interval 10 \
  --init_noise_std 0.03 \
  --residual_slew_rate 0.015 \
  --residual_penalty_scale 45 \
  --slew_penalty_scale 10 \
  --safety_gate_error_m 0.08 \
  --teacher_checkpoint \
  logs/rsl_rl/quadhopper_planner_random_two_hop_v36_full_relative_next_smooth_distance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_fixed_100/2026-08-23_12-24-40/model_90.pt
```

输出目录:

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v54_phase_residual_ppo/2026-08-24_15-47-22
```

训练正常完成,无 CUDA assert。保存 checkpoint:

```text
model_0.pt, model_0_zero_residual.pt, model_10.pt, model_20.pt,
model_30.pt, model_40.pt, model_50.pt, model_60.pt, model_70.pt, model_79.pt
```

训练统计观察:

- residual 幅度很小:`phase_residual_motor_abs_mean` 约 `0.0012-0.0014`
- `phase_residual_motor_abs_max` 约 `0.006-0.008`
- training pair rate 后段约 `0.125-0.132`
- reward 持续上升,但 pair success 没有同步上升,说明逐帧/单跳 shaping
  仍然主导优化方向。

严格 3000-step full-range evaluation:

```text
model_79:
  touchdown_count=3928
  target_hit_rate=0.347505
  touchdown_error_m=0.143792
  max_consecutive_hits=2.046875
  conditional_second_hit_rate=0.408305
  two_hop_pair_success_rate=0.125133

model_50:
  touchdown_count=4247
  target_hit_rate=0.336944
  touchdown_error_m=0.144373
  max_consecutive_hits=2.128906
  conditional_second_hit_rate=0.381173
  two_hop_pair_success_rate=0.120901

model_10:
  touchdown_count=4021
  target_hit_rate=0.346183
  touchdown_error_m=0.144409
  max_consecutive_hits=2.210938
  conditional_second_hit_rate=0.403175
  two_hop_pair_success_rate=0.131470
```

Decision:

- v54 首轮 PPO 不替换 accepted v49 + v50 baseline。
- 最好候选为 `model_10`,但 pair `0.131470` 仍低于 accepted
  `0.146438`。
- v54 wrapper 结构保留;训练方式需要调整。

下一步建议:

- 不继续加长这条逐帧 PPO 训练。
- 将 v54 phase residual 接入 Semi-MDP 事件级训练,复用 v46/v50 的
  pair-credit 思路,使每个 stored transition 绑定完整一跳或两跳 pair
  回报。
- 或先做 phase residual action search/BC:在相同 route context 下 fan-out
  stance/ascent residual sequence,用 pair success 排序,再训 scorer/selector。
  但最直接的是 Semi-MDP phase residual,因为当前失败信号已经很清楚:
  新执行权限有了,优化目标没有对齐。

## 重启后的命令

先检查:

```bash
nvidia-smi
```

### v54 冒烟

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac

/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  train_two_hop_residual.py \
  --headless \
  --phase_residual \
  --num_envs 16 \
  --iterations 1 \
  --curriculum_iterations 8 \
  --save_interval 1 \
  --teacher_checkpoint \
  logs/rsl_rl/quadhopper_planner_random_two_hop_v36_full_relative_next_smooth_distance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_fixed_100/2026-08-23_12-24-40/model_90.pt
```

### v54 正式首轮训练

建议先跑保守首轮,避免 residual 过早破坏 teacher:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  train_two_hop_residual.py \
  --headless \
  --phase_residual \
  --num_envs 256 \
  --iterations 80 \
  --curriculum_iterations 50 \
  --save_interval 10 \
  --init_noise_std 0.03 \
  --residual_slew_rate 0.015 \
  --residual_penalty_scale 45 \
  --slew_penalty_scale 10 \
  --safety_gate_error_m 0.08 \
  --teacher_checkpoint \
  logs/rsl_rl/quadhopper_planner_random_two_hop_v36_full_relative_next_smooth_distance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_fixed_100/2026-08-23_12-24-40/model_90.pt
```

### 评估

把 `<RUN_DIR>/model_N.pt` 替换成训练输出中的候选 checkpoint:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  train_two_hop_residual.py \
  --headless \
  --phase_residual \
  --num_envs 256 \
  --eval_checkpoint <RUN_DIR>/model_N.pt \
  --eval_steps 3000 \
  --curriculum_iterations 0 \
  --teacher_checkpoint \
  logs/rsl_rl/quadhopper_planner_random_two_hop_v36_full_relative_next_smooth_distance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_fixed_100/2026-08-23_12-24-40/model_90.pt
```

## 当前 accepted baseline

继续保留 v49 planner + v50 second-hop scorer 作为已接受基线:

- planner:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v49_second_hop_adapter_semimdp_ppo/2026-08-24_10-58-40/model_321.pt`
- scorer:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v50_action_scorer/scorer_best_with_offsets.pt`

v54 只有在 3000-step full-range evaluation 中超过该组合时才替换。
