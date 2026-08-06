# Quadhopper 圆轨迹阶段汇总（截至 v10）

更新时间：2026-08-06

这份文档用于给后续开发者或 AI 恢复项目上下文。当前阶段已经从稳定跳跃基线发展到：Quadhopper 在 XY 平面沿半径 2 m 的圆形落点序列连续跳跃，策略同时观察后续两个落点，由数值 planner 联合规划两个完整跳跃周期。v10 进一步针对确定性仿真中的落点精度进行了训练，当前 play 效果已经由用户确认可用。

## 1. 当前可用成果

- 任务入口：`train_planner_circular.py`、`play_planner_circular.py`
- 环境：`Quadhopper_Planner_Circular/planner_circular_env.py`
- Planner：`Quadhopper_Planner_Circular/trajectory_planner.py`
- PPO 配置：`Quadhopper_Planner_Circular/rsl_rl_ppo_cfg.py`
- 当前实验名：`quadhopper_planner_circular_v10`
- 当前最新完整训练：`logs/rsl_rl/quadhopper_planner_circular_v10/2026-08-06_11-10-23`
- 当前最新 checkpoint：`logs/rsl_rl/quadhopper_planner_circular_v10/2026-08-06_11-10-23/model_518.pt`
- v10 前置短测：`logs/rsl_rl/quadhopper_planner_circular_v10/2026-08-06_11-05-20/model_19.pt`

`model_518.pt` 是目前默认 play 自动选择的最新模型。用户已通过可视化确认这一轮效果达到当前阶段要求。这里的“可用”指 nominal、确定性仿真效果；尚不能据此认定随机动力学、扰动环境或真实硬件也同样稳定。

## 2. 保持不变的 Quadhopper 基线

硬件模型来自 `Quadhopper_Stable`，其源头是用户提供的六个稳定跳跃文件：

- `Create-Hopper-Model.py`
- `Jump+Base.stl`
- `Jump+Leg.stl`
- `my_hopper_cfg.py`
- `quadhopper_env.py`
- `rsl_rl_ppo_cfg.py`

本任务没有改变以下物理契约：

- `HopperAsset.usd`、机体质量/惯量、弹簧腿与接触模型；
- 4 个直接电机动作及 `F1,F2,F3,F4` 顺序；
- 推力/力矩拟合曲线和 100 Hz 控制频率；
- 动作延迟、一级电机滞后和功率模型的定义；
- RSL-RL recurrent actor/critic 的基本结构。

不要改回主分支旧的 140 g/restitution hopper，也不要把 `experiment/higher-jump` 当作当前参数源。权威冻结副本是 `Quadhopper_Stable`。

## 3. 新圆轨迹任务的定义

### 目标与路径

- 圆半径：`2.0 m`
- 相邻落点弦长：约 `0.22 m`
- 一圈：`58` 个成功落点
- episode：`150 s`
- 期望 root 跳跃最高点：默认 `1.30 m`
- 落地 root 高度：`0.38 m`
- 落点成功 XY 容差：`0.10 m`
- 落点精度核宽度：`0.06 m`

`P_t` 和 `P_{t+1}` 都是地面上的落点：

- `P_t`：下一跳落点；
- `P_{t+1}`：下下跳落点；
- target height：每次跳跃希望达到的最高 root 高度，而不是第三个空中路径点。

### 双周期 Planner

任务不再使用人为计算的解析抛物线。Planner 使用离散 direct-collocation/KKT 最小 jerk 求解，在同一次规划中联合生成：

1. 当前状态到 `P_t` 的完整跳跃周期；
2. `P_t` 到 `P_{t+1}` 的完整跳跃周期。

下降阶段还使用弹道落点投影，为接触前的 XY 位置和切向速度修正提供 dense guidance。

### 42 维 observation 契约

策略输入固定为：

```text
[稳定跳跃基线 37 维,
 P_t 在 body frame 下的 XY 误差 2 维,
 P_t+1 在 body frame 下的 XY 误差 2 维,
 相对目标高度 1 维]
= 42 维
```

动作仍为 4 维直接电机动作。任何改变 observation 顺序、缩放或维数的后续任务，都不能无适配地直接加载本阶段 checkpoint。

### 可视化颜色

- 绿色：本次跳跃的 apex 高度目标；
- 橙色：地面落点 `P_t`；
- 紫色：地面落点 `P_{t+1}`；
- 蓝色点：当前到 `P_t` 的 planner 轨迹；
- 青色点：`P_t` 到 `P_{t+1}` 的 planner 轨迹；
- 黄色细标记：完整地面圆周。

## 4. 训练版本演进与解决的问题

### v4/v5/v6：双周期规划与落点塑形

- 从稳定 37 维策略迁移到 42 维，新输入列初始化为零；
- 将两个跳跃周期放入一个联合 planner；
- 明确 `P_t`、`P_{t+1}` 为地面点；
- 增加 touchdown precision、landing error、下降弹道预测等奖励；
- 修正可视化和 `env.reset()` 后再 `step()` 的播放流程。

### v7/v8：让完整一圈在时间上可达

- 根据 2 m 半径与 0.22 m 弦长定义一圈为 58 点；
- episode 从不足的时长调整为 150 s；
- 降低 entropy，并在迁移时重置过大的 action std；
- 增加 `successful_waypoints` 与 `circle_completion` 指标。

v8 能累计完成一圈，但中间漏点后补跳仍会被算作最终完成，因此不能证明每个目标都是第一次命中。

### v9：严格连续命中

- 完成条件改为连续 58 个首次落点命中；
- 任意 miss 都将连续命中计数清零；
- 增加 miss penalty 和 streak progress reward；
- 主要选择指标改为 `Metrics/max_consecutive_hits`。

v9 确定性 play 仍常见每圈约 4–5 个点需要补跳。训练中的随机质量、惯量、时延和 observation noise，与单环境 play 的 nominal 条件存在差异，是继续提高确定性落点精度的主要障碍之一。

### v10：nominal 落点精度微调

v10 专门匹配单环境 play 条件：

- 质量和惯量倍率固定为 `1.0`；
- 电机时间常数固定为 `0.125 s`；
- 动作延迟固定为 3 个 control step；
- observation noise 降为 `0.002`；
- play 中 observation noise 为 `0.0`；
- PPO entropy coefficient 为 `0.0005`；
- 从旧版本迁移时 action std 重置为 `0.2`，并丢弃旧 optimizer；
- 从 v10 checkpoint 继续训练时精确恢复 optimizer。

这些配置只覆盖圆轨迹 v10。`Quadhopper_Stable` 默认仍保留动力学与时延随机化，因此没有修改基线物理参数含义。

## 5. 当前训练、测试和监控命令

先进入项目：

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac
```

从当前模型继续训练：

```bash
bash run_train_planner_circular.sh 64 \
  --iterations 500 \
  --checkpoint logs/rsl_rl/quadhopper_planner_circular_v10/2026-08-06_11-10-23/model_518.pt
```

自动播放最新 checkpoint：

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  play_planner_circular.py
```

播放当前明确记录的模型：

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  play_planner_circular.py \
  --checkpoint logs/rsl_rl/quadhopper_planner_circular_v10/2026-08-06_11-10-23/model_518.pt \
  --num_envs 1
```

TensorBoard：

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  -m tensorboard.main \
  --logdir logs/rsl_rl/quadhopper_planner_circular_v10 \
  --port 6006 --reload_interval 2
```

浏览器访问 `http://localhost:6006`。

## 6. 评价模型时看什么

优先级如下：

1. `Metrics/max_consecutive_hits`：目标为 58；
2. `Metrics/circle_completion`：严格连续一圈完成率；
3. `Metrics/mean_touchdown_error_m`：实际 touchdown XY 误差；
4. play 中是否仍发生针对同一个 `P_t` 的补跳；
5. 高度、姿态、动作平滑度和 action std。

不要只凭总 reward、value loss 或累计 waypoint 数判断效果。`successful_waypoints` 很高仍可能包含 miss 后补跳；loss 上升也不能单独说明策略退化。

## 7. Checkpoint 兼容规则

- 稳定基线 37-D checkpoint：必须通过 migration 扩展 recurrent 输入矩阵到 42 维；
- planner v4–v9：作为 v10 policy initialization，重置 std 和 optimizer；
- planner v10：可以 exact resume，包括 optimizer；
- 若未来新增钻环观测或改变动作接口：建立新版本/新实验目录，写显式迁移，不能直接假定兼容。

## 8. 后续建议

当前阶段先冻结 `model_518.pt` 作为圆轨迹 nominal checkpoint。下一阶段若进入钻环或其他任务，应在它之上分级训练：

1. 保持硬件、42-D 共有输入语义和稳定跳跃能力；
2. 新增环中心、法向、孔径/clearance 等任务观测；
3. 先用固定大圆环训练合法穿越，再逐步随机化位置、方向与孔径；
4. 将 nominal 精度与 domain-randomized robustness 分成不同实验版本；
5. 保存一份 nominal 最佳模型，避免鲁棒性训练覆盖当前成果。

在开展新任务前，先阅读：

- `skills/quadhopper-baseline/SKILL.md`
- `Quadhopper_Stable/BASELINE.md`
- `Quadhopper_Planner_Circular/README.md`
- 本文档

