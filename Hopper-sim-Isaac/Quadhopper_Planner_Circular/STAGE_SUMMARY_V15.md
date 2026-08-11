# Quadhopper 固定与周期可变高度阶段汇总（v11–v22）

更新时间：2026-08-09

本文承接 `STAGE_SUMMARY_V10.md`，记录从已验收的固定高跳圆轨迹模型向
固定 `0.70 m` root-apex 以及周期可变高度控制扩展的实际结果。第 3–8 节
保留 v11–v15 的历史失败记录；第 9 节以后是 2026-08-09 的最新状态。本文只记录
已经由代码、TensorBoard 或 Isaac 播放验证的事实，不把训练最新 checkpoint
等同于最佳 checkpoint。

## 1. 当前结论

- 固定 `0.70 m` 已由 v17/v18 学会；关键是从 37-D stable baseline 重新训练，
  而不是从固定 `1.30 m` 的 v10 专家降高。
- 真正的高度条件控制已由 v21 从 stable baseline 直接训练得到。用户确认
  `0.70/0.90 m` 周期交替回放效果很好。
- 当前推荐模型是 v22 `0.70/1.00 m` 周期交替模型，用户已确认该版本可用：
  `logs/rsl_rl/quadhopper_planner_circular_v22_expand_alternate_070_100/2026-08-09_15-13-43/model_100.pt`。
- v10、v15 和 v19 的失败说明：固定单高度专家容易忽略高度 observation，
  之后微调会产生高度折中，不能把固定高度性能当作高度泛化能力。
- v15 第二轮 `model_498.pt` 仍属于崩坏 checkpoint，禁止使用。后文所有
  `v15 未达到门槛` 的表述只代表当时的历史结论。

## 2. 保持不变的硬件契约

本阶段仍继承 `Quadhopper_Stable`，没有改变：

- `HopperAsset.usd`、质量、惯量、弹簧刚度/阻尼/行程；
- 四电机顺序、推力和力矩拟合曲线；
- 100 Hz 控制、电机一阶滞后、动作延迟；
- 功率模型、接触模型和零 restitution；
- recurrent actor/critic 基本结构。

低高度训练失败不能通过修改物理参数掩盖。

## 3. v11–v14 暴露的问题

早期实验直接从固定约 `1.30 m` 的 v10 策略切换到低高度或交替高度，同时
改变 planner 时序、奖励和 observation。这造成严重分布偏移：旧策略仍采用
高跳动作，强行降低推力后又破坏水平落点。

已确认并修正的代码问题包括：

1. 目标高度语义统一为世界坐标中的 root apex；落地 root 约为 `0.38 m`，
   因此 `0.70 m` 表示约 `0.32 m` 的相对增高。
2. observation 从 42 维扩展为 43 维：稳定 37 维、`P_t` XY、`P_(t+1)` XY、
   `H_t` 和 `H_(t+1)`。
3. 42→43 迁移把 v10 固定高度列的常量贡献折叠进 LSTM bias，新高度列从零
   开始学习，避免把常量输入误认为高度泛化能力。
4. planner 根据实际起跳、apex 和落地高度计算每跳飞行时长；接触/stance 与
   flight 分段。它仍是运动学参考生成器，不是含电机、弹簧和接触约束的完整
   动力学 direct collocation。
5. reset 后必须先观察到真实弹簧压缩接触，之后才允许第一次 liftoff，避免
   第一跳 planner 时钟提前启动。
6. XY 与 Z planner reward 已拆分，避免较大的高度误差屏蔽水平学习信号。
7. 增加对称 apex error 和 airborne overshoot 惩罚。

这些修改解决了接口和事件错误，但没有自动产生合格的低跳策略。

## 4. v15 单变量降高课程

v15 不训练交替高度。它从 v10 `model_518.pt` 开始，保持圆轨迹、落点间距和
硬件不变，只用 300 iterations 的 cosine schedule 将固定 absolute apex 从
`1.30 m` 降到 `0.70 m`：

```text
iter 0:   1.30 m
iter 50:  约 1.26 m
iter 100: 约 1.15 m
iter 150: 约 1.00 m
iter 200: 约 0.85 m
iter 250: 约 0.74 m
iter 300: 0.70 m
```

低高度阶段使用 learning rate `1e-4`、entropy `2e-4`，每 50 iterations 保存。
TensorBoard 新增 `Metrics/command_apex_height_m`。

## 5. 实际训练结果

### 第一轮降高

目录：
`logs/rsl_rl/quadhopper_planner_circular_v15_descend_to_070/2026-08-07_20-42-30`

| checkpoint | command apex | measured apex | touchdown XY error | max streak | 结论 |
|---|---:|---:|---:|---:|---|
| model_50 | 1.262 m | 1.416 m | 0.060 m | 16.0 | 圆轨迹能力尚在，但明显超高 |
| model_100 | 1.152 m | 1.301 m | 0.017 m | 8.0 | 落点好，高度仍偏高 |
| model_150 | 1.004 m | 1.232 m | 0.130 m | 3.0 | 高度与落点开始退化 |
| model_200 | 0.852 m | 1.207 m | 0.077 m | 0.18 | 仍沿用高跳步态 |
| model_250 | 0.740 m | 0.818 m | 0.083 m | 0.13 | 最接近低跳，连续命中不合格 |
| model_299 | 0.700 m | 1.117 m | 0.142 m | 0.0 | 最终阶段反弹并失去落点 |

训练过程中 `circle_completion` 始终为 `0`。因此不能仅凭 model_250 的平均
落点误差小于 `0.10 m` 判定成功。

### 固定 0.70 m 继续训练

目录：
`logs/rsl_rl/quadhopper_planner_circular_v15_descend_to_070/2026-08-08_02-00-58`

最新 `model_498.pt`：

```text
command apex            0.700 m
mean measured apex      0.294 m
mean touchdown XY error 0.821 m
mean max streak         1.73
circle completion       0
```

该 checkpoint 已明显崩坏，不能继续训练或作为 high/alternate 初始化。

## 6. v15 当时的 Isaac 可视化状态（历史）

2026-08-08 已启动以下诊断播放：

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  play_planner_circular.py --height_stage low --num_envs 1 \
  --checkpoint logs/rsl_rl/quadhopper_planner_circular_v15_descend_to_070/2026-08-07_20-42-30/model_250.pt
```

选择 model_250 是因为它在训练统计中最接近低高度且平均落点误差尚可；视觉
检查仍需关注实际 hopper 是否跟随蓝色当前轨迹、是否首次命中橙色 `P_t`、
是否反复补跳，以及 apex 是否明显超过绿色方块。

## 7. v15 当时的 Checkpoint 使用规则（历史）

- 正式圆轨迹 baseline：只使用 v10 `model_518.pt`。
- 低高度视觉诊断：可使用 v15 `model_250.pt`，但必须标记为未验收。
- 禁止使用 v15 `model_299.pt` 或第二轮 `model_498.pt` 开启变化高度。
- 43-D checkpoint 不能由旧 42-D play 环境直接加载。
- 后续实验必须使用新的 log namespace，不能覆盖 v10。

## 8. v15 当时的下一步门槛（历史）

在满足以下全部条件之前，不训练 `0.70/1.00 m` 交替高度：

```text
fixed command apex          0.70 m
mean measured apex error   <= 0.08 m
mean touchdown XY error    < 0.10 m
max consecutive hits       接近 58
circle completion          > 0
Isaac play                 首跳命中且不依赖补跳
```

当前 v15 未达到门槛。后续实验已按此结论重新设计，结果见下文。

## 9. v16–v18：固定 0.70 m 与落点精度

### v16：从 v10 固定高跳专家直接训练（失败）

v16 将命令从第一轮固定为 `0.70 m`，不再使用 1.30→0.70 cosine schedule，
但仍从 v10 `model_518.pt` 初始化。训练到约 206 轮后停止。`model_200` 的
apex 约 `0.979 m`、落点误差约 `0.153 m`，高度在约 `0.45–1.09 m` 间大幅
波动，证明 v10 存在明显的固定高跳负迁移。

### v17：从 stable baseline 直接学习固定 0.70 m（成功）

初始化模型：
`logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt`

目录：
`logs/rsl_rl/quadhopper_planner_circular_v17_fixed_070_from_stable/2026-08-08_14-46-42`

推荐 checkpoint：`model_500.pt`。

```text
command apex             0.700 m
mean measured apex       0.678 m
mean touchdown XY error  0.0369 m
mean max streak          56.24
circle completion        0.777
```

单环境回放曾连续命中 58 跳并触发整圈完成。回放 CSV 在成功触地当帧已经切换
到下一个 waypoint，因此不能用触地当帧的 `Target_X/Y` 计算当前落点误差；
必须使用触地前一帧目标，否则会多算一个约 `0.22 m` 的 hop distance。

### v18：固定 0.70 m 首跳落点精度（成功）

目录：
`logs/rsl_rl/quadhopper_planner_circular_v18_fixed_070_accuracy/2026-08-08_17-10-49`

推荐 checkpoint：`model_100.pt`。该阶段从 v17 `model_500.pt` 初始化，使用
`5e-5` learning rate，并在训练中令首次 miss 直接结束 episode，禁止策略用
多次修正跳命中同一个 waypoint。用户回放确认高度与落点效果明显改善，能够
无失误完成大半圈。

## 10. v19：从固定 0.70 m 模型训练 0.70/0.80 m（未通过）

目录：
`logs/rsl_rl/quadhopper_planner_circular_v19_alternate_070_080_accuracy/2026-08-08_22-10-43`

v19 从固定高度 v18 `model_100.pt` 初始化。虽然训练中的平均 apex 和落点看似
可用，用户回放确认 `0.80 m` 档没有明显达到目标。原因是固定高度训练期间高度
输入始终为常量，网络可以忽略该 observation；同时 v19 继承了落点专项阶段较弱
的高度奖励，最终倾向于保持熟悉的约 `0.70–0.74 m`，而不是真正条件化高度。

因此 v19 checkpoint 不作为后续可变高度模型。曾规划的 v20 强高度奖励迁移
方案没有正式训练，随后改为从 stable baseline 直接学习周期可变高度。

## 11. v21：从 stable baseline 直接训练 0.70/0.90 m（成功）

目录：
`logs/rsl_rl/quadhopper_planner_circular_v21_direct_alternate_070_090/2026-08-08_23-35-23`

初始化模型仍是 37-D stable baseline `model_498.pt`，迁移为 43-D 后从第一轮
开始按成功 waypoint 周期交替：

```text
0.70 -> 0.90 -> 0.70 -> 0.90 -> ...
```

前期允许 miss 后重试，先学习高度条件和完整圆轨迹；没有经过任何固定高度专家。
训练 700 轮，关键进展如下：

| iteration | touchdown error | max streak | circle completion | 结论 |
|---:|---:|---:|---:|---|
| 250 | 约 0.067–0.094 m | 约 3–4 | 0 | 两档高度已分离，路径仍弱 |
| 450 | 约 0.050–0.073 m | 约 8–9 | 0 | 完整圆轨迹继续改善 |
| 598 | 约 0.048 m | 44.67 | 0 | 接近整圈 |
| 626 | 约 0.068 m | 36.91 | 0.0195 | 首次完成整圈 |
| 660 | -- | -- | 1.0 | 训练批次完成率峰值 |
| 668 | 0.0255 m | 54.95 | 0.8359 | 稳定高完成率区间 |
| 675 | 0.0106 m | 56.48 | 0.2383 | 保存的推荐 checkpoint |

`model_675.pt` 的分档 apex error 约为：高档 `0.0258 m`、低档接近 `0`。
用户回放确认该模型效果很好。这个实验验证了：直接向通用 stable baseline 暴露
多高度命令，比先训练单高度专家再扩展更容易学到真正的高度条件控制。

## 12. v22：将高档从 0.90 m 扩展到 1.00 m（成功，当前推荐）

目录：
`logs/rsl_rl/quadhopper_planner_circular_v22_expand_alternate_070_100/2026-08-09_15-13-43`

v22 从 v21 `model_675.pt` 初始化，重置 optimizer，使用 learning rate `5e-5`、
entropy `1e-4`，并强化对称 apex tracking。训练阶段仍允许落点重试，没有同时
引入 miss terminal。命令周期为：

```text
0.70 -> 1.00 -> 0.70 -> 1.00 -> ...
```

约 100 轮已经达到最佳区间，因此在约 110 轮安全停止：

```text
recommended checkpoint       model_100.pt
circle completion            1.000（iteration 99 batch）
max consecutive hits         58
mean touchdown XY error      0.0241 m
1.00 m command apex error    0.0243 m
0.70 m command apex error    0.0058 m
```

用户随后确认该版本模型可用。当前正式推荐 checkpoint：

```text
logs/rsl_rl/quadhopper_planner_circular_v22_expand_alternate_070_100/
2026-08-09_15-13-43/model_100.pt
```

## 13. 当前训练与播放命令

播放当前推荐模型：

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac

/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  play_planner_circular.py \
  --height_stage alternate \
  --height_low 0.70 \
  --height_high 1.00 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/quadhopper_planner_circular_v22_expand_alternate_070_100/2026-08-09_15-13-43/model_100.pt
```

v22 训练入口使用 `--expand_variable_height`；v21 从 stable baseline 直接训练时
使用 `--direct_variable_height`。两者都写入独立 namespace，不覆盖历史模型。

## 14. 后续工作规则

- 后续可变高度实验优先从 v22 `model_100.pt` 或 v21 `model_675.pt` 开始，
  不再从 v10 固定 1.30 m 专家或 v18 固定 0.70 m 专家开始。
- 若扩展为连续随机高度，必须从训练第一轮就让高度 observation 变化，并分别
  统计不同高度区间，不能只看平均 apex。
- 高度扩展和首跳落点精度不要在同一阶段同时收紧。先学高度范围，再用低学习率
  和 miss-terminal 做落点专项微调。
- TensorBoard 分档绝对 apex 在某批没有对应高度样本时可能被零值稀释；优先看
  分档 apex error、连续命中、完成率，并以单环境 Isaac 回放作最终验收。
- Isaac Sim 训练进程退出后偶尔无法立即重新建立 CUDA context；checkpoint 不受
  影响。必要时恢复 NVIDIA driver 或重启后再运行播放。
