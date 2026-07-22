# Isaac Lab QuadHopper 圆形轨迹连续跳跃训练方案

## 1. 任务目标

在现有 Isaac Lab `HopperEnv` 原地连续跳跃策略的基础上，实现三维平面内的轨迹条件连续跳跃。第一阶段不加入圆环或其他障碍物；机器人需要通过一系列离散落脚点，沿给定圆形路径稳定前进。

本阶段的目标不是让机器人在空中逐时刻贴合一条连续圆曲线，而是把圆形路径离散为连续的落脚目标：

1. 在支撑相产生朝向下一个目标点的起跳速度；
2. 在飞行相保持姿态稳定并修正水平速度；
3. 在目标点附近落地；
4. 落地稳定后更新到下一个目标点；
5. 连续完成一圈，并保持可重复的跳跃极限环。

这种定义符合 Hopper 的混合动力学，也与 `Hopper_simu/Quadhopper` 当前的“逐落点规划—起飞—飞行—落地—更新目标”结构一致。

---

## 2. 现有代码基础

### 2.1 Isaac Lab 主环境

主要修改文件：

```text
Hopper-sim-Isaac/Quadhopper_Isaac/my_hopper_env.py
Hopper-sim-Isaac/Quadhopper_Isaac/rsl_rl_ppo_cfg.py
Hopper-sim-Isaac/Quadhopper_Isaac/agents/rsl_rl_ppo_cfg.py
```

当前 `HopperEnv` 已具备：

- Isaac Lab `DirectRLEnv` 和 RSL-RL PPO；
- 4 电机归一化动作；
- 一阶电机延迟与一步控制延迟；
- 实测推力曲线；
- 质量、惯量、电机参数随机化；
- 原地连续跳跃；
- upright、水平位置保持、动作平滑等奖励；
- 200 Hz 物理频率、100 Hz 策略频率。

应保留已有电机、推力、力矩、延迟和 domain randomization 实现。第一版只改任务生成、观测、奖励、事件检测和日志，不要同时重构底层动力学。

### 2.2 二维参考实现

重点参考：

```text
Hopper_simu/Quadhopper/trajectory.py
Hopper_simu/Quadhopper/env.py
Hopper_simu/Quadhopper/reward.py
Hopper_simu/Quadhopper/feasibility.py
```

需要迁移的思想：

- `TrajectoryPlanner.plan()`：每一跳根据当前状态和下一个落点生成弹道参考；
- liftoff/touchdown/apex 三类事件；
- 起飞瞬间匹配规划速度，而不是只奖励逐步接近目标；
- touchdown 时计算落点误差；
- 命中目标后才推进目标索引；
- 先做可达性 sweep，再进行大规模 PPO 训练；
- 对每个 reward term 单独累计并写入 TensorBoard。

不能直接照搬的部分：

- 二维模型只有一个水平轴，Isaac 中必须使用世界系 XY 平面；
- 二维 `target_x` 必须变成每个环境独立的 `target_pos_w[:, 0:2]`；
- 圆轨迹不能依赖全局固定 X 正方向；所有目标误差和期望速度应在机体系或局部切向坐标系表达；
- 原有“超过当前 target_x 即失败”的逻辑不能用于圆形路径。

---

## 3. 圆形路径与离散落脚点

对每个并行环境保存：

```python
circle_center_w   # [num_envs, 2]
circle_radius     # [num_envs]
path_angle        # [num_envs]，当前已完成的圆周角
angle_direction   # [num_envs]，+1 逆时针，-1 顺时针
target_pos_w      # [num_envs, 2]，下一落脚目标
hop_index         # [num_envs]
successful_hops   # [num_envs]
```

圆形路径为：

\[
\mathbf p(\phi)=
\begin{bmatrix}
c_x+R\cos\phi\\
c_y+R\sin\phi
\end{bmatrix}
\]

其中：

- \((c_x,c_y)\) 为圆心，单位 m；
- \(R\) 为圆半径，单位 m；
- \(\phi\) 为路径相位，单位 rad。

不要一开始按固定时间推进 \(\phi\)。只有成功落地后才推进路径相位，避免机器人摔倒或漏跳后目标继续向前移动。

### 3.1 每跳目标间距

设期望相邻落脚点弦长为 \(d_{hop}\)，每跳圆周角增量为：

\[
\Delta\phi = 2\arcsin\left(\frac{d_{hop}}{2R}\right)
\]

下一目标为：

\[
\phi_{k+1}=\phi_k+s\Delta\phi,\qquad s\in\{-1,+1\}
\]

第一版建议：

| 参数 | 初始值 | 后续范围 |
|---|---:|---:|
| 圆半径 `R` | 2.0 m | 1.0–3.0 m |
| 每跳弦长 `d_hop` | 0.20 m | 0.15–0.45 m |
| 落点容差 | 0.18 m | 最终缩到 0.08–0.12 m |
| 单圈目标数 | 约 63 | 随半径与跳距变化 |
| 方向 | 固定逆时针 | 后期随机 ±1 |

如果当前硬件/仿真稳定跳跃的水平可达距离尚不明确，必须先进行直线目标 sweep，确定可靠的 `d_hop`，再训练圆形轨迹。

### 3.2 局部几何量

在当前位置 \(\mathbf p_{xy}\) 计算：

\[
\mathbf e_{target}=\mathbf p_{target}-\mathbf p_{xy}
\]

圆的径向误差：

\[
e_r=\|\mathbf p_{xy}-\mathbf c\|-R
\]

目标点切向单位向量：

\[
\mathbf t(\phi)=s[-\sin\phi,\ \cos\phi]^T
\]

将 `e_target` 和参考水平速度从世界系旋转到机体系后输入 actor。这样策略不需要分别学习圆的四个象限，且更容易迁移到任意圆心和起始朝向。

---

## 4. 单跳弹道参考

第一版采用与二维 `TrajectoryPlanner` 一致的逐跳弹道参考，不必先引入复杂轨迹优化器。

设起跳参考位置为 \(\mathbf p_0\)，落脚目标为 \(\mathbf p_f\)，预期飞行时间为 \(T\)。忽略飞行期间推力时：

\[
\mathbf v_{xy}^{ref}=\frac{\mathbf p_{f,xy}-\mathbf p_{0,xy}}{T}
\]

\[
v_z^{ref}=\frac{z_f-z_0+\frac{1}{2}gT^2}{T}
\]

更方便的做法是指定相对起跳点的目标顶点高度 \(h_a\)：

\[
v_z^{ref}=\sqrt{2gh_a}
\]

然后由预期回落高度求飞行时间，再计算 `v_xy_ref`。初期可固定：

```text
apex height above takeoff = 0.18–0.25 m
desired flight time       = 0.35–0.50 s
```

实际范围应通过仿真 sweep 确定。规划结果至少保存：

```python
takeoff_pos_w
target_pos_w
v_xy_ref_w
vz_ref
flight_time_ref
apex_height_ref
plan_valid
```

起飞事件发生时计算速度误差。不要在整个支撑相都强行奖励世界系目标速度；支撑初期速度低是正常的，持续惩罚会干扰压缩—回弹过程。

---

## 5. 混合相位与事件检测

建议显式维护三种相位：

```text
STANCE  -> LIFTOFF/FLIGHT -> TOUCHDOWN/SETTLE -> STANCE
```

最好给脚部或最低接触部件增加 Isaac Lab `ContactSensor`。事件定义：

- `touching`：接触法向力超过小阈值，并持续至少 2–3 个物理步；
- `liftoff_event`：上一策略步 touching、当前非 touching；
- `touchdown_event`：上一策略步非 touching、当前 touching；
- `apex_event`：飞行相中上一时刻 \(v_z>0\)，当前 \(v_z\le0\)；
- `settled`：touchdown 后保持接触若干步，倾角与角速度未超限。

若暂时无法增加 ContactSensor，可用根节点高度、竖直速度和接触几何高度组合构造事件，但不要只用单一 `z` 阈值；restitution 接触会在阈值附近产生抖动。

目标更新条件：

```python
target_hit = (
    touchdown_event
    & (landing_error < target_tolerance)
    & (tilt < landing_tilt_limit)
    & (vertical_speed < landing_vz_limit)
)
```

命中后更新 `path_angle` 和 `target_pos_w`。未命中时不要立即更换目标；允许 1 次恢复跳，或在严重越界时结束 episode。

---

## 6. Observation 设计

当前 26 维 observation 包括机体速度、角速度、四元数、高度、XY 位置误差和 3 步动作历史。圆轨迹版本建议改为约 34–38 维，具体维数必须与 `HopperEnvCfg.observation_space` 严格一致。

推荐 actor observation：

| 分量 | 维数 | 坐标系/说明 |
|---|---:|---|
| body linear velocity | 3 | 机体系 |
| body angular velocity | 3 | 机体系 |
| projected gravity | 3 | 机体系；可替代原始 quaternion，避免符号二义性 |
| root height | 1 | 相对地面 |
| target position error | 2 | 世界 XY 误差旋转到机体系 |
| reference horizontal velocity error | 2 | 机体系，仅支撑末段/起飞附近有效 |
| reference vertical velocity error | 1 | 世界 Z |
| radial path error | 1 | 标量，可裁剪/归一化 |
| tangent direction | 2 | 机体系 |
| contact flag | 1 | 0/1 |
| phase encoding | 2 | `sin(phase), cos(phase)` 或 one-hot |
| time since touchdown/liftoff | 1–2 | 归一化并裁剪 |
| previous action history | 12 | 保留现有 3×4 |

必须遵循：

1. 不把 `hop_index` 或绝对世界坐标直接作为策略必要输入；
2. 位置误差按典型跳距归一化；
3. 速度按预期最大速度归一化；
4. 角速度、误差和时间全部裁剪；
5. actor 只能使用以后实机可获得的量；若 critic 使用额外仿真状态，可采用 asymmetric actor-critic。

第一版可以继续使用 quaternion，减少代码改动；但若训练不稳定，优先改成 projected gravity，而不是继续堆奖励权重。

---

## 7. Action 设计

保留当前四电机动作：

```python
action.shape == (num_envs, 4)
action in [-1, 1]
```

继续使用现有：

- action → motor command 映射；
- 一步 delay；
- 一阶 motor lag；
- 推力曲线；
- 每电机 thrust randomization；
- roll/pitch/yaw 力矩计算。

不要在第一版同时改成高级 action（总推力 + 姿态目标）或 residual action。先证明直接四电机 PPO 能完成小步距定向跳跃；若探索困难，再增加低层姿态控制器或 residual policy。

---

## 8. Reward 设计

奖励应围绕“每跳成功”而不是“每个仿真步贴着圆走”。建议总奖励为：

\[
r=r_{alive}+r_{upright}+r_{hop}+r_{launch}+r_{progress}
+r_{landing}+r_{path}-r_{rate}-r_{smooth}-r_{failure}
\]

所有逐步 reward 乘 `step_dt`；liftoff、apex、touchdown、target hit 等事件奖励不乘 `step_dt`。

### 8.1 保留并调整现有奖励

- `survival`：保留但权重较小；
- `upright`：保留；
- `action_rate`、`action_smooth`：保留；
- `hop_velocity`：课程前期保留，后期降低；
- `ground_bonus`：降低，避免策略为了奖励拒绝水平前进；
- 现有 `xy_pos`：由“回到出生点”改为“接近当前落脚目标”；
- 现有 `xy_vel`：不能继续只惩罚所有水平速度，应改为参考速度误差或落地阶段速度惩罚。

### 8.2 起飞事件奖励

在 `liftoff_event`：

\[
r_{launch}=w_{vxy}\exp\left(-k_{vxy}\|\mathbf v_{xy}-\mathbf v_{xy}^{ref}\|^2\right)
+w_{vz}\exp\left(-k_{vz}(v_z-v_z^{ref})^2\right)
\]

这是从二维 `Quadhopper/reward.py` 迁移的主要训练信号。

### 8.3 朝目标推进奖励

使用势函数差，而不是直接奖励绝对距离：

\[
r_{progress}=w_p(d_{t-1}-d_t)
\]

其中 \(d_t=\|\mathbf p_{target}-\mathbf p_{xy,t}\|\)。势函数差可减少“停在目标附近持续刷奖励”的问题。

### 8.4 落地与命中奖励

在 touchdown：

\[
r_{landing}=w_l\exp(-k_l\|\mathbf p_{foot}-\mathbf p_{target}\|^2)
\]

同时加入：

- 低倾角奖励；
- 低角速度奖励；
- 适度的落地竖直速度奖励/惩罚；
- 命中目标的一次性 bonus；
- 连续命中 streak bonus，但保持有上限。

### 8.5 圆轨迹奖励

径向误差只作为弱 shaping：

\[
r_{path}=w_r\exp(-k_re_r^2)
\]

不要把径向奖励设得比落点和起飞奖励更大，否则策略可能在圆周附近原地跳，而不沿圆前进。

### 8.6 建议的初始权重比例

以下是相对量级，不是最终超参数：

| Reward term | 建议相对权重 |
|---|---:|
| target hit event | +50 |
| touchdown proximity event | +15 |
| liftoff XY velocity match event | +10 |
| liftoff Z velocity match event | +8 |
| upright dense | +/−4 |
| progress potential | +3 |
| radial path shaping | +1 |
| survival dense | +0.2 |
| action rate | −0.2 |
| action smoothness | −0.1 |
| crash / flip / fly-away | −30 至 −60 |

每次修改权重后必须查看各项 episode sum，避免某一 dense term 的累计量压过事件奖励。

---

## 9. Episode 终止条件

终止条件建议包括：

- roll 或 pitch 超过安全倾角；
- 机体非预期部位长时间触地；
- 高度超过 `max_hop_height`；
- 距离圆形工作区过远；
- 连续若干秒没有发生有效 liftoff；
- 错过目标且距离继续增加；
- 完成规定的目标数量或一整圈。

不要因为一次落点误差略大就立刻结束；否则策略难以学会恢复。严重摔倒和明显飞走才立即终止。

建议 episode 目标先设为 4–8 跳，而不是一开始要求完整一圈。随着成功率提高逐步延长。

---

## 10. Curriculum

按以下阶段训练，不要直接从原地策略跳到随机圆形路径：

### Stage 0：复现原地跳跃

- 加载或复现现有稳定原地跳跃 checkpoint；
- 确认接触事件、apex 和跳跃周期检测正确；
- 记录平均跳高、周期、倾角和落点漂移。

### Stage 1：单个固定方向小步跳

- 目标固定在机体前方 0.10–0.15 m；
- episode 只要求命中一个目标；
- 从原地策略 checkpoint fine-tune；
- 先关闭或缩小 domain randomization。

### Stage 2：随机方向单跳

- 目标方位随机 \([0,2\pi)\)；
- 距离从 0.10 m 逐渐扩到可靠范围；
- observation 必须使用机体系目标误差；
- 验证策略是否具有旋转等变性，而非记住世界轴。

### Stage 3：直线连续多跳

- 2 跳 → 4 跳 → 8 跳；
- 命中后更新下一个落点；
- 加入连续成功率和恢复能力指标。

### Stage 4：大半径圆弧

- 半径 3 m，较小 `d_hop`；
- 只训练 1/4 圈；
- 固定逆时针方向。

### Stage 5：完整圆形路径

- 半径降至 2 m；
- 训练半圈，再扩到一圈；
- 同时训练顺时针与逆时针。

### Stage 6：随机路径参数与 sim-to-real

- 随机圆心、半径、初始相位和方向；
- 逐步恢复并扩大质量、惯量、推力、延迟和接触参数随机化；
- 加入 Vicon 噪声、状态延迟、外力扰动与地面摩擦变化。

课程升级条件建议使用最近评估窗口的成功率：

```text
单跳成功率 > 85% 才增加距离
连续 4 跳成功率 > 75% 才增加目标数
圆弧成功率 > 70% 才缩小半径或增加随机化
```

训练 curriculum 的评估环境应关闭随机探索噪声，并使用固定种子集合。

---

## 11. Feasibility Sweep

在投入 PPO 前，增加一个与 `Hopper_simu/Quadhopper/feasibility.py` 类似的 Isaac 扫描脚本。至少扫描：

- 目标距离：0.05–0.50 m；
- 目标方位：0°、45°、90°、135°；
- 起跳总推力脉冲；
- roll/pitch 差动推力；
- 脉冲持续时间；
- 不同 restitution、质量和推力倍率。

记录：

```text
liftoff time
flight time
apex height
touchdown position
landing error
maximum tilt
body collision
energy / integrated motor command
```

输出每个目标距离的最优可行控制和成功范围。课程初始跳距应选在 nominal dynamics 下有明显裕量的区域，而不是理论最大距离。

---

## 12. PPO 训练策略

优先从现有原地跳跃 checkpoint 继续训练，不要从零开始。

建议：

- 第一阶段使用 1024–4096 个并行环境（按显存调整）；
- actor/critic MLP 可先使用 `[256, 256, 128]`；
- 保持 observation normalization；
- 初始 exploration noise 不宜过低，因为定向跳需要打破四电机对称动作；
- 逐步降低 entropy，而不是一开始使用很小的探索；
- 新增 observation 后旧 checkpoint 的第一层维度不匹配时，编写权重迁移：复制已有状态和动作历史相关权重，新输入列初始化为零或小随机值；
- 如果迁移成本过高，可保留 26 维布局，优先把原 `xy_err_b` 替换成 `target_err_b`，再利用暂未使用或低价值维度逐步加入参考量。

检查现有两份 PPO 配置：

```text
Quadhopper_Isaac/rsl_rl_ppo_cfg.py
Quadhopper_Isaac/agents/rsl_rl_ppo_cfg.py
```

避免只修改其中一份但训练入口加载另一份。启动训练时打印实际使用的 config class、observation dimension 和 checkpoint path。

---

## 13. 日志与验收指标

TensorBoard 至少记录：

### 事件率

- `liftoff_rate`；
- `touchdown_rate`；
- `target_hit_rate`；
- `consecutive_hops_mean`；
- `full_arc_success_rate`；
- `full_circle_success_rate`。

### 误差

- liftoff XY velocity error；
- liftoff Z velocity error；
- touchdown position error；
- radial path RMSE；
- maximum/mean tilt；
- touchdown vertical speed。

### 控制与安全

- mean/max motor command；
- action first difference；
- action second difference；
- body collision count；
- fly-away termination rate；
- mean episode length。

阶段性验收：

| 阶段 | 最低标准 |
|---|---|
| 单方向单跳 | 100 个固定种子中成功率 ≥ 90% |
| 随机方向单跳 | 成功率 ≥ 80%，各象限差异 < 10% |
| 连续 4 跳 | episode 成功率 ≥ 75% |
| 1/4 圆弧 | 成功率 ≥ 70%，径向 RMSE < 0.15 m |
| 完整圆 | nominal 成功率 ≥ 70%，随机化后 ≥ 50% |

必须保存成功和失败 rollout 的视频、XY 轨迹图、Z/速度/姿态曲线以及四电机命令。只看总 reward 不能判断策略是否真的沿圆跳跃。

---

## 14. 建议的代码拆分

避免把所有功能继续堆进 `my_hopper_env.py`。建议逐步形成：

```text
Quadhopper_Isaac/
├── my_hopper_env.py          # DirectRLEnv 生命周期、张量状态、reward 汇总
├── trajectory_commands.py   # 圆路径、目标推进、坐标变换
├── ballistic_planner.py      # 单跳参考速度与可行性检查
├── hop_events.py             # contact/liftoff/apex/touchdown 状态机
├── rewards.py                # 各 reward term，纯 Torch 张量函数
├── feasibility_sweep.py      # 开环/启发式单跳扫描
└── agents/
    └── rsl_rl_ppo_cfg.py
```

这些模块必须批量处理 `[num_envs, ...]` Torch tensor，禁止在每个 policy step 中对环境逐个执行 Python 循环。

---

## 15. 第一轮最小实现清单

训练电脑上的第一轮实现只完成以下内容：

1. 为 Hopper 增加可靠的 contact、liftoff、apex、touchdown 检测；
2. 将原固定 `_desired_pos_w` 改为每个环境独立的可更新 `target_pos_w`；
3. 先实现“机体前方 0.12 m 的单个目标”，不要立即上圆；
4. observation 中的 `xy_err_b` 改成当前目标误差；
5. 将 `xy_pos` 改成目标进度和 touchdown 落点奖励；
6. 将 `xy_vel` 改为起飞参考速度匹配，避免惩罚所有水平运动；
7. 从原地跳跃 checkpoint fine-tune；
8. 完成随机方向单跳后，实现圆形 waypoint generator；
9. 先训练 4 个连续圆弧目标，再扩展到完整圆；
10. 最后恢复完整 domain randomization。

最小成功定义：机器人能从地面连续完成 4 次跳跃，每次落点沿半径 2 m 的圆弧推进约 0.20 m，落点误差均小于 0.18 m，且没有机身触地或飞走。

---

## 16. 常见失败模式

### 原地刷跳跃奖励

原因：`hop_velocity`、`ground_bonus` 或径向奖励过强，前进奖励不足。

处理：降低原地奖励；增加 liftoff 水平速度匹配和 target-hit 事件奖励；目标未推进时不累计路径完成度。

### 飞行而不是跳跃

原因：持续总推力的收益大于回到地面的收益。

处理：保留最大高度终止；限制飞行相持续高推力；检查事件奖励是否只在真实 liftoff/touchdown 触发。

### 沿圆切线飞出

原因：只奖励切向速度，没有下一落点或径向约束。

处理：以目标点误差为主要任务量，切向速度只作为辅助；touchdown 必须靠近离散 waypoint。

### 在圆周附近不前进

原因：径向奖励可被持续刷取。

处理：减小 `r_path`；使用 progress potential；只有命中下一个 waypoint 才增加完成进度。

### 四个方向表现不同

原因：策略使用世界系目标误差，或训练目标方向不均匀。

处理：将目标、切向和速度参考旋转到机体系；均匀采样方位；分别统计各象限成功率。

### touchdown 检测重复触发

原因：restitution 引起接触抖动。

处理：接触力阈值、迟滞、最小 flight/stance 时间和 2–3 步 debounce 同时使用。

### Reward 上升但落点没有改善

原因：dense reward 的累计量压过 touchdown event。

处理：记录每项 reward 的 episode sum；用无探索评估的落点误差和命中率选择 checkpoint，而不是只按 mean reward。

---

## 17. 后续扩展

完成圆形离散落脚点后，再按以下顺序扩展：

1. 任意二维 waypoint 曲线（S 形、8 字形、样条曲线）；
2. 速度可调的轨迹命令；
3. 在线轨迹优化器输出每跳落点与起飞速度；
4. residual RL 修正模型误差；
5. 在圆形轨迹上加入圆环或障碍物；
6. 使用 Vicon 提供实机位置与路径误差；
7. ONNX 导出与 Crazyflie 控制频率、延迟和动作限幅对齐。

在加入圆环以前，必须先证明策略能够稳定完成随机方向单跳、连续多跳和完整圆形 waypoint 跟踪；否则圆环碰撞会把任务失败原因混合在一起，难以诊断。
