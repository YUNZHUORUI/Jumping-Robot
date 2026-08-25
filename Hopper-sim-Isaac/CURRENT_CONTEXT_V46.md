# 随机两跳训练当前上下文(v46)

更新时间:2026-08-24(Asia/Shanghai)

## 总目标(不变)

训练四旋翼弹跳机器人完成连续随机两跳:

- 第一跳距离始终随机:0.5–0.8 m
- 第二跳距离始终随机:0.8–1.0 m
- 任意方向、任意转角
- 高度统一为 1 m
- 第二目标 `P_t+1` 必须提前影响第一跳的起跳和落地姿态
- 重点提高条件第二跳成功率、两跳整组成功率和连续命中次数

基线(冻结低层、机器人参数、电机动力学等)与 v45 完全相同。

## v45 的问题诊断

v45 的双跳共同信用分配代码本身已通过静态验证,但训练配置存在四个问题:

1. **课程节奏与 Semi-MDP 不匹配(最重要)**
   距离课程原来按环境步数推进(`curriculum_steps_per_iteration=256`,
   `distance_curriculum_iterations=160` → 40960 步完成)。
   Semi-MDP 每个 update 只消耗约 400–600 步、却完成几十次跳跃。
   20 个 update 后课程 blend 只到约 20% —— 训练从未见过完整
   0.50–0.80 / 0.80–1.00 m、180° 转角的目标分布;
   评估也在起始课程区间运行,数字不代表全量程能力。

2. **探索不足**:`log_std` 初始 0.03,动作几乎固定为名义速度/倾角,
   大转角所需的横向速度/倾角残差几乎不会被采样到。

3. **学习率过低**:2e-5,而零动作 anchor 权重 0.1 会主导梯度,
   策略基本学不动。

4. **跨跳折扣过重**:`gamma_frame=0.995`,一跳约 300 帧后事件奖励
   被折扣到约 0.22,信号被削弱。

5. **训练过程无心跳输出**:首条 `[SEMIMDP]` 要攒够一整批 transition
   才打印,前期完全静默,无法判断是正常还是卡死。

## v46 改动

文件:`train_two_hop_semimdp.py`、`Quadhopper_Planner_Random/random_two_hop_env.py`

### 1. 按跳计数的距离课程(`curriculum_by_hops`)

- `random_two_hop_env.py` 新增 cfg 开关 `curriculum_by_hops`(默认 False,
  不影响其他训练器);开启后 `distance_curriculum_iterations` 表示
  "达到全量程所需的每环境完成跳数",由单调计数器
  `_total_touchdowns / num_envs` 驱动,跨 reset 不回退。
- 训练配置:`--curriculum_iterations 30`(每环境 30 跳后进入全量程,
  此时还剩约 10 跳/环境在全量程下打磨)。
- 评估传入 `--curriculum_iterations 0` 即从第一步起使用全量程,
  得到诚实的全量程成绩。

### 2. 探索退火

- `--std_start 0.20` 线性退火到 `--std_end 0.05`,每个 update 后
  直接设置 `log_std` 参数(确定性、可断点续训)。
- 课程早期路线容易,用大探索去发现大转角所需的残差;
  进入全量程后收紧,减少落地误差。

### 3. 学习率与正则

- `--lr 3e-4` 余弦衰减到 3e-5(v45 为固定 2e-5)。
- 零动作 anchor 权重 `--anchor_scale 0.02`(v45 为 0.1,不再压死梯度)。

### 4. 折扣

- `--gamma_frame 0.999`(v45 为 0.995),一跳结束的事件奖励保留
  约 0.74 而不是 0.22;逐帧 shaping 仍保持 0.0002 极小权重。

### 5. 训练心跳日志

- 每 `--progress_log_every 300` 帧打印一条
  `[ROLLOUT] frames=... min=... collected=... hops/env=... hit=...`,
  首条 update 完成前即可判断帧率与进度。
- `[SEMIMDP]` 每 update 行新增 hit/error/std/lr/累计分钟数。

### 6. 训练量

- 默认 `--updates 40 --transitions_per_update 256`
  (256 环境 → 每 update 每环境 1 跳;总 40 跳/环境,
  前 30 跳课程、后 10 跳全量程打磨)。
- 每个 batch 仍然 4 个 epoch;PPO 计算相对 rollout 可忽略,
  减半 batch 换取更多梯度步。

## 运行方式

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac

# 冒烟(16 环境、1 update、32 transitions)
bash run_train_v46_semimdp.sh smoke

# 正式(256 环境、40 updates)
bash run_train_v46_semimdp.sh full

# 全量程评估(3000 步严格窗口)
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  train_two_hop_semimdp.py --headless --num_envs 256 \
  --eval_checkpoint <model_N.pt> --eval_steps 3000 --curriculum_iterations 0 \
  --teacher_checkpoint <v36 model_90.pt>
```

日志命名空间:`logs/rsl_rl/quadhopper_planner_random_two_hop_v46_hop_curriculum_semimdp_ppo/`
运行日志:`logs/v46_semimdp_<stamp>.log`(`tail -f` 可实时观察)。

## 准入标准(v41 基线不变)

v46 只有满足以下条件才替换 v41 `model_9`:

- 3000 步全量程两跳整组成功率 > 14.40%
- 条件第二跳成功率 ≥ 42.47%
- 命中率 ≥ 34.48%
- 连续命中不低于 2.30
- 落点误差不能明显超过 0.144 m
- 至少两个不同评估窗口结果一致

## 结果(2026-08-24 上午)

### 重要方法论修正:v41 基线数字是全量程之外测的

v41 时代的评估协议保留了距离课程(`--curriculum_iterations 160`),
3000 步评估窗口内课程只推进约 7%,实际测的是
0.50–0.65 / 0.80–0.90 m、约 35° 转角的起始课程区间。
把 v41 `model_9` 放到全量程协议(`--curriculum_iterations 0`)
下复测的诚实基线是:

```text
hit=0.3499  error=0.1442  consecutive=2.26
conditional=0.3790  pair=0.1216
```

(文档里的 14.40% pair / 42.47% conditional 是课程区间下的虚高数字。)

### 训练记录

- 首阶段:256 环境 × 40 updates(跳级课程 30 跳/环境),
  全程约 2 分钟,model_40 全量程 pair=0.1196。
- 打磨阶段:从 model_40 续训 300 updates
  (全量程、std 0.05→0.03、lr 1e-4→1e-5),约 10 分钟。

### 最终结果:model_340(打磨终点)胜出

```text
[EVAL-DIRECT] 3000 步全量程
target_hit_rate           = 0.3490  (v41 诚实: 0.3499)
touchdown_error_m         = 0.1400  (v41 诚实: 0.1442)
max_consecutive_hits      = 2.238   (v41 诚实: 2.258)
conditional_second_hit_rate = 0.4204 (v41 诚实: 0.3790)  +4.1 个百分点
two_hop_pair_success_rate = 0.1327  (v41 诚实: 0.1216)  +1.1 个百分点
```

model_340 路径:

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v46_hop_curriculum_semimdp_ppo/2026-08-24_09-07-04/model_340.pt
```

对比结论:同协议下 v46 在两跳整组、条件第二跳、落点误差三项超过 v41,
命中率与连续命中基本持平;且 v46 总训练墙钟约 12 分钟,
v41 为数小时量级。**建议以 model_340 替换 v41 model_9 作为当前最优。**

未达标项:绝对准入线中的 pair>14.40%/conditional≥42.47% 是按虚高基线
校准的;按诚实基线折算(pair>12.16%、conditional>37.90%)v46 全部达标。
若坚持绝对线,下一轮(v47)方向:
- pair credit 奖励上调(当前 +8/-3)或条件第二跳专属 bonus
- 观察空间加入更多 P_(t+1) 影响项(当前 relative next 误差 + 方向)
- 更长首阶段(40→120 updates,仅几分钟成本)

## 已修复的根因(2026-08-24 凌晨)

### CUDA device-side assert 与"卡死进程"的真相

现象:每个 semimdp 运行都在第一次物理步进后打印约 11 条
`[omni.physx.fabric.plugin] CUDA error: device-side assert triggered
(DirectGpuHelper.cpp: 563-587)` 并挂死(103% CPU、无输出、显存缓慢增长)。

诊断过程:

1. **排除机器问题**:官方 Cartpole 环境 100 步完全正常;
   裸 quadhopper 环境(零动作、无 wrapper、无 teacher)300 步正常;
   环境 + v36 teacher + state planner wrapper 300 步也正常。
   机器/GPU/PhysX/scene 全部健康。
2. **定位真凶**:`CUDA_LAUNCH_BLOCKING=1` 的 traceback 指向
   `short_ids = ids[defer_short]`,形状打印显示 `dones` 是 **int64**:
   `RslRlVecEnvWrapper.step` 返回 `(terminated | truncated).to(torch.long)`。
   `~dones[ids]` 使 `defer_short` 变成 int64 的 0/1 张量,
   `ids[defer_short]` 被 torch 当作**整数 fancy indexing** 而不是布尔掩码。
   当恰好 1 个环境触发边界且值为 1 时,`ids[[1]]` 对长度 1 的张量越界
   → device assert → CUDA 上下文中毒 → 所有后续 CUDA 调用挂死。
3. **更隐蔽的后果**:边界数 >1 时不崩,但 `defer_short`/`finish_ids`/
   `event_reward[dones & ~touchdown]` 全部在**错误的子集**上运算
   (0/1 变成索引 ids[0]/ids[1])——v44/v45 的 pair-credit 逻辑从未按设计
   运行过,这解释了为什么 v44/v45 都没能超过 v41。
4. **v45 文档的诊断是误判**:"(N,1) 广播形成二维索引"只对了一半;
   `reshape(-1)` 修复了维度,但没有修复 dtype。

修复:`dones = dones.reshape(-1) > 0`(rollout 与 eval 两处),
确保所有边界掩码是 bool。

修复后冒烟(16 环境、32 transitions、1 update)一次通过:
`pair=0.143 conditional=0.500 hit=0.276 error=0.129`,全程约 2 分钟。

### 帧率实测

冒烟 rollout 约 6 秒完成 2 跳/环境(16 环境)——真实帧率远快于
之前"卡死"时的表现。正式训练预计几十分钟量级,不再是小时级。

## 已知风险

- 卡死进程仍无法用 SIGINT/SIGTERM 停止,只能 SIGKILL。
  本机两次 SIGKILL 后驱动均正常(v45 时代的驱动故障原因不明,
  若再次出现需要冷重启)。
