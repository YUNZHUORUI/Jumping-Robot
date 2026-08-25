# 随机两跳训练当前上下文（v45）

更新时间：2026-08-24（Asia/Shanghai）

## 总目标

训练四旋翼弹跳机器人完成连续随机两跳：

- 第一跳距离始终随机：0.5–0.8 m
- 第二跳距离始终随机：0.8–1.0 m
- 任意方向、任意转角
- 高度统一为 1 m
- 不牺牲距离随机化和泛化能力
- 第二目标 `P_t+1` 必须提前影响第一跳的起跳和落地姿态
- 重点提高条件第二跳成功率、两跳整组成功率和连续命中次数

## 保持不变的部分

以下基线没有修改：

- 机器人 USD、质量、惯量和弹簧参数
- 四电机动力学、延迟、时间常数和推力曲线
- 冻结的 v36 低层策略
- 电机顺序与坐标系约定
- 每跳距离随机化

冻结低层检查点：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v36_full_relative_next_smooth_distance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_fixed_100/2026-08-23_12-24-40/model_90.pt
```

## 已完成的策略迭代

### v40：逐帧高层状态规划器

加入当前和下一目标、预计落点、落地速度、落地姿态和下一跳方向预备控制。

结果：

- 命中率：32.18%
- 条件第二跳：42.02%
- 两跳整组：13.12%
- 落点误差：0.1398 m

问题：逐帧 PPO 动作容易偏离解析先验并发生退化。

### v41：每跳锁存一次

高层只在每跳开始时决策一次。

当前严格 3000 步评估下的实际最佳检查点：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v41_hop_latched_bc_ppo_curriculum/2026-08-23_22-14-04/model_9.pt
```

结果：

- 命中率：34.48%
- 条件第二跳：42.47%
- 两跳整组：14.40%
- 平均最长连续命中：2.30
- 落点误差：0.1442 m

问题：标准 RSL-RL PPO 仍记录大量没有执行的飞行动作，动作与回报不匹配。

### v42：支撑期因果 EMA

支撑期所有动作都会更新计划，起飞后冻结，从而减少未执行动作。

结果没有超过 v41：

- model_9 条件第二跳：41.56%
- 两跳整组：12.47%
- 落点误差：0.1399 m

因此没有替换 v41。

### v43：按跳 Semi-MDP PPO

新增独立训练器：

```text
train_two_hop_semimdp.py
```

结构：

```text
落地/重置
  → 采样一次高层动作
  → 整跳保持
  → 累计整跳折扣回报
  → 下一次落地
  → 写入一条 PPO transition
```

解决了飞行期伪动作问题，但最初仍累计全部低层逐帧 shaping，导致高层“两跳整组成功”奖励被淹没。

### v44：事件级 Semi-MDP 奖励

高层回报改为：

- 单跳命中/失败
- 连续落点误差奖励
- 第一跳成功后第二跳成功的大额奖励
- 极小比例的低层 shaping

短窗口 model_10 曾达到：

- 2000 步整组成功率：15.43%
- 条件第二跳：43.26%

但严格 3000 步复核只有：

- 命中率：33.64%
- 条件第二跳：38.72%
- 两跳整组：13.36%
- 落点误差：0.1412 m

没有超过 v41，未接受。

### v45：双跳共同信用分配

发现 v44 的关键问题：第二跳整组奖励只回传给第二跳动作，没有回传给决定第一跳落地姿态的第一跳动作。

v45 已修改为：

```text
第一跳 transition 暂存
        ↓
等待第二跳完成
        ↓
计算两跳整组结果
        ↓
将整组成功/失败奖励同时分配给第一跳和第二跳动作
```

这对应“后一个点应提前影响前一跳起跳和落地姿态”的要求。

同时修复：

- `dones` 实际形状为 `(N, 1)`
- 与 `(N,)` 事件掩码广播后形成二维索引
- 最终导致 CUDA index out of bounds

修复方式：所有 Semi-MDP 边界计算前将 `dones` 展平为 `(N,)`。

## 目前卡住的位置

代码已经修复，但尚未完成 v45 冒烟验证。

连续 CUDA device assert 后留下两个卡死的 Isaac 进程，各占约 2.32 GB 显存。强制清理后 NVIDIA 驱动进入故障状态：

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

当前阻塞点：GPU 驱动失效，需要重启电脑；不是代码编译或单元测试失败。

当前静态验证：

- Python 编译通过
- `git diff --check` 通过
- 单元测试 5/5 通过

## 重启后的第一步

先检查：

```bash
nvidia-smi
```

正常后运行 v45 冒烟：

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac

/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
train_two_hop_semimdp.py \
--headless \
--num_envs 16 \
--updates 1 \
--transitions_per_update 32 \
--curriculum_iterations 160 \
--teacher_checkpoint \
logs/rsl_rl/quadhopper_planner_random_two_hop_v36_full_relative_next_smooth_distance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_fixed_100/2026-08-23_12-24-40/model_90.pt
```

冒烟通过后运行正式训练：

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
train_two_hop_semimdp.py \
--headless \
--num_envs 256 \
--updates 20 \
--transitions_per_update 512 \
--curriculum_iterations 160 \
--teacher_checkpoint \
logs/rsl_rl/quadhopper_planner_random_two_hop_v36_full_relative_next_smooth_distance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_fixed_100/2026-08-23_12-24-40/model_90.pt
```

新日志命名空间：

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v45_pair_credit_semimdp_ppo/
```

## 下一检查点准入标准

v45 只有满足以下条件才替换 v41：

- 3000 步两跳整组成功率 > 14.40%
- 条件第二跳成功率 ≥ 42.47%
- 命中率 ≥ 34.48%
- 连续命中不低于 2.30
- 落点误差不能明显超过 0.144 m
- 至少两个不同评估窗口结果一致

如果 v45 仍然退化，保留 v41 `model_9`，不继续扩大课程。
# 2026-08-24 v46/v47 continuation

- GPU recovered after reboot and all Isaac Sim GPU evaluations completed normally.
- v46 full-range long training regressed. Best screened candidate was `model_300.pt`, but its strict 3000-step pair success was only `0.128342`.
- Added conservative v47 defaults in `train_two_hop_semimdp.py`: learning rate `2e-5`, action std `0.03 -> 0.01`, zero-residual anchor `0.05`; random distance ranges and full `180 deg` turns remain unchanged.
- v47 continued from v46 model 300 for 20 updates. Output directory:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v47_conservative_pair_credit_semimdp_ppo/2026-08-24_10-34-23`
- v47 `model_320.pt` strict 3000-step evaluation (1030 touchdowns):
  - target hit rate: `0.379612`
  - touchdown error: `0.138196 m`
  - mean max consecutive hits: `2.437500`
  - conditional second-hop hit rate: `0.422131`
  - two-hop pair success: `0.136605`
- Decision: v47 model 320 improves single-hop hit rate, error, and maximum streak over v41, but does **not** replace v41 as the continuous two-hop best because pair success is below `0.1440` and conditional second-hop success is slightly below `0.4247`.
- Tests after the change: `5 passed`; static compilation and diff whitespace checks passed.

## v48/v49 second-hop and pair-objective work

- Added an explicit Markov pair context to the state-planner observation. The
  observation is now 69-D and appends `[short_phase, long_phase,
  first_hop_hit]`; old 66-D checkpoints migrate by copying all old input
  weights and zero-initializing the three new columns.
- Added eligibility-aware delayed pair credit to both actions. A second-hop
  action receives pair credit only when hop one hit, and eligible second-hop
  samples receive configurable PPO weighting.
- Added a zero-output second-hop adapter and `--second_hop_adapter_only` mode.
  In this mode the migrated shared actor is frozen, the first-hop action stays
  exactly unchanged, and only long-hop actions receive the learned adapter.
- Fixed checkpoint migration to reset incompatible 66-D Adam moments while
  preserving model weights.
- v48 full shared-actor run:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v48_markov_pair_objective_semimdp_ppo/2026-08-24_10-45-47`
  Its early short-eval peak did not survive a 3000-step test (`model_321` pair
  `0.126162`, conditional `0.387755`).
- v49 frozen-shared-actor adapter run:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v49_second_hop_adapter_semimdp_ppo/2026-08-24_10-58-40`
  Strict 3000-step `model_321` result (1024 touchdowns): hit `0.345703`, error
  `0.137530 m`, max consecutive `2.156250`, conditional second hit
  `0.414847`, pair success `0.126330`.
- Decision: neither v48 nor v49 replaces v41. Structural fixes are retained,
  but the next method should use success/failure-balanced pair data and
  supervised advantage/action distillation instead of more plain on-policy
  PPO updates.

## v50 decomposed touchdown objective and batched action search

- Added `Quadhopper_Planner_Random/pair_objective.py`, decomposing touchdown
  quality into position, next-hop velocity, attitude/angular velocity, and
  spring position/velocity components.
- Added `search_second_hop_actions.py`. Evaluation-mode environments are
  grouped into cloned relative routes with an identical first-hop context;
  each group fans out over different second-hop actions and records the full
  69-D state, action, decomposed costs, total score, hit, context ID, and
  candidate ID.
- Added `build_second_hop_pairs.py`, which ranks candidates within the same
  context (hit first, dense score second) and emits preferred/rejected action
  pairs.
- Formal grouped dataset:
  `outputs/second_hop_search/v50_candidates_grouped_5120.pt`
  contains 5118 completed candidates from 640 contexts (8 candidates per
  context; only two timeouts).
- Preference dataset:
  `outputs/second_hop_search/v50_preference_pairs.pt`
  contains 640 explicit preferred/rejected pairs.
- Baseline candidate hit rate in these controlled second-hop contexts was
  `0.526563`; best-of-8 hit coverage was `0.915625`. This is strong evidence
  that action selection, rather than physical controllability, is the main
  second-hop bottleneck.
- The exact-touchdown spring objective is poorly calibrated (mean normalized
  spring cost about 43). Prefer an empirical successful-touchdown spring
  target or a short post-touchdown settling window before using this component
  as a regression target. Pair ranking remains hit-first, so this did not
  corrupt the generated preferred/rejected labels.
- Tests: 6 passed; compile and diff whitespace checks passed.
- Direct preferred-action MSE distillation was rejected: held-out preference
  accuracy was only `0.4609`, consistent with a multimodal best-action target
  whose average need not be successful. The rejected artifact is stored at
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v50_preference_distilled/model_distilled.pt`.
- Added a learned state-action scorer (`second_hop_scorer.py` and
  `train_second_hop_scorer.py`) instead. Context-held-out top-1 candidate hit
  rate peaked at `0.632812`, versus `0.526563` for the baseline candidate.
  Best scorer checkpoint:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v50_action_scorer/scorer_best.pt`.
- Next required step: integrate the scorer into second-hop inference, sample a
  small deterministic candidate set around the planner action, choose the
  highest-scoring action, and run the standard 3000-step full-range pair eval.
- Online best-of-8 selector was integrated into `train_two_hop_semimdp.py` via
  `--action_scorer`. It only changes long-hop actions; first-hop actions remain
  the planner output. The scorer checkpoint now stores the exact candidate
  offsets recovered from the collection dataset.
- Dense-quality ranking (`hit_logit + 0.1 * quality`) was inferior because the
  exact-touchdown spring target was miscalibrated. The accepted selector uses
  hit logit only (`--scorer_quality_weight 0`, now the default).
- Accepted strict 3000-step full-range evaluation for v49 model 321 + v50
  hit-only scorer (1168 touchdowns):
  - target hit rate: `0.370719`
  - touchdown error: `0.137038 m`
  - mean max consecutive hits: `2.437500`
  - conditional second-hop hit rate: `0.433594`
  - two-hop pair success: `0.146438`
- This combination narrowly but consistently passes every v41 gate (v41:
  hit `0.3448`, error `0.1442 m`, max streak `2.30`, conditional `0.4247`,
  pair `0.1440`). It is the new accepted full-range candidate.
- Planner:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v49_second_hop_adapter_semimdp_ppo/2026-08-24_10-58-40/model_321.pt`
- Scorer:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v50_action_scorer/scorer_best_with_offsets.pt`

## v51 first-hop pair selector attempt

- Pair decomposition of the accepted v50 strict result shows the main cap:
  estimated first-hop hit rate is `0.146438 / 0.433594 = 0.3377`. About two
  thirds of pairs therefore fail before the second-hop selector can help.
- Extended `search_second_hop_actions.py` with `--search_phase first`. It fans
  out first-hop actions from cloned initial contexts, then executes the
  accepted v50 hit-only selector on hop two, recording first hit, second hit,
  and final pair hit.
- Dataset `outputs/first_hop_search/v51_first_candidates_grouped_5120.pt`
  contains all 5120 candidates from 640 contexts. Controlled baseline pair
  rate was `0.21875`, best-of-8 oracle pair coverage `0.426562`, and best-of-8
  first-hit coverage `0.4625`.
- First-hop pair scorer checkpoint:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v51_first_pair_scorer/scorer_best_with_offsets.pt`.
  Context-held-out top-1 pair rate peaked at `0.257812`.
- Online 1000-step dual-selector test failed: hit `0.334135`, conditional
  `0.400000`, pair `0.116667`, error `0.144749 m`. Reject this scorer and keep
  the accepted v50 second-hop-only hit selector.
- Diagnosis: the 73-D first-hop state-action scorer overfit 640 contexts and a
  single fixed candidate-offset set. Future first-hop data must span multiple
  route seeds, candidate-offset sets, and substantially more contexts before
  retraining.

## v52 dynamic first-hop scorer and candidate-count ablation

- Collected `outputs/first_hop_search/v52_dynamic_multiseed_candidates_15360.pt`:
  15358 completed candidates, 1920 contexts, with candidate offsets and route
  seeds resampled between trials. Controlled baseline pair rate was `0.195414`
  and best-of-8 oracle coverage was `0.424479`.
- A 32-candidate deployment scorer reached only `0.250000` held-out top-1 pair
  rate and regressed online pair success to `0.136546`; rejected.
- Tested whether maximization over 32 noisy predictions caused the regression.
  Retrained the same scorer with an independently generated 8-candidate bank:
  `logs/rsl_rl/quadhopper_planner_random_two_hop_v52_dynamic_first_pair_scorer/scorer_best_8.pt`.
  Held-out top-1 remained `0.250000` and the learned conservative threshold was
  zero.
- Online 1000-step full-range dual-selector result for the 8-candidate scorer:
  hit `0.372159`, touchdown error `0.137924 m`, max consecutive hits `1.648438`,
  conditional second hit `0.445183`, pair success `0.137718` (1760 touchdowns).
  This also regresses pair success and continuity, so it is rejected without a
  3000-step evaluation.
- Conclusion: candidate count is not the primary failure. The first-hop
  one-step state-action scorer cannot reliably estimate delayed pair success;
  offline oracle coverage (`42.45%`) is high while learned selection (`25%`)
  is weak. Keep v49 + v50 second-hop-only selector as the accepted policy.
- Next method should replace point-estimate first-hop ranking with either an
  ensemble lower-confidence-bound selector plus abstention, or train a
  sequence/world-model critic using the intermediate touchdown state. Do not
  spend another round merely changing the number or scale of static offsets.

## v53 environment-state and hop-command audit

- Found and fixed a duplicate `_reset_idx` definition in
  `random_two_hop_env.py`. The later definition had overwritten pair-state
  cleanup, allowing `_first_hop_hit_for_pair` and pair counters to survive an
  environment reset and contaminate the Markov state, delayed credit, and
  reported metrics.
- Found cross-hop command contamination in `two_hop_state_planner_wrapper.py`:
  a new hop was initialized by blending only 10% of its decoded command into
  90% of the preceding hop command. This is especially harmful for random
  turns up to 180 degrees. A new command is now latched immediately on the
  first stance step after reset/touchdown; later stance updates remain smooth.
- Same-seed 1000-step evaluation of v49 + v50 after reset fix, before immediate
  latching: hit `0.355171`, error `0.138348 m`, conditional second `0.480000`,
  pair `0.144737`.
- Same evaluation after immediate per-hop latching: hit `0.371380`, error
  `0.135731 m`, conditional second `0.439189`, pair `0.149597`. This is a real
  direct improvement in total hit, pair success, and landing error, despite
  using the old checkpoint.
- A low-LR continuation from model 321 was stopped at model 353 because online
  training statistics regressed (pair about `0.08`, error about `0.160 m`). Do
  not promote checkpoints from `2026-08-24_13-39-51`.
- Deeper strategy bottleneck: the planner action only requests touchdown
  velocity/tilt; `_motor_correction` applies it only during descent and clips
  each motor residual to `0.03`. It cannot materially change stance impulse or
  early-flight trajectory, so most first-hop misses are outside its control.
  Further scorer/reward tuning is unlikely to produce a large gain.
- Next structural version should keep the frozen teacher but learn a small
  phase-conditioned motor residual active during stance/ascent/descent, with
  phase-specific limits, zero initialization, teacher anchoring, and a landing
  projection safety gate. This gives the policy authority over takeoff range
  while retaining the stable teacher fallback.
