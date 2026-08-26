# CURRENT_CONTEXT_V57

Date: 2026-08-24

## Route Semantics

Random two-hop now uses a continuous rolling waypoint queue by default:

- execute the current target `P_t`
- observe and prepare for `P_(t+1)`
- every touchdown consumes `P_t`, shifts old `P_(t+1)` into `P_t`, and appends a new future point
- hop radii alternate short/long: `0.50-0.80 m`, then `0.80-1.00 m`

Legacy pair restart behavior remains available with `--pair_restart_queue`.

## Landing-Only Preparation

v57 introduces `--correction_mode landing` in
`experiments/random_two_hop/train_two_hop_semimdp.py`.
This keeps high-level touchdown-state correction out of the high airborne phase
and allows it only near landing / spring-contact stance:

- correction mode: `landing`
- correction height: `0.20 m` above landing root height
- motor correction limit: `0.014`
- velocity feedback gain: `0.020`
- attitude feedback gain: `0.100`

This matches the intended behavior: use thrust around landing/contact to prepare
the body attitude and horizontal velocity for the next hop, without broad
mid-flight trajectory chasing.

## Best Checkpoint

Best strict 3000-step full-range landing-only model so far:

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo/2026-08-24_16-54-47/model_446.pt
```

Evaluation command:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh experiments/random_two_hop/train_two_hop_semimdp.py \
  --correction_mode landing \
  --eval_checkpoint logs/rsl_rl/quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo/2026-08-24_16-54-47/model_446.pt \
  --eval_steps 3000 \
  --num_envs 256 \
  --curriculum_iterations 0 \
  --teacher_checkpoint logs/rsl_rl/quadhopper_planner_random_two_hop_v36_full_relative_next_smooth_distance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_fixed_100/2026-08-23_12-24-40/model_90.pt \
  --prepared_reward 8.0 \
  --touchdown_attitude_penalty 3.0 \
  --touchdown_next_velocity_penalty 2.0 \
  --state_reward_scale 180 \
  --motor_correction_limit 0.014 \
  --velocity_feedback_gain 0.020 \
  --attitude_feedback_gain 0.100 \
  --landing_correction_height 0.20
```

Strict eval results:

```text
touchdown_count                 6340
target_hit_rate                 0.392587
touchdown_error_m               0.131814
max_consecutive_hits            3.125000
conditional_second_hit_rate     0.480111
two_hop_pair_success_rate       0.166186
prepared_landing_rate           0.098580
touchdown_attitude_error_rad    0.151751
touchdown_next_velocity_error   0.629700
```

## Comparison

Previous v57 final:

```text
model_421 pair                  0.160347
```

Fine-tuned candidates:

```text
model_446 pair                  0.166186
model_472 pair                  0.162304
model_501 pair                  0.162864
```

Previous v55 default descent correction:

```text
pair                            0.166021
prepared_landing_rate           0.108457
touchdown_attitude_error_rad    0.149420
```

`model_446` slightly exceeds v55 pair success while keeping the better landing-only
correction semantics.

## Short/Medium Distance Retargeting - 2026-08-25

Goal: improve accuracy and stability on the easier deployment range:

```text
first/short hop radius: 0.20-0.50 m
second/long hop radius: 0.50-0.80 m
```

Code changes:

- `experiments/random_two_hop/train_two_hop_semimdp.py` now accepts `--short_radius_min`,
  `--short_radius_max`, `--long_radius_min`, and `--long_radius_max`.
- `experiments/random_two_hop/play_two_hop_semimdp.py` accepts the same radius flags for matching
  visualization.
- Defaults remain the original `0.50-0.80 / 0.80-1.00` range, so old commands
  are unchanged unless the new flags are passed.

Fine-tune source:

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo/2026-08-24_16-54-47/model_446.pt
```

Training run:

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo/2026-08-25_08-42-38
```

Recommended balanced checkpoint:

```text
saved_checkpoints/v57_short020_050_long050_080_balanced/model_466.pt
sha256 f61170fda04e44c29f404e49bb973b7baeabce60d24ba97d125ba7552653cc4d
```

Evaluation with `--eval_steps 3000 --num_envs 256` on the new radius range:

```text
model_466:
  target_hit_rate                 0.693966
  touchdown_error_m               0.082959
  max_consecutive_hits            6.781250
  conditional_second_hit_rate     0.753976
  two_hop_pair_success_rate       0.484375
  prepared_landing_rate           0.475670
  touchdown_attitude_error_rad    0.113235
  touchdown_next_velocity_error   0.419705

model_476:
  target_hit_rate                 0.689351
  touchdown_error_m               0.083099
  max_consecutive_hits            6.648438
  conditional_second_hit_rate     0.750472
  two_hop_pair_success_rate       0.478065
  prepared_landing_rate           0.459468
  touchdown_attitude_error_rad    0.113234
  touchdown_next_velocity_error   0.422246

model_486:
  target_hit_rate                 0.691513
  touchdown_error_m               0.083857
  max_consecutive_hits            6.730469
  conditional_second_hit_rate     0.770071
  two_hop_pair_success_rate       0.487079
  prepared_landing_rate           0.457267
  touchdown_attitude_error_rad    0.113526
  touchdown_next_velocity_error   0.422637
```

Use `model_466` when prioritizing stable, accurate continuous jumping. Use
`model_486` only if optimizing the reported two-hop pair rate by a very small
margin matters more than landing preparation and mean error.

## Landing Precision Follow-up - 2026-08-25

Added precision-tuning and diagnosis flags to
`experiments/random_two_hop/train_two_hop_semimdp.py`:

- `--target_tolerance`
- `--precision_reward_scale`
- `--precision_sigma`
- `--landing_error_penalty`
- `--action_offset`
- `--offset_grid_search`
- `--offset_grid_span`

Also added `--action_offset` to `experiments/random_two_hop/play_two_hop_semimdp.py`.

Two PPO precision fine-tunes from the balanced checkpoint were tested but should
not be used:

```text
strict 7 cm tolerance:
logs/rsl_rl/quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo/2026-08-25_09-00-27
final training error about 0.105 m

soft 10 cm tolerance with narrower precision reward:
logs/rsl_rl/quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo/2026-08-25_09-02-13
model_467 eval error 0.083714 m, worse than saved model_466 at 0.082959 m
```

Policy-offset diagnosis:

```text
offset grid +/-0.12 found a noisy best at action_offset=-0.12,0,+0.12,0
single eval of -0.12,0,+0.12,0:
  target_hit_rate                 0.693804
  touchdown_error_m               0.082860
  two_hop_pair_success_rate       0.486178
  prepared_landing_rate           0.462287
```

This offset is only a tiny improvement in mean error, with slightly worse
landing preparation than the saved balanced checkpoint. Use it only for visual
A/B testing; the recommended checkpoint is still
`saved_checkpoints/v57_short020_050_long050_080_balanced/model_466.pt`.

## Local Action Search / Selector - 2026-08-25

Implemented the search-and-selector route for precision diagnosis:

- `experiments/random_two_hop/search_second_hop_actions.py` now accepts the short/long radius flags,
  target tolerance, and landing-only correction flags.
- `experiments/random_two_hop/train_second_hop_scorer.py` now has `--selection_quality_weight`.
- `experiments/random_two_hop/play_two_hop_semimdp.py` can use `--action_scorer`,
  `--first_action_scorer`, and `--scorer_quality_weight`.

Collected candidate datasets:

```text
outputs/second_hop_search/v57_short020_050_long050_080_second_candidates.pt
outputs/second_hop_search/v57_short020_050_long050_080_first_candidates.pt
```

Oracle analysis from the collected data:

```text
second-hop candidates:
  base err                         0.075524 m
  oracle min-position err          0.032137 m
  oracle max-score err             0.056485 m

first-hop candidates:
  base err                         0.066251 m
  oracle min-position err          0.053845 m
  oracle max-score pair_hit         0.680990
```

Trained scorers:

```text
second scorer:
logs/rsl_rl/quadhopper_planner_random_two_hop_v57_second_precision_scorer/scorer_quality_w05.pt
heldout_top1_hit                   0.857143

first scorer:
logs/rsl_rl/quadhopper_planner_random_two_hop_v57_first_precision_scorer/scorer_quality_w05.pt
heldout result was weak; do not use by default
```

Selector eval on the saved balanced checkpoint:

```text
second scorer only:
  target_hit_rate                  0.692653
  touchdown_error_m                0.082826
  max_consecutive_hits             7.019531
  two_hop_pair_success_rate        0.483774
  prepared_landing_rate            0.472692

first + second scorers:
  target_hit_rate                  0.681567
  touchdown_error_m                0.083648
  max_consecutive_hits             6.535156
  two_hop_pair_success_rate        0.470694
  prepared_landing_rate            0.463597
```

Use only the second-hop scorer for A/B visualization. It gives a very small
mean-error improvement and a noticeable max-consecutive-hits improvement, but
does not yet unlock the large oracle gap. The first-hop scorer is not reliable
enough with the current small dataset.

## Training Notes

The fine-tune that produced `model_446` started from:

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo/2026-08-24_16-49-44/model_421.pt
```

Key training flags:

```text
--correction_mode landing
--curriculum_iterations 0
--std_start 0.008
--std_end 0.004
--lr 4e-6
--anchor_scale 0.25
--hit_reward 4.0
--miss_penalty -2.5
--pair_success_reward 20.0
--pair_first_miss_penalty -5.0
--pair_second_miss_penalty -10.0
--eligible_second_weight 3.0
--prepared_reward 8.0
--touchdown_attitude_penalty 3.0
--touchdown_next_velocity_penalty 2.0
--state_reward_scale 180
--motor_correction_limit 0.014
--velocity_feedback_gain 0.020
--attitude_feedback_gain 0.100
--landing_correction_height 0.20
```

## 2026-08-25 Precision Follow-Up

Comparable 500-env baseline for the saved short/medium model:

```text
checkpoint:
saved_checkpoints/v57_short020_050_long050_080_balanced/model_466.pt

eval flags:
--eval_steps 3000 --num_envs 500 --curriculum_iterations 0
--target_tolerance 0.10
--short_radius_min 0.2 --short_radius_max 0.5
--long_radius_min 0.5 --long_radius_max 0.8

target_hit_rate                   0.681428
touchdown_error_m                 0.084122
max_consecutive_hits              6.626000
conditional_second_hit_rate       0.758806
two_hop_pair_success_rate         0.477378
prepared_landing_rate             0.458167
short_hit_rate                    0.628929
short_touchdown_error_m           0.090069
long_hit_rate                     0.734380
long_touchdown_error_m            0.078123
```

Static short-hop offset grid did not produce a large gain. The only candidate
that survived a direct 500-env eval was:

```text
--short_action_offset=0.24,0,-0.12,0

target_hit_rate                   0.680934
touchdown_error_m                 0.083896
max_consecutive_hits              6.880000
conditional_second_hit_rate       0.759484
two_hop_pair_success_rate         0.480462
prepared_landing_rate             0.459571
short_hit_rate                    0.632012
short_touchdown_error_m           0.089912
long_hit_rate                     0.730308
long_touchdown_error_m            0.077824
```

This is a tiny but real improvement over baseline under the same evaluation
settings. Use it only as an A/B visualization/deployment option; keep
`model_466.pt` as the main checkpoint.

Two PPO fine-tune attempts from `model_466.pt` were stopped or rejected:

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo/2026-08-25_23-44-34
  dynamics-randomized short-precision PPO; model_467 direct eval worsened:
  target_hit_rate 0.669837, error 0.085164, pair 0.457948

logs/rsl_rl/quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo/2026-08-25_23-52-00
  nominal short-precision PPO with --disable_train_randomization;
  model_467 direct eval worsened:
  target_hit_rate 0.675222, error 0.084820, pair 0.464297
```

Conclusion: do not keep training PPO blindly from `model_466.pt` with the
current reward variants. For now the safest model remains `model_466.pt`; the
smallest deployable precision tweak is the short-hop runtime offset above.

## 2026-08-26 Error Tail Optimization

Added tail metrics and a hinge tail penalty to
`experiments/random_two_hop/train_two_hop_semimdp.py`.

New metrics printed in eval/training:

```text
short_tail10_rate / long_tail10_rate: fraction of touchdowns with error > 0.10 m
short_tail15_rate / long_tail15_rate: fraction of touchdowns with error > 0.15 m
```

New training args:

```text
--tail_error_threshold
--tail_error_penalty
--short_tail_error_penalty
--long_tail_error_penalty
```

Baseline tail eval for `model_466.pt`, 500 envs, 3000 steps:

```text
target_hit_rate                   0.681428
touchdown_error_m                 0.084122
two_hop_pair_success_rate         0.477378
short_hit_rate                    0.628929
short_touchdown_error_m           0.090069
short_tail10_rate                 0.366951
short_tail15_rate                 0.115807
long_hit_rate                     0.734380
long_touchdown_error_m            0.078123
long_tail10_rate                  0.264697
long_tail15_rate                  0.078947
```

Conservative PPO tail fine-tune was attempted from `model_466.pt`:

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v57_landing_prep_continuous_queue_semimdp_ppo/2026-08-26_10-06-43
```

It did not improve direct eval. Rejected candidates:

```text
model_475.pt:
  target_hit_rate                 0.677392
  touchdown_error_m               0.084972
  pair                            0.468452
  short_tail10_rate               0.367755
  short_tail15_rate               0.121263

model_477.pt:
  target_hit_rate                 0.676960
  touchdown_error_m               0.084687
  pair                            0.464758
  short_tail10_rate               0.371529
  short_tail15_rate               0.125877
```

A short-hop offset tail grid with `--offset_grid_phase short
--offset_grid_span 0.36` found promising per-grid candidates, but the best
tail candidate `--short_action_offset=-0.18,0,-0.18,0` did not reproduce in
direct eval and should not be used:

```text
target_hit_rate                   0.665243
touchdown_error_m                 0.085748
pair                              0.455288
short_tail10_rate                 0.387265
short_tail15_rate                 0.124903
```

The only tail/precision tweak that still reproduces is the previous small
short-hop offset:

```text
--short_action_offset=0.24,0,-0.12,0

target_hit_rate                   0.680934
touchdown_error_m                 0.083896
two_hop_pair_success_rate         0.480462
prepared_landing_rate             0.459571
short_hit_rate                    0.632012
short_touchdown_error_m           0.089912
short_tail10_rate                 0.364787
short_tail15_rate                 0.112957
long_hit_rate                     0.730308
long_touchdown_error_m            0.077824
long_tail10_rate                  0.268769
long_tail15_rate                  0.073077
```

Conclusion: error-tail instrumentation is now in place. PPO tail penalties did
not yet produce a better checkpoint; for visualization/deployment A/B, keep
`model_466.pt` and use `--short_action_offset=0.24,0,-0.12,0` as the only
currently verified tail-reducing tweak.

## 2026-08-26 Standard RSL-RL Runner Control

Added a standard RSL-RL PPO control experiment:

```text
experiments/random_two_hop/train_two_hop_rslrl.py
```

This keeps the same frozen 43-D teacher, 69-D state-planner observation,
continuous P_t/P_(t+1) random route, short/long distances, landing-only
correction, and tail metrics, but uses stock `OnPolicyRunner.learn()` instead
of the custom Semi-MDP PPO loop. The robot physics and teacher remain
unchanged.

The wrapper now logs tail metrics for standard RSL-RL training:

```text
Metrics/short_tail10_rate
Metrics/short_tail15_rate
Metrics/long_tail10_rate
Metrics/long_tail15_rate
```

Pilot command:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  experiments/random_two_hop/train_two_hop_rslrl.py \
  --teacher_checkpoint saved_checkpoints/professor_approved_v36_teacher/model_90.pt \
  --iterations 30 \
  --num_envs 256 \
  --num_steps_per_env 128 \
  --save_interval 10 \
  --target_tolerance 0.10 \
  --short_radius_min 0.2 \
  --short_radius_max 0.5 \
  --long_radius_min 0.5 \
  --long_radius_max 0.8 \
  --curriculum_iterations 0 \
  --correction_mode landing \
  --motor_correction_limit 0.014 \
  --velocity_feedback_gain 0.020 \
  --attitude_feedback_gain 0.100 \
  --landing_correction_height 0.20 \
  --disable_train_randomization \
  --tail_error_threshold 0.10 \
  --short_tail_error_penalty 2.0 \
  --long_tail_error_penalty 1.0 \
  --init_noise_std 0.02 \
  --lr 1e-5 \
  --entropy_coef 0.0
```

Pilot log:

```text
logs/rsl_rl/quadhopper_planner_random_two_hop_v58_rslrl_state_planner_continuous_queue_ppo/2026-08-26_12-48-14
```

The standard runner restores the full RSL-RL iteration printout. Training
metrics briefly looked strong early, but direct eval did not beat the current
Semi-MDP checkpoint.

Direct eval, 500 envs, 3000 steps:

```text
RSL-RL model_0.pt:
  target_hit_rate                 0.680577
  touchdown_error_m               0.084732
  pair                            0.470833
  short_tail10_rate               0.369057
  short_tail15_rate               0.121388

RSL-RL model_10.pt:
  target_hit_rate                 0.671877
  touchdown_error_m               0.084915
  pair                            0.460154
  short_tail10_rate               0.375267
  short_tail15_rate               0.124733

RSL-RL model_20.pt:
  target_hit_rate                 0.673768
  touchdown_error_m               0.084957
  pair                            0.461917
  short_tail10_rate               0.374656
  short_tail15_rate               0.121527

RSL-RL model_29.pt:
  target_hit_rate                 0.671955
  touchdown_error_m               0.084906
  pair                            0.455371
  short_tail10_rate               0.376489
  short_tail15_rate               0.119084
```

Conclusion: the standard runner is useful as a logging/stability control, but
this first state-planner pilot did not outperform `model_466.pt`. The likely
issue is not the runner itself; it is that step-level state-planner reward still
does not give a clean enough touchdown/pair credit signal. Keep the RSL-RL
script for further controlled experiments, but do not use these pilot
checkpoints for deployment.

## 2026-08-26 Offline Tail-Selector Experiment

Goal: test whether the large visible landing-error tail can be reduced by an
offline state-action selector/refiner without retraining the main planner.

Tools added:

```text
experiments/random_two_hop/analyze_tail_candidates.py
experiments/random_two_hop/train_second_hop_scorer.py --label_key not_tail10|not_tail15 --use_dataset_candidate_offsets
```

Oracle analysis on the existing v57 short/long candidate datasets shows that
there is real recoverable tail error, especially on the second hop:

```text
first-hop candidates:
  base_error                  0.066251
  oracle_min_error            0.053845
  base_tail10                 0.167969
  oracle_tail10               0.105469
  tail_rescued_to_10          0.372093

second-hop candidates:
  base_error                  0.075524
  oracle_min_error            0.032137
  base_tail10                 0.238281
  oracle_tail10               0.011719
  tail_rescued_to_10          0.950820
```

Two online direct-eval checks were run with a learned second-hop `not_tail10`
scorer:

```text
random deployment offsets, threshold override 0.0:
  target_hit_rate             0.672663
  touchdown_error_m           0.085241
  pair                        0.465456

dataset candidate offsets, learned threshold 0.100:
  target_hit_rate             0.680362
  touchdown_error_m           0.084359
  pair                        0.473992
  short_tail10_rate           0.366371
  long_tail10_rate            0.268698

dataset candidate offsets + short_action_offset=0.24,0,-0.12,0:
  target_hit_rate             0.678232
  touchdown_error_m           0.084456
  pair                        0.470688
  short_tail10_rate           0.368309
  long_tail10_rate            0.270195
```

Conclusion: the offline oracle proves that the tail is fixable, but the current
learned selector should not be enabled for deployment. It moves actions often
enough to lose pair success, even when using the same eight candidate offsets as
the search data. The better next route is to recycle the saved candidate search
failures into PPO fine-tuning or train a stricter failure gate that intervenes
only when the original action is predicted to be in the tail.
