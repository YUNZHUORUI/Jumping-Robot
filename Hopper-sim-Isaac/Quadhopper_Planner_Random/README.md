# Arbitrary-direction two-hop waypoint training

This task replaces only the circular ground-waypoint generator. The robot
USD, spring/contact dynamics, four motor actions, thrust/torque curves, motor
lag, command delay, power model, 100 Hz controller, direct-collocation
reference, reward implementation, and recurrent PPO architecture are inherited
unchanged from the accepted planner-circular task.

## Route command

For each environment, let `X_0` be the reset position. The target sequence is

```text
X_1 = X_0 + r_short [cos(theta_1), sin(theta_1)]
X_2 = X_1 + r_long  [cos(theta_2), sin(theta_2)]
X_3 = X_2 + r_short [cos(theta_3), sin(theta_3)]
...
```

where every heading is sampled independently and uniformly from `[0, 2 pi)`,
`r_short ~ U(0.5, 0.8) m`, and `r_long ~ U(0.8, 1.0) m`. The command keeps a
rolling two-point queue, so the policy and planner always receive both `P_t`
and `P_(t+1)`. A new point is appended only after a successful landing.

The task retains the 43-D v22 policy interface:

```text
stable observations (37)
+ body-frame XY error to P_t (2), divided by 0.75 m
+ body-frame XY error to P_(t+1) (2), divided by 0.75 m
+ current and next absolute apex commands (2), divided by 2.0 m
```

The input width and ordering are checkpoint-compatible with v22, but the XY
command distribution and scale are intentionally different. Transfer the v22
policy with a reset optimizer; do not use exact optimizer resume across tasks.

## Train and play

Train the distance curriculum one stage at a time. Advance only after the
monitor reports `PASS`:

| Stage | Both hop ranges | Fixed action std |
|---|---:|---:|
| `direction` | 0.22--0.30 m | 0.12 |
| `medium` | 0.30--0.50 m | 0.10 |
| `short` | 0.50--0.80 m | 0.08 |
| `full` | short 0.50--0.80 m, long 0.80--1.00 m | 0.06 |

The current random-route campaign uses a fixed 1.00 m apex for both hops so
that height alternation does not add difficulty while the XY route is learned.
The wrapper defaults to `--height_stage high`; pass `--height_stage alternate`
explicitly only when height alternation is wanted again.

To train the requested full two-hop geometry directly, with the first circle
at 0.50--0.80 m and the second circle at 0.80--1.00 m:

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper-sim-Isaac

bash run_train_random_curriculum.sh full 256 \
  --iterations 300 \
  --checkpoint /path/to/a/compatible/model_N.pt
```

Each stage gets an isolated `quadhopper_planner_random_two_hop_v24_*` log
namespace. Transfer the selected checkpoint to the next stage without
`--resume_optimizer`; exact optimizer resume is only for continuing the same
stage. Check progress with:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  monitor_random_two_hop.py \
  logs/rsl_rl/quadhopper_planner_random_two_hop_v24_direction_alternate_070_100 \
  --stage direction
```

Once a distance stage has learned the route but still lacks consecutive
first-attempt hits, run its isolated accuracy sub-stage. A miss terminates the
episode, landing penalties are tightened, and action std is fixed at 0.06:

```bash
bash run_train_random_curriculum.sh direction 256 \
  --accuracy_finetune \
  --iterations 200 \
  --checkpoint logs/rsl_rl/quadhopper_planner_random_two_hop_v24_direction_alternate_070_100/<run>/model_<N>.pt
```

Accuracy logs use the `quadhopper_planner_random_two_hop_v25_*_accuracy_*`
namespace. Do not expand distance until deterministic evaluation confirms the
stage gate.

Playback uses the same generator and draws both sampled-radius circles:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  play_planner_circular.py \
  --route random_two_hop \
  --distance_stage full \
  --height_stage high \
  --height_high 1.00 \
  --num_envs 1 \
  --checkpoint /path/to/a/compatible/model_N.pt
```

`direction` is deliberately a small-radius warm-up: it overrides both circles
to 0.22--0.30 m. It must not be used when visually checking the final
0.50--0.80 m / 0.80--1.00 m geometry. Playback prints both configured ranges
and the actually sampled distances at startup.

Useful selection metrics are `Metrics/episode_touchdown_error_m`,
`Metrics/target_hit_rate`, the short/long error and hit-rate splits,
`Metrics/max_consecutive_hits`, `Metrics/route_completion`, and the low/high
apex errors. The inherited `Metrics/circle_completion` key is retained as an
alias so existing dashboards do not break; in this task it means completion of
58 consecutive arbitrary-direction targets, not a circle.

Before a full run, instantiate 1--4 environments and verify that the requested
0.8--1.0 m low-apex hops are physically feasible with the preserved hardware
model. The kinematic planner itself does not enforce motor or contact limits.
