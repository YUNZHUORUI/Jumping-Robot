# Planner-conditioned circular hopping

## Version 15: first learn a fixed 0.70 m gait

Version 15 keeps the proven circular trajectory task fixed and changes only
one variable at a time. Starting from v10, the absolute apex command follows a
300-iteration cosine curriculum from 1.30 m down to 0.70 m. This preserves the
working horizontal/circular gait while the controller learns progressively
lower spring/thrust timing. Variable-height training must not start until a
fixed 0.70 m checkpoint completes the circle reliably.

The policy uses a 43-D
command-conditioned policy: stable 37 dimensions, body-frame XY errors to
`P_t` and `P_(t+1)`, then absolute root-apex commands `H_t/2` and `H_(t+1)/2`.
At the nominal grounded root height of 0.38 m, 0.70 m is a 0.32 m rise (close
to the old circular task) and 1.00 m is a 0.62 m rise.

The collocation reference now covers flight only. Its duration is computed
from measured takeoff height, requested apex, landing height, and gravity.
Ground contact/stance is a separate phase with a grounded, zero-velocity
reference. This remains a kinematic reference planner; PPO handles spring,
contact, motor lag, attitude, and thrust feasibility.

Train strictly in order and inspect each stage before advancing:

```bash
bash run_train_variable_height.sh low 64 --iterations 300 \
  --checkpoint logs/rsl_rl/quadhopper_planner_circular_v10/2026-08-06_11-10-23/model_518.pt

bash run_train_variable_height.sh high 64 --iterations 20 \
  --checkpoint logs/rsl_rl/quadhopper_planner_circular_v15_descend_to_070/<run>/model_<N>.pt

bash run_train_variable_height.sh alternate 64 --iterations 20 \
  --checkpoint logs/rsl_rl/quadhopper_planner_circular_v15_high_100/<run>/model_<N>.pt
```

Use `--resume_optimizer` only to continue the same stage. Cross-stage transfer
resets optimizer state and action standard deviation. Playback uses
`play_planner_circular.py --height_stage low|high|alternate --checkpoint ...`.

TensorBoard can be launched with:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  -m tensorboard.main --logdir logs/rsl_rl --port 6006
```

This is the next curriculum level after `Quadhopper_Stable`. It does not reuse the old hand-authored ballistic/parabolic reference from `Hopper_Circular_Trajectory_Isaac`.

The accepted fixed-height circular baseline is recorded in
[STAGE_SUMMARY_V10.md](STAGE_SUMMARY_V10.md). The attempted low/variable-height
extension, including failed checkpoints that must not be reused, is recorded in
[STAGE_SUMMARY_V15.md](STAGE_SUMMARY_V15.md). Read both before continuing.

## Contract

- unchanged: USD, spring/contact model, four direct motor actions, thrust/torque curves, 100 Hz control, action delay, motor lag, power model, domain randomization, recurrent PPO architecture;
- planner: discrete direct-collocation minimum-jerk solve with start/apex/landing position and velocity constraints;
- command: `P_t` and `P_{t+1}` are the next and following **ground landing points** on the circle;
- target height: requested root apex height in the world Z frame (`1.30 m` by default);
- observation: original stable 37 dimensions followed by `[P_t error body XY (2), P_t+1 error body XY (2), relative target height (1)]`, total 42;
- checkpoint transfer: the two recurrent input matrices expand from 37 to 42 columns; the five new columns start at zero while all stable weights are preserved.

Playback colors: green is this hop's apex target, orange is ground landing point `P_t`, purple is the following ground landing point `P_{t+1}`, blue points are the first optimized cycle, cyan points are the simultaneously optimized second cycle, and the thin yellow markers trace the complete ground circle.

Version 6 uses an isolated `quadhopper_planner_circular_v6` log directory. One joint KKT solve plans both `current→P_t` and `P_t→P_(t+1)` cycles. Landing success requires XY error below `0.10 m` and apex above `1.15 m`; touchdown precision uses a `0.06 m` kernel. During descent below `0.90 m`, a ballistic touchdown projection supplies dense landing-position and tangent-velocity guidance so the controller can correct before contact. A v4/v5 checkpoint may initialize the policy, but its optimizer is reset because the landing objective changed.

Version 7 fixes the full-circle task horizon: the 2 m radius and 0.22 m chord require 58 successful waypoint advances, so episodes last 75 s and terminate successfully only after one complete revolution. It logs successful waypoints and circle completion and adds a sparse full-circle bonus. Hardware, 42-D observations, height target, and the two-cycle planner remain unchanged.

Version 8 uses the measured closed-loop hop cadence rather than the planner flight duration: episodes last 150 s so 58 advances are reachable. PPO entropy is reduced to `0.002`; transfer resets action standard deviation to `0.4` and discards the old optimizer to prevent v7's increasing exploration noise from destroying landing consistency.

Version 9 requires 58 consecutive first-attempt waypoint hits. A miss may still trigger a physical recovery hop, but it resets the completion streak to zero. The miss penalty is increased and each hit receives a bonus proportional to the current streak. `Metrics/max_consecutive_hits` is the primary stability metric; total waypoint advances alone no longer establish full-circle success.

Planner playback disables the baseline observation noise (`observation_noise_std=0`) so repeated evaluation is deterministic. Training retains the canonical `0.01` Gaussian observation noise for robustness; this does not alter observation ordering or checkpoint shape.

Version 10 is the nominal-precision fine-tuning stage. It fixes training dynamics to the same mass, inertia, `0.125 s` motor time constant, and three-step action delay used by one-environment playback; observation noise is `0.002`, entropy coefficient is `0.0005`, and transferred action std is `0.2`. Baseline defaults remain randomized, and no physical parameter value is changed.

## Train

Smoke test from the latest stable model:

```bash
bash run_train_planner_circular.sh 64 --iterations 20 \
  --checkpoint logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt
```

Full first run:

```bash
bash run_train_planner_circular.sh 256 --iterations 1000 \
  --checkpoint logs/rsl_rl/quadhopper_stable_baseline/2026-08-05_01-28-58/model_498.pt
```

Playback automatically selects the newest task checkpoint:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh play_planner_circular.py
```

Experimental alternating-height playback keeps the v10 42-D checkpoint
contract. The first and second optimized cycles use separate apex commands,
and the policy observes the command for its current hop:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  play_planner_circular.py --alternate_heights --height_high 1.0 --height_low 0.7
```

This tests zero-shot height generalization of a policy trained at `1.30 m`;
it is not evidence that the model has learned the full height range. Train a
separate curriculum before treating variable-height tracking as a completed
capability.

Variable-height v11 training starts from the accepted nominal v10 model but
uses a new log namespace and resets the optimizer and action std. Vectorized
environments randomize whether their first command is high or low, then
alternate the two heights after each successful landing:

```bash
bash run_train_planner_circular.sh 64 \
  --iterations 500 \
  --checkpoint logs/rsl_rl/quadhopper_planner_circular_v10/2026-08-06_11-10-23/model_518.pt \
  --alternate_heights --height_high 1.0 --height_low 0.7
```

The output is isolated under
`logs/rsl_rl/quadhopper_planner_circular_v11_variable_height`. Resume a v11
checkpoint with the same flags to restore its optimizer exactly.
