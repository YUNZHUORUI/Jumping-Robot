# Planner-conditioned circular hopping

This is the next curriculum level after `Quadhopper_Stable`. It does not reuse the old hand-authored ballistic/parabolic reference from `Hopper_Circular_Trajectory_Isaac`.

Current milestone, checkpoint, version history, commands, and handoff context are recorded in [STAGE_SUMMARY_V10.md](STAGE_SUMMARY_V10.md). Read it before extending this task or starting ring traversal.

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
