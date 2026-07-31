# Circular Hopper Isaac

Isaac Lab/RSL-RL project for waypoint-based circular trajectory hopping.

The robot asset, motor model, thrust curve, torque model, control delay, motor lag, and domain randomization are migrated from `Hopper-sim-Isaac/Quadhopper_Isaac`.

## Task

The first training target is a short circular arc rather than a full circle:

- radius: `2.0 m`
- chord distance per hop: `0.22 m`
- landing tolerance: `0.14 m`
- desired per-hop apex: `0.30 m`
- episode success: `24` consecutive target hits

Targets advance only after a touchdown near the current waypoint. This follows the training plan in `docs/Isaac_Circular_Trajectory_Training_Plan.md` and avoids moving the target forward while the robot is falling behind.

## Run

```bash
bash run_train.sh --num_envs 256
python play.py --checkpoint logs/rsl_rl/circular_hopper_spring/<run>/model_<iter>.pt
```

Increase `CircularHopperEnvCfg.max_successful_hops` after the 4-hop arc succeeds consistently.
