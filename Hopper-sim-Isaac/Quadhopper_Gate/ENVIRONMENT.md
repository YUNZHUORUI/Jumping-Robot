# Quadhopper Gate Environment

This package implements the first executable environment from `Quadhopper_Updated_Training_Roadmap.md`. Its current hardware source of truth is `/home/terry/Desktop/workspace/Create-Hopper-Model.py`, `my_hopper_cfg.py`, and `quadhopper_env.py`.

## Policy contract

- Action (4-D): normalized collective command, desired body roll rate, desired body pitch rate, desired body yaw rate.
- Observation (50-D): body linear/angular velocity, quaternion, projected gravity, spring position/velocity/contact, body-frame gate position/normal, radius/clearance, body-frame landing target, three-phase one-hot state, and five actions of history.
- Control rate: 100 Hz; physics rate: 100 Hz.
- Asset: `Quadhopper_Isaac/model/HopperAsset.usd` (183 g, 604 N/m spring, 0.08 m travel).
- Actuator: authoritative thrust curve, randomized 20--40 ms training delay, 100--140 ms training motor time constant, and 125 ms deterministic inspection time constant.
- Energy: `quadhopper_memory_power.pt` receives normalized motor history in `[0,1]`, plus the SOC-like tracker.

Existing 26-D `myhopper`, 37-D `higher-jump`, and earlier 42-D gate checkpoints are not input-compatible with this 50-D task.
New logs use `logs/rsl_rl/quadhopper_gate_synced/` so all earlier gate checkpoints remain isolated.

## Curriculum stages

Stages 0–6 correspond to stable hopping, landing-target control, large/medium/narrow/extreme gates, and a multi-gate scaffold. Start with Stage 0 and advance only after success and landing metrics stabilize.

Inspect the higherjump mechanical model before training or loading a policy:

```bash
/path/to/isaac-sim/python.sh Hopper-sim-Isaac/inspect_gate_model.py
```

This repeatedly drops one robot with its motors off. Add `--powered` to inspect the neutral `u=0.5` motor command instead.

```bash
bash Hopper-sim-Isaac/run_train_gate.sh 0 64 --iterations 20
```

Continue the next stage from a compatible checkpoint (all new stages preserve the same 50-D/4-D policy contract):

```bash
bash Hopper-sim-Isaac/run_train_gate.sh 1 256 \
  --checkpoint /path/to/stage_0/model_N.pt
```

Visualize a checkpoint:

```bash
/path/to/isaac-sim/python.sh Hopper-sim-Isaac/play_gate.py \
  --stage 2 --checkpoint /path/to/model_N.pt
```

With no arguments, `play_gate.py` automatically selects the newest checkpoint under `logs/rsl_rl/quadhopper_gate_synced/` and infers `stage_N` from its path.

The first version uses an analytic circular gate crossing/collision test for fast parallel training. It does not yet spawn a rigid visual ring. Stage 6 currently exposes the observation/reward contract but still uses one gate per episode; sequence state and multiple gate instances are the next implementation layer.
