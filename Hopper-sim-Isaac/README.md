# Hopper-sim-Isaac

Isaac Lab + RSL-RL based Quadhopper simulation workspace. The current active branch is
`experiment/higher-jump`, used to save and continue the higher-jump experiment state.

## Current Experiment Snapshot

Latest saved snapshot:

```text
fbc4366 chore(isaac): save current hopper experiment state
```

This snapshot includes the generated hopper USD asset and the TorchScript power model
required by the current environment:

- `Quadhopper_Isaac/model/HopperAsset.usd`
- `Quadhopper_Isaac/model/quadhopper_memory_power.pt`

The environment now loads `HopperAsset.usd` directly and uses the neural-network power
model inside the reward calculation.

## Main Files

- `Quadhopper_Isaac/quadhopper_env.py` - Isaac Lab direct RL environment.
- `Quadhopper_Isaac/my_hopper_cfg.py` - articulation config and USD asset path.
- `Quadhopper_Isaac/rsl_rl_ppo_cfg.py` - PPO runner and recurrent actor-critic config.
- `Quadhopper_Isaac/model/Create-Model.py` - current USD asset generation script.
- `train.py` - train PPO with RSL-RL.
- `play.py` - load a checkpoint and visualize the policy.
- `export_onnx.py` - export a trained checkpoint to ONNX for deployment.
- `deploy/` - Windows real-robot deployment package.

## Current Model Notes

- Asset path: `Quadhopper_Isaac/model/HopperAsset.usd`
- Observation size: `37`
- Action size: `4`
- Policy type: recurrent actor-critic
- Control rate: 100 Hz (`dt=0.01`, `decimation=1`)
- Episode length: 15 s for training
- Power model input is built from recent PWM history and an accumulated SOC-like tracker.

Current thrust curve in simulation:

```python
F = -0.2371 * u ** 2 + 0.8130 * u + 0.0113
```

## Train

Run from this directory with the Isaac Sim Python environment:

```bash
python train.py --num_envs 256
```

Resume from the newest checkpoint:

```bash
python train.py --num_envs 256 --resume
```

Resume from a specific checkpoint:

```bash
python train.py --num_envs 256 --resume --checkpoint logs/rsl_rl/myhopper/<run>/model_<iter>.pt
```

Training logs are written under:

```text
logs/rsl_rl/myhopper/
```

## Play

Use the latest checkpoint:

```bash
python play.py --num_envs 16
```

Use a specific checkpoint:

```bash
python play.py --checkpoint logs/rsl_rl/myhopper/<run>/model_<iter>.pt
```

## Export ONNX

Export the latest checkpoint:

```bash
python export_onnx.py
```

Export a specific checkpoint:

```bash
python export_onnx.py --checkpoint logs/rsl_rl/myhopper/<run>/model_<iter>.pt
```

The exported ONNX policy is saved next to the checkpoint.

## Known Notes

- `HopperAsset.usd` and `quadhopper_memory_power.pt` are normally ignored by `.gitignore`,
  but this experiment snapshot intentionally tracks them so the saved state is reproducible.
- `Quadhopper_Isaac/model/DroneAsset_physics_override.usda` and `Drop-Test.py` still reference
  the older `DroneAsset` flow. Treat those as legacy/drop-test utilities until updated.
- Some comments and scripts still contain old `MyQuadcopter` names. The active gym environment
  registration is `myhopper`.
