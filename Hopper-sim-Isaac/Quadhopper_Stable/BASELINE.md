# Canonical Stable-Jump Baseline

This package is a frozen copy of the six user-provided files from `/home/terry/Downloads` plus the required TorchScript power model. The copied environment, PPO configuration, asset configuration, STL files, and model generator retain their original hashes. `model/HopperAsset.usd` was regenerated locally from the frozen model generator and STL files.

Only `train_stable.py` and `play_stable.py` override the original Windows paths for the power model and CSV output. The environment equations and training parameters are unchanged.

Policy contract: 37 observations, four direct motor actions, five actions of history, recurrent actor-critic. Logs are isolated under `logs/rsl_rl/quadhopper_stable_baseline/`.

Train the supplied 250-iteration PPO configuration:

```bash
bash run_train_stable.sh 256
```

Run a short smoke test:

```bash
bash run_train_stable.sh 64 --iterations 20
```

After training, play the newest checkpoint with no arguments:

```bash
/path/to/isaac-sim/python.sh play_stable.py
```

Resume exact-compatible training from an earlier level:

```bash
bash run_train_stable.sh 256 --checkpoint /absolute/path/to/model_N.pt
```

The next task level must preserve this package's robot model, 37-D observation ordering, four-action motor interface, and recurrent network if it intends to resume the complete PPO checkpoint. If observations change, use an explicit transfer adapter instead of silently loading an incompatible checkpoint.
