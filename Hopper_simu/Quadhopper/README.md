# QuadHopper RL code package

This folder contains the modular QuadHopper reinforcement learning code used for training and evaluating a hopping robot policy.

## Quick start

1. Install dependencies
   ```bash
   cd Hopper_simu/Quadhopper
   python -m pip install -r requirements.txt
   ```

2. Train a model from the parent directory
   ```bash
   cd ../
   python -m Quadhopper.train --mode train --backend auto --device auto
   ```

   - `--backend auto` will use the native Torch backend on CUDA and the Stable-Baselines3 backend on CPU.
   - `--device auto` will use GPU when available.

3. Test the trained model
   ```bash
   cd ../
   python -m Quadhopper.train --mode test --model models/quadhopper_policy
   ```

4. Evaluate multiple episodes
   ```bash
   cd ../
   python -m Quadhopper.train --mode eval --model models/quadhopper_policy --episodes 20
   ```

## Important notes

- The package is intended to be run from the `Hopper_simu` directory so Python can resolve the `Quadhopper` module.
- The default model path is configured in `config.py` and can be overridden with `--model`.
- The training script supports several options:
  - `--target-count`
  - `--timesteps`
  - `--resume-model`
  - `--num-envs`
  - `--rollout-steps`
  - `--batch-size`

## Main files

- `config.py`: hyperparameters and environment settings
- `env.py`: Gym-style environment implementation
- `train.py`: CLI entry point for train/test/eval/demo
- `torch_ppo.py`: native PyTorch PPO implementation
- `torch_env.py`: vectorized torch environment
- `renderer.py`: rollout rendering and plot generation

## Suggested modification points

- Adjust reward shaping in `reward.py`
- Change physics or contact behavior in `physics.py`
- Tune training hyperparameters in `config.py`
- Modify the observation/action interface in `env.py`

If you want, you can also add a new model checkpoint path or switch the training backend from `auto` to `torch` or `sb3` depending on the machine.
