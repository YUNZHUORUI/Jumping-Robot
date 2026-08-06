# Reuse, training, and validation workflows

## Table of contents

- Create a new task project
- Static validation
- Runtime smoke test
- Train and play
- Checkpoint and deployment compatibility

## Create a new task project

1. Preserve the original `Quadhopper_Isaac` package as the comparison baseline.
2. Create a clearly named sibling package or environment module for the new task.
3. Reuse/copy together:
   - `model/` assets and their license/provenance;
   - corrected asset config;
   - motor/action dynamics from `_pre_physics_step` and `_apply_action`;
   - baseline randomization ranges;
   - PPO config as an initial hyperparameter baseline.
4. Give the new task a unique Gym ID and PPO `experiment_name`.
5. Add task commands and observations deliberately; never append fields without updating `observation_space` and documenting ordering.
6. Create task-specific reward, termination, reset, curriculum, train, and play configuration.
7. Keep output logs in a new experiment directory so checkpoints cannot be confused.

## Static validation

Before starting Isaac Sim:

- resolve every asset path with `os.path.exists` or `Path.exists`;
- verify the registered entry-point modules/classes exist;
- verify runner config exports resolve;
- calculate observation dimension from concatenated tensors;
- check action count stays 4 unless the motor interface changes;
- check all reward tensors have shape `(num_envs,)`;
- verify body name `body` and articulation prim assumptions against the USD;
- search for hard-coded task IDs, experiment names, device names, and absolute paths;
- compile Python modules using the Isaac Sim/Isaac Lab Python environment, not arbitrary system Python when imports are required.

## Runtime smoke test

Use a staged test:

1. Instantiate 1–4 environments for a few steps with zero actions.
2. Run random bounded actions and check NaN/Inf, tensor shape, resets, contacts, and joint travel.
3. Verify thrust direction, roll/pitch/yaw signs, action delay, and motor response.
4. Visualize one environment and confirm target/obstacle frames.
5. Run a short PPO job with 16–64 environments.
6. Only then launch the full environment count and iterations.

Record Isaac Lab/Isaac Sim/RSL-RL versions and GPU because APIs and behavior are version-sensitive.

## Train and play

The captured machine uses:

```bash
bash Hopper-sim-Isaac/run_train.sh 256
```

The equivalent explicit form is:

```bash
/path/to/isaac-sim/python.sh Hopper-sim-Isaac/train.py --headless --num_envs 256
```

Playback requires the same Isaac Python environment:

```bash
/path/to/isaac-sim/python.sh Hopper-sim-Isaac/play.py \
  --num_envs 16 --checkpoint /path/to/model_N.pt
```

Treat these as templates. Update task ID, environment config, runner config, and log root for a new task.

## Checkpoint and deployment compatibility

A checkpoint is directly reusable only if action count, observation count/order/scaling, actor architecture, and normalization state are compatible. Reward changes do not prevent loading but change what further training optimizes. Physics changes may make a policy load successfully yet behave incorrectly.

For transfer learning, explicitly choose among:

- exact resume: all interfaces and PPO state compatible;
- policy initialization: load compatible actor weights, reset optimizer/training state;
- partial transfer: map shared observation encoder/layers and initialize new input/output weights;
- train from scratch: preferred when observation/action semantics change substantially.

For ONNX export, save beside the model: input ordering and shape, normalization behavior, action post-processing, motor order, control frequency, and source checkpoint/revision.
