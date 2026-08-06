# Architecture and source map

## Project entry points

The reusable project root is `Hopper-sim-Isaac`. The current source files are the three authoritative files directly under `/home/terry/Desktop/workspace/`; `experiment/higher-jump` is only historical context.

- higherjump:`train.py`: constructs `QuadhopperEnvCfg`, makes Gym task `myhopper`, wraps it for RSL-RL, and trains PPO.
- higherjump:`play.py`: loads a checkpoint and runs the higherjump policy.
- `train_gate.py` / `play_gate.py`: train and visualize the new gate curriculum without replacing the hardware baseline.
- `train_planner_circular.py` / `play_planner_circular.py`: two-cycle look-ahead circular tracking driven by a numerical direct-collocation planner.
- `Quadhopper_Isaac/__init__.py`: Gym registration.

## Runtime data flow

```text
policy command
  -> randomized 20--40 ms command delay
  -> command-to-motor mixer (direct motors in higherjump; rate mixer in gate task)
  -> 100--140 ms training motor lag (125 ms deterministic inspection)
  -> authoritative fitted thrust and torque curves
  -> force/torque on rigid body Body
  -> HopperAsset spring/contact dynamics
  -> state, center_spring_joint, task commands, action history
  -> reward, TorchScript power estimate, and termination
  -> recurrent RSL-RL PPO
```

## File ownership

| Concern | Authoritative file | Main symbols/assets |
|---|---|---|
| Hardware environment | higherjump:`Quadhopper_Isaac/quadhopper_env.py` | `QuadhopperEnvCfg`, `QuadhopperEnv` |
| Robot spawn config | higherjump:`Quadhopper_Isaac/my_hopper_cfg.py` | `MY_HOPPER_CFG` |
| PPO baseline | higherjump:`Quadhopper_Isaac/rsl_rl_ppo_cfg.py` | `QuadhopperPPORunnerCfg` |
| Physical model | higherjump:`Quadhopper_Isaac/model/HopperAsset.usd` | `Body`, `SpringLeg`, `center_spring_joint` |
| Energy model | higherjump:`Quadhopper_Isaac/model/quadhopper_memory_power.pt` | 49-D input, predicted power |
| Model generation | higherjump:`Quadhopper_Isaac/model/Create-Model.py` | mass, inertia, geometry, spring, collision |
| Gate task | `Quadhopper_Gate/gate_env.py` | `QuadhopperGateEnvCfg`, `QuadhopperGateEnv` |
| Planner circular task | `Quadhopper_Planner_Circular/planner_circular_env.py` | `PlannerCircularEnvCfg`, `PlannerCircularEnv` |

## Coordinate and tensor conventions

- World is Z-up and Isaac root quaternion order is `wxyz`.
- Thrust is positive body Z and force/torque is applied to body `Body`.
- `root_lin_vel_b` and `root_ang_vel_b` are body-frame state.
- Gate and landing commands are transformed to body frame.
- Motor order is `F1,F2,F3,F4`; preserve the higherjump mixer signs.
- The spring coordinate is `center_spring_joint`; higherjump treats `q>0.002 m` as contact/compression activity.
- The higherjump action history contains five four-dimensional actions.
