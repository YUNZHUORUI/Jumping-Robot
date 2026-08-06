# Authoritative Quadhopper hardware baseline contract

## Source of truth

Use the six user-supplied files in `/home/terry/Downloads` as the authoritative stable-jump definitions:

- `Create-Hopper-Model.py`, `Jump+Base.stl`, and `Jump+Leg.stl`;
- `my_hopper_cfg.py` and `quadhopper_env.py`;
- `rsl_rl_ppo_cfg.py`.

The verified frozen copy is `Hopper-sim-Isaac/Quadhopper_Stable`; use its isolated log namespace `quadhopper_stable_baseline` as curriculum level 1.

The `experiment/higher-jump` branch is an older snapshot: its model geometry is equivalent, but its motor time-constant range, yaw coefficient, power input scaling, and power rewards differ. Do not use the main-branch `QuadhopperAsset.usd`, `my_drone_cfg.py`, or `my_hopper_env.py`; they describe an older 140 g/restitution-assisted robot.

## Model and simulation

| Parameter | Higherjump value |
|---|---:|
| Asset | `model/HopperAsset.usd` |
| Root/default prim | `/Drone` |
| Wrench body | `Body` |
| Leg body | `SpringLeg` |
| Spring joint | `center_spring_joint` |
| Total mass | 0.183 kg |
| Body mass | 0.173 kg |
| Numerical leg mass | 0.010 kg |
| Body inertia | `(1.231252e-3, 1.286169e-3, 2.305957e-3)` kg m² |
| Leg inertia | `(1e-5, 1e-5, 1e-5)` kg m² |
| Spring stiffness | 604 N/m |
| Spring damping | 2 N s/m |
| Spring travel | 0.08 m |
| Static body-root height | 0.2733543068 m |
| Physics/control step | 0.01 s / 0.01 s (100 Hz) |
| Ground friction | static 1.5, dynamic 1.5 |
| Restitution | 0.0 |

The USD custom-data field `hopper_measurements_json` records the generated geometry, mass, spring, collision, and static-pose measurements. Read it when exact dimensions are needed.

## Actuator contract

- Four motor channels use the preserved motor order and arm length `L=0.0813 m`.
- Higherjump direct-motor mapping is `target_u = clamp(0.5*action + 0.5, 0, 1)`.
- Randomly select an action delayed by 2–4 control samples (20–40 ms) in multi-environment training; use three samples for deterministic single-environment inspection.
- Apply first-order lag `alpha=dt/(tm+dt)`. Training resets use `tm=[0.10,0.14] s`; single-environment inspection uses `0.125 s`.
- Per-motor thrust is `F=-0.2371*u²+0.8130*u+0.0113` N.
- Roll is `L*(F1+F4-F2-F3)`.
- Pitch is `L*(F1+F2-F3-F4)`.
- Yaw is `0.054*(-u1²-u3²+u2²+u4²)`.
- Equivalent mass randomization divides thrust by `[0.95,1.05]`.
- Equivalent inertia randomization divides each torque axis by `[0.9,1.1]`.

For roadmap tasks, a high-level rate/thrust policy may sit above this actuator layer. Explicitly distinguish that intentional policy-interface change from the preserved motor physics.

## Power model contract

Load `model/quadhopper_memory_power.pt` as a TorchScript model. Maintain 25 samples of four normalized motor commands in `[0,1]`; do not multiply them by 1000 before inference. Build its 49-D input as:

- last 10 samples flattened: 40;
- mean of last 15 samples: 4;
- mean of all 25 samples: 4;
- accumulated SOC-like tracker: 1.

Update the tracker by `sum(u)/10000` each control step. Reset both histories every episode. The current environment scales the legacy power rewards by 40: power `-0.08`, spring-release synergy `+0.8`, and spring-compression anti-synergy `-0.8`.

## Higherjump policy/training snapshot

The original higherjump task has 37 observations, four direct-motor actions, five actions of history, a 15 s episode, and an `ActorCriticRecurrent` PPO policy. Its PPO baseline uses actor `[256,128]`, critic `[256,256]`, ELU, initial noise 0.4, value coefficient 0.5, entropy 0.01, eight minibatches, learning rate `3e-4`, desired KL 0.015, gamma 0.99, and lambda 0.95.

The 37-D observation uses Gaussian noise with configurable standard deviation `observation_noise_std`, defaulting to the canonical training value `0.01`. Deterministic single-environment playback sets it to `0.0`; this changes neither observation semantics nor checkpoint compatibility.

Do not load its 37-D checkpoint directly into a task with a different observation ordering/dimension. Reuse the physical and optimizer baseline, then train a new policy or write an explicit partial-transfer adapter.

## Required preflight

- Confirm `HopperAsset.usd` and `quadhopper_memory_power.pt` exist and match the chosen higherjump revision.
- Find body `Body` and joint `center_spring_joint` after spawning.
- Keep restitution at zero; elastic behavior comes from the physical spring joint.
- Preserve the 100 Hz actuator assumptions unless all delay and time-constant calculations are retuned.
- Record any intentional difference from higherjump in the new task documentation.
