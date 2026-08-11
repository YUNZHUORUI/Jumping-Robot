# Quadhopper v22 real-robot deployment

Accepted source checkpoint: v22 `model_100.pt`, alternating absolute root-apex commands
`0.70 -> 1.00 -> 0.70 -> 1.00 m` at 100 Hz.

## What the ONNX model contains

The ONNX file contains the deterministic recurrent actor only. It has three inputs
(`obs`, `h_in`, `c_in`) and three outputs (`actions`, `h_out`, `c_out`). Carry the
hidden and cell states between 100 Hz calls. Reset both states to zero on boot, manual
disarm, fall detection, estimator reset, or a new mission.

The policy does **not** contain state estimation, waypoint generation, direct-collocation
reference generation, contact/liftoff/touchdown detection, ESC calibration, the simulated
motor lag, or a safety supervisor. Those components must run around the policy.

## Required 43-D observation

| Slice | Signal | Convention |
|---|---|---|
| 0:3 | linear velocity | body frame, m/s |
| 3:6 | angular velocity | body frame, rad/s |
| 6:10 | attitude quaternion | world orientation, `wxyz` |
| 10:13 | planner reference position error | body frame, m |
| 13 | root height | world Z, m |
| 14 | contact | `spring_position > 0.002 m` |
| 15 | spring position | m |
| 16 | spring velocity | m/s |
| 17:37 | five previous raw policy actions | oldest to newest, each `F1..F4` |
| 37:39 | current landing XY error | body frame, divided by `0.22 m` |
| 39:41 | next landing XY error | body frame, divided by `0.22 m` |
| 41 | current absolute apex command | divided by `2.0 m` |
| 42 | next absolute apex command | divided by `2.0 m` |

Do not reorder quaternion or motors. The trained motor order is `F1,F2,F3,F4`.

## Action and actuator contract

Clamp policy actions to `[-1,1]`, then calculate
`u_target = clamp(0.5 * action + 0.5, 0, 1)`. Simulation inspection used a three-sample
command delay and `0.125 s` motor time constant. The real ESC command corresponding to
`u` must be identified from a guarded thrust-stand calibration; `u*1000` is not a safe
or universal PWM mapping.

The trained thrust fit was `F=-0.2371*u^2+0.8130*u+0.0113 N` per motor. Measure the real
curve, delay, and time constant before free flight. Large mismatch requires system-ID and
randomized fine-tuning, not an ad-hoc gain inserted after the policy.

## Planner integration

The package includes `direct_collocation_planner.py` and `waypoint_command.py` as reference
implementations. Port their state machine and coordinate conventions to the flight computer,
or run the high-level planner on a companion computer and send the resulting observation
terms at 100 Hz. Replanning occurs around liftoff/touchdown; trajectory sampling occurs every
policy tick.

Required estimator signals are body velocity, body angular velocity, `wxyz` attitude, world
position/height, spring position/velocity, and reliable contact. Motion capture is strongly
recommended for the first experiments; onboard VIO can follow only after latency and frame
conventions are verified.

## Bring-up sequence

1. Remove propellers and compare the 43-D real observation stream against logged simulation.
2. Run the included ONNX inference example and verify hidden-state reset behavior.
3. Use a thrust stand to map normalized `u` to ESC commands and measure delay/time constant.
4. Tether the robot with a hardware kill switch; test fixed `0.70 m` in place.
5. Test one short XY hop at `0.70 m`, then a fixed `1.00 m` command.
6. Test alternating `0.70/1.00 m`; only then enable the full circular waypoint sequence.

Never perform the first test untethered. Add independent attitude, position, action-saturation,
timeout, estimator-health, and operator-disarm limits outside the neural policy.

## Dependencies

Export uses the Isaac Sim Python environment. Deployment inference requires Python 3 with
`numpy` and `onnxruntime`, or an equivalent ONNX Runtime/TensorRT C++ integration supporting
ONNX opset 18 LSTM.
