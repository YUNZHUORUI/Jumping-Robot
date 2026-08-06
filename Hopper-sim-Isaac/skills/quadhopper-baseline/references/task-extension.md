# Extending to trajectories, obstacles, and ring traversal

## Table of contents

- Separate robot and task
- Command design
- Observation design
- Reward design
- Termination and reset
- Curriculum
- Ring-traversal checklist

## Separate robot and task

Keep the robot layer invariant first: USD/model, 4-motor dynamics, action convention, motor lag, force application, and baseline randomization. Put course geometry, goals, command generation, phase tracking, rewards, success logic, and curricula in the task layer.

Do not call a reward-shaped PPO policy “trajectory optimization” without clarifying the method. Distinguish:

- reference trajectory tracking with RL;
- waypoint/command-conditioned RL;
- model-predictive control or direct collocation;
- offline trajectory generation followed by policy tracking.

## Command design

Represent task goals in frames that generalize across environments. For a target or ring, useful commands include:

- target position in body frame;
- target velocity or time-to-go;
- target/ring orientation relative to the robot;
- ring-plane normal in body frame;
- phase or next-waypoint index;
- desired crossing point and clearance.

Transform world targets per environment. Never feed raw tiled-world coordinates unless intentionally learning environment indices.

## Observation design

Start from the 26-D baseline and add only task-relevant, observable quantities. Document each slice, frame, units, scale/clipping, noise, and latency. For each addition:

- update `observation_space`;
- update actor input/checkpoint policy;
- decide whether the critic gets privileged information separately;
- normalize distances and velocities to comparable numeric ranges;
- avoid leaking future randomized trajectory information unavailable on hardware.

## Reward design

Use a small, interpretable set of terms and log each separately. A typical staged structure is:

- progress toward the next target or through the ring plane;
- alignment with desired velocity/ring normal;
- upright/attitude and angular-rate stability;
- clearance/collision penalty;
- action-rate and smoothness penalties;
- sparse success bonus for a valid crossing;
- failure penalties for collision, bypassing, or leaving the workspace.

Prefer progress differences or potential-based shaping over a large constant proximity reward that can encourage hovering beside a goal. Define a ring crossing using a sign change across the ring plane plus radial distance inside the aperture; proximity alone is not success.

## Termination and reset

Define success separately from failure and timeout. Randomize initial pose and course geometry within a curriculum. Ensure reset clears action/motor history, phase, previous plane-side state, success flags, and episodic accumulators.

For higher jumps or flight phases, revisit the baseline `z > 0.4 m` termination; it will incorrectly kill many trajectory/ring tasks. Changing this is task logic, not necessarily a robot-physics change.

## Curriculum

Increase difficulty only after measurable success:

1. fixed close target/ring, generous aperture, mild pose noise;
2. random XY/Z and yaw, still generous aperture;
3. narrower aperture and greater distance;
4. varied approach direction, speed, disturbances, and dynamics;
5. multiple rings/waypoints and sim-to-real randomization.

Track success rate, valid-crossing rate, collisions, minimum clearance, time-to-cross, XY drift, peak tilt, action saturation, and reward components.

The planner-circular curriculum uses a 42-D checkpoint contract: the canonical stable 37 dimensions remain first, followed by body-frame XY errors to ground landing points `P_t` and `P_{t+1}` and one requested apex-height command. Version 5 jointly optimizes both complete jump cycles in one discrete direct-collocation KKT solve and uses a 0.10 m landing-success tolerance; it does not use the historical analytic ballistic arc.

Version 8 defines a complete circular episode from geometry and measured closed-loop cadence. With radius `2.0 m` and chord `0.22 m`, `steps_per_revolution=58`; the measured controller needs about 2 s per successful advance, so the episode horizon is `150 s`. Completion occurs only after all 58 waypoint advances. PPO entropy is `0.002`, and v7 transfer must reset action std to `0.4` instead of inheriting its unstable `>1.17` value.

Version 9 tightens completion to 58 consecutive first-attempt hits. Any missed touchdown resets the hit streak, although the controller may physically retry the waypoint. Train and select checkpoints using `Metrics/max_consecutive_hits` and `Metrics/circle_completion`, not accumulated waypoint count alone.

Version 10 is a nominal-precision fine-tuning stage: circular training matches single-environment playback dynamics (unit mass/inertia multipliers, `0.125 s` motor time constant, three-step delay), uses observation noise `0.002`, PPO entropy `0.0005`, and transfer std `0.2`. This is separate from randomized robustness training and does not alter canonical physical parameters.

The completed v10 milestone, latest accepted checkpoint, commands, version history, and next-task handoff are recorded in `Quadhopper_Planner_Circular/STAGE_SUMMARY_V10.md`. Read that summary before extending planner-circular into ring traversal or another downstream curriculum.

## Ring-traversal checklist

- Model ring collision geometry separately from visualization.
- Define ring center, plane normal, local in-plane axes, inner radius/shape, and safety margin.
- Express relative geometry in body frame for the policy.
- Detect crossing between consecutive simulation steps to avoid tunneling past the plane.
- Require crossing direction when the course is directional.
- Reject crossings outside the aperture even if the plane was crossed.
- Consider continuous collision detection or smaller simulation steps at high speed.
- Visualize command frames, crossing point, and clearance during debugging.
- Preserve a task version/config with every checkpoint.
