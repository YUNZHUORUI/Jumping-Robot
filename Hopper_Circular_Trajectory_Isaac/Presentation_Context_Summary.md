# Hopper Circular Trajectory Tracking: Progress Summary

## 1. Project Goal

This project trains a hopping robot in Isaac Lab to follow a planned 2D trajectory through continuous jumping.

The current target trajectory is a circle. The hopper should jump from waypoint to waypoint along the circular path and keep its flight trajectory close to a generated ballistic reference.

The task evolved from basic hopping into fixed-horizon trajectory tracking:

- Stable takeoff and landing
- Accurate landing on circular waypoints
- Continuous progression along the circle
- Fixed 4-5 hop planning horizon instead of replanning every jump
- Reduced in-air correction and smoother ballistic tracking

## 2. Project Directory

Main project:

```text
/home/terry/Desktop/workspace/Jumping-Robot/Hopper_Circular_Trajectory_Isaac
```

Important files:

```text
CircularHopper_Isaac/circular_hopper_env.py
CircularHopper_Isaac/trajectory_commands.py
CircularHopper_Isaac/ballistic_planner.py
CircularHopper_Isaac/rsl_rl_ppo_cfg.py
train.py
play.py
run_train.sh
run_play.sh
migrate_horizon_checkpoint.py
```

The robot model and environment were migrated from the previous Quadhopper / Hopper simulation work. The current implementation uses the corrected spring hopper model instead of the earlier unstable model.

## 3. Simulation and Robot Setup

Framework:

```text
Isaac Lab + RSL-RL PPO
```

Action space:

```text
action_space = 4
```

The four actions represent motor/thrust commands. Actions are clamped to `[-1, 1]` and mapped to `[0, 1]`.

Thrust model:

```python
force = -0.2371 * u**2 + 0.8130 * u + 0.0113
```

Simulation timing:

```text
sim dt = 0.01
decimation = 2
policy dt = 0.02 s
```

Terrain:

```text
plane terrain
static_friction = 1.5
dynamic_friction = 1.5
restitution = 0.0
```

## 4. Circular Trajectory Task

Default circular task configuration:

```text
circle_radius = 2.0
hop_distance = 0.22
planning_horizon_hops = 5
fixed_ccw = True
```

Common training preset:

```text
hop_distance = 0.14
target_tolerance = 0.18 or 0.12
apex_height_ref = 0.18
flight_time_ref = 0.55
max_successful_hops = 8
episode_length_s = 12
```

## 5. Trajectory Planning Evolution

### Early Version

The initial implementation replanned the current hop after every liftoff or touchdown using the robot's actual current position.

Problem:

- The hopper was not following a fixed future trajectory.
- Each jump generated a new local plan.
- The behavior looked like waypoint chasing rather than multi-hop trajectory tracking.

### Current Version

The current implementation generates a fixed multi-hop segment:

```text
segment_horizon = 5
```

At reset, the command generator creates 5 future hops along the circle and caches the waypoints. The current target advances through this cached segment. A new segment is generated only after the current 5-hop segment is completed.

Key state in `trajectory_commands.py`:

```text
segment_waypoints_w: [num_envs, 6, 2]
segment_index
current_start_pos_w
target_pos_w
successful_hops
```

The planner no longer replans from the actual root position at liftoff. Instead, it uses fixed segment start and target positions.

## 6. Ballistic Reference

Each hop uses a simple ballistic reference between a fixed start point and target point.

Horizontal reference:

```text
planned_xy = takeoff_xy + (target_xy - takeoff_xy) * phase
```

Vertical reference:

```text
planned_z = base_z + 4 * phase * (1 - phase) * apex_height_ref
```

Phase:

```text
phase = time_since_liftoff / flight_time_ref
```

Reference horizontal velocity:

```text
v_xy_ref = (target_xy - start_xy) / flight_time_ref
```

Reference vertical velocity:

```text
vz_ref = sqrt(2 * 9.81 * apex_height_ref)
```

## 7. Observation Design

Current observation dimension:

```text
observation_space = 50
```

The original base observation is 40 dimensions. The current version appends 5 future waypoint errors in the robot body frame:

```text
40 + 5 * 2 = 50
```

Observation components:

```text
root linear velocity in body frame: 3
root angular velocity in body frame: 3
root quaternion: 4
root z: 1
root vz: 1
current target error in body frame: 2
reference horizontal velocity error: 2
reference vertical velocity error: 1
radial error to circle: 1
circle tangent direction in body frame: 2
spring joint position: 1
spring joint velocity: 1
contact flag: 1
phase one-hot: 3
time since liftoff and touchdown: 2
action history, 3 frames * 4 actions: 12
future horizon waypoint errors, 5 * 2: 10
```

Small observation noise is added during training:

```text
linear velocity noise = 0.02
angular velocity noise = 0.01
quaternion noise = 0.005
target error noise = 0.02
horizon error noise = 0.02
```

## 8. Jump Phase and Contact Logic

The environment tracks a simple jump phase:

```text
phase = 0: stance / ground
phase = 1: flight
phase = 2: touchdown settling
```

Contact and liftoff use both spring joint state and root motion:

```text
stance_root_height = 0.38
liftoff_root_clearance = 0.06
contact_vz_threshold = 0.20
liftoff_vz_threshold = 0.35
```

Important events:

```text
liftoff_event
touchdown_event
apex_event
target_hit
target_miss
advance_event
```

Current advancement logic:

```text
advance_on_touchdown = True
```

This means stable touchdown advances to the next planned waypoint. Missing the exact target still receives a penalty, but the hopper does not remain stuck on the old target.

## 9. Reward Design

The reward is divided into several groups.

### Survival and Attitude

```text
survival_scale = 0.2
upright_scale = -4.0
flight_attitude_scale = -2.5
```

### Takeoff

```text
hop_velocity_scale = 4.0
stance_relaunch_scale = 8.0
launch_vxy_scale = 14.0
launch_vz_scale = 18.0
```

### Trajectory Tracking

```text
progress_scale = 0.25
radial_path_scale = 1.0
flight_vxy_scale = 12.0
flight_traj_xy_scale = 8.0
flight_traj_height_scale = 28.0
low_flight_penalty_scale = -35.0
apex_height_scale = 70.0
```

Recent change:

- `progress` only applies outside flight.
- During flight, the policy is encouraged to follow fixed ballistic velocity and trajectory.
- In-air attitude and angular velocity are penalized to reduce aggressive mid-air corrections.

### Landing and Target Hit

```text
touchdown_scale = 45.0
target_hit_scale = 120.0
landing_precision_scale = 35.0
landing_error_scale = -45.0
touchdown_miss_scale = -60.0
landing_vxy_scale = -4.0
```

### Action Smoothness

```text
action_rate_scale = -0.35
action_smooth_scale = -0.25
```

### Failure / Stall

```text
stance_stall_scale = -8.0
failure_scale = -40.0
```

## 10. Success and Termination Conditions

Target hit requires:

```text
touchdown_event
stable_landing
landing_error < target_tolerance
```

Stable landing requires:

```text
tilt < landing_tilt_limit
abs(vz) < touchdown_vz_limit
hop height within [min_target_hop_height, max_target_hop_height]
flight time >= min_flight_time_for_hit
under arc error <= max_under_arc_error_for_hit
```

Important thresholds:

```text
landing_tilt_limit = 0.35
touchdown_vz_limit = 1.2
max_hop_height = 0.75
max_workspace_error = 1.4
max_stance_time = 0.65
```

## 11. PPO Configuration

The current policy uses MLP instead of LSTM. LSTM was tested earlier but caused CUDA out-of-memory issues.

Experiment:

```text
experiment_name = circular_hopper_horizon
num_steps_per_env = 128
max_iterations = 2000
save_interval = 100
empirical_normalization = True
```

Actor network:

```text
input = 50
hidden = [256, 128]
output = 4
activation = ELU
init_noise_std = 0.2
```

Critic network:

```text
input = 50
hidden = [256, 256]
output = 1
activation = ELU
```

PPO algorithm:

```text
value_loss_coef = 0.5
clip_param = 0.2
entropy_coef = 0.001
num_learning_epochs = 5
num_mini_batches = 8
learning_rate = 3e-4
schedule = adaptive
gamma = 0.99
lam = 0.95
desired_kl = 0.015
max_grad_norm = 1.0
```

## 12. Training and Play Commands

Training:

```bash
cd /home/terry/Desktop/workspace/Jumping-Robot/Hopper_Circular_Trajectory_Isaac
bash run_train.sh --num_envs 512 --checkpoint latest --task_preset track_feasible --max_iterations 1500
```

Play:

```bash
bash run_play.sh --checkpoint latest --num_envs 1 --task_preset track_feasible --print_state --print_dones
```

Checkpoint options:

```text
--checkpoint latest
--checkpoint migrated
--checkpoint path/to/model.pt
```

When the observation changed from 40 to 50 dimensions, a migration script was added to expand the first actor/critic layer. The new 10 horizon inputs were initialized with zero weights so older checkpoints could still be used for finetuning.

## 13. Current Progress

Completed:

- Created a new Isaac Lab circular trajectory tracking project.
- Migrated corrected hopper model and configuration.
- Replaced unstable early model.
- Switched from LSTM to MLP to avoid GPU memory issues.
- Implemented circular waypoint generation.
- Added target, circle, waypoint, and trajectory arc visualization.
- Added per-hop ballistic reference.
- Expanded observation from 40 to 50 dimensions with 5-hop horizon information.
- Added fixed 5-hop segment generation.
- Removed liftoff-time replanning from actual robot position.
- Added trajectory tracking rewards.
- Added landing precision and miss penalties.
- Added touchdown-based advancement so the hopper does not keep chasing the same old target.
- Added in-air velocity and attitude rewards to reduce excessive aerial correction.

Current observed behavior:

- The hopper can jump normally.
- It can move along the circular path direction.
- It can visualize fixed future waypoints and reference arcs.
- Tracking is better than the earliest waypoint-chasing version.
- However, trajectory tracking is still not fully stable.
- In-air correction is still sometimes too large.
- Landing error often remains around `0.1-0.25 m`.
- Occasional timeout or reset still happens.

## 14. Current Bottlenecks

The current bottleneck is no longer basic hopping. The main issue is accurate and stable fixed-horizon trajectory tracking.

Main problems:

- In-air lateral correction can still be too aggressive.
- `z` and `z_ref` are sometimes not perfectly synchronized.
- Fixed 5-hop planning introduces accumulated error.
- Landing precision is not yet consistent.
- The policy may still inherit some old waypoint-chasing behavior from previous checkpoints.

Likely causes:

- The policy is still learning both planning compensation and control.
- The hopper is underactuated and has limited ability to correct in flight.
- Reward terms for landing accuracy, trajectory tracking, height, and stability are sensitive to weighting.
- The current reference is a simple parabola, not a full dynamics-feasible trajectory optimization result.

## 15. Future Work

Short-term:

- Continue finetuning the fixed-horizon policy.
- Use curriculum learning:
  - single-hop tracking
  - 2-hop horizon
  - 5-hop horizon
  - stricter landing tolerance
- Tune `flight_time_ref` and `apex_height_ref`.
- Monitor TensorBoard metrics:
  - `target_hit_rate`
  - `landing_error_mean`
  - `done_tilt_rate`
  - `done_repeated_miss_rate`
  - `flight_vxy`
  - `flight_attitude`

Mid-term:

- Add full future trajectory samples to observation instead of only future waypoints.
- Use time-indexed reference trajectory tracking reward.
- Add segment-level trajectory error instead of only per-hop reward.
- Add phase-dependent control or explicit phase encoding.
- Use asymmetric critic with privileged state.

Long-term:

- Add MPC or a trajectory optimizer to generate dynamically feasible multi-hop trajectories.
- Use RL mainly as a tracking controller.
- Support arbitrary 2D paths:
  - circle
  - ellipse
  - figure-eight
  - custom user path
- Improve sim-to-real readiness with domain randomization, delay modeling, and system identification.
- Compare:
  - single-step planning
  - receding horizon planning
  - fixed multi-hop horizon planning

## 16. One-Sentence Summary

The project has progressed from basic hopper stabilization to fixed 5-hop horizon circular trajectory tracking in Isaac Lab. The system now includes multi-hop waypoint generation, ballistic references, 50-dimensional horizon observations, visualization, and PPO training; the main remaining challenge is improving fixed-horizon tracking accuracy and reducing aggressive in-air corrections.
