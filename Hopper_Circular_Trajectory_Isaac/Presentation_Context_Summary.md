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

## 16. Connection to the Successful 2D Hopper

The Isaac circular hopping task is strongly related to the earlier successful 2D QuadHopper target-jumping task.

The 2D version can be treated as the conceptual prototype for the current Isaac trajectory-tracking work. In 2D, the robot already demonstrated that a jumping task can be learned more reliably if the problem is decomposed into:

```text
1. Plan a ballistic trajectory to the next target.
2. During stance, generate the correct takeoff velocity and takeoff angle.
3. During flight, mainly preserve attitude and prepare for touchdown.
4. At touchdown, check whether the foot landed close to the target.
5. Advance to the next target.
```

This is directly relevant to the Isaac version because the current Isaac hopper also needs to avoid simply chasing the current target point in the air. Instead, it should follow a planned ballistic segment.

### 2D Branch / Code Location

The successful 2D implementation is in the 2D QuadHopper code:

```text
Hopper_simu/Quadhopper
```

The related branch is:

```text
feature/cuda-2d-quadhopper-simulation
```

Main files:

```text
Hopper_simu/Quadhopper/config.py
Hopper_simu/Quadhopper/env.py
Hopper_simu/Quadhopper/trajectory.py
Hopper_simu/Quadhopper/reward.py
Hopper_simu/Quadhopper/torch_env.py
Hopper_simu/Quadhopper/torch_ppo.py
```

### 2D Physical Model

The 2D hopper is a pitch-plane model with two thrust inputs:

```text
action_space = 2
```

The two actions correspond to left-side and right-side thrust. Each action is mapped from `[-1, 1]` to motor command `[0, 1]`.

Important physical parameters:

```text
dt = 0.01 s
mass = 0.180 kg
rotor_span = 0.1626 m
leg_length = 0.30 m
stroke_length = 0.08 m
inertia = 7.667e-4 kg m^2
gravity = 9.81 m/s^2
```

The thrust curve is matched to the real / Isaac motor model:

```text
F(u) = thrust_a * u^2 + thrust_b * u + thrust_c, for u <= 0.64
F(u) = thrust_offset_high + (u - 0.64) * thrust_slope_high, for u > 0.64
```

Parameters:

```text
thrust_a = -0.9715
thrust_b = 1.2578
thrust_c = -0.0577
thrust_breakpoint = 0.64
thrust_offset_high = 0.349
thrust_slope_high = 0.6139
thrust_max_per_motor = 0.6 N
n_motors_per_side = 2
```

Motor lag:

```text
motor_tau = 0.065 s
motor_tau_min = 0.055 s
motor_tau_max = 0.075 s
```

The stance phase uses a SLIP-style spring model:

```text
use_slip_stance = True
k_slip = 400.0 N/m
c_slip = 1.5 N s/m
spring_preload = 1.8 N
min_stance_substeps = 8
```

The 2D simulation also includes a gentle flight attitude assist:

```text
flight_attitude_assist = True
flight_att_kp = 0.04
flight_att_kd = 0.018
flight_att_tau_limit = 0.020 N m
```

### 2D Target Task

The 2D task is target jumping along the x direction.

Default target setup:

```text
target_count = 6
target_spacing = 0.5 m
targets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
target_tolerance = 0.15 m
```

Episode and reset:

```text
max_episode_steps = 1500
reset_mode = ground
ground_init_leg_compression = 0.050 m
ground_init_theta range = [-6 deg, 6 deg]
```

The successful training commonly used a five-target version:

```bash
python -m Quadhopper.train --mode train --backend torch --device cuda --target-count 5 --timesteps 8000000 --model artifacts/models/ppo_quadhopper_v7_cuda_five_hop
```

Evaluation:

```bash
python -m Quadhopper.train --mode eval --backend torch --device cuda --target-count 5 --episodes 512 --model artifacts/models/ppo_quadhopper_v7_cuda_five_hop
```

Rendering:

```bash
python -m Quadhopper.train --mode test --backend torch --device cuda --target-count 5 --test-steps 800 --model artifacts/models/ppo_quadhopper_v7_cuda_five_hop
```

### 2D Observation Design

The 2D environment uses a 15-dimensional observation:

```text
observation_space = 15
```

Observation components:

```text
0  theta           body pitch angle
1  dtheta          pitch angular velocity
2  l_norm          normalized leg length, l / l_nominal - 1
3  dl              leg extension velocity
4  vx_com          COM horizontal velocity
5  vy_com          COM vertical velocity
6  com_y           COM height
7  height_err      target jump height - COM height
8  vx_deficit      planned vx - actual vx, stance only
9  vy_deficit      planned vy - actual vy, stance only
10 dx_target       horizontal distance from foot to current target
11 is_touching     contact flag
12 task_phase      reserved, currently 0
13 theta_err_td    theta error to touchdown target angle
14 stance_ratio    consecutive stance steps / max stance steps
```

Important point:

```text
The 2D policy observes the planned velocity deficit during stance.
```

This is one of the main ideas borrowed by the Isaac version. In Isaac, the policy now observes:

```text
vxy_ref_err
vz_ref_err
future horizon waypoint errors
```

### 2D Ballistic Planner

The 2D planner computes a ballistic path from the current COM position to the next target.

It first converts foot target to COM target using the expected touchdown angle:

```text
x_target_com = x_target_foot - leg_length * sin(landing_theta)
```

Then it solves for a ballistic launch velocity:

```text
vx_nom
vy_nom
theta_opt = atan2(vy_nom, vx_nom)
```

The planner uses a higher-than-minimum-energy trajectory to reduce aggressive touchdown compression:

```text
traj_apex_scale = 1.22
traj_apex_clearance = 0.22
traj_apex_height = 1.0
traj_min_vx = 0.25
traj_max_vx = 8.0
```

The generated trajectory is:

```text
y(x) = a * dx^2 + b * dx + y0
```

with:

```text
a = -g / (2 * vx_nom^2)
b = vy_nom / vx_nom
```

The stance attitude target is linked to the ballistic launch angle:

```text
takeoff_theta_target = pi / 2 - theta_opt
```

This is a key difference from simple waypoint chasing: the 2D controller learns to create the correct takeoff condition, not to keep steering toward the target during flight.

### 2D Reward Design

The 2D reward is centered around the liftoff event. The main training signal is whether the robot produces the correct ballistic launch velocity.

Reward groups:

#### Liftoff Event

The reward checks the actual COM velocity at liftoff:

```text
v0_actual = sqrt(vx_com^2 + vy_com^2)
v0_nom = sqrt(vx_nom^2 + vy_nom^2)
alpha_actual = atan2(vy_com, vx_com)
alpha_nom = atan2(vy_nom, vx_nom)
```

Launch angle sector:

```text
alpha_min_deg = 5.0
alpha_max_deg = 85.0
```

Reward weights:

```text
liftoff_v_weight = 80.0
liftoff_v_sharpness = 10.0
liftoff_angle_weight = 20.0
liftoff_angle_sharpness = 4.0
```

This reward encourages the robot to produce the correct ballistic initial condition at takeoff.

#### Stance Phase

The stance reward encourages an inverted-pendulum and spring-loading motion:

```text
stance_pendulum_weight = 3.0
stance_spring_weight = 1.0
stance_theta_pos_weight = 2.0
stance_stall_penalty = 0.05
stance_timeout_penalty = 100.0
```

Specific behavior:

- reward `dtheta > 0`
- reward leg compression when `theta < 0`
- reward fast leg extension when `theta > 0`
- penalize staying in stance too long

#### Touchdown Event

Touchdown target angle:

```text
phi_td_target_deg = -3.0
touchdown_weight = 40.0
touchdown_sharpness = 8.0
touchdown_bad_penalty = 20.0
```

The reward encourages a slight backward lean at touchdown so the next stance starts with a good attack angle.

#### Flight Phase

Flight reward is deliberately lighter than liftoff reward:

```text
flight_attitude_weight = 2.0
flight_attitude_sharpness = 3.0
flight_thrust_penalty = 0.15
flight_height_weight = 1.0
flight_height_sharpness = 6.0
overheight_penalty_weight = 18.0
target_height = 1.0
```

The policy is not encouraged to aggressively chase the target in flight. It mainly maintains height and prepares touchdown attitude.

#### Apex Height

```text
apex_height_weight = 50.0
apex_height_sharpness = 20.0
apex_height_error_penalty = 150.0
```

#### Dense Progress and Landing

```text
forward_progress_weight = 0.35
backward_progress_penalty = 0.20
landing_proximity_weight = 30.0
landing_proximity_sharpness = 8.0
```

#### Target Success and Termination

```text
target_hit_reward = 150.0
all_targets_bonus = 300.0
termination_penalty = 60.0
out_of_bounds_penalty = 60.0
max_tilt_rad = 1.1
max_height = 1.7
min_height = 0.04
max_overshoot = 0.4
```

### 2D PPO Configuration

The successful 2D version supports a CUDA-native PPO backend where physics rollout, GAE, and PPO update remain on GPU.

Training config:

```text
total_timesteps = 8,000,000
learning_rate = 3e-4
n_steps = 2048
batch_size = 256
ent_coef = 0.02
gamma = 0.995
```

CUDA backend:

```text
cuda_n_envs = 2048
cuda_rollout_steps = 128
cuda_batch_size = 8192
cuda_update_epochs = 10
cuda_hidden_size = 128
cuda_ent_coef = 0.001
cuda_vf_coef = 0.5
cuda_clip_range = 0.2
cuda_gae_lambda = 0.95
cuda_max_grad_norm = 0.5
cuda_log_std_init = -1.5
cuda_reward_scale = 0.01
```

Network:

```text
Actor: 15 -> 128 -> 128 -> 2
Critic: 15 -> 128 -> 128 -> 1
Activation: Tanh
Policy distribution: Gaussian
```

### What Isaac Can Borrow from 2D

The 2D success suggests that the Isaac version should focus less on direct in-air target chasing and more on generating correct takeoff conditions.

Specific ideas to borrow:

- Use liftoff velocity matching as a stronger reward term.
- Reward takeoff angle sector instead of only landing distance.
- Treat flight as mostly ballistic, with light attitude correction only.
- Use touchdown angle / landing attitude to prepare the next hop.
- Use multi-hop success as a sequence objective, not independent single-hop targets.
- Add full reference trajectory state to observation, not only future waypoints.
- Curriculum from 1-hop to 2-hop to 5-hop fixed horizon.

The current Isaac changes already move in this direction:

```text
flight_vxy reward
fixed 5-hop segment
ballistic planner
future waypoint observation
reduced in-flight progress reward
flight attitude penalty
```

The next step is to make Isaac's reward more explicitly match the successful 2D principle:

```text
During stance: learn to create the correct launch velocity.
During flight: track the planned ballistic arc with minimal correction.
At touchdown: land close and prepare the next stance.
```

## 17. One-Sentence Summary

The project has progressed from basic hopper stabilization to fixed 5-hop horizon circular trajectory tracking in Isaac Lab. The system now includes multi-hop waypoint generation, ballistic references, 50-dimensional horizon observations, visualization, and PPO training; the main remaining challenge is improving fixed-horizon tracking accuracy and reducing aggressive in-air corrections.
