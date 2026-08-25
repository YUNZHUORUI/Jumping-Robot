# Experiment Scripts

This directory keeps research scripts out of the project root while preserving
the stable Quadhopper task packages.

## Layout

- `random_two_hop/`: continuous two-hop planning, Semi-MDP policies, local
  action search, and scorer training.
- `height/`: height residual and specialist experiments built on the circular
  planner teacher.

The project root intentionally keeps only the main task entry points such as
`train.py`, `play.py`, `train_gate.py`, `play_gate.py`,
`train_planner_circular.py`, and `play_planner_circular.py`.

## Current Random Two-Hop Entry Points

Use the Isaac Sim Python executable from the project root:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  experiments/random_two_hop/play_two_hop_semimdp.py \
  --teacher_checkpoint saved_checkpoints/professor_approved_v36_teacher/model_90.pt \
  --checkpoint saved_checkpoints/v57_short020_050_long050_080_balanced/model_466.pt \
  --short_radius_min 0.2 --short_radius_max 0.5 \
  --long_radius_min 0.5 --long_radius_max 0.8 \
  --correction_mode landing \
  --num_envs 1
```

For A/B visualization of the learned second-hop selector:

```bash
/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh \
  experiments/random_two_hop/play_two_hop_semimdp.py \
  --teacher_checkpoint saved_checkpoints/professor_approved_v36_teacher/model_90.pt \
  --checkpoint saved_checkpoints/v57_short020_050_long050_080_balanced/model_466.pt \
  --action_scorer logs/rsl_rl/quadhopper_planner_random_two_hop_v57_second_precision_scorer/scorer_quality_w05.pt \
  --scorer_quality_weight 0.50 \
  --short_radius_min 0.2 --short_radius_max 0.5 \
  --long_radius_min 0.5 --long_radius_max 0.8 \
  --correction_mode landing \
  --num_envs 1
```

Large generated artifacts remain ignored by Git: `logs/`, `outputs/`,
`saved_checkpoints/`, and model checkpoints (`*.pt`, `*.pth`, `*.ckpt`).
