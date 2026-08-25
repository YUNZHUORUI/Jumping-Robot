#!/usr/bin/env bash
# v46 hop-curriculum Semi-MDP PPO training launcher.
#
# Usage:
#   bash run_train_v46_semimdp.sh smoke   # 16 envs, 1 update - validates code + measures frame rate
#   bash run_train_v46_semimdp.sh full    # 256 envs, 40 updates
#
# Stdout is mirrored to logs/v46_semimdp_<stamp>.log; watch it with:
#   tail -f logs/v46_semimdp_*.log
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh
TEACHER=logs/rsl_rl/quadhopper_planner_random_two_hop_v36_full_relative_next_smooth_distance_turn_180_tol_10_spd_030_tilt_06_arw_100_lcg_000_fixed_100/2026-08-23_12-24-40/model_90.pt
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG="logs/v46_semimdp_${STAMP}.log"

case "${1:-full}" in
  smoke)
    ARGS=(--headless --num_envs 16 --updates 1 --transitions_per_update 32
          --curriculum_iterations 30)
    ;;
  full)
    ARGS=(--headless --num_envs 256 --updates 40 --transitions_per_update 256
          --curriculum_iterations 30)
    ;;
  *)
    echo "unknown mode: $1 (use smoke or full)" >&2
    exit 1
    ;;
esac

echo "[LAUNCH] mode=$1 log=$LOG"
echo "[LAUNCH] $PYTHON experiments/random_two_hop/train_two_hop_semimdp.py ${ARGS[*]} --teacher_checkpoint $TEACHER"
exec "$PYTHON" experiments/random_two_hop/train_two_hop_semimdp.py "${ARGS[@]}" --teacher_checkpoint "$TEACHER" 2>&1 | tee "$LOG"
