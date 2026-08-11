#!/bin/bash
set -euo pipefail

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_train_variable_height.sh {low|high|alternate} NUM_ENVS [train options]"
  echo "Example: bash run_train_variable_height.sh low 64 --iterations 20 --checkpoint logs/.../model.pt"
  exit 2
fi

HEIGHT_STAGE="$1"
NUM_ENVS="$2"
shift 2

if [[ "$HEIGHT_STAGE" != "low" && "$HEIGHT_STAGE" != "high" && "$HEIGHT_STAGE" != "alternate" ]]; then
  echo "height stage must be low, high, or alternate"
  exit 2
fi

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_planner_circular.py" \
  --headless --height_stage "$HEIGHT_STAGE" --num_envs "$NUM_ENVS" "$@"
