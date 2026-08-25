#!/bin/bash
set -euo pipefail

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# The current random-route campaign holds both apex commands at 1.0 m.
# Pass --height_stage alternate explicitly to opt back into alternating heights.
DEFAULT_HEIGHT_STAGE="${RANDOM_ROUTE_HEIGHT_STAGE:-high}"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_train_random_curriculum.sh {direction|medium|short|bridge|full} NUM_ENVS [train options]"
  exit 2
fi

DISTANCE_STAGE="$1"
NUM_ENVS="$2"
shift 2

if [[ "$DISTANCE_STAGE" != "direction" && "$DISTANCE_STAGE" != "medium" && "$DISTANCE_STAGE" != "short" && "$DISTANCE_STAGE" != "bridge" && "$DISTANCE_STAGE" != "full" ]]; then
  echo "distance stage must be direction, medium, short, bridge, or full"
  exit 2
fi

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_planner_circular.py" \
  --headless --route random_two_hop --height_stage "$DEFAULT_HEIGHT_STAGE" \
  --distance_stage "$DISTANCE_STAGE" --num_envs "$NUM_ENVS" "$@"
