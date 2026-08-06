#!/bin/bash
set -euo pipefail

ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$ISAAC_SIM_ROOT/python.sh" "$SCRIPT_DIR/train_gate.py" \
  --headless --stage "${1:-0}" --num_envs "${2:-256}" "${@:3}"
