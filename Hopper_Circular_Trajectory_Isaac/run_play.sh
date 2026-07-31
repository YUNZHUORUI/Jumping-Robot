#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ISAAC_PYTHON="${ISAAC_PYTHON:-/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh}"
PYTHON_BIN="${PYTHON_BIN:-${ISAAC_PYTHON}}"

"${PYTHON_BIN}" play.py "$@"
