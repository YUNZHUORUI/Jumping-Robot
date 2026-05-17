#!/bin/bash
# Usage:
#   bash run_train.sh                       # 256 envs, from scratch
#   bash run_train.sh 1024                  # 1024 envs, from scratch
#   bash run_train.sh 256 --resume          # 256 envs, resume from latest checkpoint
#   bash run_train.sh 256 --resume --checkpoint path/to/model_999.pt
set -e
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "ERROR: in conda env '$CONDA_DEFAULT_ENV' — run 'conda deactivate' first."
    echo "       Isaac Sim's bundled Python segfaults when conda's libs are on LD_LIBRARY_PATH."
    exit 1
fi
ISAAC=/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64
SCRIPT=$(dirname "$0")/train.py
NUM_ENVS="${1:-256}"
"$ISAAC/python.sh" "$SCRIPT" --headless --num_envs "$NUM_ENVS" "${@:2}"
