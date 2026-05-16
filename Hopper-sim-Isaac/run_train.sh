#!/bin/bash
ISAAC=/home/terry/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64
SCRIPT=$(dirname "$0")/train.py
$ISAAC/python.sh "$SCRIPT" --headless --num_envs "${1:-256}"
