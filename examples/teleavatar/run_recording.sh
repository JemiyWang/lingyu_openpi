#!/bin/bash
# Wrapper script for teleavatar recording with LeRobot support
# This script ensures the correct Python environment is used

# Source ROS2
source /opt/ros/humble/setup.bash

# Set rosdomain (if needed)
rosdomain 2>/dev/null || true

# Option 1: Use conda environment's Python directly
# Uncomment and modify the path to match your system
CONDA_PYTHON="/home/lingyu/anaconda3/envs/lerobot/bin/python"

# Add lerobot to PYTHONPATH (for system python3 fallback)
export PYTHONPATH="/home/lingyu/project/lerobot:$PYTHONPATH"

# Option 2: Activate conda environment and use python
# Uncomment these lines if Option 1 doesn't work
# source /home/lingyu/anaconda3/etc/profile.d/conda.sh
# conda activate lerobot
# CONDA_PYTHON="python"

# Check if conda python exists
if [ -x "$CONDA_PYTHON" ]; then
    echo "[INFO] Using Python: $CONDA_PYTHON"
    exec $CONDA_PYTHON "$@"
else
    echo "[ERROR] Conda Python not found at: $CONDA_PYTHON"
    echo "[FALLBACK] Using system python3 (LeRobot may not be available)"
    exec python3 "$@"
fi
