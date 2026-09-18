#!/bin/bash
# Alternative wrapper: Set PYTHONPATH to include lerobot
# NOTE: Only works if system python3 version matches conda environment (both Python 3.10)

# Source ROS2
source /opt/ros/humble/setup.bash

# Set rosdomain
rosdomain 2>/dev/null || true

# LeRobot paths (editable install + site-packages)
export PYTHONPATH="/home/lingyu/project/lerobot:$PYTHONPATH"
export PYTHONPATH="/home/lingyu/VLA-project/lerobot:$PYTHONPATH"
export PYTHONPATH="/home/lingyu/anaconda3/envs/lerobot/lib/python3.10/site-packages:$PYTHONPATH"

# Print debug info
echo "[INFO] PYTHONPATH set to include lerobot"
echo "[INFO] System Python: $(python3 --version)"

# Run the script
exec python3 "$@"
