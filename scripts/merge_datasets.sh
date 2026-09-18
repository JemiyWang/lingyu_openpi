#!/bin/bash

cd /mnt/mnt/data/lingyu/openpi
source ./.venv/bin/activate
uv run scripts/merge_lerobot_datasets.py \
    --source-dir /mnt/mnt/data/Real_FPF_workspace/datasets/food/backup \
    --pattern "labeled_*" \
    --output /mnt/mnt/data/Real_FPF_workspace/datasets/food/labeled_staged_food