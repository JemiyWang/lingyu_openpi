#!/bin/bash
# --config-name pi0_teleavatar
cd /mnt/mnt/data/lingyu/openpi
source ./.venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
uv run scripts/compute_norm_stats.py \
    --config-name pi0_teleavatar_norm_stats_compute 
    
    