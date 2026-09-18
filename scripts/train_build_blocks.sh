#!/bin/bash

cd /mnt/mnt/data/lingyu/openpi
source ./.venv/bin/activate
wandb login 727ad8430431f284b7d54606b4b820a70b350a56

# 防止 OpenBLAS/OMP 多线程初始化冲突导致 Segmentation fault
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python scripts/train.py pi0_teleavatar \
    --exp_name=finetune_build_blocks_20fps \
    --batch_size=64 \
    --num_train_steps=20000 \
    --save_interval=2000 \
    --weight_loader.params_path=/mnt/mnt/data/behavior/checkpoints/pi0_base/params \
    --num_workers=16 \
    --fsdp_devices=8 \
    --overwrite


