#!/usr/bin/env bash
set -euo pipefail

cd /mnt/mnt/data/lingyu/openpi

# The supplied W&B credential is currently invalid.  Offline mode still writes
# complete local W&B runs, which can be synchronized later with `wandb sync`.
export WANDB_MODE=offline
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false

tasks=(
  collect_food
  fold_towels
  pick_flowers
  pick_up_paper_rolls
  pick_up_trash
  stack_blocks
)

for task in "${tasks[@]}"; do
  free_kb=$(df -Pk /mnt/mnt/data | awk 'NR==2 {print $4}')
  free_gb=$((free_kb / 1024 / 1024))
  if (( free_gb < 35 )); then
    echo "Stopping before ${task}: only ${free_gb} GiB remain on /mnt/mnt/data" >&2
    exit 2
  fi
  echo "===== START ${task} ($(date -Is)) ====="
  ./.venv/bin/python scripts/train_sed_robot_lora.py "${task}" --overwrite
  echo "===== DONE ${task} ($(date -Is)) ====="
done

echo "All SED Robot PI05 LoRA tasks completed."
