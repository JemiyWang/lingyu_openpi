#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="/mnt/mnt/data/lingyu/openpi"
cd "$OPENPI_ROOT"
mkdir -p logs

export WANDB_MODE=offline
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export OPENPI_DATALOADER_START_METHOD=forkserver
export OPENPI_SED_DECODE_THREADS=14
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 TASK [TASK ...]" >&2
  exit 2
fi

for task in "$@"; do
  log="logs/sed_robot_${task}.log"
  echo "===== START ${task} $(date --iso-8601=seconds) =====" >> "$log"
  ./.venv/bin/python scripts/train_sed_robot_lora.py "$task" --overwrite >> "$log" 2>&1
  echo "===== DONE ${task} $(date --iso-8601=seconds) =====" >> "$log"
done
