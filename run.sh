#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/wsl/.conda/envs/GraphEEG/bin/python}"
DATA_PATH="${DATA_PATH:-./data/features/}"
GPU="${GPU:-0}"
SEED="${SEED:-666}"

if [[ "${DATA_PATH}" != */ ]]; then
  DATA_PATH="${DATA_PATH}/"
fi

echo "python=${PYTHON}"
echo "data-path=${DATA_PATH}"
echo "gpu=${GPU} seed=${SEED}"

for LAB in A V L; do
  TAG="EAGLE-${LAB}-s${SEED}"
  echo "========== ${TAG} $(date -Iseconds) =========="
  "${PYTHON}" -u main.py \
    --data-path "${DATA_PATH}" \
    --data-using DEAP \
    --dataset DEAP \
    --label-type "${LAB}" \
    --random-seed "${SEED}" \
    --gpu "${GPU}" \
    --run-tag "${TAG}"
done

echo "========== ALL DONE $(date -Iseconds) =========="
