#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN)}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN)}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m PaST.train_q_sequence \
  --smoke_test \
  --device cpu \
  --num_cpu_threads 0 \
  --num_collection_workers 0 \
  --num_dataloader_workers 0 \
  --collection_batch_size 16
