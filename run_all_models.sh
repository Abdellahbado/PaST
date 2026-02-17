#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Be conservative on shared/HPC nodes: avoid thread oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# -----------------------------
# nohup launcher (default on)
#
# By default, the script re-launches itself under nohup in the background so
# terminal/SSH disconnects won't stop the run.
#
# Disable:
#   NOHUP=0 bash PaST/run_all_models.sh
# -----------------------------

if [[ "${NOHUP:-1}" == "1" && -z "${IN_NOHUP:-}" ]]; then
  cd "$(dirname "$0")"
  mkdir -p "ADP/logs/nohup"
  TS="$(date +"%Y%m%d_%H%M%S")"
  LOG_FILE="ADP/logs/nohup/$(basename "$0" .sh)_${TS}.log"
  echo "[nohup] launching in background; log: $LOG_FILE"
  IN_NOHUP=1 NOHUP=0 nohup bash "$0" "$@" >"$LOG_FILE" 2>&1 &
  echo "[nohup] pid=$!"
  exit 0
fi

# Common parameters
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-16}"
TRAIN_SEEDS="0-49"          # 50 instances
SAMPLES=4000                # x 4000 samples = 200k total
TRAIN_N=35
TRAIN_D=8
TRAIN_PMAX=5

EVAL_SEEDS="300-319"        # 20 held-out instances
EVAL_SAME_N=40
EVAL_LARGE_N=60
EVAL_D=12

LOG_DIR="ADP/logs/mixed_models"
MODEL_DIR="ADP/models/mixed_models"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

echo "========================================================================"
echo "TRAINING & EVALUATING ALL MODELS ON MIXED SIZES (pmax=$TRAIN_PMAX)"
echo "========================================================================"

for MODEL in linear poly mlp lgbm; do
    echo ""
    echo ">>> Running $MODEL model..."

    if [[ "$MODEL" == "lgbm" ]]; then
      if ! "$PYTHON_BIN" -c "import lightgbm" >/dev/null 2>&1; then
        echo "    [skip] lightgbm is not installed for $PYTHON_BIN"
        echo ">>> lgbm skipped."
        continue
      fi
    fi
    
    # 1. Train on N=35 pmax=5 + Eval on Same Size (N=40)
    echo "    Training & Eval (Same Size)..."
    "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
      --D $TRAIN_D --N $TRAIN_N --pmax $TRAIN_PMAX \
      --workers "$WORKERS" \
      --train-seeds "$TRAIN_SEEDS" --samples-per-instance $SAMPLES \
      --eval-seeds "$EVAL_SEEDS" \
      --eval-D $EVAL_D --eval-N $EVAL_SAME_N --eval-pmax $TRAIN_PMAX \
      --transferable-features --normalize --normalize-labels \
      --model-type $MODEL --beams 2,5 \
      --save-model "$MODEL_DIR/vhat_$MODEL.npz" \
      --out-csv "$LOG_DIR/eval_${MODEL}_same.csv" \
      2>&1 | tee "$LOG_DIR/run_${MODEL}_same.log"

    # 2. Eval on Larger Size (N=60) using the SAVED model
    echo "    Eval on Larger Size (N=$EVAL_LARGE_N)..."
    "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
      --load-model "$MODEL_DIR/vhat_$MODEL.npz" \
      --workers "$WORKERS" \
      --eval-seeds "$EVAL_SEEDS" \
      --eval-D $EVAL_D --eval-N $EVAL_LARGE_N --eval-pmax $TRAIN_PMAX \
      --beams 2,5 \
      --out-csv "$LOG_DIR/eval_${MODEL}_large.csv" \
      2>&1 | tee "$LOG_DIR/run_${MODEL}_large.log"
      
    echo ">>> $MODEL done."
done

echo "Done. Results in $LOG_DIR"
