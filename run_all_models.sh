#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Common parameters
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
      if ! conda run -n new-ml-env python -c "import lightgbm" >/dev/null 2>&1; then
        echo "    [skip] lightgbm is not installed in conda env 'new-ml-env'"
        echo ">>> lgbm skipped."
        continue
      fi
    fi
    
    # 1. Train on N=35 pmax=5 + Eval on Same Size (N=40)
    echo "    Training & Eval (Same Size)..."
    conda run -n new-ml-env python sandbox/eval_pooled_vhat.py \
      --D $TRAIN_D --N $TRAIN_N --pmax $TRAIN_PMAX \
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
    conda run -n new-ml-env python sandbox/eval_pooled_vhat.py \
      --load-model "$MODEL_DIR/vhat_$MODEL.npz" \
      --eval-seeds "$EVAL_SEEDS" \
      --eval-D $EVAL_D --eval-N $EVAL_LARGE_N --eval-pmax $TRAIN_PMAX \
      --beams 2,5 \
      --out-csv "$LOG_DIR/eval_${MODEL}_large.csv" \
      2>&1 | tee "$LOG_DIR/run_${MODEL}_large.log"
      
    echo ">>> $MODEL done."
done

echo "Done. Results in $LOG_DIR"
