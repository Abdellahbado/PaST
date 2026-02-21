#!/bin/bash
set -euo pipefail

# Run from this script's directory
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="ADP/logs/deploy_epsilon_profiles_all_sizes"
MODEL_DIR="ADP/models/deploy_epsilon_profiles_all_sizes"

mkdir -p "$LOG_DIR" "$MODEL_DIR"

CATEGORY="medium"
PROFILE="daily_tou"
MODEL="factored_mlp"

CKPT="$MODEL_DIR/vhat_${CATEGORY}_${PROFILE}_${MODEL}.npz"
POOLED_CSV="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.csv"
POOLED_LOG="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.log"

EPS_CSV="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.csv"
EPS_LOG="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.log"

# The pre-collected dataset
POOLED_DATA="ADP/Data/Pooled Medium Daily Optimal Path Training Data.npz"

if [[ ! -f "$POOLED_DATA" ]]; then
    echo "Error: Pooled data file not found at $POOLED_DATA"
    exit 1
fi

echo "========================================================================="
echo ">>> TRAIN pooled: category=$CATEGORY model=$MODEL profile=$PROFILE"
echo "========================================================================="

"$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
  --model-type "$MODEL" \
  --transferable-features --normalize --normalize-labels \
  --pmax 12 --target-util 0.85 \
  --train-seeds "0-199" --eval-seeds "400-429" \
  --train-N-range "100-200" --train-D-range "5-15" \
  --eval-N-range "100-200" --eval-D-range "5-15" \
  --label-mode "optimal_path" \
  --daily-price-profile "$PROFILE" \
  --load-pooled-data "$POOLED_DATA" \
  --save-model "$CKPT" \
  --out-csv "$POOLED_CSV" \
  --mlp-max-epochs 600 --mlp-patience 40 \
  --mlp-batch-size 4096 --mlp-lr 1e-3 \
  --beams 2,5 \
  --eval-time-limit 30.0 \
  2>&1 | tee "$POOLED_LOG"

echo ""
echo "========================================================================="
echo ">>> DEPLOY epsilon-sim: category=$CATEGORY model=$MODEL profile=$PROFILE"
echo "========================================================================="

"$PYTHON_BIN" sandbox/eval_epsilon_constraint_sim.py \
  --category "$CATEGORY" \
  --N 40 --N-range "100-200" \
  --M 5 --M-range "8-16" \
  --D 3 --D-range "5-15" \
  --replicates 8 --seed 123 \
  --pmax 12 --target-util 0.85 \
  --price-mode "$PROFILE" \
  --daily-price-profile "$PROFILE" \
  --assign-alpha 1.2 --assign-uniform-mix 0.15 \
  --guided --beam 80 --prune-factor 2.0 \
  --load-model "$CKPT" \
  --transferable-features --normalize \
  --eps-dp-time-limit 300 \
  --out-csv "$EPS_CSV" \
  2>&1 | tee "$EPS_LOG"

echo "Done. Logs: $LOG_DIR  Models: $MODEL_DIR"
