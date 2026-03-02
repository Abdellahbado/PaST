#!/usr/bin/env bash
# ============================================================================
# Experiment G — Forecast Bias Stress
#
# Goal: Test robustness to SYSTEMATIC forecast errors, not just random noise.
# Real-world price forecasts often have:
#   - Peak underprediction (peak prices forecast too low)
#   - Peak timing errors (peak window shifted by 1-2 hours)
#
# The learned model receives the BIASED forecast for feature extraction, but
# the DP costs are computed on TRUE prices. A robust model should still
# outperform the price heuristic because:
#   1. Price-Vhat uses the SAME biased forecast as its cost estimate
#   2. Learned-Vhat captures ordering patterns that persist under bias
#
# Design:
#   - bias_factor ∈ {0.0, 0.1, 0.2, 0.3, 0.5}: fraction by which peak (top
#     tertile) prices are underpredicted in the forecast
#   - bias_shift ∈ {0, 1, 2}: number of slots the peak window is shifted
#
# Eval-only: loads models from Exp B-hard.
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-1}"
WORKERS="${WORKERS:-2}"

LOG_DIR="ADP/logs/exp_forecast_bias"
MODEL_DIR="ADP/models/exp_hard_profile_sweep"
mkdir -p "$LOG_DIR"

PMAX=12
TARGET_UTIL="0.95"
FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"

# ── Bias grid ───────────────────────────────────────────────────────────
BIAS_FACTORS=("0.0" "0.1" "0.2" "0.3" "0.5")
BIAS_SHIFTS=("0" "1" "2")

# ── Profiles × Models ──────────────────────────────────────────────────
PROFILES=("ramp" "double_peak" "daily_tou")
MODELS=("mlp" "elasticnet")

# ── Eval config: HARD instances ─────────────────────────────────────────
EVAL_SEEDS="200-219"
EVAL_D_RANGE="3-5"
EVAL_N_RANGE="8-20"
BEAMS="5,10,20"

echo "============================================================"
echo " Experiment G — Forecast Bias Stress"
echo " Bias factors: ${BIAS_FACTORS[*]}"
echo " Bias shifts: ${BIAS_SHIFTS[*]}"
echo " Profiles: ${PROFILES[*]}"
echo " Models: ${MODELS[*]}"
echo " Eval: D∈[$EVAL_D_RANGE] N∈[$EVAL_N_RANGE] util=$TARGET_UTIL"
echo "============================================================"

RUN_IDX=0

for PROFILE in "${PROFILES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="$MODEL_DIR/vhat_${PROFILE}_${MODEL}.npz"
    if [[ ! -f "$MODEL_PATH" ]]; then
      echo "[WARN] Model not found: $MODEL_PATH — skipping"
      continue
    fi

    for BIAS_FACTOR in "${BIAS_FACTORS[@]}"; do
      for BIAS_SHIFT in "${BIAS_SHIFTS[@]}"; do
        RUN_IDX=$(( RUN_IDX + 1 ))
        TAG="${PROFILE}_${MODEL}_bf${BIAS_FACTOR}_bs${BIAS_SHIFT}"
        OUT_CSV="$LOG_DIR/${TAG}.csv"

        if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
          echo "[$RUN_IDX] SKIP (exists) $OUT_CSV"
          continue
        fi

        echo "[$RUN_IDX] $TAG"
        $PYTHON_BIN sandbox/eval_pooled_vhat.py \
          --daily-price-profile "$PROFILE" \
          --model-type "$MODEL" \
          --pmax "$PMAX" \
          --target-util "$TARGET_UTIL" \
          --load-model "$MODEL_PATH" \
          --eval-seeds "$EVAL_SEEDS" \
          --eval-D-range "$EVAL_D_RANGE" --eval-N-range "$EVAL_N_RANGE" \
          --beams "$BEAMS" \
          --workers "$WORKERS" \
          --eval-price-mode forecast_bias \
          --eval-price-bias-factor "$BIAS_FACTOR" \
          --eval-price-bias-shift "$BIAS_SHIFT" \
          --out-csv "$OUT_CSV" \
          $FEAT_FLAGS \
          2>&1 | tee "$LOG_DIR/${TAG}.log"

      done
    done
  done
done

echo ""
echo "============================================================"
echo " Experiment G complete."
echo " Results: $LOG_DIR/"
echo "============================================================"
