#!/usr/bin/env bash
# ============================================================================
# Experiment D — Combined Noise + Profile Stress
#
# Goal: Find the practical operating envelope — the (profile, noise) region
# where guided beam DP remains within 5% of optimal.
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

LOG_DIR="ADP/logs/exp_noise_profile_combined"
MODEL_DIR="ADP/models/exp_profile_sweep"
mkdir -p "$LOG_DIR"

PMAX=12

FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"

# ── Stress grid ─────────────────────────────────────────────────────
PROFILES=("two_block" "ramp" "double_peak" "weekend_weekday")
SIGMAS=("1.0" "2.0" "4.0")
RHO="0.9"
# extreme spikes
SPIKE_PROB="0.05"
SPIKE_MAG="4.0"
SPIKE_DUR="3"

MODELS=("poly" "elasticnet" "lgbm")

EVAL_SEEDS="200-219"
BEAMS="2,5,10"

echo "============================================================"
echo " Experiment D — Combined Noise + Profile Stress"
echo " Grid: ${#PROFILES[@]} profiles x ${#SIGMAS[@]} sigmas x ${#MODELS[@]} models"
echo " Total: $(( ${#PROFILES[@]} * ${#SIGMAS[@]} * ${#MODELS[@]} )) runs"
echo "============================================================"

RUN_IDX=0

for PROFILE in "${PROFILES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="$MODEL_DIR/vhat_${PROFILE}_${MODEL}.npz"
    if [[ ! -f "$MODEL_PATH" ]]; then
      echo "[WARN] Model not found: $MODEL_PATH — skipping"
      continue
    fi

    for SIGMA in "${SIGMAS[@]}"; do
      RUN_IDX=$(( RUN_IDX + 1 ))
      TAG="${PROFILE}_${MODEL}_s${SIGMA}"
      OUT_CSV="$LOG_DIR/${TAG}.csv"

      if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
        echo "[$RUN_IDX] SKIP (exists) $OUT_CSV"
        continue
      fi

      echo "[$RUN_IDX] $TAG"
      $PYTHON_BIN -m PaST.sandbox.eval_pooled_vhat \
        --daily-price-profile "$PROFILE" \
        --model-type "$MODEL" \
        --pmax "$PMAX" \
        --load-model "$MODEL_PATH" \
        --eval-seeds "$EVAL_SEEDS" \
        --beams "$BEAMS" \
        --workers "$WORKERS" \
        --eval-price-mode forecast_realized \
        --eval-price-noise-sigma "$SIGMA" \
        --eval-price-noise-rho "$RHO" \
        --eval-price-spike-prob "$SPIKE_PROB" \
        --eval-price-spike-mag "$SPIKE_MAG" \
        --eval-price-spike-dur "$SPIKE_DUR" \
        --out-csv "$OUT_CSV" \
        $FEAT_FLAGS \
        2>&1 | tee "$LOG_DIR/${TAG}.log"

    done
  done
done

echo ""
echo "============================================================"
echo " Experiment D complete."
echo " Results: $LOG_DIR/"
echo "============================================================"
