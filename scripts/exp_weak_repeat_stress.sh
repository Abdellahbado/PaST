#!/usr/bin/env bash
# ============================================================================
# Experiment F — Weakly Repeating Price Stress (Drifting Amplitude)
#
# Goal: Test how the learned model degrades as the repeating-profile assumption
# weakens. The realized prices have day-to-day amplitude drift while the
# forecast (used for model features) stays as the original repeating profile.
#
# Design: AR(1) multiplicative drift per day:
#   realized_day_d = forecast_day * (1 + alpha_d)
#   alpha_d = rho * alpha_{d-1} + sigma * z_d
#
# As drift_sigma increases, realized prices diverge more from forecast →
# the learned model's feature context becomes increasingly wrong.
# Price-Vhat should degrade before learned-Vhat (learned may be more robust
# due to having seen varied price structures during training).
#
# Eval-only: loads models from Exp B-hard. No retraining.
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

LOG_DIR="ADP/logs/exp_weak_repeat_stress"
MODEL_DIR="ADP/models/exp_hard_profile_sweep"
mkdir -p "$LOG_DIR"

PMAX=12
TARGET_UTIL="0.95"
FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"

# ── Drift grid ──────────────────────────────────────────────────────────
# sigma: amplitude of per-day drift (0.0 = pure repeating)
# rho: autocorrelation (0.9 = persistent drift, 0.0 = independent)
DRIFT_SIGMAS=("0.0" "0.05" "0.10" "0.20" "0.40" "0.60")
DRIFT_RHOS=("0.0" "0.9")

# ── Profiles × Models ──────────────────────────────────────────────────
# Focus on profiles where L showed advantage over P
PROFILES=("ramp" "double_peak" "daily_tou")
MODELS=("mlp" "elasticnet")

# ── Eval config: HARD instances ─────────────────────────────────────────
EVAL_SEEDS="200-219"
EVAL_D_RANGE="3-5"
EVAL_N_RANGE="8-20"
BEAMS="5,10,20"

echo "============================================================"
echo " Experiment F — Weakly Repeating Price Stress"
echo " Drift sigmas: ${DRIFT_SIGMAS[*]}"
echo " Drift rhos: ${DRIFT_RHOS[*]}"
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

    for DRIFT_SIGMA in "${DRIFT_SIGMAS[@]}"; do
      for DRIFT_RHO in "${DRIFT_RHOS[@]}"; do
        RUN_IDX=$(( RUN_IDX + 1 ))
        TAG="${PROFILE}_${MODEL}_dsig${DRIFT_SIGMA}_drho${DRIFT_RHO}"
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
          --eval-price-mode drifting_amplitude \
          --eval-price-drift-sigma "$DRIFT_SIGMA" \
          --eval-price-drift-rho "$DRIFT_RHO" \
          --out-csv "$OUT_CSV" \
          $FEAT_FLAGS \
          2>&1 | tee "$LOG_DIR/${TAG}.log"

      done
    done
  done
done

echo ""
echo "============================================================"
echo " Experiment F complete."
echo " Results: $LOG_DIR/"
echo "============================================================"
