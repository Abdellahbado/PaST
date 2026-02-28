#!/usr/bin/env bash
# ============================================================================
# Experiment A — Noise Stress Test
#
# Goal: Find the noise level at which the method breaks (gap > 1%, 2%, 5%).
# No retraining — noise is applied at evaluation time only via forecast_realized
# price mode.  All trained models are loaded from Experiment B checkpoints.
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-1}"

# ── Workers: auto-scale for HPC (eval-only, small instances = low memory) ──
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
WORKERS="${WORKERS_SMALL:-$(( NCPU > 32 ? 32 : NCPU ))}"

LOG_DIR="ADP/logs/exp_noise_stress"
MODEL_DIR_B="ADP/models/exp_profile_sweep"   # reuse models from Exp B
MODEL_DIR_C="ADP/models/exp_regularization_features"
mkdir -p "$LOG_DIR"

PMAX=12

FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"

# ── Noise grid ──────────────────────────────────────────────────────
SIGMAS=("0.0" "0.25" "0.5" "1.0" "2.0" "4.0")
RHOS=("0.5" "0.9")
# Spike configs: "prob,mag,dur"
SPIKE_CONFIGS=("0,0,0" "0.02,2.0,2" "0.05,4.0,3")

# ── Models & profiles to test ──────────────────────────────────────
PROFILES=("daily_tou" "double_peak")
MODELS=("poly" "elasticnet" "lgbm")

# ── Eval config ─────────────────────────────────────────────────────
EVAL_SEEDS="200-219"
BEAMS="2,5,10"

echo "============================================================"
echo " Experiment A — Noise Stress Test"
echo " Noise combos: $(( ${#SIGMAS[@]} * ${#RHOS[@]} * ${#SPIKE_CONFIGS[@]} )) = 36"
echo " Per-combo models: ${#PROFILES[@]} x ${#MODELS[@]} = $(( ${#PROFILES[@]} * ${#MODELS[@]} ))"
echo "============================================================"

RUN_IDX=0

for PROFILE in "${PROFILES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    # Find model checkpoint from Experiment B
    MODEL_PATH="$MODEL_DIR_B/vhat_${PROFILE}_${MODEL}.npz"
    if [[ ! -f "$MODEL_PATH" ]]; then
      echo "[WARN] Model not found: $MODEL_PATH — skipping $PROFILE/$MODEL"
      continue
    fi

    for SIGMA in "${SIGMAS[@]}"; do
      for RHO in "${RHOS[@]}"; do
        for SPIKE_LINE in "${SPIKE_CONFIGS[@]}"; do
          IFS=',' read -r SPIKE_PROB SPIKE_MAG SPIKE_DUR <<< "$SPIKE_LINE"

          RUN_IDX=$(( RUN_IDX + 1 ))
          TAG="${PROFILE}_${MODEL}_s${SIGMA}_r${RHO}_sp${SPIKE_PROB}"
          OUT_CSV="$LOG_DIR/${TAG}.csv"

          if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
            echo "[$RUN_IDX] SKIP (exists) $OUT_CSV"
            continue
          fi

          NOISE_ARGS="--eval-price-mode forecast_realized"
          NOISE_ARGS="$NOISE_ARGS --eval-price-noise-sigma $SIGMA"
          NOISE_ARGS="$NOISE_ARGS --eval-price-noise-rho $RHO"
          if [[ "$SPIKE_PROB" != "0" ]]; then
            NOISE_ARGS="$NOISE_ARGS --eval-price-spike-prob $SPIKE_PROB"
            NOISE_ARGS="$NOISE_ARGS --eval-price-spike-mag $SPIKE_MAG"
            NOISE_ARGS="$NOISE_ARGS --eval-price-spike-dur $SPIKE_DUR"
          fi

          echo "[$RUN_IDX] $TAG"
          $PYTHON_BIN sandbox/eval_pooled_vhat.py \
            --daily-price-profile "$PROFILE" \
            --model-type "$MODEL" \
            --pmax "$PMAX" \
            --load-model "$MODEL_PATH" \
            --eval-seeds "$EVAL_SEEDS" \
            --beams "$BEAMS" \
            --workers "$WORKERS" \
            --out-csv "$OUT_CSV" \
            $FEAT_FLAGS $NOISE_ARGS \
            2>&1 | tee "$LOG_DIR/${TAG}.log"

        done
      done
    done
  done
done

echo ""
echo "============================================================"
echo " Experiment A complete."
echo " Results: $LOG_DIR/"
echo "============================================================"
