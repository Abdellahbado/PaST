#!/usr/bin/env bash
# ============================================================================
# Experiment B — Profile Complexity Sweep
#
# Goal: Quantify how pricing-profile complexity affects approximation quality
# and cross-profile generalization.
#
# Uses ALL named profiles (daily_tou, flat, two_block, ramp, double_peak,
# weekend_weekday) plus generate_data.
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

LOG_DIR="ADP/logs/exp_profile_sweep"
MODEL_DIR="ADP/models/exp_profile_sweep"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

PMAX=12

# ── Profiles ────────────────────────────────────────────────────────
PROFILES=("daily_tou" "flat" "two_block" "ramp" "double_peak" "weekend_weekday" "generate_data")

# ── Models ──────────────────────────────────────────────────────────
MODELS=("poly" "elasticnet" "lasso" "mlp" "lgbm")

# ── Feature set: enriched + extra for all models ────────────────────
FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"

# ── Small training ──────────────────────────────────────────────────
TRAIN_SEEDS_SMALL="0-199"
SAMPLES_SMALL=400
TRAIN_D_RANGE_SMALL="2-4"
TRAIN_N_RANGE_SMALL="20-60"
TARGET_UTIL_SMALL="0.80"
DP_TIME_LIMIT_SMALL="${DP_TIME_LIMIT_SMALL:-60}"

# ── Eval configs ────────────────────────────────────────────────────
EVAL_SEEDS_SMALL="200-224"
EVAL_SEEDS_MEDIUM="400-414"
EVAL_D_MEDIUM="8"
EVAL_N_MEDIUM="150"

BEAMS="2,5,10"

# ── ElasticNet/LASSO defaults ───────────────────────────────────────
EN_ALPHA="1e-3"
EN_L1_RATIO="0.5"

# ── MLP defaults ────────────────────────────────────────────────────
MLP_EPOCHS=500
MLP_PATIENCE=35

echo "============================================================"
echo " Experiment B — Profile Complexity Sweep"
echo " Profiles: ${PROFILES[*]}"
echo " Models: ${MODELS[*]}"
echo " Total runs: $(( ${#PROFILES[@]} * ${#MODELS[@]} ))"
echo "============================================================"

RUN_IDX=0
TOTAL=$(( ${#PROFILES[@]} * ${#MODELS[@]} ))

for PROFILE in "${PROFILES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    TAG="${PROFILE}_${MODEL}"

    # Model-specific CLI args
    MODEL_ARGS=""
    case "$MODEL" in
      elasticnet) MODEL_ARGS="--elasticnet-alpha $EN_ALPHA --elasticnet-l1-ratio $EN_L1_RATIO" ;;
      lasso)      MODEL_ARGS="--elasticnet-alpha $EN_ALPHA" ;;
      mlp)        MODEL_ARGS="--mlp-max-epochs $MLP_EPOCHS --mlp-patience $MLP_PATIENCE" ;;
      lgbm)       MODEL_ARGS="" ;;
      poly)       MODEL_ARGS="--l2 1e-3" ;;
    esac

    # ── Train on small, eval on small ──────────────────────────────
    OUT_CSV="$LOG_DIR/${TAG}_train_small_eval_small.csv"
    if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
      echo "[$RUN_IDX/$TOTAL] SKIP (exists) $OUT_CSV"
    else
      echo "[$RUN_IDX/$TOTAL] TRAIN small → EVAL small : $TAG"
      $PYTHON_BIN sandbox/eval_pooled_vhat.py \
        --daily-price-profile "$PROFILE" \
        --model-type "$MODEL" \
        --pmax "$PMAX" \
        --train-seeds "$TRAIN_SEEDS_SMALL" \
        --train-D-range "$TRAIN_D_RANGE_SMALL" --train-N-range "$TRAIN_N_RANGE_SMALL" \
        --target-util "$TARGET_UTIL_SMALL" \
        --samples-per-instance "$SAMPLES_SMALL" \
        --label-mode optimal_path \
        --dp-time-limit "$DP_TIME_LIMIT_SMALL" --require-optimal-labels \
        --eval-seeds "$EVAL_SEEDS_SMALL" \
        --beams "$BEAMS" \
        --workers "$WORKERS" \
        --save-model "$MODEL_DIR/vhat_${TAG}.npz" \
        --out-csv "$OUT_CSV" \
        $FEAT_FLAGS $MODEL_ARGS \
        2>&1 | tee "$LOG_DIR/${TAG}_train_small_eval_small.log"
    fi

    # ── Eval on medium (cross-size) ────────────────────────────────
    MODEL_PATH="$MODEL_DIR/vhat_${TAG}.npz"
    OUT_CSV_MED="$LOG_DIR/${TAG}_train_small_eval_medium.csv"
    if [[ -f "$OUT_CSV_MED" && "$RESUME" == "1" ]]; then
      echo "[$RUN_IDX/$TOTAL] SKIP (exists) $OUT_CSV_MED"
    elif [[ ! -f "$MODEL_PATH" ]]; then
      echo "[$RUN_IDX/$TOTAL] SKIP (no model) $OUT_CSV_MED"
    else
      echo "[$RUN_IDX/$TOTAL] EVAL medium (cross-size) : $TAG"
      $PYTHON_BIN sandbox/eval_pooled_vhat.py \
        --daily-price-profile "$PROFILE" \
        --model-type "$MODEL" \
        --pmax "$PMAX" \
        --load-model "$MODEL_PATH" \
        --eval-D "$EVAL_D_MEDIUM" --eval-N "$EVAL_N_MEDIUM" \
        --eval-seeds "$EVAL_SEEDS_MEDIUM" \
        --beams "$BEAMS" \
        --workers "$WORKERS" \
        --dp-time-limit 120 \
        --out-csv "$OUT_CSV_MED" \
        $FEAT_FLAGS $MODEL_ARGS \
        2>&1 | tee "$LOG_DIR/${TAG}_train_small_eval_medium.log"
    fi

  done
done

echo ""
echo "============================================================"
echo " Experiment B complete."
echo " Results: $LOG_DIR/"
echo "============================================================"
