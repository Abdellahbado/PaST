#!/usr/bin/env bash
# ============================================================================
# Experiment C — Regularization & Feature Enrichment Comparison
#
# Goal: Compare Ridge vs LASSO vs ElasticNet on degree-2 polynomials with
# varying feature richness.  Train on small instances, evaluate on small
# and medium to measure generalization.
#
# Profile: daily_tou only (isolates regularization effect from profile).
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

LOG_DIR="ADP/logs/exp_regularization_features"
MODEL_DIR="ADP/models/exp_regularization_features"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

PROFILE="daily_tou"
PMAX=12

# ── Training config ─────────────────────────────────────────────────
TRAIN_SEEDS="0-199"        # 200 small instances
SAMPLES=400                # per instance
TRAIN_D_RANGE="2-4"
TRAIN_N_RANGE="20-60"
TARGET_UTIL="0.80"

# ── Eval configs ────────────────────────────────────────────────────
EVAL_SEEDS_SMALL="200-224"
EVAL_SEEDS_MEDIUM="400-414"
EVAL_D_MEDIUM="8"
EVAL_N_MEDIUM="150"

BEAMS="2,5,10"

DP_TIME_LIMIT="${DP_TIME_LIMIT:-120}"

# ── Feature configurations ──────────────────────────────────────────
declare -A FEAT_CONFIGS
FEAT_CONFIGS[default]=""
FEAT_CONFIGS[all_flags]="--feat-len-hist --feat-price-shape --feat-meta"
FEAT_CONFIGS[all_extra]="--feat-len-hist --feat-price-shape --feat-meta --feat-extra"
FEAT_CONFIGS[transferable_all]="--transferable-features --feat-len-hist --feat-price-shape --feat-meta --feat-extra"

# ── Regularization configurations ───────────────────────────────────
# format: "model-type|extra-cli-args"
declare -a REG_CONFIGS=(
  "poly|--l2 1e-3"
  "poly|--l2 1e-1"
  "elasticnet|--elasticnet-alpha 1e-3 --elasticnet-l1-ratio 0.5"
  "elasticnet|--elasticnet-alpha 1e-2 --elasticnet-l1-ratio 0.5"
  "lasso|--elasticnet-alpha 1e-3"
  "lasso|--elasticnet-alpha 1e-2"
)

echo "============================================================"
echo " Experiment C — Regularization & Feature Enrichment"
echo " Feature configs: ${!FEAT_CONFIGS[@]}"
echo " Regularization configs: ${#REG_CONFIGS[@]}"
echo " Total runs: $(( ${#FEAT_CONFIGS[@]} * ${#REG_CONFIGS[@]} ))"
echo "============================================================"

RUN_IDX=0
TOTAL=$(( ${#FEAT_CONFIGS[@]} * ${#REG_CONFIGS[@]} ))

for FEAT_NAME in "${!FEAT_CONFIGS[@]}"; do
  FEAT_FLAGS="${FEAT_CONFIGS[$FEAT_NAME]}"

  for REG_LINE in "${REG_CONFIGS[@]}"; do
    IFS='|' read -r MODEL_TYPE REG_FLAGS <<< "$REG_LINE"
    REG_TAG="${MODEL_TYPE}_$(echo "$REG_FLAGS" | sed 's/[- ]/_/g; s/__*/_/g; s/^_//; s/_$//')"

    RUN_IDX=$(( RUN_IDX + 1 ))
    TAG="${FEAT_NAME}__${REG_TAG}"

    # ── Train on small, eval on small ────────────────────────────
    OUT_CSV="$LOG_DIR/${TAG}_train_small_eval_small.csv"
    if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
      echo "[$RUN_IDX/$TOTAL] SKIP (exists) $OUT_CSV"
    else
      echo "[$RUN_IDX/$TOTAL] TRAIN small → EVAL small : $TAG"
      $PYTHON_BIN sandbox/eval_pooled_vhat.py \
        --daily-price-profile "$PROFILE" \
        --model-type "$MODEL_TYPE" \
        --normalize --normalize-labels \
        --pmax "$PMAX" \
        --train-seeds "$TRAIN_SEEDS" \
        --train-D-range "$TRAIN_D_RANGE" --train-N-range "$TRAIN_N_RANGE" \
        --target-util "$TARGET_UTIL" \
        --samples-per-instance "$SAMPLES" \
        --label-mode optimal_path \
        --dp-time-limit "$DP_TIME_LIMIT" --require-optimal-labels \
        --eval-seeds "$EVAL_SEEDS_SMALL" \
        --beams "$BEAMS" \
        --workers "$WORKERS" \
        --save-model "$MODEL_DIR/vhat_${TAG}.npz" \
        --out-csv "$OUT_CSV" \
        $FEAT_FLAGS $REG_FLAGS \
        2>&1 | tee "$LOG_DIR/${TAG}_train_small_eval_small.log"
    fi

    # ── Eval same model on medium (cross-size generalization) ────
    OUT_CSV_MED="$LOG_DIR/${TAG}_train_small_eval_medium.csv"
    MODEL_PATH="$MODEL_DIR/vhat_${TAG}.npz"
    if [[ -f "$OUT_CSV_MED" && "$RESUME" == "1" ]]; then
      echo "[$RUN_IDX/$TOTAL] SKIP (exists) $OUT_CSV_MED"
    elif [[ ! -f "$MODEL_PATH" ]]; then
      echo "[$RUN_IDX/$TOTAL] SKIP (no model) $OUT_CSV_MED"
    else
      echo "[$RUN_IDX/$TOTAL] EVAL medium (cross-size) : $TAG"
      $PYTHON_BIN sandbox/eval_pooled_vhat.py \
        --daily-price-profile "$PROFILE" \
        --model-type "$MODEL_TYPE" \
        --normalize --normalize-labels \
        --pmax "$PMAX" \
        --load-model "$MODEL_PATH" \
        --eval-D "$EVAL_D_MEDIUM" --eval-N "$EVAL_N_MEDIUM" \
        --eval-seeds "$EVAL_SEEDS_MEDIUM" \
        --beams "$BEAMS" \
        --workers "$WORKERS" \
        --dp-time-limit "$DP_TIME_LIMIT" \
        --out-csv "$OUT_CSV_MED" \
        $FEAT_FLAGS $REG_FLAGS \
        2>&1 | tee "$LOG_DIR/${TAG}_train_small_eval_medium.log"
    fi

  done
done

echo ""
echo "============================================================"
echo " Experiment C complete."
echo " Results: $LOG_DIR/"
echo "============================================================"
