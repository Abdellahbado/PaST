#!/usr/bin/env bash
# ============================================================================
# Experiment B2 — Corrected Daily-Repeating Baseline (variable sizes)
#
# Re-runs the core repeating experiments at BOTH small AND medium scales,
# using optimal_path labels + all feature enrichments, to serve as a
# properly matched baseline for experiments I (NR-honest) and J (weekly).
#
# Profiles tested: ramp, double_peak, daily_tou, generate_data
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-1}"
WORKERS_SMALL="${WORKERS_SMALL:-64}"
WORKERS_MEDIUM="${WORKERS_MEDIUM:-16}"

LOG_DIR="ADP/logs/exp_repeating_v2"
MODEL_DIR="ADP/models/exp_repeating_v2"
DATA_DIR="ADP/data/exp_repeating_v2"
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$DATA_DIR"

TARGET_UTIL="0.95"
FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"
LABEL_SMALL="--label-mode subproblem"
LABEL_MEDIUM="--label-mode optimal_path --optimal-path-n-paths 2 --optimal-path-topup-max -1 --optimal-path-topup-dp-time-limit 1.0"
DP_TIME_LIMIT="${DP_TIME_LIMIT:-120}"

MODELS=("mlp" "poly" "poly_mlp" "factored_mlp")
PROFILES=("ramp" "double_peak" "daily_tou" "generate_data")
BEAMS="5,10,20"

TRAIN_SEEDS="0-299"
EVAL_SEEDS="500-549"
SAMPLES=1000

# ── Size configs ─────────────────────────────────────────────────────
SMALL_D="4" ; SMALL_N="20" ; SMALL_PMAX="5"
SMALL_TRAIN_D_RANGE="3-6"  ; SMALL_TRAIN_N_RANGE="10-40"
SMALL_EVAL_D_RANGE="3-6"   ; SMALL_EVAL_N_RANGE="10-40"

MED_D="8" ; MED_N="40" ; MED_PMAX="8"
MED_TRAIN_D_RANGE="6-10"   ; MED_TRAIN_N_RANGE="20-60"
MED_EVAL_D_RANGE="6-10"    ; MED_EVAL_N_RANGE="20-60"

EN_ALPHA="1e-3" ; EN_L1_RATIO="0.5"
MLP_EPOCHS=500  ; MLP_PATIENCE=35

echo "============================================================"
echo " Experiment B2 — Repeating Baseline (small + medium)"
echo " Profiles: ${PROFILES[*]}"
echo " Models: ${MODELS[*]}"
echo "============================================================"

# Collect training data once per (profile, size) and cache to DATA_DIR.
collect_data() {
  local PROFILE="$1" SIZE="$2"
  local CACHE="$DATA_DIR/pooled_${PROFILE}_${SIZE}.npz"

  if [[ -f "$CACHE" && "$RESUME" == "1" ]]; then
    echo "  DATA (cached) $CACHE"; return 0
  fi

  local D N PMAX TDR TNR LABEL W
  if [[ "$SIZE" == "small" ]]; then
    D=$SMALL_D; N=$SMALL_N; PMAX=$SMALL_PMAX
    TDR=$SMALL_TRAIN_D_RANGE; TNR=$SMALL_TRAIN_N_RANGE
    LABEL="$LABEL_SMALL"; W="$WORKERS_SMALL"
  else
    D=$MED_D; N=$MED_N; PMAX=$MED_PMAX
    TDR=$MED_TRAIN_D_RANGE; TNR=$MED_TRAIN_N_RANGE
    LABEL="$LABEL_MEDIUM"; W="$WORKERS_MEDIUM"
  fi

  echo "  DATA $PROFILE/$SIZE  (workers=$W, samples=$SAMPLES/inst) → $CACHE"
  $PYTHON_BIN sandbox/eval_pooled_vhat.py \
    --daily-price-profile "$PROFILE" \
    --model-type mlp \
    --D "$D" --N "$N" --pmax "$PMAX" \
    --target-util "$TARGET_UTIL" \
    --train-seeds "$TRAIN_SEEDS" \
    --train-D-range "$TDR" --train-N-range "$TNR" \
    --samples-per-instance "$SAMPLES" \
    $LABEL \
    --dp-time-limit "$DP_TIME_LIMIT" --require-optimal-labels \
    --workers "$W" \
    --save-pooled-data "$CACHE" \
    --save-pooled-data-only \
    $FEAT_FLAGS \
    2>&1 | tee "$DATA_DIR/collect_${PROFILE}_${SIZE}.log"
}

# Train one model using cached pooled data + run beam eval.
run_one() {
  local TAG="$1" SIZE="$2" MODEL="$3" PROFILE="$4"
  local CACHE="$DATA_DIR/pooled_${PROFILE}_${SIZE}.npz"

  local D N PMAX EDR ENR W
  if [[ "$SIZE" == "small" ]]; then
    D=$SMALL_D; N=$SMALL_N; PMAX=$SMALL_PMAX
    EDR=$SMALL_EVAL_D_RANGE; ENR=$SMALL_EVAL_N_RANGE
    W="$WORKERS_SMALL"
  else
    D=$MED_D; N=$MED_N; PMAX=$MED_PMAX
    EDR=$MED_EVAL_D_RANGE; ENR=$MED_EVAL_N_RANGE
    W="$WORKERS_MEDIUM"
  fi

  MODEL_ARGS=""
  case "$MODEL" in
    mlp|poly_mlp|factored_mlp) MODEL_ARGS="--mlp-max-epochs $MLP_EPOCHS --mlp-patience $MLP_PATIENCE" ;;
    poly) MODEL_ARGS="--l2 1e-3" ;;
  esac

  OUT_CSV="$LOG_DIR/${TAG}.csv"
  if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
    echo "  SKIP (exists) $OUT_CSV"; return 0
  fi

  echo "  RUN $TAG : $PROFILE / $MODEL / $SIZE  (from cache)"
  $PYTHON_BIN sandbox/eval_pooled_vhat.py \
    --daily-price-profile "$PROFILE" \
    --model-type "$MODEL" \
    --D "$D" --N "$N" --pmax "$PMAX" \
    --target-util "$TARGET_UTIL" \
    --train-seeds "$TRAIN_SEEDS" \
    --load-pooled-data "$CACHE" \
    --eval-seeds "$EVAL_SEEDS" \
    --eval-D-range "$EDR" --eval-N-range "$ENR" \
    --beams "$BEAMS" \
    --workers "$W" \
    --save-model "$MODEL_DIR/vhat_${TAG}.npz" \
    --out-csv "$OUT_CSV" \
    $FEAT_FLAGS $MODEL_ARGS \
    2>&1 | tee "$LOG_DIR/${TAG}.log"
}

# Cross-size: train small → eval medium
run_cross_size() {
  local TAG="$1" MODEL="$2" MODEL_PATH="$3" PROFILE="$4"

  OUT_CSV="$LOG_DIR/${TAG}.csv"
  if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
    echo "  SKIP (exists) $OUT_CSV"; return 0
  fi
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "  SKIP (no model) $MODEL_PATH"; return 0
  fi

  echo "  CROSS $TAG : $MODEL  (small→medium, workers=$WORKERS_MEDIUM)"
  $PYTHON_BIN sandbox/eval_pooled_vhat.py \
    --daily-price-profile "$PROFILE" \
    --model-type "$MODEL" \
    --D "$MED_D" --N "$MED_N" --pmax "$MED_PMAX" \
    --target-util "$TARGET_UTIL" \
    --load-model "$MODEL_PATH" \
    --eval-seeds "$EVAL_SEEDS" \
    --eval-D-range "$MED_EVAL_D_RANGE" --eval-N-range "$MED_EVAL_N_RANGE" \
    --beams "$BEAMS" \
    --workers "$WORKERS_MEDIUM" \
    --out-csv "$OUT_CSV" \
    $FEAT_FLAGS \
    2>&1 | tee "$LOG_DIR/${TAG}.log"
}

# ── Main loop ───────────────────────────────────────────────────────
for PROFILE in "${PROFILES[@]}"; do
  echo ""
  echo ">>> Profile: $PROFILE"
  for SIZE in small medium; do
    # Collect data once for this (profile, size), shared by all models
    collect_data "$PROFILE" "$SIZE"
    for MODEL in "${MODELS[@]}"; do
      run_one "B2_${PROFILE}_${SIZE}_${MODEL}" "$SIZE" "$MODEL" "$PROFILE"
    done
  done
  # Cross-size
  for MODEL in "${MODELS[@]}"; do
    MP="$MODEL_DIR/vhat_B2_${PROFILE}_small_${MODEL}.npz"
    run_cross_size "B2_${PROFILE}_cross_${MODEL}" "$MODEL" "$MP" "$PROFILE"
  done
done

echo ""
echo "============================================================"
echo " Experiment B2 complete.  Results: $LOG_DIR/"
echo "============================================================"
