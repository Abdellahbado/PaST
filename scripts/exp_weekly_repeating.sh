#!/usr/bin/env bash
# ============================================================================
# Experiment J — Weekly-Repeating Price Profile (Variable Period)
#
# Goal: Test the method when the cycle length is NOT a single day (H=20)
#   but an entire WEEK (H=140, i.e., 7 unique days × 20 hours).
#   This is a realistic setting: electricity prices often have weekly
#   periodicity (weekday/weekend patterns) but differ day-to-day.
#
# Sub-experiments (each at two scales):
#   J1-small:  Train weekly → Eval weekly (small, D=7-14, so 1-2 weeks)
#   J1-medium: Train weekly → Eval weekly (medium, D=14-21, 2-3 weeks)
#   J2:        Train daily-repeating → Eval weekly (transfer, medium)
#   J3:        Train weekly-small → Eval weekly-medium (cross-size)
#
# Expected: The method should work naturally since the existing feature
#   extraction just uses a longer cycle.  But medium instances (D=14-21)
#   become T=280-420, hitting DP harder → learning value is higher.
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-1}"
WORKERS="${WORKERS:-16}"

LOG_DIR="ADP/logs/exp_weekly"
MODEL_DIR="ADP/models/exp_weekly"
MODEL_DIR_DAILY="ADP/models/exp_hard_profile_sweep"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

TARGET_UTIL="0.95"
FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"
LABEL="--label-mode optimal_path --optimal-path-n-paths 2"
DP_TIME_LIMIT="${DP_TIME_LIMIT:-180}"

MODELS=("mlp" "elasticnet" "poly")
BEAMS="5,10,20"

TRAIN_SEEDS="0-59"
EVAL_SEEDS="200-224"
SAMPLES=5000

# ── Size configs (D must be multiples-of-7 friendly) ────────────────────
# Small: 1-2 weeks  (D in 7-14, T in 140-280)
SMALL_D="7" ; SMALL_N="30" ; SMALL_PMAX="5"
SMALL_TRAIN_D_RANGE="7-14" ; SMALL_TRAIN_N_RANGE="15-60"
SMALL_EVAL_D_RANGE="7-14"  ; SMALL_EVAL_N_RANGE="15-60"

# Medium: 2-3 weeks (D in 14-21, T in 280-420)
MED_D="14" ; MED_N="60" ; MED_PMAX="8"
MED_TRAIN_D_RANGE="14-21" ; MED_TRAIN_N_RANGE="30-100"
MED_EVAL_D_RANGE="14-21"  ; MED_EVAL_N_RANGE="30-100"

EN_ALPHA="1e-3" ; EN_L1_RATIO="0.5"
MLP_EPOCHS=500  ; MLP_PATIENCE=35

echo "============================================================"
echo " Experiment J — Weekly-Repeating (H_cycle=140)"
echo " Models: ${MODELS[*]}"
echo "============================================================"

run_one() {
  local TAG="$1" SIZE="$2" MODEL="$3"

  if [[ "$SIZE" == "small" ]]; then
    D=$SMALL_D; N=$SMALL_N; PMAX=$SMALL_PMAX
    TDR=$SMALL_TRAIN_D_RANGE; TNR=$SMALL_TRAIN_N_RANGE
    EDR=$SMALL_EVAL_D_RANGE;  ENR=$SMALL_EVAL_N_RANGE
  else
    D=$MED_D; N=$MED_N; PMAX=$MED_PMAX
    TDR=$MED_TRAIN_D_RANGE; TNR=$MED_TRAIN_N_RANGE
    EDR=$MED_EVAL_D_RANGE;  ENR=$MED_EVAL_N_RANGE
  fi

  MODEL_ARGS=""
  case "$MODEL" in
    elasticnet) MODEL_ARGS="--elasticnet-alpha $EN_ALPHA --elasticnet-l1-ratio $EN_L1_RATIO" ;;
    mlp)        MODEL_ARGS="--mlp-max-epochs $MLP_EPOCHS --mlp-patience $MLP_PATIENCE" ;;
    poly)       MODEL_ARGS="--l2 1e-3" ;;
  esac

  OUT_CSV="$LOG_DIR/${TAG}.csv"
  if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
    echo "  SKIP (exists) $OUT_CSV"; return 0
  fi

  echo "  RUN $TAG : $MODEL / $SIZE"
  $PYTHON_BIN sandbox/eval_pooled_vhat.py \
    --daily-price-profile weekly_repeating \
    --model-type "$MODEL" \
    --D "$D" --N "$N" --pmax "$PMAX" \
    --target-util "$TARGET_UTIL" \
    --train-seeds "$TRAIN_SEEDS" \
    --train-D-range "$TDR" --train-N-range "$TNR" \
    --samples-per-instance "$SAMPLES" \
    $LABEL \
    --dp-time-limit "$DP_TIME_LIMIT" --require-optimal-labels \
    --eval-seeds "$EVAL_SEEDS" \
    --eval-D-range "$EDR" --eval-N-range "$ENR" \
    --beams "$BEAMS" \
    --workers "$WORKERS" \
    --save-model "$MODEL_DIR/vhat_${TAG}.npz" \
    --out-csv "$OUT_CSV" \
    $FEAT_FLAGS $MODEL_ARGS \
    2>&1 | tee "$LOG_DIR/${TAG}.log"
}

run_eval_only() {
  local TAG="$1" SIZE="$2" MODEL="$3" MODEL_PATH="$4"

  if [[ "$SIZE" == "small" ]]; then
    D=$SMALL_D; N=$SMALL_N; PMAX=$SMALL_PMAX
    EDR=$SMALL_EVAL_D_RANGE; ENR=$SMALL_EVAL_N_RANGE
  else
    D=$MED_D; N=$MED_N; PMAX=$MED_PMAX
    EDR=$MED_EVAL_D_RANGE; ENR=$MED_EVAL_N_RANGE
  fi

  OUT_CSV="$LOG_DIR/${TAG}.csv"
  if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
    echo "  SKIP (exists) $OUT_CSV"; return 0
  fi

  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "  SKIP (no model) $MODEL_PATH"; return 0
  fi

  echo "  EVAL $TAG : $MODEL / $SIZE"
  $PYTHON_BIN sandbox/eval_pooled_vhat.py \
    --daily-price-profile weekly_repeating \
    --model-type "$MODEL" \
    --D "$D" --N "$N" --pmax "$PMAX" \
    --target-util "$TARGET_UTIL" \
    --load-model "$MODEL_PATH" \
    --eval-seeds "$EVAL_SEEDS" \
    --eval-D-range "$EDR" --eval-N-range "$ENR" \
    --beams "$BEAMS" \
    --workers "$WORKERS" \
    --out-csv "$OUT_CSV" \
    $FEAT_FLAGS \
    2>&1 | tee "$LOG_DIR/${TAG}.log"
}

# ────────────────────────────────────────────────────────────────────────
# J1: Train weekly → Eval weekly  (small + medium)
# ────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> J1: Train weekly → Eval weekly"
for SIZE in small medium; do
  for MODEL in "${MODELS[@]}"; do
    run_one "J1_${SIZE}_${MODEL}" "$SIZE" "$MODEL"
  done
done

# ────────────────────────────────────────────────────────────────────────
# J2: Eval daily-repeating-trained models on weekly (zero-shot transfer)
# ────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> J2: Zero-shot daily → weekly"
DAILY_PROFILES=("ramp" "double_peak" "daily_tou")
for SRC in "${DAILY_PROFILES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    MP="$MODEL_DIR_DAILY/vhat_${SRC}_${MODEL}.npz"
    run_eval_only "J2_${SRC}_${MODEL}" "medium" "$MODEL" "$MP"
  done
done

# ────────────────────────────────────────────────────────────────────────
# J3: Train weekly-small → Eval weekly-medium (cross-size generalization)
# ────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> J3: Cross-size weekly: train small → eval medium"
for MODEL in "${MODELS[@]}"; do
  MP="$MODEL_DIR/vhat_J1_small_${MODEL}.npz"
  run_eval_only "J3_small_to_med_${MODEL}" "medium" "$MODEL" "$MP"
done

echo ""
echo "============================================================"
echo " Experiment J complete.  Results: $LOG_DIR/"
echo "============================================================"
