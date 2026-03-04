#!/usr/bin/env bash
# ============================================================================
# Experiment I — Non-Repeating with NR-Honest Features
#
# Goal: The corrected non-repeating experiment.  Features are now built from
#   the ACTUAL full-trajectory prices (honest regime, exact window costs,
#   forward-looking distances, local Fourier) instead of the old day-1
#   copy-and-repeat proxy.
#
# Sub-experiments (each at two scales):
#   I1-small:  Train NR → Eval NR (small instances, D=3-6, N=10-40)
#   I1-medium: Train NR → Eval NR (medium instances, D=6-10, N=20-60)
#   I2-small:  Train repeating → Eval NR (zero-shot transfer, small)
#   I2-medium: Train repeating → Eval NR (zero-shot transfer, medium)
#   I3:        Train NR-small → Eval NR-medium (cross-size generalization)
#
# Expected: I1 should outperform old NR (experiment H) because features
#   actually reflect the true price trajectory.  I2 zero-shot may degrade.
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

LOG_DIR="ADP/logs/exp_nr_honest"
MODEL_DIR="ADP/models/exp_nr_honest"
MODEL_DIR_REP="ADP/models/exp_hard_profile_sweep"   # repeating models for I2
DATA_DIR="ADP/data/exp_nr_honest"
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$DATA_DIR"

TARGET_UTIL="0.95"
FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"
LABEL_SMALL="--label-mode subproblem"
LABEL_MEDIUM="--label-mode optimal_path --optimal-path-n-paths 2 --optimal-path-topup-max -1 --optimal-path-topup-dp-time-limit 1.0"
DP_TIME_LIMIT="${DP_TIME_LIMIT:-120}"

MODELS=("mlp" "poly" "poly_mlp" "factored_mlp")
BEAMS="2,5,10"

TRAIN_SEEDS="0-299"
EVAL_SEEDS="500-549"
SAMPLES=1000

# ── Size configs ─────────────────────────────────────────────────────────
# Small
SMALL_D="4" ; SMALL_N="20" ; SMALL_PMAX="5"
SMALL_TRAIN_D_RANGE="3-6"  ; SMALL_TRAIN_N_RANGE="10-40"
SMALL_EVAL_D_RANGE="3-6"   ; SMALL_EVAL_N_RANGE="10-40"

# Medium
MED_D="8" ; MED_N="40" ; MED_PMAX="8"
MED_TRAIN_D_RANGE="6-10"   ; MED_TRAIN_N_RANGE="20-60"
MED_EVAL_D_RANGE="6-10"    ; MED_EVAL_N_RANGE="20-60"

# ── Model-specific defaults ────────────────────────────────────────────
EN_ALPHA="1e-3" ; EN_L1_RATIO="0.5"
MLP_EPOCHS=500  ; MLP_PATIENCE=35

echo "============================================================"
echo " Experiment I — NR-Honest Features (corrected non-repeating)"
echo " Models: ${MODELS[*]}"
echo "============================================================"

# Collect training data once per size and cache to DATA_DIR.
collect_data() {
  local SIZE="$1"
  local CACHE="$DATA_DIR/pooled_non_repeating_${SIZE}.npz"

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

  echo "  DATA non_repeating/$SIZE  (workers=$W, samples=$SAMPLES/inst) → $CACHE"
  $PYTHON_BIN sandbox/eval_pooled_vhat.py \
    --daily-price-profile non_repeating \
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
    2>&1 | tee "$DATA_DIR/collect_non_repeating_${SIZE}.log"
}

# Helper: train one model using cached pooled data + run beam eval.
run_one() {
  local TAG="$1" SIZE="$2" MODEL="$3"
  local CACHE="$DATA_DIR/pooled_non_repeating_${SIZE}.npz"

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

  echo "  RUN $TAG : $MODEL / $SIZE  (from cache)"
  $PYTHON_BIN sandbox/eval_pooled_vhat.py \
    --daily-price-profile non_repeating \
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

# Helper: eval-only using a loaded model
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
    --daily-price-profile non_repeating \
    --model-type "$MODEL" \
    --D "$D" --N "$N" --pmax "$PMAX" \
    --target-util "$TARGET_UTIL" \
    --load-model "$MODEL_PATH" \
    --eval-seeds "$EVAL_SEEDS" \
    --eval-D-range "$EDR" --eval-N-range "$ENR" \
    --beams "$BEAMS" \
    --workers "$WORKERS_MEDIUM" \
    --out-csv "$OUT_CSV" \
    $FEAT_FLAGS \
    2>&1 | tee "$LOG_DIR/${TAG}.log"
}

# ──────────────────────────────────────────────────────────────────────────
# I1: Train NR → Eval NR  (small + medium)
# ────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> I1: Train NR-honest → Eval NR-honest"
for SIZE in small medium; do
  # Collect data once per size, shared by all models
  collect_data "$SIZE"
  for MODEL in "${MODELS[@]}"; do
    run_one "I1_${SIZE}_${MODEL}" "$SIZE" "$MODEL"
  done
done

# ────────────────────────────────────────────────────────────────────────
# I2: Eval repeating-trained models on NR (zero-shot)
# ────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> I2: Zero-shot repeating → NR-honest"
REPEAT_PROFILES=("ramp" "double_peak" "daily_tou")
for SIZE in small medium; do
  for SRC in "${REPEAT_PROFILES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      MP="$MODEL_DIR_REP/vhat_${SRC}_${MODEL}.npz"
      run_eval_only "I2_${SIZE}_${SRC}_${MODEL}" "$SIZE" "$MODEL" "$MP"
    done
  done
done

# ────────────────────────────────────────────────────────────────────────
# I3: Train NR-small → Eval NR-medium (cross-size generalization)
# ────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> I3: Cross-size NR: train small → eval medium"
for MODEL in "${MODELS[@]}"; do
  MP="$MODEL_DIR/vhat_I1_small_${MODEL}.npz"
  run_eval_only "I3_small_to_med_${MODEL}" "medium" "$MODEL" "$MP"
done

echo ""
echo "============================================================"
echo " Experiment I complete.  Results: $LOG_DIR/"
echo "============================================================"
