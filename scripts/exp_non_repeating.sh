#!/usr/bin/env bash
# ============================================================================
# Experiment H — Non-Repeating Price Profile
#
# Goal: Test the method when the key repeating-profile assumption is BROKEN.
# Each day has a DIFFERENT random price structure (generate_data.py style).
#
# Two sub-experiments:
#   H1: Train on non-repeating → Eval on non-repeating (full adaptation)
#   H2: Eval B-hard models on non-repeating (zero-shot, no retraining)
#
# Expected: H1 should work reasonably (model adapts to general features),
# but H2 should degrade significantly since the TOU regime features become
# meaningless. This quantifies how much the method relies on price structure.
#
# Instance generation: same difficulty as B-hard (util=0.95, pmax=12),
# but --daily-price-profile non_repeating replaces the repeating day with
# independent random days using generate_data.py-style interval sampling.
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

LOG_DIR="ADP/logs/exp_non_repeating"
MODEL_DIR_H="ADP/models/exp_non_repeating"
MODEL_DIR_B="ADP/models/exp_hard_profile_sweep"
mkdir -p "$LOG_DIR" "$MODEL_DIR_H"

PMAX=12
TARGET_UTIL="0.95"
FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"

# ── Models ──────────────────────────────────────────────────────────────
MODELS=("mlp" "elasticnet" "poly")

# ── Training: HARD instances with NON-REPEATING prices ──────────────────
TRAIN_SEEDS="0-199"
SAMPLES=5000
TRAIN_D_RANGE="3-5"
TRAIN_N_RANGE="8-20"
DP_TIME_LIMIT="${DP_TIME_LIMIT:-120}"

# ── Eval config ─────────────────────────────────────────────────────────
EVAL_SEEDS="200-224"
EVAL_D_RANGE="3-5"
EVAL_N_RANGE="8-20"
BEAMS="5,10,20"

# ── Model-specific defaults ─────────────────────────────────────────────
EN_ALPHA="1e-3"
EN_L1_RATIO="0.5"
MLP_EPOCHS=500
MLP_PATIENCE=35

echo "============================================================"
echo " Experiment H — Non-Repeating Price Profile"
echo " Models: ${MODELS[*]}"
echo " Train+Eval: D∈[$TRAIN_D_RANGE] N∈[$TRAIN_N_RANGE] util=$TARGET_UTIL"
echo "============================================================"

# ────────────────────────────────────────────────────────────────────────
# H1: Train on non-repeating → Eval on non-repeating
# ────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> H1: Train on non-repeating → Eval on non-repeating"
echo ""

RUN_IDX=0
for MODEL in "${MODELS[@]}"; do
  RUN_IDX=$(( RUN_IDX + 1 ))
  TAG="nonrep_${MODEL}"

  MODEL_ARGS=""
  case "$MODEL" in
    elasticnet) MODEL_ARGS="--elasticnet-alpha $EN_ALPHA --elasticnet-l1-ratio $EN_L1_RATIO" ;;
    mlp)        MODEL_ARGS="--mlp-max-epochs $MLP_EPOCHS --mlp-patience $MLP_PATIENCE" ;;
    poly)       MODEL_ARGS="--l2 1e-3" ;;
  esac

  OUT_CSV="$LOG_DIR/${TAG}_train_eval.csv"
  if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
    echo "[H1-$RUN_IDX] SKIP (exists) $OUT_CSV"
  else
    echo "[H1-$RUN_IDX] TRAIN+EVAL non-repeating : $MODEL"
    $PYTHON_BIN sandbox/eval_pooled_vhat.py \
      --daily-price-profile non_repeating \
      --model-type "$MODEL" \
      --pmax "$PMAX" \
      --target-util "$TARGET_UTIL" \
      --train-seeds "$TRAIN_SEEDS" \
      --train-D-range "$TRAIN_D_RANGE" --train-N-range "$TRAIN_N_RANGE" \
      --samples-per-instance "$SAMPLES" \
      --label-mode subproblem \
      --dp-time-limit "$DP_TIME_LIMIT" --require-optimal-labels \
      --eval-seeds "$EVAL_SEEDS" \
      --eval-D-range "$EVAL_D_RANGE" --eval-N-range "$EVAL_N_RANGE" \
      --beams "$BEAMS" \
      --workers "$WORKERS" \
      --save-model "$MODEL_DIR_H/vhat_${TAG}.npz" \
      --out-csv "$OUT_CSV" \
      $FEAT_FLAGS $MODEL_ARGS \
      2>&1 | tee "$LOG_DIR/${TAG}_train_eval.log"
  fi
done

# ────────────────────────────────────────────────────────────────────────
# H2: Eval B-hard models on non-repeating (zero-shot transfer)
# ────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> H2: Eval B-hard repeating-trained models on non-repeating data"
echo ""

# Test models trained on repeating profiles
B_PROFILES=("ramp" "double_peak" "daily_tou")

RUN_IDX=0
for SRC_PROFILE in "${B_PROFILES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    MODEL_PATH="$MODEL_DIR_B/vhat_${SRC_PROFILE}_${MODEL}.npz"
    if [[ ! -f "$MODEL_PATH" ]]; then
      echo "[H2-$RUN_IDX] SKIP (no model) $MODEL_PATH"
      continue
    fi

    TAG="zeroshot_${SRC_PROFILE}_${MODEL}_on_nonrep"
    OUT_CSV="$LOG_DIR/${TAG}.csv"
    if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
      echo "[H2-$RUN_IDX] SKIP (exists) $OUT_CSV"
      continue
    fi

    echo "[H2-$RUN_IDX] EVAL $SRC_PROFILE/$MODEL → non-repeating"
    $PYTHON_BIN sandbox/eval_pooled_vhat.py \
      --daily-price-profile non_repeating \
      --model-type "$MODEL" \
      --pmax "$PMAX" \
      --target-util "$TARGET_UTIL" \
      --load-model "$MODEL_PATH" \
      --eval-seeds "$EVAL_SEEDS" \
      --eval-D-range "$EVAL_D_RANGE" --eval-N-range "$EVAL_N_RANGE" \
      --beams "$BEAMS" \
      --workers "$WORKERS" \
      --out-csv "$OUT_CSV" \
      $FEAT_FLAGS \
      2>&1 | tee "$LOG_DIR/${TAG}.log"
  done
done

echo ""
echo "============================================================"
echo " Experiment H complete."
echo " Results: $LOG_DIR/"
echo " H1 models: $MODEL_DIR_H/"
echo "============================================================"
