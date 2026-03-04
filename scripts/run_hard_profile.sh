#!/usr/bin/env bash
# ============================================================================
# run_hard_profile.sh — Collect + train on a single hard profile (small size)
#
# Usage:
#   # Step 1 — collect data only:
#   PROFILE=generate_data  CACHE_ONLY=1 bash scripts/run_hard_profile.sh
#   PROFILE=two_block      CACHE_ONLY=1 bash scripts/run_hard_profile.sh
#   PROFILE=non_repeating  CACHE_ONLY=1 bash scripts/run_hard_profile.sh
#
#   # Step 2 — train & eval all models (reuses cached data):
#   PROFILE=generate_data  NO_COLLECT=1 bash scripts/run_hard_profile.sh
#   PROFILE=two_block      NO_COLLECT=1 bash scripts/run_hard_profile.sh
#   PROFILE=non_repeating  NO_COLLECT=1 bash scripts/run_hard_profile.sh
#
# Environment variables:
#   PROFILE        — which profile to run (required)
#   CACHE_ONLY=1   — collect & save data, then exit (no training)
#   NO_COLLECT=1   — assume cache exists, skip data collection (fail if missing)
#   RESUME=1       — skip model runs whose CSV already exists (default: 1)
#   WORKERS=16     — parallel workers for DP data collection
#   DP_TIME_LIMIT  — per-solve DP timeout in seconds (default: 120)
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-}"
if [[ -z "$PROFILE" ]]; then
  echo "ERROR: set PROFILE=generate_data | two_block | non_repeating"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-1}"
NO_COLLECT="${NO_COLLECT:-0}"
CACHE_ONLY="${CACHE_ONLY:-0}"
WORKERS="${WORKERS:-16}"
DP_TIME_LIMIT="${DP_TIME_LIMIT:-120}"

# ── Directory layout (shared with exp_hard_profile_sweep so models reuse) ──
LOG_DIR="ADP/logs/exp_hard_profile_sweep"
MODEL_DIR="ADP/models/exp_hard_profile_sweep"
DATA_DIR="ADP/data/exp_hard_profile_sweep"
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$DATA_DIR"

# ── Instance settings ────────────────────────────────────────────────────────
# non_repeating uses B2-style (D=3-6, N=10-40, pmax=5, 300 seeds)
# generate_data / two_block use hard-small settings (D=3-5, N=8-20, pmax=12, 200 seeds)
if [[ "$PROFILE" == "non_repeating" ]]; then
  PMAX=5
  TRAIN_SEEDS="0-299"
  TRAIN_D_RANGE="3-6"
  TRAIN_N_RANGE="10-40"
  EVAL_SEEDS="500-549"
  EVAL_D_RANGE="3-6"
  EVAL_N_RANGE="10-40"
  LABEL_MODE="subproblem"
  TARGET_UTIL="0.95"
else
  PMAX=12
  TRAIN_SEEDS="0-199"
  TRAIN_D_RANGE="3-5"
  TRAIN_N_RANGE="8-20"
  EVAL_SEEDS="200-224"
  EVAL_D_RANGE="3-5"
  EVAL_N_RANGE="8-20"
  LABEL_MODE="subproblem"
  TARGET_UTIL="0.95"
fi

SAMPLES=3000
FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"
MODELS=("mlp" "poly" "poly_mlp" "factored_mlp")
BEAMS="2,5,10"
MLP_EPOCHS=500
MLP_PATIENCE=35

CACHE="$DATA_DIR/pooled_${PROFILE}_small.npz"

echo "============================================================"
echo " Hard profile: $PROFILE  (small size)"
echo " Cache: $CACHE"
echo " CACHE_ONLY=$CACHE_ONLY  NO_COLLECT=$NO_COLLECT  RESUME=$RESUME"
echo "============================================================"

# ── Step 1: Collect data ────────────────────────────────────────────────────
if [[ -f "$CACHE" ]]; then
  echo "[data] Cache already exists → skipping collection."
elif [[ "$NO_COLLECT" == "1" ]]; then
  echo "ERROR: cache missing and NO_COLLECT=1 — run with CACHE_ONLY=1 first."
  exit 1
else
  echo "[data] Collecting pooled data → $CACHE"
  $PYTHON_BIN sandbox/eval_pooled_vhat.py \
    --daily-price-profile "$PROFILE" \
    --model-type mlp \
    --pmax "$PMAX" \
    --target-util "$TARGET_UTIL" \
    --train-seeds "$TRAIN_SEEDS" \
    --train-D-range "$TRAIN_D_RANGE" --train-N-range "$TRAIN_N_RANGE" \
    --samples-per-instance "$SAMPLES" \
    --label-mode "$LABEL_MODE" \
    --dp-time-limit "$DP_TIME_LIMIT" --require-optimal-labels \
    --workers "$WORKERS" \
    --save-pooled-data "$CACHE" \
    --save-pooled-data-only \
    $FEAT_FLAGS \
    2>&1 | tee "$DATA_DIR/collect_${PROFILE}_small.log"
  echo "[data] Done."
fi

if [[ "$CACHE_ONLY" == "1" ]]; then
  echo ""
  echo "============================================================"
  echo " Cache-only complete. Next:"
  echo "   PROFILE=$PROFILE NO_COLLECT=1 bash scripts/run_hard_profile.sh"
  echo "============================================================"
  exit 0
fi

# ── Step 2: Train + eval each model ────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
  TAG="${PROFILE}_${MODEL}"
  OUT_CSV="$LOG_DIR/${TAG}_hard_small.csv"

  if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
    echo "[run] SKIP (exists) $OUT_CSV"
    continue
  fi

  MODEL_ARGS=""
  case "$MODEL" in
    mlp|poly_mlp|factored_mlp) MODEL_ARGS="--mlp-max-epochs $MLP_EPOCHS --mlp-patience $MLP_PATIENCE" ;;
    poly)                       MODEL_ARGS="--l2 1e-3" ;;
  esac

  echo "[run] $MODEL  →  $OUT_CSV"
  $PYTHON_BIN sandbox/eval_pooled_vhat.py \
    --daily-price-profile "$PROFILE" \
    --model-type "$MODEL" \
    --pmax "$PMAX" \
    --target-util "$TARGET_UTIL" \
    --train-seeds "$TRAIN_SEEDS" \
    --load-pooled-data "$CACHE" \
    --eval-seeds "$EVAL_SEEDS" \
    --eval-D-range "$EVAL_D_RANGE" --eval-N-range "$EVAL_N_RANGE" \
    --beams "$BEAMS" \
    --workers "$WORKERS" \
    --save-model "$MODEL_DIR/vhat_${TAG}.npz" \
    --out-csv "$OUT_CSV" \
    $FEAT_FLAGS $MODEL_ARGS \
    2>&1 | tee "$LOG_DIR/${TAG}_hard_small.log"
done

echo ""
echo "============================================================"
echo " Done: $PROFILE (small)"
echo " Results: $LOG_DIR/${PROFILE}_*_hard_small.csv"
echo "============================================================"
