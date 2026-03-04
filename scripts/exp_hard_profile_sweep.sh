#!/usr/bin/env bash
# ============================================================================
# Experiment B-hard — Profile Sweep with HARD Instances
#
# Goal: Retrain on properly difficult instances and evaluate. The previous
# Experiment B used target_util=0.80 with N∈[20,60], which produced trivially
# solvable instances (96.7% zero gap). This experiment fixes that by:
#   1. Higher utilization (0.95) → tighter horizons
#   2. Matched N ranges → job sizes remain diverse (no collapse to p_j=1)
#   3. Profiles that showed differentiation: ramp, double_peak, generate_data
#   4. Larger beams (5,10,20) since beam=2 is still often optimal
#
# Instance difficulty calibration (pmax=12, E[p]=6.5, target_util=0.95):
#   D=3: T=60, cap=57, N∈[8,14] → sum(p)∈[52,91], moderate reduction
#   D=4: T=80, cap=76, N∈[10,18] → sum(p)∈[65,117], moderate reduction
#   D=5: T=100, cap=95, N∈[12,20] → sum(p)∈[78,130], moderate reduction
#
# Medium eval (higher D):
#   D=6: T=120, cap=114, N∈[15,22] → sum(p)∈[97,143], light reduction
#   D=8: T=160, cap=152, N∈[20,28] → sum(p)∈[130,182], light reduction
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-1}"
NO_COLLECT="${NO_COLLECT:-0}"
CACHE_ONLY="${CACHE_ONLY:-0}"
WORKERS="${WORKERS:-16}"

LOG_DIR="ADP/logs/exp_hard_profile_sweep"
MODEL_DIR="ADP/models/exp_hard_profile_sweep"
DATA_DIR="ADP/data/exp_hard_profile_sweep"
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$DATA_DIR"

PMAX=12
TARGET_UTIL="0.95"

# ── Profiles (only the ones that showed non-trivial gaps in round 1) ────
PROFILES=("double_peak" "generate_data" "daily_tou" "two_block")

# ── Models (top performers from round 1 + lightweight baseline) ─────────
MODELS=("mlp" "poly" "elasticnet")

# ── Feature set: enriched + extra (same as v1) ─────────────────────────
FEAT_FLAGS="--feat-len-hist --feat-price-shape --feat-meta --feat-extra --normalize --normalize-labels"

# ── Training: HARD small instances ──────────────────────────────────────
TRAIN_SEEDS="0-199"
SAMPLES=3000
TRAIN_D_RANGE="3-5"
TRAIN_N_RANGE="8-20"
DP_TIME_LIMIT="${DP_TIME_LIMIT:-120}"

# ── Eval: small-hard (same difficulty as training) ──────────────────────
EVAL_SEEDS_SMALL="200-224"
EVAL_D_RANGE_SMALL="3-5"
EVAL_N_RANGE_SMALL="8-20"

# ── Eval: medium-hard (cross-size generalization) ───────────────────────
EVAL_SEEDS_MEDIUM="400-414"
EVAL_D_RANGE_MEDIUM="6-8"
EVAL_N_RANGE_MEDIUM="15-28"
DP_TIME_LIMIT_MEDIUM="${DP_TIME_LIMIT_MEDIUM:-300}"

BEAMS="2,5,10"

# ── Model-specific defaults ─────────────────────────────────────────────
EN_ALPHA="1e-3"
EN_L1_RATIO="0.5"
MLP_EPOCHS=500
MLP_PATIENCE=35

echo "============================================================"
echo " Experiment B-hard — Profile Sweep (Hard Instances)"
echo " Profiles: ${PROFILES[*]}"
echo " Models: ${MODELS[*]}"
echo " target_util=$TARGET_UTIL pmax=$PMAX"
echo " Train: D∈[$TRAIN_D_RANGE] N∈[$TRAIN_N_RANGE]"
echo " Eval small: D∈[$EVAL_D_RANGE_SMALL] N∈[$EVAL_N_RANGE_SMALL]"
echo " Eval medium: D∈[$EVAL_D_RANGE_MEDIUM] N∈[$EVAL_N_RANGE_MEDIUM]"
echo " Total runs: $(( ${#PROFILES[@]} * ${#MODELS[@]} ))"
echo "============================================================"

# Collect training data once per profile and cache to DATA_DIR.
collect_data() {
  local PROFILE="$1"
  local CACHE="$DATA_DIR/pooled_${PROFILE}_small.npz"

  if [[ -f "$CACHE" ]]; then
    echo "  DATA (cached) $CACHE"; return 0
  fi
  if [[ "$NO_COLLECT" == "1" ]]; then
    echo "  ERROR: missing cache $CACHE (NO_COLLECT=1)"; return 1
  fi

  echo "  DATA $PROFILE/small  (workers=$WORKERS, samples=$SAMPLES/inst) → $CACHE"
  $PYTHON_BIN sandbox/eval_pooled_vhat.py \
    --daily-price-profile "$PROFILE" \
    --model-type mlp \
    --pmax "$PMAX" \
    --target-util "$TARGET_UTIL" \
    --train-seeds "$TRAIN_SEEDS" \
    --train-D-range "$TRAIN_D_RANGE" --train-N-range "$TRAIN_N_RANGE" \
    --samples-per-instance "$SAMPLES" \
    --label-mode subproblem \
    --dp-time-limit "$DP_TIME_LIMIT" --require-optimal-labels \
    --workers "$WORKERS" \
    --save-pooled-data "$CACHE" \
    --save-pooled-data-only \
    $FEAT_FLAGS \
    2>&1 | tee "$DATA_DIR/collect_${PROFILE}_small.log"
}

RUN_IDX=0
TOTAL=$(( ${#PROFILES[@]} * ${#MODELS[@]} ))

for PROFILE in "${PROFILES[@]}"; do
  CACHE="$DATA_DIR/pooled_${PROFILE}_small.npz"
  collect_data "$PROFILE"

  if [[ "$CACHE_ONLY" == "1" ]]; then
    continue
  fi

  for MODEL in "${MODELS[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    TAG="${PROFILE}_${MODEL}"

    # Model-specific CLI args
    MODEL_ARGS=""
    case "$MODEL" in
      elasticnet) MODEL_ARGS="--elasticnet-alpha $EN_ALPHA --elasticnet-l1-ratio $EN_L1_RATIO" ;;
      mlp)        MODEL_ARGS="--mlp-max-epochs $MLP_EPOCHS --mlp-patience $MLP_PATIENCE" ;;
      poly)       MODEL_ARGS="--l2 1e-3" ;;
    esac

    # ── Train on hard-small from cache, eval on hard-small ────────────────────
    OUT_CSV="$LOG_DIR/${TAG}_hard_small.csv"
    if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
      echo "[$RUN_IDX/$TOTAL] SKIP (exists) $OUT_CSV"
    else
      echo "[$RUN_IDX/$TOTAL] TRAIN hard-small → EVAL hard-small : $TAG  (from cache)"
      $PYTHON_BIN sandbox/eval_pooled_vhat.py \
        --daily-price-profile "$PROFILE" \
        --model-type "$MODEL" \
        --pmax "$PMAX" \
        --target-util "$TARGET_UTIL" \
        --train-seeds "$TRAIN_SEEDS" \
        --load-pooled-data "$CACHE" \
        --eval-seeds "$EVAL_SEEDS_SMALL" \
        --eval-D-range "$EVAL_D_RANGE_SMALL" --eval-N-range "$EVAL_N_RANGE_SMALL" \
        --beams "$BEAMS" \
        --workers "$WORKERS" \
        --save-model "$MODEL_DIR/vhat_${TAG}.npz" \
        --out-csv "$OUT_CSV" \
        $FEAT_FLAGS $MODEL_ARGS \
        2>&1 | tee "$LOG_DIR/${TAG}_hard_small.log"
    fi

    # ── Eval on medium-hard (cross-size) ───────────────────────────
    MODEL_PATH="$MODEL_DIR/vhat_${TAG}.npz"
    OUT_CSV_MED="$LOG_DIR/${TAG}_hard_medium.csv"
    if [[ -f "$OUT_CSV_MED" && "$RESUME" == "1" ]]; then
      echo "[$RUN_IDX/$TOTAL] SKIP (exists) $OUT_CSV_MED"
    elif [[ ! -f "$MODEL_PATH" ]]; then
      echo "[$RUN_IDX/$TOTAL] SKIP (no model) $OUT_CSV_MED"
    else
      echo "[$RUN_IDX/$TOTAL] EVAL medium-hard (cross-size) : $TAG"
      $PYTHON_BIN sandbox/eval_pooled_vhat.py \
        --daily-price-profile "$PROFILE" \
        --model-type "$MODEL" \
        --pmax "$PMAX" \
        --target-util "$TARGET_UTIL" \
        --load-model "$MODEL_PATH" \
        --eval-D-range "$EVAL_D_RANGE_MEDIUM" --eval-N-range "$EVAL_N_RANGE_MEDIUM" \
        --eval-seeds "$EVAL_SEEDS_MEDIUM" \
        --beams "$BEAMS" \
        --workers "$WORKERS" \
        --dp-time-limit "$DP_TIME_LIMIT_MEDIUM" \
        --out-csv "$OUT_CSV_MED" \
        $FEAT_FLAGS $MODEL_ARGS \
        2>&1 | tee "$LOG_DIR/${TAG}_hard_medium.log"
    fi

  done
done

if [[ "$CACHE_ONLY" == "1" ]]; then
  echo ""
  echo "============================================================"
  echo " Experiment B-hard cache-only complete.  Cached data: $DATA_DIR/"
  echo "============================================================"
  exit 0
fi

echo ""
echo "============================================================"
echo " Experiment B-hard complete."
echo " Results: $LOG_DIR/"
echo " Models: $MODEL_DIR/"
echo "============================================================"
