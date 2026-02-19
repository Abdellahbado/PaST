#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Be conservative on shared/HPC nodes: avoid thread oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Run from this script's directory (PaST/)
cd "$(dirname "$0")"

# ============================================================
# LIGHT sweep: MLP feature-embedding variants
#              (mlp_hist, mlp_price, mlp_meta, mlp_all)
#              medium + large sizes only
#              profile: daily_tou only (20-hour repeating TOU)
#
# Design rationale vs the full script:
#   1. ENOUGH DATA: medium uses 100 train seeds × 400 samples,
#      large uses 60 train seeds × 600 samples — enough to
#      distinguish variant quality from data-quantity effects.
#
#   2. SINGLE PROFILE: only daily_tou (deterministic 20-hour
#      repeating TOU pattern). No generate_data profile.
#
#   3. MODERATE DP TIME LIMIT + DISCARD TIMEOUTS:
#      medium 120 s, large 300 s per DP solve.
#      --require-optimal-labels ensures any DP that hits the
#      time limit is silently DISCARDED (not used as a label).
#      We move on to the next instance — easy instances
#      accumulate labels quickly, slow/hard ones are skipped.
#
#   4. CROSS-SIZE GENERALIZATION:
#        - Train on medium, eval on large.
#      This mimics real deployment: a model trained on medium
#      instances is applied to slightly larger ones.
#
# nohup launcher (default on).
# Disable: NOHUP=0 bash run_mlp_variants_light_medium_large.sh
# ============================================================

if [[ "${NOHUP:-1}" == "1" && -z "${IN_NOHUP:-}" ]]; then
  mkdir -p "ADP/logs/nohup"
  TS="$(date +"%Y%m%d_%H%M%S")"
  LOG_FILE="ADP/logs/nohup/$(basename "$0" .sh)_${TS}.log"
  echo "[nohup] launching in background; log: $LOG_FILE"
  IN_NOHUP=1 NOHUP=0 nohup bash "$0" "$@" >"$LOG_FILE" 2>&1 &
  echo "[nohup] pid=$!"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

WORKERS="${WORKERS:-4}"
MAX_WORKERS="${MAX_WORKERS:-16}"

EFFECTIVE_WORKERS="$WORKERS"
if [[ "$EFFECTIVE_WORKERS" -gt "$MAX_WORKERS" ]]; then
  EFFECTIVE_WORKERS="$MAX_WORKERS"
fi

POOL_DIR="${POOL_DIR:-ADP/tmp/pool}"
mkdir -p "$POOL_DIR"

# ============================================================
# Build Cython sparse-DP extension (native C hash maps).
# Most efficient available solver — compiles once, cached after.
# ============================================================
echo "[build] Compiling Cython sparse-DP extension ..."
( cd solvers && "$PYTHON_BIN" setup_cython.py build_ext --inplace 2>&1 \
    | grep -vE '^(running|building|copying)' || true )
echo "[build] done"

RESUME="${RESUME:-1}"

# ============================================================
# Profile: daily_tou only (20-hour repeating TOU pattern).
# ============================================================
PROFILE="daily_tou"
PROFILE_ARGS=(--daily-price-profile daily_tou)

NEW_MODELS=("mlp_hist" "mlp_price" "mlp_meta" "mlp_all")

# ============================================================
# DATA: enough seeds × samples to distinguish variant quality.
#   medium: 100 train seeds × 400 samples ≈ 40,000 labeled states
#   large:   60 train seeds × 600 samples ≈ 36,000 labeled states
# ============================================================
TRAIN_SEEDS_MEDIUM="0-99"
TRAIN_SEEDS_LARGE="0-59"

EVAL_SEEDS_MEDIUM="400-424"    # 25 held-out eval seeds
EVAL_SEEDS_LARGE="500-514"     # 15 held-out eval seeds

SAMPLES_MEDIUM=400
SAMPLES_LARGE=600

# ============================================================
# DP TIME LIMITS: moderate — easy instances finish, hard ones
# are discarded via --require-optimal-labels.
#   medium: 120 s  (most medium instances finish quickly with
#           Cython; the 120 s cap catches occasional blow-ups)
#   large:  300 s  (large instances vary widely; 300 s keeps
#           the easy-to-moderate majority usable)
# ============================================================
DP_TIME_LIMIT_MEDIUM="120"
DP_TIME_LIMIT_LARGE="300"

EPS_DP_TIME_LIMIT_MEDIUM="60"
EPS_DP_TIME_LIMIT_LARGE="120"

DP_MAX_STATES_MEDIUM="8000000"
DP_MAX_STATES_LARGE="15000000"

# Model training knobs
MLP_MAX_EPOCHS=500
MLP_PATIENCE=35
MLP_BATCH_SIZE=4096
MLP_LR=1e-3

# Deployment sim
REPLICATES_MEDIUM=6
REPLICATES_LARGE=4

EPS_SEED=123
BEAM=60
PRUNE_FACTOR=2.0
ASSIGN_ALPHA=1.2
ASSIGN_UNIFORM_MIX=0.15

PMAX=12

LOG_DIR="ADP/logs/mlp_variants_light_medium_large"
MODEL_DIR="ADP/models/mlp_variants_light_medium_large"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

_size_params() {
  local category="$1"
  case "$category" in
    medium)
      echo "100-200" "5-15" "8-16" "0.85" "$TRAIN_SEEDS_MEDIUM" "$EVAL_SEEDS_MEDIUM" "$SAMPLES_MEDIUM" "$REPLICATES_MEDIUM"
      ;;
    large)
      echo "250-500" "10-30" "25-40" "0.90" "$TRAIN_SEEDS_LARGE" "$EVAL_SEEDS_LARGE" "$SAMPLES_LARGE" "$REPLICATES_LARGE"
      ;;
    *)
      echo "Unknown CATEGORY=$category" 1>&2; exit 1 ;;
  esac
}

# ============================================================
# Phase 1: within-size train + eval (medium, then large)
# ============================================================
for CATEGORY in "medium" "large"; do
  read -r N_RANGE D_RANGE M_RANGE TARGET_UTIL TRAIN_SEEDS EVAL_SEEDS SAMPLES REPLICATES <<< "$(_size_params "$CATEGORY")"

  POOL_ARGS=(--pool-on-disk --pool-dtype float32 --pool-dir "$POOL_DIR")
  LABEL_MODE="optimal_path"
  OPT_PATH_N_PATHS="2"
  OPT_PATH_TOPUP_MAX="0"
  OPT_PATH_TOPUP_TL="-1"
  # --require-optimal-labels: discard any instance where DP
  # timed out; only well-solved instances contribute labels.
  REQUIRE_OPTIMAL_ARGS=(--require-optimal-labels)

  if [[ "$CATEGORY" == "medium" ]]; then
    DP_TIME_LIMIT="$DP_TIME_LIMIT_MEDIUM"
    EPS_DP_TIME_LIMIT="$EPS_DP_TIME_LIMIT_MEDIUM"
    DP_MAX_STATES="$DP_MAX_STATES_MEDIUM"
    EVAL_TIME_LIMIT="30.0"
  else
    DP_TIME_LIMIT="$DP_TIME_LIMIT_LARGE"
    EPS_DP_TIME_LIMIT="$EPS_DP_TIME_LIMIT_LARGE"
    DP_MAX_STATES="$DP_MAX_STATES_LARGE"
    EVAL_TIME_LIMIT="60.0"
  fi

  echo "========================================================================"
  echo "PHASE 1 | CATEGORY=$CATEGORY  N_RANGE=$N_RANGE  D_RANGE=$D_RANGE  M_RANGE=$M_RANGE  target_util=$TARGET_UTIL"
  echo "PROFILE=$PROFILE  train_seeds=$TRAIN_SEEDS  samples=$SAMPLES"
  echo "dp_time_limit=${DP_TIME_LIMIT}s  dp_max_states=$DP_MAX_STATES  (timed-out instances discarded)"
  echo "========================================================================"

  for MODEL in "${NEW_MODELS[@]}"; do
    CKPT="$MODEL_DIR/vhat_${CATEGORY}_${PROFILE}_${MODEL}.npz"
    POOLED_CSV="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.csv"
    POOLED_LOG="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.log"
    POOLED_DONE="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.done"
    EPS_CSV="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.csv"
    EPS_LOG="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.log"
    EPS_DONE="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.done"

    echo ""
    echo ">>> TRAIN pooled: category=$CATEGORY model=$MODEL"

    if [[ "$RESUME" == "1" && -f "$POOLED_DONE" && -s "$CKPT" && -s "$POOLED_CSV" ]]; then
      echo "[resume] skip pooled train/eval (exists): ckpt=$CKPT csv=$POOLED_CSV"
    else
      rm -f "$POOLED_DONE"
      "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
        --D 3 --N 40 --pmax "$PMAX" \
        --workers "$EFFECTIVE_WORKERS" \
        --maxtasksperchild 1 \
        --train-seeds "$TRAIN_SEEDS" --samples-per-instance "$SAMPLES" \
        --train-N-range "$N_RANGE" --train-D-range "$D_RANGE" \
        --eval-seeds "$EVAL_SEEDS" \
        --eval-N-range "$N_RANGE" --eval-D-range "$D_RANGE" --eval-pmax "$PMAX" \
        --transferable-features --normalize --normalize-labels \
        --label-mode "$LABEL_MODE" \
        --optimal-path-n-paths "$OPT_PATH_N_PATHS" \
        --optimal-path-topup-max "$OPT_PATH_TOPUP_MAX" \
        --optimal-path-topup-dp-time-limit "$OPT_PATH_TOPUP_TL" \
        "${REQUIRE_OPTIMAL_ARGS[@]}" \
        --dp-time-limit "$DP_TIME_LIMIT" \
        --eval-time-limit "$EVAL_TIME_LIMIT" \
        --target-util "$TARGET_UTIL" \
        "${POOL_ARGS[@]}" \
        --mlp-max-epochs "$MLP_MAX_EPOCHS" --mlp-patience "$MLP_PATIENCE" \
        --mlp-batch-size "$MLP_BATCH_SIZE" --mlp-lr "$MLP_LR" \
        "${PROFILE_ARGS[@]}" \
        --model-type "$MODEL" --beams 2,5 \
        --save-model "$CKPT" \
        --out-csv "$POOLED_CSV" \
        2>&1 | tee "$POOLED_LOG"

      touch "$POOLED_DONE"
    fi

    echo ">>> DEPLOY epsilon-sim (within-size): category=$CATEGORY model=$MODEL"

    if [[ "$RESUME" == "1" && -f "$EPS_DONE" && -s "$EPS_CSV" ]]; then
      echo "[resume] skip epsilon-sim (exists): csv=$EPS_CSV"
    else
      rm -f "$EPS_DONE"
      "$PYTHON_BIN" sandbox/eval_epsilon_constraint_sim.py \
        --category "$CATEGORY" \
        --N 40 --N-range "$N_RANGE" \
        --M 5 --M-range "$M_RANGE" \
        --D 3 --D-range "$D_RANGE" \
        --replicates "$REPLICATES" --seed "$EPS_SEED" \
        --pmax "$PMAX" --target-util "$TARGET_UTIL" \
        --price-mode daily_tou \
        "${PROFILE_ARGS[@]}" \
        --assign-alpha "$ASSIGN_ALPHA" --assign-uniform-mix "$ASSIGN_UNIFORM_MIX" \
        --guided --beam "$BEAM" --prune-factor "$PRUNE_FACTOR" \
        --load-model "$CKPT" --transferable-features --normalize \
        --eps-dp-time-limit "$EPS_DP_TIME_LIMIT" \
        --dp-max-states "$DP_MAX_STATES" \
        --out-csv "$EPS_CSV" \
        2>&1 | tee "$EPS_LOG"

      touch "$EPS_DONE"
    fi

    echo ">>> Done: category=$CATEGORY model=$MODEL"
  done

done

# ============================================================
# Phase 2: cross-size generalization
#   Train on medium, evaluate on large.
#   This mimics the real-world case: a model trained on
#   medium instances is deployed on larger instances.
# ============================================================
echo ""
echo "========================================================================"
echo "PHASE 2 | CROSS-SIZE GENERALIZATION: train=medium -> eval=large"
echo "========================================================================"

SRC="medium"
TGT="large"

N_RANGE_T="250-500"; D_RANGE_T="10-30"; TARGET_UTIL_T="0.90"
EVAL_SEEDS_CROSS="$EVAL_SEEDS_LARGE"

for MODEL in "${NEW_MODELS[@]}"; do
  CKPT="$MODEL_DIR/vhat_${SRC}_${PROFILE}_${MODEL}.npz"
  CROSS_CSV="$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.csv"
  CROSS_LOG="$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.log"
  CROSS_DONE="$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.done"

  echo ""
  echo ">>> CROSS EVAL: train=$SRC -> eval=$TGT model=$MODEL"

  if [[ "$RESUME" == "1" && -f "$CROSS_DONE" && -s "$CROSS_CSV" ]]; then
    echo "[resume] skip cross eval (exists): csv=$CROSS_CSV"
  else
    rm -f "$CROSS_DONE"
    "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
      --load-model "$CKPT" \
      --D 3 --N 40 --pmax "$PMAX" \
      --workers "$EFFECTIVE_WORKERS" \
      --maxtasksperchild 1 \
      --eval-seeds "$EVAL_SEEDS_CROSS" \
      --eval-N-range "$N_RANGE_T" --eval-D-range "$D_RANGE_T" --eval-pmax "$PMAX" \
      --transferable-features --normalize \
      --eval-time-limit "60.0" \
      --target-util "$TARGET_UTIL_T" \
      "${PROFILE_ARGS[@]}" \
      --beams 2,5 \
      --out-csv "$CROSS_CSV" \
      2>&1 | tee "$CROSS_LOG"

    touch "$CROSS_DONE"
  fi
done

echo ""
echo "Done. Logs: $LOG_DIR  Models: $MODEL_DIR"
