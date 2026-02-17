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

# -----------------------------
# nohup launcher (default on)
#
# By default, the script re-launches itself under nohup in the background so
# terminal/SSH disconnects won't stop the run.
#
# Disable:
#   NOHUP=0 bash PaST/run_all_new_mlp_variants_generalization.sh
# -----------------------------

if [[ "${NOHUP:-1}" == "1" && -z "${IN_NOHUP:-}" ]]; then
  mkdir -p "ADP/logs/nohup"
  TS="$(date +"%Y%m%d_%H%M%S")"
  LOG_FILE="ADP/logs/nohup/$(basename "$0" .sh)_${TS}.log"
  echo "[nohup] launching in background; log: $LOG_FILE"
  IN_NOHUP=1 NOHUP=0 nohup bash "$0" "$@" >"$LOG_FILE" 2>&1 &
  echo "[nohup] pid=$!"
  exit 0
fi

# ============================================================
# Train + evaluate the NEW MLP feature-embedding variants:
#   - mlp_hist, mlp_price, mlp_meta, mlp_all
# across sizes (small/medium/large) and profiles (daily_tou/generate_data).
#
# For each (size, profile, model):
#   1) Train pooled Vhat on variable D/N ranges
#   2) Evaluate on held-out same-size instances (different seeds)
#   3) Run epsilon-constraint deployment sim on the same size
#
# Cross-size generalization:
#   - Train on small, evaluate on medium
#   - Train on medium, evaluate on large
# (pooled evaluation only; epsilon-sim is kept within-size for runtime)
# ============================================================

# Python executable to use (override if needed):
#   PYTHON_BIN=/path/to/python bash PaST/run_all_new_mlp_variants_generalization.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

# Multiprocessing workers for pooled data collection.
WORKERS="${WORKERS:-4}"

# Hard cap to avoid spawning too many heavy Python worker processes on HPC.
# Override:
#   MAX_WORKERS=8 WORKERS=64 bash PaST/run_all_new_mlp_variants_generalization.sh
MAX_WORKERS="${MAX_WORKERS:-16}"

# Effective workers used by all calls.
EFFECTIVE_WORKERS="$WORKERS"
if [[ "$EFFECTIVE_WORKERS" -gt "$MAX_WORKERS" ]]; then
  EFFECTIVE_WORKERS="$MAX_WORKERS"
fi

# Directory for on-disk pooling (memmap). Using an explicit path avoids relying
# on system temp dirs (which can be small/slow on some clusters).
POOL_DIR="${POOL_DIR:-ADP/tmp/pool}"
mkdir -p "$POOL_DIR"

# Resume behavior:
# - RESUME=1 (default): skip steps whose outputs already exist
# - RESUME=0: force re-run everything (overwrites CSVs; checkpoints overwritten by training)
RESUME="${RESUME:-1}"

# Start point for long runs.
# Default: start from medium so reruns don't waste time on already-finished small.
# Override:
#   START_CATEGORY=small  bash PaST/run_all_new_mlp_variants_generalization.sh
#   START_CATEGORY=large  bash PaST/run_all_new_mlp_variants_generalization.sh
START_CATEGORY="${START_CATEGORY:-medium}"

PROFILES=("daily_tou" "generate_data")
NEW_MODELS=("mlp_hist" "mlp_price" "mlp_meta" "mlp_all")
CATEGORIES=("small" "medium" "large")

# Train/eval seeds
TRAIN_SEEDS_SMALL="0-99"
TRAIN_SEEDS_MEDIUM="0-199"
TRAIN_SEEDS_LARGE="0-149"

# Held-out same-size evaluation seeds (disjoint from training)
EVAL_SEEDS_SMALL="300-339"
EVAL_SEEDS_MEDIUM="400-429"
EVAL_SEEDS_LARGE="500-519"

# Samples per training instance (category-scaled for runtime)
SAMPLES_SMALL=4000
SAMPLES_MEDIUM=600
SAMPLES_LARGE=1200

# Model training knobs (early stopping guards total runtime)
MLP_MAX_EPOCHS=600
MLP_PATIENCE=40
MLP_BATCH_SIZE=4096
MLP_LR=1e-3

# Deployment sim knobs
REPLICATES_SMALL=10
REPLICATES_MEDIUM=8
REPLICATES_LARGE=5

EPS_SEED=123
BEAM=80
PRUNE_FACTOR=2.0
ASSIGN_ALPHA=1.2
ASSIGN_UNIFORM_MIX=0.15

PMAX=12

LOG_DIR="ADP/logs/new_mlp_variants_generalization"
MODEL_DIR="ADP/models/new_mlp_variants_generalization"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

_profile_args() {
  local profile="$1"
  if [[ "$profile" == "generate_data" ]]; then
    echo "--daily-price-profile generate_data --gd-seed 20260109 --gd-ck-low 1 --gd-ck-high 8"
  else
    echo "--daily-price-profile daily_tou"
  fi
}

# Helper to choose size ranges
_size_params() {
  local category="$1"
  case "$category" in
    small)
      echo "20-60" "2-4" "3-7" "0.80" "$TRAIN_SEEDS_SMALL" "$EVAL_SEEDS_SMALL" "$SAMPLES_SMALL" "$REPLICATES_SMALL"
      ;;
    medium)
      echo "100-200" "5-15" "8-16" "0.85" "$TRAIN_SEEDS_MEDIUM" "$EVAL_SEEDS_MEDIUM" "$SAMPLES_MEDIUM" "$REPLICATES_MEDIUM"
      ;;
    large)
      echo "250-500" "10-30" "25-40" "0.90" "$TRAIN_SEEDS_LARGE" "$EVAL_SEEDS_LARGE" "$SAMPLES_LARGE" "$REPLICATES_LARGE"
      ;;
    *)
      echo "Unknown CATEGORY=$category" 1>&2
      exit 1
      ;;
  esac
}

# ------------------------
# In-size train + eval
# ------------------------
for CATEGORY in "${CATEGORIES[@]}"; do
  if [[ "${STARTED:-0}" == "0" ]]; then
    if [[ "$CATEGORY" != "$START_CATEGORY" ]]; then
      echo "[skip] CATEGORY=$CATEGORY (START_CATEGORY=$START_CATEGORY)"
      continue
    fi
    STARTED=1
  fi
  read -r N_RANGE D_RANGE M_RANGE TARGET_UTIL TRAIN_SEEDS EVAL_SEEDS SAMPLES REPLICATES <<< "$(_size_params "$CATEGORY")"

  LABEL_MODE="subproblem"
  DP_TIME_LIMIT="-1"
  OPT_PATH_N_PATHS="1"
  OPT_PATH_TOPUP_MAX="0"
  OPT_PATH_TOPUP_TL="0.2"
  REQUIRE_OPTIMAL_ARGS=()
  EVAL_TIME_LIMIT="-1"
  if [[ "$CATEGORY" == "medium" ]]; then
    LABEL_MODE="optimal_path"
    DP_TIME_LIMIT="-1"
    OPT_PATH_N_PATHS="2"
    OPT_PATH_TOPUP_MAX="0"
    OPT_PATH_TOPUP_TL="-1"
    REQUIRE_OPTIMAL_ARGS=(--require-optimal-labels)
    EVAL_TIME_LIMIT="30.0"
  elif [[ "$CATEGORY" == "large" ]]; then
    LABEL_MODE="optimal_path"
    DP_TIME_LIMIT="-1"
    OPT_PATH_N_PATHS="2"
    OPT_PATH_TOPUP_MAX="0"
    OPT_PATH_TOPUP_TL="-1"
    REQUIRE_OPTIMAL_ARGS=(--require-optimal-labels)
    EVAL_TIME_LIMIT="60.0"
  fi

  # Use on-disk pooling for bigger categories to avoid RAM spikes with many workers.
  POOL_ARGS=()
  if [[ "$CATEGORY" == "medium" || "$CATEGORY" == "large" ]]; then
    POOL_ARGS=(--pool-on-disk --pool-dtype float32 --pool-dir "$POOL_DIR")
  fi

  echo "========================================================================"
  echo "CATEGORY=$CATEGORY  N_RANGE=$N_RANGE  D_RANGE=$D_RANGE  M_RANGE=$M_RANGE  target_util=$TARGET_UTIL"
  echo "========================================================================"

  for PROFILE in "${PROFILES[@]}"; do
    PROFILE_ARGS=( $(_profile_args "$PROFILE") )

    echo "------------------------------------------------------------------------"
    echo "PROFILE=$PROFILE"
    echo "------------------------------------------------------------------------"

    for MODEL in "${NEW_MODELS[@]}"; do
      CKPT="$MODEL_DIR/vhat_${CATEGORY}_${PROFILE}_${MODEL}.npz"
      POOLED_CSV="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.csv"
      POOLED_LOG="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.log"
      POOLED_DONE="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.done"
      EPS_CSV="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.csv"
      EPS_LOG="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.log"
      EPS_DONE="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.done"

      echo ""
      echo ">>> TRAIN pooled: category=$CATEGORY model=$MODEL profile=$PROFILE"

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

      echo ">>> DEPLOY epsilon-sim (within-size): category=$CATEGORY model=$MODEL profile=$PROFILE"

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
          --out-csv "$EPS_CSV" \
          2>&1 | tee "$EPS_LOG"

        touch "$EPS_DONE"
      fi

      echo ">>> Done: category=$CATEGORY model=$MODEL profile=$PROFILE"
    done
  done

done

# ------------------------
# Cross-size generalization eval (pooled eval only)
# ------------------------
# Train small -> eval medium
# Train medium -> eval large
# (Uses the already-trained checkpoints from above.)

echo "========================================================================"
echo "CROSS-SIZE GENERALIZATION (pooled eval only)"
echo "========================================================================"

for PROFILE in "${PROFILES[@]}"; do
  PROFILE_ARGS=( $(_profile_args "$PROFILE") )

  # Eval-time limit for cross-size eval depends on the TARGET size.
  _eval_time_limit_for() {
    local category="$1"
    case "$category" in
      small) echo "-1" ;;
      medium) echo "30.0" ;;
      large) echo "60.0" ;;
      *) echo "-1" ;;
    esac
  }

  # small -> medium
  SRC="small"; TGT="medium"
  read -r N_RANGE_S D_RANGE_S _M_RANGE_S TARGET_UTIL_S _TRAIN_SEEDS_S _EVAL_SEEDS_S _SAMPLES_S _REPL_S <<< "$(_size_params "$SRC")"
  read -r N_RANGE_T D_RANGE_T _M_RANGE_T TARGET_UTIL_T EVAL_SEEDS_T _REST <<< "$(_size_params "$TGT")"

  for MODEL in "${NEW_MODELS[@]}"; do
    CKPT="$MODEL_DIR/vhat_${SRC}_${PROFILE}_${MODEL}.npz"
    CROSS_CSV="$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.csv"
    CROSS_LOG="$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.log"

    echo ""
    echo ">>> CROSS EVAL: train=${SRC} eval=${TGT} model=$MODEL profile=$PROFILE"

    if [[ "$RESUME" == "1" && -f "$CROSS_CSV" ]]; then
      echo "[resume] skip cross eval (exists): csv=$CROSS_CSV"
    else
      rm -f "$CROSS_CSV.done"
      "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
        --load-model "$CKPT" \
        --D 3 --N 40 --pmax "$PMAX" \
        --workers "$EFFECTIVE_WORKERS" \
        --maxtasksperchild 1 \
        --eval-seeds "$EVAL_SEEDS_T" \
        --eval-N-range "$N_RANGE_T" --eval-D-range "$D_RANGE_T" --eval-pmax "$PMAX" \
        --transferable-features --normalize \
        --eval-time-limit "$(_eval_time_limit_for "$TGT")" \
        --target-util "$TARGET_UTIL_T" \
        "${PROFILE_ARGS[@]}" \
        --beams 2,5 \
        --out-csv "$CROSS_CSV" \
        2>&1 | tee "$CROSS_LOG"

      touch "$CROSS_CSV.done"
    fi
  done

  # medium -> large
  SRC="medium"; TGT="large"
  read -r N_RANGE_S D_RANGE_S _M_RANGE_S TARGET_UTIL_S _TRAIN_SEEDS_S _EVAL_SEEDS_S _SAMPLES_S _REPL_S <<< "$(_size_params "$SRC")"
  read -r N_RANGE_T D_RANGE_T _M_RANGE_T TARGET_UTIL_T EVAL_SEEDS_T _REST <<< "$(_size_params "$TGT")"

  for MODEL in "${NEW_MODELS[@]}"; do
    CKPT="$MODEL_DIR/vhat_${SRC}_${PROFILE}_${MODEL}.npz"
    CROSS_CSV="$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.csv"
    CROSS_LOG="$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.log"

    echo ""
    echo ">>> CROSS EVAL: train=${SRC} eval=${TGT} model=$MODEL profile=$PROFILE"

    if [[ "$RESUME" == "1" && -f "$CROSS_CSV" ]]; then
      echo "[resume] skip cross eval (exists): csv=$CROSS_CSV"
    else
      rm -f "$CROSS_CSV.done"
      "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
        --load-model "$CKPT" \
        --D 3 --N 40 --pmax "$PMAX" \
        --workers "$EFFECTIVE_WORKERS" \
        --maxtasksperchild 1 \
        --eval-seeds "$EVAL_SEEDS_T" \
        --eval-N-range "$N_RANGE_T" --eval-D-range "$D_RANGE_T" --eval-pmax "$PMAX" \
        --transferable-features --normalize \
        --eval-time-limit "$(_eval_time_limit_for "$TGT")" \
        --target-util "$TARGET_UTIL_T" \
        "${PROFILE_ARGS[@]}" \
        --beams 2,5 \
        --out-csv "$CROSS_CSV" \
        2>&1 | tee "$CROSS_LOG"

      touch "$CROSS_CSV.done"
    fi
  done

done

echo "Done. Logs: $LOG_DIR  Models: $MODEL_DIR"
