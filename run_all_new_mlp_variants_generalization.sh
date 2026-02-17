#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Run from this script's directory (PaST/)
cd "$(dirname "$0")"

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

PROFILES=("daily_tou" "generate_data")
NEW_MODELS=("mlp_hist" "mlp_price" "mlp_meta" "mlp_all")
CATEGORIES=("small" "medium" "large")

# Train/eval seeds
TRAIN_SEEDS_SMALL="0-99"
TRAIN_SEEDS_MEDIUM="0-79"
TRAIN_SEEDS_LARGE="0-59"

# Held-out same-size evaluation seeds (disjoint from training)
EVAL_SEEDS_SMALL="300-339"
EVAL_SEEDS_MEDIUM="400-429"
EVAL_SEEDS_LARGE="500-519"

# Samples per training instance (category-scaled for runtime)
SAMPLES_SMALL=4000
SAMPLES_MEDIUM=2000
SAMPLES_LARGE=1000

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
  read -r N_RANGE D_RANGE M_RANGE TARGET_UTIL TRAIN_SEEDS EVAL_SEEDS SAMPLES REPLICATES <<< "$(_size_params "$CATEGORY")"

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

      echo ""
      echo ">>> TRAIN pooled: category=$CATEGORY model=$MODEL profile=$PROFILE"

      "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
        --D 3 --N 40 --pmax "$PMAX" \
        --train-seeds "$TRAIN_SEEDS" --samples-per-instance "$SAMPLES" \
        --train-N-range "$N_RANGE" --train-D-range "$D_RANGE" \
        --eval-seeds "$EVAL_SEEDS" \
        --eval-N-range "$N_RANGE" --eval-D-range "$D_RANGE" --eval-pmax "$PMAX" \
        --transferable-features --normalize --normalize-labels \
        --target-util "$TARGET_UTIL" \
        --mlp-max-epochs "$MLP_MAX_EPOCHS" --mlp-patience "$MLP_PATIENCE" \
        --mlp-batch-size "$MLP_BATCH_SIZE" --mlp-lr "$MLP_LR" \
        "${PROFILE_ARGS[@]}" \
        --model-type "$MODEL" --beams 2,5 \
        --save-model "$CKPT" \
        --out-csv "$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.csv" \
        2>&1 | tee "$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.log"

      echo ">>> DEPLOY epsilon-sim (within-size): category=$CATEGORY model=$MODEL profile=$PROFILE"

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
        --out-csv "$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.csv" \
        2>&1 | tee "$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.log"

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

  # small -> medium
  SRC="small"; TGT="medium"
  read -r N_RANGE_S D_RANGE_S _M_RANGE_S TARGET_UTIL_S _TRAIN_SEEDS_S _EVAL_SEEDS_S _SAMPLES_S _REPL_S <<< "$(_size_params "$SRC")"
  read -r N_RANGE_T D_RANGE_T _M_RANGE_T TARGET_UTIL_T EVAL_SEEDS_T _REST <<< "$(_size_params "$TGT")"

  for MODEL in "${NEW_MODELS[@]}"; do
    CKPT="$MODEL_DIR/vhat_${SRC}_${PROFILE}_${MODEL}.npz"

    echo ""
    echo ">>> CROSS EVAL: train=${SRC} eval=${TGT} model=$MODEL profile=$PROFILE"

    "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
      --load-model "$CKPT" \
      --D 3 --N 40 --pmax "$PMAX" \
      --eval-seeds "$EVAL_SEEDS_T" \
      --eval-N-range "$N_RANGE_T" --eval-D-range "$D_RANGE_T" --eval-pmax "$PMAX" \
      --transferable-features --normalize \
      --target-util "$TARGET_UTIL_T" \
      "${PROFILE_ARGS[@]}" \
      --beams 2,5 \
      --out-csv "$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.csv" \
      2>&1 | tee "$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.log"
  done

  # medium -> large
  SRC="medium"; TGT="large"
  read -r N_RANGE_S D_RANGE_S _M_RANGE_S TARGET_UTIL_S _TRAIN_SEEDS_S _EVAL_SEEDS_S _SAMPLES_S _REPL_S <<< "$(_size_params "$SRC")"
  read -r N_RANGE_T D_RANGE_T _M_RANGE_T TARGET_UTIL_T EVAL_SEEDS_T _REST <<< "$(_size_params "$TGT")"

  for MODEL in "${NEW_MODELS[@]}"; do
    CKPT="$MODEL_DIR/vhat_${SRC}_${PROFILE}_${MODEL}.npz"

    echo ""
    echo ">>> CROSS EVAL: train=${SRC} eval=${TGT} model=$MODEL profile=$PROFILE"

    "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
      --load-model "$CKPT" \
      --D 3 --N 40 --pmax "$PMAX" \
      --eval-seeds "$EVAL_SEEDS_T" \
      --eval-N-range "$N_RANGE_T" --eval-D-range "$D_RANGE_T" --eval-pmax "$PMAX" \
      --transferable-features --normalize \
      --target-util "$TARGET_UTIL_T" \
      "${PROFILE_ARGS[@]}" \
      --beams 2,5 \
      --out-csv "$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.csv" \
      2>&1 | tee "$LOG_DIR/cross_${SRC}_to_${TGT}_${PROFILE}_${MODEL}.log"
  done

done

echo "Done. Logs: $LOG_DIR  Models: $MODEL_DIR"
