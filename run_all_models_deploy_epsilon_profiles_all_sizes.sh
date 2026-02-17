#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Run from this script's directory (PaST/)
cd "$(dirname "$0")"

# ============================================================
# Comprehensive deployment-like sweep:
#   sizes (small/medium/large) × profiles (2) × models (4)
#   Train pooled Vhat (within-profile) on variable N/D ranges,
#   then evaluate under epsilon-constraint multi-machine deployment.
# ============================================================

# Python executable to use (override if needed):
#   PYTHON_BIN=/path/to/python bash PaST/run_all_models_deploy_epsilon_profiles_all_sizes.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

# ============================================================
# Resume behavior
#
# Default is resume-friendly: rerunning the script will skip
# steps that already produced non-empty outputs.
#
# Override from the shell if needed:
#   RESUME=0 bash PaST/run_all_models_deploy_epsilon_profiles_all_sizes.sh
#   FORCE_TRAIN=1 bash PaST/run_all_models_deploy_epsilon_profiles_all_sizes.sh
#   FORCE_EVAL=1 bash PaST/run_all_models_deploy_epsilon_profiles_all_sizes.sh
# ============================================================

RESUME="${RESUME:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

# Two profiles (train+eval within profile; no cross-profile testing here)
PROFILES=("daily_tou" "generate_data")
MODELS=("linear" "poly" "mlp" "lgbm")
CATEGORIES=("small" "medium" "large")

# Train/eval seeds
# - More seeds reduces overfitting risk (better than cranking epochs).
# - You can shrink these if runtime is too high.
TRAIN_SEEDS_SMALL="0-99"
TRAIN_SEEDS_MEDIUM="0-79"
TRAIN_SEEDS_LARGE="0-59"

EVAL_SEEDS_SMALL="300-339"
EVAL_SEEDS_MEDIUM="400-429"
EVAL_SEEDS_LARGE="500-519"

# Samples per training instance (category-scaled for runtime)
SAMPLES_SMALL=4000
SAMPLES_MEDIUM=2000
SAMPLES_LARGE=1000

# Model training knobs (long-ish but guarded by early stopping)
MLP_MAX_EPOCHS=600
MLP_PATIENCE=40
MLP_BATCH_SIZE=4096
MLP_LR=1e-3

LGBM_N_ESTIMATORS=2000
LGBM_MAX_DEPTH=5
LGBM_LR=0.05

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

LOG_DIR="ADP/logs/deploy_epsilon_profiles_all_sizes"
MODEL_DIR="ADP/models/deploy_epsilon_profiles_all_sizes"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

# Convenience: LightGBM availability check
HAS_LGBM=1
if ! "$PYTHON_BIN" -c "import lightgbm" >/dev/null 2>&1; then
  HAS_LGBM=0
fi

for CATEGORY in "${CATEGORIES[@]}"; do
  case "$CATEGORY" in
    small)
      N_RANGE="20-60";  D_RANGE="2-4";   M_RANGE="3-7";   TARGET_UTIL="0.80";
      TRAIN_SEEDS="$TRAIN_SEEDS_SMALL"; EVAL_SEEDS="$EVAL_SEEDS_SMALL";
      SAMPLES="$SAMPLES_SMALL"; REPLICATES="$REPLICATES_SMALL";
      ;;
    medium)
      N_RANGE="100-200"; D_RANGE="5-15";  M_RANGE="8-16";  TARGET_UTIL="0.85";
      TRAIN_SEEDS="$TRAIN_SEEDS_MEDIUM"; EVAL_SEEDS="$EVAL_SEEDS_MEDIUM";
      SAMPLES="$SAMPLES_MEDIUM"; REPLICATES="$REPLICATES_MEDIUM";
      ;;
    large)
      N_RANGE="250-500"; D_RANGE="10-30"; M_RANGE="25-40"; TARGET_UTIL="0.90";
      TRAIN_SEEDS="$TRAIN_SEEDS_LARGE"; EVAL_SEEDS="$EVAL_SEEDS_LARGE";
      SAMPLES="$SAMPLES_LARGE"; REPLICATES="$REPLICATES_LARGE";
      ;;
    *)
      echo "Unknown CATEGORY=$CATEGORY"; exit 1;
      ;;
  esac

  echo "========================================================================"
  echo "CATEGORY=$CATEGORY  N_RANGE=$N_RANGE  D_RANGE=$D_RANGE  M_RANGE=$M_RANGE  target_util=$TARGET_UTIL"
  echo "========================================================================"

  for PROFILE in "${PROFILES[@]}"; do
    PROFILE_ARGS=()
    if [[ "$PROFILE" == "generate_data" ]]; then
      PROFILE_ARGS+=(--daily-price-profile generate_data --gd-seed 20260109 --gd-ck-low 1 --gd-ck-high 8)
    else
      PROFILE_ARGS+=(--daily-price-profile daily_tou)
    fi

    echo "------------------------------------------------------------------------"
    echo "PROFILE=$PROFILE (20-hour repeating)"
    echo "------------------------------------------------------------------------"

    for MODEL in "${MODELS[@]}"; do
      if [[ "$MODEL" == "lgbm" && "$HAS_LGBM" == "0" ]]; then
        echo "[skip] lightgbm not installed for $PYTHON_BIN"
        continue
      fi

      CKPT="$MODEL_DIR/vhat_${CATEGORY}_${PROFILE}_${MODEL}.npz"
      POOLED_CSV="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.csv"
      POOLED_LOG="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.log"
      EPS_CSV="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.csv"
      EPS_LOG="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.log"

      echo ""
      echo ">>> TRAIN pooled: category=$CATEGORY model=$MODEL profile=$PROFILE"

      if [[ "$RESUME" == "1" && "$FORCE_TRAIN" == "0" && -s "$CKPT" && -s "$POOLED_CSV" ]]; then
        echo "[resume] skip TRAIN (ckpt+csv exist): $CKPT"
      else
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
          --lgbm-n-estimators "$LGBM_N_ESTIMATORS" --lgbm-max-depth "$LGBM_MAX_DEPTH" --lgbm-learning-rate "$LGBM_LR" \
          "${PROFILE_ARGS[@]}" \
          --model-type "$MODEL" --beams 2,5 \
          --save-model "$CKPT" \
          --out-csv "$POOLED_CSV" \
          2>&1 | tee "$POOLED_LOG"
      fi

      echo ">>> DEPLOY epsilon-sim: category=$CATEGORY model=$MODEL profile=$PROFILE"

      if [[ "$RESUME" == "1" && "$FORCE_EVAL" == "0" && -s "$EPS_CSV" ]]; then
        echo "[resume] skip DEPLOY (csv exists): $EPS_CSV"
      else
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
      fi

      echo ">>> Done: category=$CATEGORY model=$MODEL profile=$PROFILE"
    done
  done

done

echo "Done. Logs: $LOG_DIR  Models: $MODEL_DIR"
