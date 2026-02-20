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
# LIGHT sweep: all models (linear, poly, mlp, lgbm)
#              medium + large sizes only
#              profile: daily_tou only (20-hour repeating TOU)
#
# Design rationale vs the full script:
#   1. ENOUGH DATA: medium uses 100 train seeds × 400 samples,
#      large uses 60 train seeds × 600 samples — enough gradient
#      signal to distinguish model quality from data quality.
#
#   2. SINGLE PROFILE: only daily_tou (deterministic 20-hour
#      repeating TOU pattern). No generate_data profile.
#
#   3. MODERATE DP TIME LIMIT + DISCARD TIMEOUTS:
#      medium 120 s, large 300 s per DP solve.
#      --require-optimal-labels ensures any DP that hits the
#      time limit is silently DISCARDED (not used as a label).
#      We move on to the next instance — easy instances accumulate
#      labels quickly, slow/hard ones are skipped.
#
#   4. CROSS-SIZE GENERALIZATION: after within-size training we
#      evaluate the medium-trained models on large eval seeds,
#      mimicking real deployment where a model trained on one
#      size is transferred to a larger one.
#
# nohup launcher (default on).
# Disable: NOHUP=0 bash run_models_light_medium_large.sh
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

POOL_DIR="${POOL_DIR:-ADP/tmp/pool}"
mkdir -p "$POOL_DIR"

# ============================================================
# Build Cython sparse-DP extension (native C hash maps).
# Most efficient available solver — compiles once, cached after.
# ============================================================
echo "[build] Compiling Cython sparse-DP extension ..."
if ! command -v gcc &> /dev/null; then
  echo "[build] gcc not found. Attempting to install toolchain via conda-forge..."
  conda install -y -c conda-forge compilers || true
  conda install -y -c conda-forge gcc_linux-64 gxx_linux-64 || true
  conda install -y -c conda-forge c-compiler cxx-compiler || true
fi

# Some conda images provide only the wrapper compiler (x86_64-conda-linux-gnu-cc)
# without a `gcc` symlink. Set CC/CXX so setuptools can compile extensions.
if ! command -v gcc &> /dev/null; then
  if command -v x86_64-conda-linux-gnu-cc &> /dev/null; then
    export CC="$(command -v x86_64-conda-linux-gnu-cc)"
    if command -v x86_64-conda-linux-gnu-c++ &> /dev/null; then
      export CXX="$(command -v x86_64-conda-linux-gnu-c++)"
    fi
    echo "[build] Using conda wrapper compiler: CC=$CC"
  elif command -v cc &> /dev/null; then
    export CC="$(command -v cc)"
    echo "[build] Using CC=$CC"
  fi
fi
( cd solvers && "$PYTHON_BIN" setup_cython.py build_ext --inplace 2>&1 \
    | grep -vE '^(running|building|copying)' || true )
if "$PYTHON_BIN" - <<'PY'
import importlib.util, sys
sys.path.insert(0, 'solvers')
ok = importlib.util.find_spec('_sparse_dp_cython') is not None
print('[build] cython_sparse_dp_import=' + ('OK' if ok else 'FAIL'))
raise SystemExit(0 if ok else 1)
PY
then
  echo "[build] done"
else
  echo "[build] WARNING: Cython sparse-DP extension not available; falling back to slower solver."
fi

RESUME="${RESUME:-1}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

# ============================================================
# Profile: daily_tou only (20-hour repeating TOU pattern).
# ============================================================
PROFILE="daily_tou"
PROFILE_ARGS=(--daily-price-profile daily_tou)

MODELS=("linear" "poly" "mlp" "lgbm")

# ============================================================
# Feature set
#
# Default: enriched (hist + price shape + meta) while still using
# --transferable-features (fixed dimension). This makes it much harder for the
# learning to fail purely due to missing state signal.
#
# Disable:
#   USE_ENHANCED_FEATURES=0 bash run_models_light_medium_large.sh
# ============================================================
USE_ENHANCED_FEATURES="${USE_ENHANCED_FEATURES:-1}"
FEATURE_ARGS=()
FEATURE_TAG="base"
if [[ "$USE_ENHANCED_FEATURES" == "1" ]]; then
  FEATURE_ARGS=(--feat-len-hist --feat-price-shape --feat-meta)
  FEATURE_TAG="all"
fi

# ============================================================
# DATA: enough seeds × samples to distinguish model quality.
#   medium: 100 train seeds × 400 samples ≈ 40,000 labeled states
#   large:   60 train seeds × 600 samples ≈ 36,000 labeled states
#   (Many large seeds will be discarded by the DP time limit;
#   starting with 60 means we expect ~30-50 usable ones.)
# ============================================================
TRAIN_SEEDS_MEDIUM="${TRAIN_SEEDS_MEDIUM:-0-99}"
TRAIN_SEEDS_LARGE="${TRAIN_SEEDS_LARGE:-0-59}"

EVAL_SEEDS_MEDIUM="${EVAL_SEEDS_MEDIUM:-400-424}"    # 25 held-out eval seeds
EVAL_SEEDS_LARGE="${EVAL_SEEDS_LARGE:-500-514}"     # 15 held-out eval seeds

SAMPLES_MEDIUM="${SAMPLES_MEDIUM:-400}"
SAMPLES_LARGE="${SAMPLES_LARGE:-600}"

# ============================================================
# DP TIME LIMITS: moderate — easy instances finish, hard ones
# are discarded via --require-optimal-labels.
#   medium: 120 s  (most medium instances finish in < 30 s with
#           Cython; the 120 s cap catches occasional blow-ups)
#   large:  300 s  (large instances range from a few seconds to
#           thousands; 300 s keeps the easy majority usable)
# ============================================================
DP_TIME_LIMIT_MEDIUM="${DP_TIME_LIMIT_MEDIUM:-300}"
DP_TIME_LIMIT_LARGE="${DP_TIME_LIMIT_LARGE:-300}"

# Epsilon-sim DP time limits (per-machine subproblem; machines
# are smaller than the full instance, so limits are tighter).
EPS_DP_TIME_LIMIT_MEDIUM="${EPS_DP_TIME_LIMIT_MEDIUM:-60}"
EPS_DP_TIME_LIMIT_LARGE="${EPS_DP_TIME_LIMIT_LARGE:-120}"

# Memory guardrail: abort if any single DP layer exceeds this.
# Much lighter than the full script (25M / 50M) to avoid OOM.
DP_MAX_STATES_MEDIUM="${DP_MAX_STATES_MEDIUM:-8000000}"
DP_MAX_STATES_LARGE="${DP_MAX_STATES_LARGE:-15000000}"

# Model training knobs
MLP_MAX_EPOCHS=500
MLP_PATIENCE=35
MLP_BATCH_SIZE=4096
MLP_LR=1e-3

LGBM_N_ESTIMATORS=800
LGBM_MAX_DEPTH=5
LGBM_LR=0.05

# Deployment sim
REPLICATES_MEDIUM=6
REPLICATES_LARGE=4

EPS_SEED=123
BEAM=60
PRUNE_FACTOR=2.0
ASSIGN_ALPHA=1.2
ASSIGN_UNIFORM_MIX=0.15

PMAX="${PMAX:-12}"

LOG_DIR="ADP/logs/models_light_medium_large"
MODEL_DIR="ADP/models/models_light_medium_large"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

# Shared pooled dataset cache (one per size/profile/data knobs).
# This is separate from POOL_DIR (memmap scratch), so it persists across runs.
POOLED_CACHE_DIR="${POOLED_CACHE_DIR:-$LOG_DIR/pooled_cache}"
mkdir -p "$POOLED_CACHE_DIR"

HAS_LGBM=1
if ! "$PYTHON_BIN" -c "import lightgbm" >/dev/null 2>&1; then
  HAS_LGBM=0
fi

# ============================================================
# Phase 1: within-size train + eval (medium, then large)
# ============================================================
for CATEGORY in "medium" "large"; do
  case "$CATEGORY" in
    medium)
      N_RANGE="100-200"; D_RANGE="5-15";  M_RANGE="8-16";  TARGET_UTIL="0.85"
      TRAIN_SEEDS="$TRAIN_SEEDS_MEDIUM"
      EVAL_SEEDS="$EVAL_SEEDS_MEDIUM"
      SAMPLES="$SAMPLES_MEDIUM"
      REPLICATES="$REPLICATES_MEDIUM"
      POOL_ARGS=(--pool-on-disk --pool-dtype float32 --pool-dir "$POOL_DIR")
      LABEL_MODE="optimal_path"
      DP_TIME_LIMIT="$DP_TIME_LIMIT_MEDIUM"
      EPS_DP_TIME_LIMIT="$EPS_DP_TIME_LIMIT_MEDIUM"
      DP_MAX_STATES="$DP_MAX_STATES_MEDIUM"
      OPT_PATH_N_PATHS="2"
      OPT_PATH_TOPUP_MAX="1000"
      OPT_PATH_TOPUP_TL="120.0"
      # --require-optimal-labels: discard any instance where DP
      # timed out; only well-solved instances contribute labels.
      REQUIRE_OPTIMAL_ARGS=(--require-optimal-labels)
      EVAL_TIME_LIMIT="30.0"
      ;;
    large)
      N_RANGE="250-500"; D_RANGE="10-30"; M_RANGE="25-40"; TARGET_UTIL="0.90"
      TRAIN_SEEDS="$TRAIN_SEEDS_LARGE"
      EVAL_SEEDS="$EVAL_SEEDS_LARGE"
      SAMPLES="$SAMPLES_LARGE"
      REPLICATES="$REPLICATES_LARGE"
      POOL_ARGS=(--pool-on-disk --pool-dtype float32 --pool-dir "$POOL_DIR")
      LABEL_MODE="optimal_path"
      DP_TIME_LIMIT="$DP_TIME_LIMIT_LARGE"
      EPS_DP_TIME_LIMIT="$EPS_DP_TIME_LIMIT_LARGE"
      DP_MAX_STATES="$DP_MAX_STATES_LARGE"
      OPT_PATH_N_PATHS="2"
      OPT_PATH_TOPUP_MAX="1000"
      OPT_PATH_TOPUP_TL="120.0"
      REQUIRE_OPTIMAL_ARGS=(--require-optimal-labels)
      EVAL_TIME_LIMIT="60.0"
      ;;
  esac

  echo "========================================================================"
  echo "PHASE 1 | CATEGORY=$CATEGORY  N_RANGE=$N_RANGE  D_RANGE=$D_RANGE  M_RANGE=$M_RANGE  target_util=$TARGET_UTIL"
  echo "PROFILE=$PROFILE  train_seeds=$TRAIN_SEEDS  samples=$SAMPLES"
  echo "dp_time_limit=${DP_TIME_LIMIT}s  dp_max_states=$DP_MAX_STATES  (timed-out instances discarded)"
  echo "========================================================================"


    for MODEL in "${MODELS[@]}"; do
      if [[ "$MODEL" == "lgbm" && "$HAS_LGBM" == "0" ]]; then
        echo "[skip] lightgbm not installed"
        continue
      fi

      CKPT="$MODEL_DIR/vhat_${CATEGORY}_${PROFILE}_${MODEL}.npz"
      POOLED_CSV="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.csv"
      POOLED_LOG="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.log"
      POOLED_DONE="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.done"
      EPS_CSV="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.csv"
      EPS_LOG="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.log"
      EPS_DONE="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.done"

      # Cache is per (CATEGORY, PROFILE) AND per data-generation knobs,
      # so changing seeds/ranges/samples won't accidentally reuse stale data.
      POOLED_DATA_CACHE="$POOLED_CACHE_DIR/pooled_${CATEGORY}_${PROFILE}_${LABEL_MODE}_feat${FEATURE_TAG}_p${PMAX}_s${SAMPLES}_train${TRAIN_SEEDS}_N${N_RANGE}_D${D_RANGE}_tu${TARGET_UTIL}.npz"
      if [[ "$FORCE_TRAIN" == "1" ]]; then
        rm -f "$POOLED_DATA_CACHE"
      fi

      echo ""
      echo ">>> TRAIN pooled: category=$CATEGORY model=$MODEL"

      STREAM_FIT_ARGS=()
      if [[ "$MODEL" == "linear" || "$MODEL" == "poly" ]]; then
        STREAM_FIT_ARGS=(--stream-fit --fit-chunk-size 200000)
      fi

      WORKERS_EFF="$WORKERS"
      if [[ "$WORKERS_EFF" -gt "$MAX_WORKERS" ]]; then
        WORKERS_EFF="$MAX_WORKERS"
      fi

      CACHE_ARGS=()
      if [[ -s "$POOLED_DATA_CACHE" ]]; then
        CACHE_ARGS=(--load-pooled-data "$POOLED_DATA_CACHE")
      else
        CACHE_ARGS=(--save-pooled-data "$POOLED_DATA_CACHE")
      fi

      # Don't skip if the shared pooled cache is missing (other models need it).
      if [[ "$RESUME" == "1" && "$FORCE_TRAIN" == "0" && -f "$POOLED_DONE" && -s "$CKPT" && -s "$POOLED_CSV" && -s "$POOLED_DATA_CACHE" ]]; then
        echo "[resume] skip TRAIN (ckpt+csv exist): $CKPT"
      else
        rm -f "$POOLED_DONE"
        "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
          --D 3 --N 40 --pmax "$PMAX" \
          --workers "$WORKERS_EFF" \
          --maxtasksperchild 1 \
          --train-seeds "$TRAIN_SEEDS" --samples-per-instance "$SAMPLES" \
          --train-N-range "$N_RANGE" --train-D-range "$D_RANGE" \
          --eval-seeds "$EVAL_SEEDS" \
          --eval-N-range "$N_RANGE" --eval-D-range "$D_RANGE" --eval-pmax "$PMAX" \
          --transferable-features --normalize --normalize-labels \
          "${FEATURE_ARGS[@]}" \
          --label-mode "$LABEL_MODE" \
          --optimal-path-n-paths "$OPT_PATH_N_PATHS" \
          --optimal-path-topup-max "$OPT_PATH_TOPUP_MAX" \
          --optimal-path-topup-dp-time-limit "$OPT_PATH_TOPUP_TL" \
          "${REQUIRE_OPTIMAL_ARGS[@]}" \
          --dp-time-limit "$DP_TIME_LIMIT" \
          --dp-max-states "$DP_MAX_STATES" \
          --eval-time-limit "$EVAL_TIME_LIMIT" \
          --target-util "$TARGET_UTIL" \
          "${POOL_ARGS[@]}" \
          "${STREAM_FIT_ARGS[@]}" \
          "${CACHE_ARGS[@]}" \
          --mlp-max-epochs "$MLP_MAX_EPOCHS" --mlp-patience "$MLP_PATIENCE" \
          --mlp-batch-size "$MLP_BATCH_SIZE" --mlp-lr "$MLP_LR" \
          --lgbm-n-estimators "$LGBM_N_ESTIMATORS" --lgbm-max-depth "$LGBM_MAX_DEPTH" \
          --lgbm-learning-rate "$LGBM_LR" \
          "${PROFILE_ARGS[@]}" \
          --model-type "$MODEL" --beams 2,5 \
          --save-model "$CKPT" \
          --out-csv "$POOLED_CSV" \
          2>&1 | tee "$POOLED_LOG"

        touch "$POOLED_DONE"
      fi
    echo ">>> DEPLOY epsilon-sim (within-size): category=$CATEGORY model=$MODEL"

    if [[ "$RESUME" == "1" && "$FORCE_EVAL" == "0" && -f "$EPS_DONE" && -s "$EPS_CSV" ]]; then
      echo "[resume] skip DEPLOY (csv exists): $EPS_CSV"
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
#   This mimics the real-world case: a model trained on a
#   specific size is deployed on slightly larger instances.
# ============================================================
echo ""
echo "========================================================================"
echo "PHASE 2 | CROSS-SIZE GENERALIZATION: train=medium -> eval=large"
echo "========================================================================"

SRC="medium"
TGT="large"

# Eval params for the TARGET (large)
N_RANGE_T="250-500"; D_RANGE_T="10-30"; TARGET_UTIL_T="0.90"
EVAL_SEEDS_CROSS="$EVAL_SEEDS_LARGE"    # same large held-out seeds

for MODEL in "${MODELS[@]}"; do
  if [[ "$MODEL" == "lgbm" && "$HAS_LGBM" == "0" ]]; then
    echo "[skip] lightgbm not installed"
    continue
  fi

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
      --workers "$WORKERS_EFF" \
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
