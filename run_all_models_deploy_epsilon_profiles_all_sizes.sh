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
# nohup launcher (default on)
#
# By default, the script re-launches itself under nohup in the background so
# terminal/SSH disconnects won't stop the run.
#
# Disable:
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
# Comprehensive deployment-like sweep:
#   sizes (small/medium/large) × profiles (2) × models (4)
#   Train pooled Vhat (within-profile) on variable N/D ranges,
#   then evaluate under epsilon-constraint multi-machine deployment.
# ============================================================

# Python executable to use (override if needed):
#   PYTHON_BIN=/path/to/python bash PaST/run_all_models_deploy_epsilon_profiles_all_sizes.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

# Multiprocessing workers for pooled data collection.
# On HPC, mp.cpu_count() can be very large and may OOM / violate process limits.
WORKERS="${WORKERS:-4}"

# Hard cap for spawned Python worker processes (safety on memory-limited nodes).
# Override if your node has ample RAM:
#   MAX_WORKERS=128 WORKERS=128 bash ...
MAX_WORKERS="${MAX_WORKERS:-32}"

# Directory for on-disk pooling (memmap). Using an explicit path avoids relying
# on system temp dirs (which can be small/slow on some clusters).
POOL_DIR="${POOL_DIR:-ADP/tmp/pool}"
mkdir -p "$POOL_DIR"

# Optional: reuse ONE pooled dataset across multiple models for the same
# (CATEGORY, PROFILE). This avoids repeating expensive DP labeling per model.
#
# Enabled by default:
#   REUSE_POOLED_ACROSS_MODELS=1 bash ...
# Disable:
#   REUSE_POOLED_ACROSS_MODELS=0 bash ...
REUSE_POOLED_ACROSS_MODELS="${REUSE_POOLED_ACROSS_MODELS:-1}"

# ============================================================
# Build Cython sparse-DP extension (native C hash maps).
# Compiles once; subsequent runs are no-ops if .so is up to date.
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

# Start point for long runs.
# Default: start from medium so reruns don't waste time on already-finished small.
# Override:
#   START_CATEGORY=small  bash PaST/run_all_models_deploy_epsilon_profiles_all_sizes.sh
#   START_CATEGORY=large  bash PaST/run_all_models_deploy_epsilon_profiles_all_sizes.sh
START_CATEGORY="${START_CATEGORY:-medium}"

# Two profiles (train+eval within profile; no cross-profile testing here)
# Comma-separated profile list override, e.g.:
#   PROFILE_LIST_CSV=daily_tou bash ...
PROFILE_LIST_CSV="${PROFILE_LIST_CSV:-daily_tou,generate_data}"
IFS=',' read -r -a PROFILES <<< "$PROFILE_LIST_CSV"
# Comma-separated model list override, e.g.:
#   MODEL_LIST_CSV=poly,mlp,lgbm bash ...
MODEL_LIST_CSV="${MODEL_LIST_CSV:-linear,poly,mlp,lgbm}"
IFS=',' read -r -a MODELS <<< "$MODEL_LIST_CSV"
CATEGORIES=("small" "medium" "large")

# Train/eval seeds
# - More seeds reduces overfitting risk (better than cranking epochs).
# - You can shrink these if runtime is too high.
TRAIN_SEEDS_SMALL="0-99"
TRAIN_SEEDS_MEDIUM="0-199"
TRAIN_SEEDS_LARGE="0-149"

EVAL_SEEDS_SMALL="300-339"
EVAL_SEEDS_MEDIUM="400-429"
EVAL_SEEDS_LARGE="500-519"

# Samples per training instance (category-scaled for runtime)
SAMPLES_SMALL="${SAMPLES_SMALL:-4000}"
# For optimal_path labeling, samples per instance should match O(T*n_paths)
# so we don't trigger top-up. With D_hi=15, n_paths=2 -> ~600.
# With D_hi=30, n_paths=2 -> ~1200.
SAMPLES_MEDIUM="${SAMPLES_MEDIUM:-600}"
SAMPLES_LARGE="${SAMPLES_LARGE:-1200}"

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

# Shared pooled dataset cache directory (only used when REUSE_POOLED_ACROSS_MODELS=1)
POOLED_CACHE_DIR="${POOLED_CACHE_DIR:-$LOG_DIR/pooled_cache}"
mkdir -p "$POOLED_CACHE_DIR"

# Convenience: LightGBM availability check
HAS_LGBM=1
if ! "$PYTHON_BIN" -c "import lightgbm" >/dev/null 2>&1; then
  HAS_LGBM=0
fi

for CATEGORY in "${CATEGORIES[@]}"; do
  if [[ "${STARTED:-0}" == "0" ]]; then
    if [[ "$CATEGORY" != "$START_CATEGORY" ]]; then
      echo "[skip] CATEGORY=$CATEGORY (START_CATEGORY=$START_CATEGORY)"
      continue
    fi
    STARTED=1
  fi
  case "$CATEGORY" in
    small)
      N_RANGE="20-60";  D_RANGE="2-4";   M_RANGE="3-7";   TARGET_UTIL="0.80";
      TRAIN_SEEDS="$TRAIN_SEEDS_SMALL"; EVAL_SEEDS="$EVAL_SEEDS_SMALL";
      SAMPLES="$SAMPLES_SMALL"; REPLICATES="$REPLICATES_SMALL";
      POOL_ARGS=()
      LABEL_MODE="subproblem"
      DP_TIME_LIMIT="-1"
      EPS_DP_TIME_LIMIT="-1"
      DP_MAX_STATES="0"
      OPT_PATH_N_PATHS="1"
      OPT_PATH_TOPUP_MAX="${OPT_PATH_TOPUP_MAX_SMALL:-0}"
      OPT_PATH_TOPUP_TL="${OPT_PATH_TOPUP_TL_SMALL:-0.2}"
      REQUIRE_OPTIMAL_ARGS=()
      EVAL_TIME_LIMIT="-1"
      ;;
    medium)
      N_RANGE="100-200"; D_RANGE="5-15";  M_RANGE="8-16";  TARGET_UTIL="0.85";
      TRAIN_SEEDS="$TRAIN_SEEDS_MEDIUM"; EVAL_SEEDS="$EVAL_SEEDS_MEDIUM";
      SAMPLES="$SAMPLES_MEDIUM"; REPLICATES="$REPLICATES_MEDIUM";
      POOL_ARGS=(--pool-on-disk --pool-dtype float32 --pool-dir "$POOL_DIR")
      LABEL_MODE="optimal_path"
      DP_TIME_LIMIT="300"
      EPS_DP_TIME_LIMIT="300"
      DP_MAX_STATES="25000000"
      OPT_PATH_N_PATHS="2"
      OPT_PATH_TOPUP_MAX="${OPT_PATH_TOPUP_MAX_MEDIUM:-0}"
      OPT_PATH_TOPUP_TL="${OPT_PATH_TOPUP_TL_MEDIUM:--1}"
      REQUIRE_OPTIMAL_ARGS=(--require-optimal-labels)
      EVAL_TIME_LIMIT="30.0"
      ;;
    large)
      N_RANGE="250-500"; D_RANGE="10-30"; M_RANGE="25-40"; TARGET_UTIL="0.90";
      TRAIN_SEEDS="$TRAIN_SEEDS_LARGE"; EVAL_SEEDS="$EVAL_SEEDS_LARGE";
      SAMPLES="$SAMPLES_LARGE"; REPLICATES="$REPLICATES_LARGE";
      POOL_ARGS=(--pool-on-disk --pool-dtype float32 --pool-dir "$POOL_DIR")
      LABEL_MODE="optimal_path"
      DP_TIME_LIMIT="900"
      EPS_DP_TIME_LIMIT="900"
      DP_MAX_STATES="50000000"
      OPT_PATH_N_PATHS="2"
      OPT_PATH_TOPUP_MAX="${OPT_PATH_TOPUP_MAX_LARGE:-0}"
      OPT_PATH_TOPUP_TL="${OPT_PATH_TOPUP_TL_LARGE:--1}"
      REQUIRE_OPTIMAL_ARGS=(--require-optimal-labels)
      EVAL_TIME_LIMIT="60.0"
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

    # Cache is per (CATEGORY, PROFILE) AND per data-generation knobs,
    # so changing seeds/ranges/samples won't accidentally reuse stale data.
    SHARED_POOLED_NPZ="$POOLED_CACHE_DIR/pooled_${CATEGORY}_${PROFILE}_${LABEL_MODE}_p${PMAX}_s${SAMPLES}_train${TRAIN_SEEDS}_N${N_RANGE}_D${D_RANGE}_tu${TARGET_UTIL}.npz"
    if [[ "$REUSE_POOLED_ACROSS_MODELS" == "1" && "$FORCE_TRAIN" == "1" ]]; then
      rm -f "$SHARED_POOLED_NPZ"
    fi

    for MODEL in "${MODELS[@]}"; do
      if [[ "$MODEL" == "lgbm" && "$HAS_LGBM" == "0" ]]; then
        echo "[skip] lightgbm not installed for $PYTHON_BIN"
        continue
      fi

      CKPT="$MODEL_DIR/vhat_${CATEGORY}_${PROFILE}_${MODEL}.npz"
      POOLED_CSV="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.csv"
      POOLED_LOG="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.log"
      POOLED_DONE="$LOG_DIR/pooled_${CATEGORY}_${PROFILE}_${MODEL}.done"
      EPS_CSV="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.csv"
      EPS_LOG="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.log"
      EPS_DONE="$LOG_DIR/epsilon_${CATEGORY}_${PROFILE}_${MODEL}.done"

      echo ""
      echo ">>> TRAIN pooled: category=$CATEGORY model=$MODEL profile=$PROFILE"

      STREAM_FIT_ARGS=()
      if [[ "$MODEL" == "linear" || "$MODEL" == "poly" ]]; then
        # Helps keep RAM stable when pooling on disk.
        STREAM_FIT_ARGS=(--stream-fit --fit-chunk-size 200000)
      fi

      # If pooled-cache reuse is enabled, don't skip TRAIN if the shared cache
      # file is missing (other models will need it).
      if [[ "$RESUME" == "1" && "$FORCE_TRAIN" == "0" && -f "$POOLED_DONE" && -s "$CKPT" && -s "$POOLED_CSV" && ( "$REUSE_POOLED_ACROSS_MODELS" != "1" || -s "$SHARED_POOLED_NPZ" ) ]]; then
        echo "[resume] skip TRAIN (ckpt+csv exist): $CKPT"
      else
        rm -f "$POOLED_DONE"
        # Cap worker count to avoid spawning too many heavy Python processes.
        WORKERS_EFF="$WORKERS"
        if [[ "$WORKERS_EFF" -gt "$MAX_WORKERS" ]]; then
          WORKERS_EFF="$MAX_WORKERS"
        fi
        POOLED_DATA_ARGS=()
        if [[ "$REUSE_POOLED_ACROSS_MODELS" == "1" ]]; then
          if [[ -s "$SHARED_POOLED_NPZ" ]]; then
            POOLED_DATA_ARGS=(--load-pooled-data "$SHARED_POOLED_NPZ")
          else
            POOLED_DATA_ARGS=(--save-pooled-data "$SHARED_POOLED_NPZ")
          fi
        fi

        "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
          --D 3 --N 40 --pmax "$PMAX" \
          --workers "$WORKERS_EFF" \
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
          --dp-max-states "$DP_MAX_STATES" \
          --eval-time-limit "$EVAL_TIME_LIMIT" \
          --target-util "$TARGET_UTIL" \
          "${POOL_ARGS[@]}" \
          "${STREAM_FIT_ARGS[@]}" \
          "${POOLED_DATA_ARGS[@]}" \
          --mlp-max-epochs "$MLP_MAX_EPOCHS" --mlp-patience "$MLP_PATIENCE" \
          --mlp-batch-size "$MLP_BATCH_SIZE" --mlp-lr "$MLP_LR" \
          --lgbm-n-estimators "$LGBM_N_ESTIMATORS" --lgbm-max-depth "$LGBM_MAX_DEPTH" --lgbm-learning-rate "$LGBM_LR" \
          "${PROFILE_ARGS[@]}" \
          --model-type "$MODEL" --beams 2,5 \
          --save-model "$CKPT" \
          --out-csv "$POOLED_CSV" \
          2>&1 | tee "$POOLED_LOG"

        touch "$POOLED_DONE"
      fi

      echo ">>> DEPLOY epsilon-sim: category=$CATEGORY model=$MODEL profile=$PROFILE"

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

      echo ">>> Done: category=$CATEGORY model=$MODEL profile=$PROFILE"
    done
  done

done

echo "Done. Logs: $LOG_DIR  Models: $MODEL_DIR"
