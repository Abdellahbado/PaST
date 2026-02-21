#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

###############################################################################
# run_train_medium_eval_large.sh
#
# Dynamic script for the "train-on-medium → evaluate-on-large" experiment.
#
# Pipeline per model variant:
#   1. TRAIN on medium-sized instances (or reuse existing checkpoint)
#   2. EVAL single-machine: learned vs exact DP on *large* instances
#   3. DEPLOY epsilon-constraint sim on *large* instances
#
# Adding a new model variant:
#   Just edit MODEL_LIST_CSV or pass it as env var:
#     MODEL_LIST_CSV=poly,mlp,lgbm,linear bash run_train_medium_eval_large.sh
#
# Key environment overrides:
#   WORKERS=8            — parallel workers (auto-capped by MAX_WORKERS)
#   SKIP_TRAIN=1         — skip training, load existing models
#   FORCE_TRAIN=1        — retrain even if checkpoint exists
#   SKIP_EVAL=1          — skip single-machine evaluation
#   SKIP_EPSILON=1       — skip epsilon-constraint simulation
#   PROFILE_LIST_CSV=... — comma-separated price profiles
#   MODEL_LIST_CSV=...   — comma-separated model types
#   EVAL_BEAMS=2,5,10    — beam widths for single-machine eval
#   DP_MAX_STATES=0      — 0=auto from RAM
#   DRY_RUN=1            — print commands without running
###############################################################################

# --- Thread tuning (avoid oversubscription on HPC) -------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$(dirname "$0")"

# --- Nohup background launch -----------------------------------------------
if [[ "${NOHUP:-1}" == "1" && -z "${IN_NOHUP:-}" ]]; then
  mkdir -p "ADP/logs/nohup"
  TS="$(date +"%Y%m%d_%H%M%S")"
  LOG_FILE="ADP/logs/nohup/$(basename "$0" .sh)_${TS}.log"
  echo "[nohup] launching in background; log: $LOG_FILE"
  IN_NOHUP=1 NOHUP=0 nohup bash "$0" "$@" >"$LOG_FILE" 2>&1 &
  echo "[nohup] pid=$!"
  exit 0
fi

# --- Python / build ---------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[build] Compiling Cython sparse-DP extension ..."
if ! command -v gcc &>/dev/null; then
  conda install -y -c conda-forge compilers 2>/dev/null || true
  conda install -y -c conda-forge gcc_linux-64 gxx_linux-64 2>/dev/null || true
  conda install -y -c conda-forge c-compiler cxx-compiler 2>/dev/null || true
fi
if ! command -v gcc &>/dev/null; then
  if command -v x86_64-conda-linux-gnu-cc &>/dev/null; then
    export CC="$(command -v x86_64-conda-linux-gnu-cc)"
    command -v x86_64-conda-linux-gnu-c++ &>/dev/null && export CXX="$(command -v x86_64-conda-linux-gnu-c++)"
    echo "[build] Using conda wrapper compiler: CC=$CC"
  elif command -v cc &>/dev/null; then
    export CC="$(command -v cc)"; echo "[build] Using CC=$CC"
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
then echo "[build] done"
else echo "[build] WARNING: Cython not available; falling back to slower solver."
fi

# ============================================================================
# Configuration
# ============================================================================

# --- Models -----------------------------------------------------------------
MODEL_LIST_CSV="${MODEL_LIST_CSV:-poly,mlp,lgbm}"
IFS=',' read -r -a MODELS <<< "$MODEL_LIST_CSV"

# --- Profiles ---------------------------------------------------------------
PROFILE_LIST_CSV="${PROFILE_LIST_CSV:-daily_tou}"
IFS=',' read -r -a PROFILES <<< "$PROFILE_LIST_CSV"

# --- Workers ----------------------------------------------------------------
WORKERS="${WORKERS:-4}"
MAX_WORKERS="${MAX_WORKERS:-32}"

# --- Pool directory (on-disk memmap) ----------------------------------------
POOL_DIR="${POOL_DIR:-ADP/tmp/pool}"
mkdir -p "$POOL_DIR"

# --- Reuse pooled data across models (avoids repeated DP labeling) ----------
REUSE_POOLED_ACROSS_MODELS="${REUSE_POOLED_ACROSS_MODELS:-1}"

# --- Phase control ----------------------------------------------------------
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EPSILON="${SKIP_EPSILON:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

# --- DP states (0 = auto from RAM) -----------------------------------------
DP_MAX_STATES="${DP_MAX_STATES:-0}"

# --- Common -----------------------------------------------------------------
PMAX=12

# ============================================================================
# TRAINING CONFIG (medium-sized instances)
# ============================================================================
TRAIN_CATEGORY="medium"
TRAIN_N_RANGE="100-200"
TRAIN_D_RANGE="5-15"
TRAIN_TARGET_UTIL="0.85"
TRAIN_SEEDS="${TRAIN_SEEDS:-0-199}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-8000}"
TRAIN_LABEL_MODE="optimal_path"
TRAIN_DP_TIME_LIMIT="300"
TRAIN_OPT_PATH_N_PATHS="2"
TRAIN_OPT_PATH_TOPUP_MAX="${TRAIN_OPT_PATH_TOPUP_MAX:-0}"
TRAIN_OPT_PATH_TOPUP_TL="${TRAIN_OPT_PATH_TOPUP_TL:--1}"

# Model hyperparameters
MLP_MAX_EPOCHS="${MLP_MAX_EPOCHS:-600}"
MLP_PATIENCE="${MLP_PATIENCE:-40}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-4096}"
MLP_LR="${MLP_LR:-1e-3}"
LGBM_N_ESTIMATORS="${LGBM_N_ESTIMATORS:-2000}"
LGBM_MAX_DEPTH="${LGBM_MAX_DEPTH:-5}"
LGBM_LR="${LGBM_LR:-0.05}"

# ============================================================================
# EVALUATION CONFIG (large instances)
# ============================================================================
EVAL_CATEGORY="large"
EVAL_N_RANGE="250-500"
EVAL_D_RANGE="10-30"
EVAL_M_RANGE="25-40"
EVAL_TARGET_UTIL="0.90"
EVAL_SEEDS="${EVAL_SEEDS:-500-519}"
EVAL_BEAMS="${EVAL_BEAMS:-2,5}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-120.0}"

# Epsilon-constraint knobs
EPS_REPLICATES="${EPS_REPLICATES:-5}"
EPS_SEED="${EPS_SEED:-456}"
EPS_BEAM="${EPS_BEAM:-80}"
EPS_PRUNE_FACTOR="${EPS_PRUNE_FACTOR:-2.0}"
EPS_DP_TIME_LIMIT="${EPS_DP_TIME_LIMIT:-900}"
EPS_ASSIGN_ALPHA="${EPS_ASSIGN_ALPHA:-1.2}"
EPS_ASSIGN_UNIFORM_MIX="${EPS_ASSIGN_UNIFORM_MIX:-0.15}"

# ============================================================================
# Output directories
# ============================================================================
EXPERIMENT_TAG="${EXPERIMENT_TAG:-train_medium_eval_large}"
LOG_DIR="ADP/logs/${EXPERIMENT_TAG}"
MODEL_DIR="ADP/models/${EXPERIMENT_TAG}"
POOLED_CACHE_DIR="${POOLED_CACHE_DIR:-$LOG_DIR/pooled_cache}"
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$POOLED_CACHE_DIR"

# --- LightGBM check --------------------------------------------------------
HAS_LGBM=1
if ! "$PYTHON_BIN" -c "import lightgbm" >/dev/null 2>&1; then HAS_LGBM=0; fi

# ============================================================================
# Helper: run or dry-run
# ============================================================================
run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    "$@"
  fi
}

# ============================================================================
# Print experiment plan
# ============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Train-on-Medium → Evaluate-on-Large  Experiment                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  TRAINING:   ${TRAIN_CATEGORY}  N=${TRAIN_N_RANGE}  D=${TRAIN_D_RANGE}  seeds=${TRAIN_SEEDS}  util=${TRAIN_TARGET_UTIL}"
echo "  EVALUATION: ${EVAL_CATEGORY}   N=${EVAL_N_RANGE}  D=${EVAL_D_RANGE}  seeds=${EVAL_SEEDS}  util=${EVAL_TARGET_UTIL}"
echo "  EPSILON:    M=${EVAL_M_RANGE}  replicates=${EPS_REPLICATES}  beam=${EPS_BEAM}  prune=${EPS_PRUNE_FACTOR}"
echo "  MODELS:     ${MODEL_LIST_CSV}"
echo "  PROFILES:   ${PROFILE_LIST_CSV}"
echo "  WORKERS:    ${WORKERS}  (max ${MAX_WORKERS})"
echo "  DP_MAX_STATES: ${DP_MAX_STATES} (0=auto)"
echo "  LOG_DIR:    ${LOG_DIR}"
echo "  MODEL_DIR:  ${MODEL_DIR}"
echo ""

# ============================================================================
# Main loop: PROFILE × MODEL
# ============================================================================
for PROFILE in "${PROFILES[@]}"; do
  PROFILE_ARGS=()
  if [[ "$PROFILE" == "generate_data" ]]; then
    PROFILE_ARGS+=(--daily-price-profile generate_data --gd-seed 20260109 --gd-ck-low 1 --gd-ck-high 8)
  else
    PROFILE_ARGS+=(--daily-price-profile daily_tou)
  fi

  echo "========================================================================"
  echo "PROFILE=$PROFILE"
  echo "========================================================================"

  # Shared pooled-data cache key → reuse across models for same profile
  SHARED_POOLED_NPZ="$POOLED_CACHE_DIR/pooled_${TRAIN_CATEGORY}_${PROFILE}_${TRAIN_LABEL_MODE}_p${PMAX}_s${TRAIN_SAMPLES}_train${TRAIN_SEEDS}_N${TRAIN_N_RANGE}_D${TRAIN_D_RANGE}_tu${TRAIN_TARGET_UTIL}.npz"
  if [[ "$REUSE_POOLED_ACROSS_MODELS" == "1" && "$FORCE_TRAIN" == "1" ]]; then
    rm -f "$SHARED_POOLED_NPZ"
  fi

  for MODEL in "${MODELS[@]}"; do
    if [[ "$MODEL" == "lgbm" && "$HAS_LGBM" == "0" ]]; then
      echo "[skip] lightgbm not installed"; continue
    fi

    CKPT="$MODEL_DIR/vhat_${TRAIN_CATEGORY}_${PROFILE}_${MODEL}.npz"
    TRAIN_CSV="$LOG_DIR/train_${TRAIN_CATEGORY}_${PROFILE}_${MODEL}.csv"
    TRAIN_DONE="$LOG_DIR/train_${TRAIN_CATEGORY}_${PROFILE}_${MODEL}.done"
    EVAL_CSV="$LOG_DIR/eval_${EVAL_CATEGORY}_${PROFILE}_${MODEL}.csv"
    EVAL_DONE="$LOG_DIR/eval_${EVAL_CATEGORY}_${PROFILE}_${MODEL}.done"
    EPS_CSV="$LOG_DIR/epsilon_${EVAL_CATEGORY}_${PROFILE}_${MODEL}.csv"
    EPS_DONE="$LOG_DIR/epsilon_${EVAL_CATEGORY}_${PROFILE}_${MODEL}.done"

    # ==================================================================
    # PHASE 1: TRAIN on medium
    # ==================================================================
    if [[ "$SKIP_TRAIN" == "0" ]]; then
      echo ""
      echo ">>> TRAIN: category=${TRAIN_CATEGORY} model=${MODEL} profile=${PROFILE}"

      if [[ "$RESUME" == "1" && "$FORCE_TRAIN" == "0" && -f "$TRAIN_DONE" && -s "$CKPT" ]]; then
        echo "[resume] skip TRAIN (checkpoint exists): $CKPT"
      else
        rm -f "$TRAIN_DONE"

        WORKERS_EFF="$WORKERS"
        [[ "$WORKERS_EFF" -gt "$MAX_WORKERS" ]] && WORKERS_EFF="$MAX_WORKERS"

        # Stream-fit for linear/poly to manage memory
        STREAM_FIT_ARGS=()
        if [[ "$MODEL" == "linear" || "$MODEL" == "poly" ]]; then
          STREAM_FIT_ARGS=(--stream-fit --fit-chunk-size 200000)
        fi

        # Pooled data reuse
        POOLED_DATA_ARGS=()
        if [[ "$REUSE_POOLED_ACROSS_MODELS" == "1" ]]; then
          if [[ -s "$SHARED_POOLED_NPZ" ]]; then
            POOLED_DATA_ARGS=(--load-pooled-data "$SHARED_POOLED_NPZ")
          else
            POOLED_DATA_ARGS=(--save-pooled-data "$SHARED_POOLED_NPZ")
          fi
        fi

        # Training: train on medium, also do brief in-domain eval for sanity
        run_cmd "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
          --D 3 --N 40 --pmax "$PMAX" \
          --workers "$WORKERS_EFF" \
          --maxtasksperchild 1 \
          --train-seeds "$TRAIN_SEEDS" --samples-per-instance "$TRAIN_SAMPLES" \
          --train-N-range "$TRAIN_N_RANGE" --train-D-range "$TRAIN_D_RANGE" \
          --eval-seeds "${EVAL_SEEDS}" \
          --eval-N-range "$EVAL_N_RANGE" --eval-D-range "$EVAL_D_RANGE" --eval-pmax "$PMAX" \
          --transferable-features --normalize --normalize-labels \
          --label-mode "$TRAIN_LABEL_MODE" \
          --optimal-path-n-paths "$TRAIN_OPT_PATH_N_PATHS" \
          --optimal-path-topup-max "$TRAIN_OPT_PATH_TOPUP_MAX" \
          --optimal-path-topup-dp-time-limit "$TRAIN_OPT_PATH_TOPUP_TL" \
          --require-optimal-labels \
          --dp-time-limit "$TRAIN_DP_TIME_LIMIT" \
          --dp-max-states "$DP_MAX_STATES" \
          --eval-time-limit "$EVAL_TIME_LIMIT" \
          --target-util "$TRAIN_TARGET_UTIL" \
          --pool-on-disk --pool-dtype float32 --pool-dir "$POOL_DIR" \
          "${STREAM_FIT_ARGS[@]}" \
          "${POOLED_DATA_ARGS[@]}" \
          --mlp-max-epochs "$MLP_MAX_EPOCHS" --mlp-patience "$MLP_PATIENCE" \
          --mlp-batch-size "$MLP_BATCH_SIZE" --mlp-lr "$MLP_LR" \
          --lgbm-n-estimators "$LGBM_N_ESTIMATORS" --lgbm-max-depth "$LGBM_MAX_DEPTH" --lgbm-learning-rate "$LGBM_LR" \
          "${PROFILE_ARGS[@]}" \
          --model-type "$MODEL" --beams "$EVAL_BEAMS" \
          --save-model "$CKPT" \
          --out-csv "$TRAIN_CSV"

        touch "$TRAIN_DONE"
      fi
    else
      echo ""
      echo ">>> [SKIP_TRAIN] model=${MODEL} profile=${PROFILE}"
      if [[ ! -s "$CKPT" ]]; then
        echo "[ERROR] SKIP_TRAIN=1 but checkpoint missing: $CKPT"
        continue
      fi
    fi

    # ==================================================================
    # PHASE 2: EVAL single-machine on large (learned vs exact DP)
    # ==================================================================
    if [[ "$SKIP_EVAL" == "0" ]]; then
      echo ""
      echo ">>> EVAL on LARGE: model=${MODEL} profile=${PROFILE}  (learned vs exact DP)"

      if [[ "$RESUME" == "1" && "$FORCE_EVAL" == "0" && -f "$EVAL_DONE" && -s "$EVAL_CSV" ]]; then
        echo "[resume] skip EVAL (csv exists): $EVAL_CSV"
      else
        rm -f "$EVAL_DONE"

        WORKERS_EFF="$WORKERS"
        [[ "$WORKERS_EFF" -gt "$MAX_WORKERS" ]] && WORKERS_EFF="$MAX_WORKERS"

        run_cmd "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
          --D 3 --N 40 --pmax "$PMAX" \
          --workers "$WORKERS_EFF" \
          --maxtasksperchild 1 \
          --load-model "$CKPT" \
          --eval-seeds "$EVAL_SEEDS" \
          --eval-N-range "$EVAL_N_RANGE" --eval-D-range "$EVAL_D_RANGE" --eval-pmax "$PMAX" \
          --transferable-features --normalize --normalize-labels \
          --dp-max-states "$DP_MAX_STATES" \
          --eval-time-limit "$EVAL_TIME_LIMIT" \
          --target-util "$EVAL_TARGET_UTIL" \
          "${PROFILE_ARGS[@]}" \
          --beams "$EVAL_BEAMS" \
          --out-csv "$EVAL_CSV"

        touch "$EVAL_DONE"
      fi
    fi

    # ==================================================================
    # PHASE 3: EPSILON-CONSTRAINT sim on large (learned vs exact)
    # ==================================================================
    if [[ "$SKIP_EPSILON" == "0" ]]; then
      echo ""
      echo ">>> EPSILON-CONSTRAINT on LARGE: model=${MODEL} profile=${PROFILE}"

      if [[ "$RESUME" == "1" && "$FORCE_EVAL" == "0" && -f "$EPS_DONE" && -s "$EPS_CSV" ]]; then
        echo "[resume] skip EPSILON (csv exists): $EPS_CSV"
      else
        rm -f "$EPS_DONE"

        run_cmd "$PYTHON_BIN" sandbox/eval_epsilon_constraint_sim.py \
          --category "$EVAL_CATEGORY" \
          --N 40 --N-range "$EVAL_N_RANGE" \
          --M 5 --M-range "$EVAL_M_RANGE" \
          --D 3 --D-range "$EVAL_D_RANGE" \
          --replicates "$EPS_REPLICATES" --seed "$EPS_SEED" \
          --pmax "$PMAX" --target-util "$EVAL_TARGET_UTIL" \
          --price-mode daily_tou \
          "${PROFILE_ARGS[@]}" \
          --assign-alpha "$EPS_ASSIGN_ALPHA" --assign-uniform-mix "$EPS_ASSIGN_UNIFORM_MIX" \
          --guided --beam "$EPS_BEAM" --prune-factor "$EPS_PRUNE_FACTOR" \
          --load-model "$CKPT" --transferable-features --normalize \
          --eps-dp-time-limit "$EPS_DP_TIME_LIMIT" \
          --dp-max-states "$DP_MAX_STATES" \
          --out-csv "$EPS_CSV"

        touch "$EPS_DONE"
      fi
    fi

    echo ""
    echo ">>> Done: model=${MODEL} profile=${PROFILE}"
    echo ""
  done
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Experiment complete!                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Logs:       $LOG_DIR"
echo "  Models:     $MODEL_DIR"
echo ""
echo "  Training CSVs (medium, in-domain sanity):"
for f in "$LOG_DIR"/train_*.csv; do
  [[ -f "$f" ]] && echo "    $f"
done
echo ""
echo "  Eval CSVs (large, learned vs exact DP):"
for f in "$LOG_DIR"/eval_*.csv; do
  [[ -f "$f" ]] && echo "    $f"
done
echo ""
echo "  Epsilon-constraint CSVs (large, multi-machine):"
for f in "$LOG_DIR"/epsilon_*.csv; do
  [[ -f "$f" ]] && echo "    $f"
done
echo ""
echo "Done."
