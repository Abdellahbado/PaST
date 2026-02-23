#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Thread tuning
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$(dirname "$0")"

# Nohup background launch
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
MAX_WORKERS="${MAX_WORKERS:-32}"
DRY_RUN="${DRY_RUN:-0}"

DP_MAX_STATES="${DP_MAX_STATES:-0}"
PMAX="${PMAX:-12}"

PROFILE="${PROFILE:-daily_tou}"
DAILY_PROFILE_MODE="${DAILY_PROFILE_MODE:-daily_tou}"

# Evaluation price modes. We always run deterministic first, then optionally a
# second pass with forecast_realized.
EVAL_PRICE_MODE_SECOND="${EVAL_PRICE_MODE_SECOND:-forecast_realized}"
EVAL_PRICE_NOISE_SIGMA="${EVAL_PRICE_NOISE_SIGMA:-0.25}"
EVAL_PRICE_NOISE_RHO="${EVAL_PRICE_NOISE_RHO:-0.9}"
EVAL_PRICE_SPIKE_PROB="${EVAL_PRICE_SPIKE_PROB:-0.02}"
EVAL_PRICE_SPIKE_MAG="${EVAL_PRICE_SPIKE_MAG:-2.0}"
EVAL_PRICE_SPIKE_DUR="${EVAL_PRICE_SPIKE_DUR:-2}"

EVAL_BEAMS="${EVAL_BEAMS:-2,5}"
EVAL_PRUNE_FACTOR="${EVAL_PRUNE_FACTOR:-2.0}"

POOL_DIR="${POOL_DIR:-ADP/tmp/pool}"
mkdir -p "$POOL_DIR"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-train_small_eval_medium_large_poly_gen}"
LOG_DIR="ADP/logs/${EXPERIMENT_TAG}"
MODEL_DIR="ADP/models/${EXPERIMENT_TAG}"
POOLED_CACHE_DIR="${POOLED_CACHE_DIR:-$LOG_DIR/pooled_cache}"
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$POOLED_CACHE_DIR"

# Small training config
TRAIN_CATEGORY="small"
TRAIN_N_RANGE="${TRAIN_N_RANGE:-20-60}"
TRAIN_D_RANGE="${TRAIN_D_RANGE:-2-4}"
TRAIN_TARGET_UTIL="${TRAIN_TARGET_UTIL:-0.80}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0-199}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-12000}"
TRAIN_LABEL_MODE="${TRAIN_LABEL_MODE:-optimal_path}"
TRAIN_DP_TIME_LIMIT="${TRAIN_DP_TIME_LIMIT:-300}"
TRAIN_OPT_PATH_N_PATHS="${TRAIN_OPT_PATH_N_PATHS:-2}"
TRAIN_OPT_PATH_TOPUP_MAX="${TRAIN_OPT_PATH_TOPUP_MAX:-0}"
TRAIN_OPT_PATH_TOPUP_TL="${TRAIN_OPT_PATH_TOPUP_TL:--1}"

# Medium eval config
MED_CATEGORY="medium"
MED_N_RANGE="${MED_N_RANGE:-100-200}"
MED_D_RANGE="${MED_D_RANGE:-5-15}"
MED_TARGET_UTIL="${MED_TARGET_UTIL:-0.85}"
MED_SEEDS="${MED_SEEDS:-400-429}"
MED_EVAL_TIME_LIMIT="${MED_EVAL_TIME_LIMIT:-30.0}"

# Large eval config
LARGE_CATEGORY="large"
LARGE_N_RANGE="${LARGE_N_RANGE:-250-500}"
LARGE_D_RANGE="${LARGE_D_RANGE:-10-30}"
LARGE_TARGET_UTIL="${LARGE_TARGET_UTIL:-0.90}"
LARGE_SEEDS="${LARGE_SEEDS:-500-519}"
LARGE_EVAL_TIME_LIMIT="${LARGE_EVAL_TIME_LIMIT:-60.0}"

# Epsilon config (parallel-machine)
EPS_REPLICATES_MED="${EPS_REPLICATES_MED:-5}"
EPS_REPLICATES_LARGE="${EPS_REPLICATES_LARGE:-5}"
EPS_SEED="${EPS_SEED:-456}"
EPS_BEAM="${EPS_BEAM:-80}"
EPS_PRUNE_FACTOR="${EPS_PRUNE_FACTOR:-2.0}"
EPS_DP_TIME_LIMIT_MED="${EPS_DP_TIME_LIMIT_MED:-300}"
EPS_DP_TIME_LIMIT_LARGE="${EPS_DP_TIME_LIMIT_LARGE:-900}"
EPS_ASSIGN_ALPHA="${EPS_ASSIGN_ALPHA:-1.2}"
EPS_ASSIGN_UNIFORM_MIX="${EPS_ASSIGN_UNIFORM_MIX:-0.15}"

MED_M_RANGE="${MED_M_RANGE:-8-16}"
LARGE_M_RANGE="${LARGE_M_RANGE:-25-40}"

# Polynomial variants
POLY_STD_L2="${POLY_STD_L2:-1e-3}"
POLY_GEN_L2="${POLY_GEN_L2:-1e-1}"
POLY_GEN_X_NOISE="${POLY_GEN_X_NOISE:-0.02}"
POLY_GEN_DROPOUT="${POLY_GEN_DROPOUT:-0.10}"
POLY_GEN_AUG_SEED="${POLY_GEN_AUG_SEED:-123}"

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    "$@"
  fi
}

WORKERS_EFF="$WORKERS"
[[ "$WORKERS_EFF" -gt "$MAX_WORKERS" ]] && WORKERS_EFF="$MAX_WORKERS"

PROFILE_ARGS=(--daily-price-profile "$DAILY_PROFILE_MODE")

EVAL_PRICE_ARGS_DET=(--eval-price-mode deterministic)

EVAL_PRICE_ARGS_SECOND=(--eval-price-mode "$EVAL_PRICE_MODE_SECOND")
if [[ "$EVAL_PRICE_MODE_SECOND" == "forecast_realized" ]]; then
  EVAL_PRICE_ARGS_SECOND+=(
    --eval-price-noise-sigma "$EVAL_PRICE_NOISE_SIGMA"
    --eval-price-noise-rho "$EVAL_PRICE_NOISE_RHO"
    --eval-price-spike-prob "$EVAL_PRICE_SPIKE_PROB"
    --eval-price-spike-mag "$EVAL_PRICE_SPIKE_MAG"
    --eval-price-spike-dur "$EVAL_PRICE_SPIKE_DUR"
  )
fi

# Shared pooled cache for SMALL training data (reused across variants)
POOLED_SMALL_NPZ="$POOLED_CACHE_DIR/pooled_${TRAIN_CATEGORY}_${PROFILE}_${TRAIN_LABEL_MODE}_p${PMAX}_s${TRAIN_SAMPLES}_train${TRAIN_SEEDS}_N${TRAIN_N_RANGE}_D${TRAIN_D_RANGE}_tu${TRAIN_TARGET_UTIL}.npz"

train_poly_variant() {
  local tag="$1"; shift
  local poly_l2="$1"; shift
  local x_noise="$1"; shift
  local dropout="$1"; shift
  local aug_seed="$1"; shift

  local ckpt="$MODEL_DIR/vhat_${TRAIN_CATEGORY}_${PROFILE}_poly_${tag}.npz"
  local train_csv="$LOG_DIR/train_${TRAIN_CATEGORY}_${PROFILE}_poly_${tag}.csv"
  local train_log="$LOG_DIR/train_${TRAIN_CATEGORY}_${PROFILE}_poly_${tag}.log"

  local robust_args=()
  if [[ "${x_noise}" != "0" && "${x_noise}" != "0.0" ]]; then
    robust_args+=(--train-x-noise-sigma "$x_noise")
  fi
  if [[ "${dropout}" != "0" && "${dropout}" != "0.0" ]]; then
    robust_args+=(--train-feature-dropout "$dropout")
  fi
  robust_args+=(--train-augment-seed "$aug_seed")

  local pooled_args=()
  if [[ -s "$POOLED_SMALL_NPZ" ]]; then
    pooled_args=(--load-pooled-data "$POOLED_SMALL_NPZ")
  else
    pooled_args=(--save-pooled-data "$POOLED_SMALL_NPZ")
  fi

  run_cmd "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
    --D 3 --N 40 --pmax "$PMAX" \
    --workers "$WORKERS_EFF" \
    --maxtasksperchild 1 \
    --train-seeds "$TRAIN_SEEDS" --samples-per-instance "$TRAIN_SAMPLES" \
    --train-N-range "$TRAIN_N_RANGE" --train-D-range "$TRAIN_D_RANGE" \
    --transferable-features --normalize --normalize-labels \
    --label-mode "$TRAIN_LABEL_MODE" \
    --optimal-path-n-paths "$TRAIN_OPT_PATH_N_PATHS" \
    --optimal-path-topup-max "$TRAIN_OPT_PATH_TOPUP_MAX" \
    --optimal-path-topup-dp-time-limit "$TRAIN_OPT_PATH_TOPUP_TL" \
    --require-optimal-labels \
    --dp-time-limit "$TRAIN_DP_TIME_LIMIT" \
    --dp-max-states "$DP_MAX_STATES" \
    --prune-factor "$EVAL_PRUNE_FACTOR" \
    --pool-on-disk --pool-dtype float32 --pool-dir "$POOL_DIR" \
    --stream-fit --fit-chunk-size 200000 \
    "${PROFILE_ARGS[@]}" \
    "${EVAL_PRICE_ARGS_DET[@]}" \
    "${pooled_args[@]}" \
    "${robust_args[@]}" \
    --model-type poly \
    --poly-l2 "$poly_l2" \
    --beams "$EVAL_BEAMS" \
    --save-model "$ckpt" \
    --out-csv "$train_csv" \
    2>&1 | tee "$train_log"

  echo "$ckpt"
}

eval_single_machine() {
  local tag="$1"; shift
  local ckpt="$1"; shift
  local tgt="$1"; shift
  local n_range="$1"; shift
  local d_range="$1"; shift
  local seeds="$1"; shift
  local target_util="$1"; shift
  local time_limit="$1"; shift
  local mode_suffix="$1"; shift
  local -a price_args=("$@")

  local out_csv="$LOG_DIR/eval_${tgt}_${PROFILE}_poly_${tag}_${mode_suffix}.csv"
  local out_log="$LOG_DIR/eval_${tgt}_${PROFILE}_poly_${tag}_${mode_suffix}.log"

  run_cmd "$PYTHON_BIN" sandbox/eval_pooled_vhat.py \
    --D 3 --N 40 --pmax "$PMAX" \
    --workers "$WORKERS_EFF" \
    --maxtasksperchild 1 \
    --load-model "$ckpt" \
    --eval-seeds "$seeds" \
    --eval-N-range "$n_range" --eval-D-range "$d_range" --eval-pmax "$PMAX" \
    --transferable-features --normalize --normalize-labels \
    --dp-max-states "$DP_MAX_STATES" \
    --eval-time-limit "$time_limit" \
    --target-util "$target_util" \
    --prune-factor "$EVAL_PRUNE_FACTOR" \
    "${PROFILE_ARGS[@]}" \
    "${price_args[@]}" \
    --beams "$EVAL_BEAMS" \
    --out-csv "$out_csv" \
    2>&1 | tee "$out_log"
}

eval_epsilon() {
  local tag="$1"; shift
  local ckpt="$1"; shift
  local tgt="$1"; shift
  local n_range="$1"; shift
  local d_range="$1"; shift
  local m_range="$1"; shift
  local target_util="$1"; shift
  local replicates="$1"; shift
  local dp_time_limit="$1"; shift
  local mode_suffix="$1"; shift
  local -a price_args=("$@")

  local out_csv="$LOG_DIR/epsilon_${tgt}_${PROFILE}_poly_${tag}_${mode_suffix}.csv"
  local out_log="$LOG_DIR/epsilon_${tgt}_${PROFILE}_poly_${tag}_${mode_suffix}.log"

  run_cmd "$PYTHON_BIN" sandbox/eval_epsilon_constraint_sim.py \
    --category "$tgt" \
    --N 40 --N-range "$n_range" \
    --M 5 --M-range "$m_range" \
    --D 3 --D-range "$d_range" \
    --replicates "$replicates" --seed "$EPS_SEED" \
    --pmax "$PMAX" --target-util "$target_util" \
    --price-mode daily_tou \
    "${PROFILE_ARGS[@]}" \
    "${price_args[@]}" \
    --assign-alpha "$EPS_ASSIGN_ALPHA" --assign-uniform-mix "$EPS_ASSIGN_UNIFORM_MIX" \
    --guided --beam "$EPS_BEAM" --prune-factor "$EPS_PRUNE_FACTOR" \
    --load-model "$ckpt" --transferable-features --normalize \
    --eps-dp-time-limit "$dp_time_limit" \
    --dp-max-states "$DP_MAX_STATES" \
    --out-csv "$out_csv" \
    2>&1 | tee "$out_log"
}

echo "[exp] TRAIN small -> EVAL medium & large (poly standard vs poly generalization)"
echo "[exp] LOG_DIR=$LOG_DIR"
echo "[exp] MODEL_DIR=$MODEL_DIR"
echo "[exp] POOLED_SMALL_NPZ=$POOLED_SMALL_NPZ"

CKPT_STD="$(train_poly_variant std "$POLY_STD_L2" 0.0 0.0 0)"
CKPT_GEN="$(train_poly_variant gen "$POLY_GEN_L2" "$POLY_GEN_X_NOISE" "$POLY_GEN_DROPOUT" "$POLY_GEN_AUG_SEED")"

# EVAL single-machine
for TAG in std gen; do
  CKPT_VAR="$CKPT_STD"; [[ "$TAG" == "gen" ]] && CKPT_VAR="$CKPT_GEN"

  # 1) Deterministic evaluation (standard)
  eval_single_machine "$TAG" "$CKPT_VAR" "$MED_CATEGORY" "$MED_N_RANGE" "$MED_D_RANGE" "$MED_SEEDS" "$MED_TARGET_UTIL" "$MED_EVAL_TIME_LIMIT" det "${EVAL_PRICE_ARGS_DET[@]}"
  eval_single_machine "$TAG" "$CKPT_VAR" "$LARGE_CATEGORY" "$LARGE_N_RANGE" "$LARGE_D_RANGE" "$LARGE_SEEDS" "$LARGE_TARGET_UTIL" "$LARGE_EVAL_TIME_LIMIT" det "${EVAL_PRICE_ARGS_DET[@]}"

  # 2) Second evaluation pass with forecast_realized (noisy realized costs)
  if [[ "$EVAL_PRICE_MODE_SECOND" != "deterministic" ]]; then
    eval_single_machine "$TAG" "$CKPT_VAR" "$MED_CATEGORY" "$MED_N_RANGE" "$MED_D_RANGE" "$MED_SEEDS" "$MED_TARGET_UTIL" "$MED_EVAL_TIME_LIMIT" "$EVAL_PRICE_MODE_SECOND" "${EVAL_PRICE_ARGS_SECOND[@]}"
    eval_single_machine "$TAG" "$CKPT_VAR" "$LARGE_CATEGORY" "$LARGE_N_RANGE" "$LARGE_D_RANGE" "$LARGE_SEEDS" "$LARGE_TARGET_UTIL" "$LARGE_EVAL_TIME_LIMIT" "$EVAL_PRICE_MODE_SECOND" "${EVAL_PRICE_ARGS_SECOND[@]}"
  fi

done

# EPSILON-CONSTRAINT sims
for TAG in std gen; do
  CKPT_VAR="$CKPT_STD"; [[ "$TAG" == "gen" ]] && CKPT_VAR="$CKPT_GEN"

  # 1) Deterministic evaluation (standard)
  eval_epsilon "$TAG" "$CKPT_VAR" "$MED_CATEGORY" "$MED_N_RANGE" "$MED_D_RANGE" "$MED_M_RANGE" "$MED_TARGET_UTIL" "$EPS_REPLICATES_MED" "$EPS_DP_TIME_LIMIT_MED" det "${EVAL_PRICE_ARGS_DET[@]}"
  eval_epsilon "$TAG" "$CKPT_VAR" "$LARGE_CATEGORY" "$LARGE_N_RANGE" "$LARGE_D_RANGE" "$LARGE_M_RANGE" "$LARGE_TARGET_UTIL" "$EPS_REPLICATES_LARGE" "$EPS_DP_TIME_LIMIT_LARGE" det "${EVAL_PRICE_ARGS_DET[@]}"

  # 2) Second evaluation pass with forecast_realized
  if [[ "$EVAL_PRICE_MODE_SECOND" != "deterministic" ]]; then
    eval_epsilon "$TAG" "$CKPT_VAR" "$MED_CATEGORY" "$MED_N_RANGE" "$MED_D_RANGE" "$MED_M_RANGE" "$MED_TARGET_UTIL" "$EPS_REPLICATES_MED" "$EPS_DP_TIME_LIMIT_MED" "$EVAL_PRICE_MODE_SECOND" "${EVAL_PRICE_ARGS_SECOND[@]}"
    eval_epsilon "$TAG" "$CKPT_VAR" "$LARGE_CATEGORY" "$LARGE_N_RANGE" "$LARGE_D_RANGE" "$LARGE_M_RANGE" "$LARGE_TARGET_UTIL" "$EPS_REPLICATES_LARGE" "$EPS_DP_TIME_LIMIT_LARGE" "$EVAL_PRICE_MODE_SECOND" "${EVAL_PRICE_ARGS_SECOND[@]}"
  fi

done

echo "[done] logs in: $LOG_DIR"
echo "[done] models in: $MODEL_DIR"
