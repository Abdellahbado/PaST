#!/usr/bin/env bash
# ============================================================================
# Experiment E — Epsilon-Constraint Boundary (Bi-Objective: Energy + Makespan)
#
# Goal: Evaluate the learned Vhat model in the epsilon-constraint (multi-machine
# decomposition) setting where it should shine. As epsilon (deadline) tightens,
# the per-machine subproblems become harder and the learned guidance becomes
# increasingly valuable vs. the price heuristic.
#
# Uses sandbox/eval_epsilon_constraint_sim.py which:
#   1. Generates parallel-machine instances (New Benchmark/new_data.py style)
#   2. Assigns jobs to machines via biased-softmax
#   3. Sweeps epsilon from K down to min feasible (accelerated: eps = Cmax-1)
#   4. Compares: exact, guided (learned-Vhat), guided (price-Vhat)
#
# Loads model checkpoints from Exp B-hard (no retraining).
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-1}"

LOG_DIR="ADP/logs/exp_epsilon_boundary"
# Models from Exp B-hard
MODEL_DIR="ADP/models/exp_hard_profile_sweep"
mkdir -p "$LOG_DIR"

# ── Instance configurations ─────────────────────────────────────────────
# Small multi-machine: M=5 machines, N=40 jobs, D=3 days → K=60 slots
# Medium multi-machine: M=8 machines, N=100 jobs, D=5 days → K=100 slots
#
# Uses new_data.py's cap formula: cap = floor(target_util * M * K).
# Per-machine after biased assignment: load ≈ cap/M, within K slots.
# As epsilon decreases, per-machine horizon shrinks → harder subproblems.

CATEGORIES=("small" "medium")
TARGET_UTILS=("0.85" "0.90" "0.95")
REPLICATES=10
PMAX=12

# Category-specific instance sizes
declare -A CAT_N CAT_M CAT_D
CAT_N[small]=40;  CAT_M[small]=5;  CAT_D[small]=3
CAT_N[medium]=100; CAT_M[medium]=8; CAT_D[medium]=5

# ── Solver config ───────────────────────────────────────────────────────
BEAMS=("20" "50")
PRUNE_FACTOR="2.0"
EPS_DP_TIME_LIMIT="120"
DP_MAX_STATES="500000"

# ── Profiles × Models to test ──────────────────────────────────────────
PROFILES=("ramp" "double_peak" "daily_tou")
MODELS=("mlp" "poly" "elasticnet")

# ── Assignment configs ──────────────────────────────────────────────────
ASSIGN_ALPHA="1.2"
ASSIGN_MIX="0.1"

FEAT_FLAGS="--transferable-features --normalize"

echo "============================================================"
echo " Experiment E — Epsilon-Constraint Boundary"
echo " Categories: ${CATEGORIES[*]}"
echo " Utils: ${TARGET_UTILS[*]}"
echo " Profiles: ${PROFILES[*]}"
echo " Models: ${MODELS[*]}"
echo " Beams: ${BEAMS[*]}"
echo " Replicates: $REPLICATES per config"
echo "============================================================"

RUN_IDX=0

for CAT in "${CATEGORIES[@]}"; do
  N_VAL="${CAT_N[$CAT]}"
  M_VAL="${CAT_M[$CAT]}"
  D_VAL="${CAT_D[$CAT]}"

  for UTIL in "${TARGET_UTILS[@]}"; do
    for PROFILE in "${PROFILES[@]}"; do
      for MODEL in "${MODELS[@]}"; do
        for BEAM in "${BEAMS[@]}"; do
          RUN_IDX=$(( RUN_IDX + 1 ))
          TAG="${CAT}_${PROFILE}_${MODEL}_u${UTIL}_b${BEAM}"

          # Find model checkpoint from Exp B-hard
          MODEL_PATH="$MODEL_DIR/vhat_${PROFILE}_${MODEL}.npz"
          if [[ ! -f "$MODEL_PATH" ]]; then
            echo "[$RUN_IDX] SKIP (no model) $MODEL_PATH"
            continue
          fi

          OUT_CSV="$LOG_DIR/${TAG}.csv"
          if [[ -f "$OUT_CSV" && "$RESUME" == "1" ]]; then
            echo "[$RUN_IDX] SKIP (exists) $OUT_CSV"
            continue
          fi

          echo "[$RUN_IDX] $TAG"
          $PYTHON_BIN sandbox/eval_epsilon_constraint_sim.py \
            --category "$CAT" \
            --N "$N_VAL" --M "$M_VAL" --D "$D_VAL" \
            --replicates "$REPLICATES" \
            --seed 42 \
            --pmax "$PMAX" \
            --target-util "$UTIL" \
            --daily-price-profile "$PROFILE" \
            --assign-alpha "$ASSIGN_ALPHA" \
            --assign-uniform-mix "$ASSIGN_MIX" \
            --guided --beam "$BEAM" --prune-factor "$PRUNE_FACTOR" \
            --load-model "$MODEL_PATH" \
            $FEAT_FLAGS \
            --eps-dp-time-limit "$EPS_DP_TIME_LIMIT" \
            --dp-max-states "$DP_MAX_STATES" \
            --out-csv "$OUT_CSV" \
            2>&1 | tee "$LOG_DIR/${TAG}.log"

        done
      done
    done
  done
done

echo ""
echo "============================================================"
echo " Experiment E complete."
echo " Results: $LOG_DIR/"
echo "============================================================"
