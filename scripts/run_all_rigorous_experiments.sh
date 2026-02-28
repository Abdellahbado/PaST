#!/usr/bin/env bash
# ============================================================================
# Master Orchestrator — Run All Rigorous Experiments
#
# Execution order (respects dependencies):
#   Phase 1: B (Profile Sweep)    — trains all models
#            C (Regularization)   — trains reg-comparison models (independent)
#   Phase 2: A (Noise Stress)     — loads B's models (eval-only)
#            D (Combined Stress)  — loads B's models (eval-only)
#
# B and C run in foreground so their models exist before A and D start.
# Nohup is applied at THIS level only.
#
# Skip flags: SKIP_PROFILE=1 SKIP_REG=1 SKIP_NOISE=1 SKIP_COMBINED=1
# Resume mode (default): RESUME=1 — skips runs with existing output CSVs
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."

# ── Nohup background launch (at orchestrator level) ─────────────────
if [[ "${NOHUP:-1}" == "1" && -z "${IN_NOHUP:-}" ]]; then
  mkdir -p "ADP/logs/nohup"
  TS="$(date +"%Y%m%d_%H%M%S")"
  LOG_FILE="ADP/logs/nohup/run_all_${TS}.log"
  echo "[nohup] launching full pipeline in background; log: $LOG_FILE"
  IN_NOHUP=1 NOHUP=0 nohup bash "$0" "$@" >"$LOG_FILE" 2>&1 &
  echo "[nohup] pid=$!"
  exit 0
fi

SKIP_PROFILE="${SKIP_PROFILE:-0}"
SKIP_REG="${SKIP_REG:-0}"
SKIP_NOISE="${SKIP_NOISE:-0}"
SKIP_COMBINED="${SKIP_COMBINED:-0}"

export RESUME="${RESUME:-1}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Workers: auto-scale based on available cores ────────────────────
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export WORKERS_SMALL="${WORKERS_SMALL:-$(( NCPU > 32 ? 32 : NCPU ))}"
export WORKERS_MEDIUM="${WORKERS_MEDIUM:-4}"

echo "============================================================"
echo " Running All Rigorous Experiments"
echo " RESUME=$RESUME"
echo " Workers: small=$WORKERS_SMALL  medium=$WORKERS_MEDIUM  (ncpu=$NCPU)"
echo " SKIP_PROFILE=$SKIP_PROFILE  SKIP_REG=$SKIP_REG"
echo " SKIP_NOISE=$SKIP_NOISE  SKIP_COMBINED=$SKIP_COMBINED"
echo "============================================================"
echo ""

START_TS=$(date +%s)

# ── Phase 1: Training experiments (foreground) ─────────────────────
# B and C produce model checkpoints that A and D depend on.

if [[ "$SKIP_PROFILE" != "1" ]]; then
  echo ">>> [Phase 1] Starting Experiment B — Profile Sweep"
  bash scripts/exp_profile_sweep.sh
  echo ""
else
  echo ">>> SKIPPED Experiment B (SKIP_PROFILE=1)"
fi

if [[ "$SKIP_REG" != "1" ]]; then
  echo ">>> [Phase 1] Starting Experiment C — Regularization & Features"
  bash scripts/exp_regularization_features.sh
  echo ""
else
  echo ">>> SKIPPED Experiment C (SKIP_REG=1)"
fi

# ── Phase 2: Eval-only experiments (depend on B's models) ──────────

if [[ "$SKIP_NOISE" != "1" ]]; then
  echo ">>> [Phase 2] Starting Experiment A — Noise Stress Test"
  bash scripts/exp_noise_stress.sh
  echo ""
else
  echo ">>> SKIPPED Experiment A (SKIP_NOISE=1)"
fi

if [[ "$SKIP_COMBINED" != "1" ]]; then
  echo ">>> [Phase 2] Starting Experiment D — Combined Stress"
  bash scripts/exp_noise_profile_combined.sh
  echo ""
else
  echo ">>> SKIPPED Experiment D (SKIP_COMBINED=1)"
fi

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
echo "============================================================"
echo " All experiments complete."
echo " Total time: ${ELAPSED}s ($(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m)"
echo " Results:"
echo "   B: ADP/logs/exp_profile_sweep/"
echo "   C: ADP/logs/exp_regularization_features/"
echo "   A: ADP/logs/exp_noise_stress/"
echo "   D: ADP/logs/exp_noise_profile_combined/"
echo "============================================================"
