#!/usr/bin/env bash
# ============================================================================
# Master Orchestrator V2 — Run All Rigorous Experiments (Hard Instances)
#
# Execution order (respects dependencies):
#
#   Phase 1: B-hard (Profile Sweep, hard instances) — TRAINS core models
#            H1 (Non-Repeating, train+eval) — TRAINS separate models
#
#   Phase 2 (eval-only, loads B-hard models):
#            E (Epsilon-Constraint Boundary)
#            F (Weakly Repeating Stress — Drifting Amplitude)
#            G (Forecast Bias Stress)
#            H2 (Non-Repeating zero-shot — inside Exp H script)
#
# Key design differences from V1:
#   - target_util=0.95 (up from 0.80) eliminates trivially solvable instances
#   - N∈[8,20] matched to D∈[3,5] prevents p_j collapse to unit-length
#   - Beams 5,10,20 (not 2,5,10) — beam=2 was often sufficient in V1
#   - Epsilon-constraint adds bi-objective dimension
#   - Price mode stress tests: drifting amplitude, forecast bias, non-repeating
#
# Skip flags: SKIP_BHARD=1 SKIP_EPSILON=1 SKIP_DRIFT=1 SKIP_BIAS=1 SKIP_NONREP=1
# Resume mode (default): RESUME=1 — skips runs with existing output CSVs
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."

# ── Nohup background launch (at orchestrator level) ─────────────────
if [[ "${NOHUP:-1}" == "1" && -z "${IN_NOHUP:-}" ]]; then
  mkdir -p "ADP/logs/nohup"
  TS="$(date +"%Y%m%d_%H%M%S")"
  LOG_FILE="ADP/logs/nohup/run_all_v2_${TS}.log"
  echo "[nohup] launching full pipeline in background; log: $LOG_FILE"
  IN_NOHUP=1 NOHUP=0 nohup bash "$0" "$@" >"$LOG_FILE" 2>&1 &
  echo "[nohup] pid=$!"
  exit 0
fi

SKIP_BHARD="${SKIP_BHARD:-0}"
SKIP_EPSILON="${SKIP_EPSILON:-0}"
SKIP_DRIFT="${SKIP_DRIFT:-0}"
SKIP_BIAS="${SKIP_BIAS:-0}"
SKIP_NONREP="${SKIP_NONREP:-0}"

export RESUME="${RESUME:-1}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Workers: auto-scale based on available cores ────────────────────
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export WORKERS="${WORKERS:-$(( NCPU > 32 ? 32 : NCPU ))}"
export WORKERS_SMALL="$WORKERS"
export WORKERS_MEDIUM="${WORKERS_MEDIUM:-$(( NCPU > 16 ? 16 : NCPU ))}"

echo "============================================================"
echo " Running All V2 Rigorous Experiments (Hard Instances)"
echo " RESUME=$RESUME"
echo " Workers: default=$WORKERS  medium=$WORKERS_MEDIUM  (ncpu=$NCPU)"
echo " SKIP_BHARD=$SKIP_BHARD  SKIP_EPSILON=$SKIP_EPSILON"
echo " SKIP_DRIFT=$SKIP_DRIFT  SKIP_BIAS=$SKIP_BIAS"
echo " SKIP_NONREP=$SKIP_NONREP"
echo "============================================================"
echo ""

START_TS=$(date +%s)

# ── Phase 1: Training experiments (foreground) ─────────────────────
# B-hard produces model checkpoints that E, F, G, H2 depend on.

if [[ "$SKIP_BHARD" != "1" ]]; then
  echo ">>> [Phase 1a] Starting Experiment B-hard — Hard Profile Sweep"
  NOHUP=0 bash scripts/exp_hard_profile_sweep.sh
  echo ""
else
  echo ">>> SKIPPED Experiment B-hard (SKIP_BHARD=1)"
fi

if [[ "$SKIP_NONREP" != "1" ]]; then
  echo ">>> [Phase 1b] Starting Experiment H — Non-Repeating Profile"
  NOHUP=0 bash scripts/exp_non_repeating.sh
  echo ""
else
  echo ">>> SKIPPED Experiment H (SKIP_NONREP=1)"
fi

# ── Phase 2: Eval-only experiments (depend on B-hard's models) ─────

if [[ "$SKIP_EPSILON" != "1" ]]; then
  echo ">>> [Phase 2] Starting Experiment E — Epsilon-Constraint Boundary"
  NOHUP=0 bash scripts/exp_epsilon_boundary.sh
  echo ""
else
  echo ">>> SKIPPED Experiment E (SKIP_EPSILON=1)"
fi

if [[ "$SKIP_DRIFT" != "1" ]]; then
  echo ">>> [Phase 2] Starting Experiment F — Weakly Repeating Stress"
  NOHUP=0 bash scripts/exp_weak_repeat_stress.sh
  echo ""
else
  echo ">>> SKIPPED Experiment F (SKIP_DRIFT=1)"
fi

if [[ "$SKIP_BIAS" != "1" ]]; then
  echo ">>> [Phase 2] Starting Experiment G — Forecast Bias Stress"
  NOHUP=0 bash scripts/exp_forecast_bias.sh
  echo ""
else
  echo ">>> SKIPPED Experiment G (SKIP_BIAS=1)"
fi

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
echo "============================================================"
echo " All V2 experiments complete."
echo " Total time: ${ELAPSED}s ($(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m)"
echo " Results:"
echo "   B-hard: ADP/logs/exp_hard_profile_sweep/"
echo "   E:      ADP/logs/exp_epsilon_boundary/"
echo "   F:      ADP/logs/exp_weak_repeat_stress/"
echo "   G:      ADP/logs/exp_forecast_bias/"
echo "   H:      ADP/logs/exp_non_repeating/"
echo "============================================================"
