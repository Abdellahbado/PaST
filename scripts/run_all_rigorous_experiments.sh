#!/usr/bin/env bash
# ============================================================================
# Master Orchestrator — Run All Rigorous Experiments
#
# Execution order:
#   C (Regularization) → B (Profile Sweep) → A (Noise Stress) → D (Combined)
#
# Skip flags: SKIP_REG=1 SKIP_PROFILE=1 SKIP_NOISE=1 SKIP_COMBINED=1
# Resume mode (default): RESUME=1 — skips runs with existing output CSVs
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."

SKIP_REG="${SKIP_REG:-0}"
SKIP_PROFILE="${SKIP_PROFILE:-0}"
SKIP_NOISE="${SKIP_NOISE:-0}"
SKIP_COMBINED="${SKIP_COMBINED:-0}"

export RESUME="${RESUME:-1}"
export WORKERS="${WORKERS:-2}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

echo "============================================================"
echo " Running All Rigorous Experiments"
echo " RESUME=$RESUME  WORKERS=$WORKERS"
echo " SKIP_REG=$SKIP_REG  SKIP_PROFILE=$SKIP_PROFILE"
echo " SKIP_NOISE=$SKIP_NOISE  SKIP_COMBINED=$SKIP_COMBINED"
echo "============================================================"
echo ""

START_TS=$(date +%s)

# ── Experiment C ────────────────────────────────────────────────────
if [[ "$SKIP_REG" != "1" ]]; then
  echo ">>> Starting Experiment C — Regularization & Features"
  bash scripts/exp_regularization_features.sh
  echo ""
else
  echo ">>> SKIPPED Experiment C (SKIP_REG=1)"
fi

# ── Experiment B ────────────────────────────────────────────────────
if [[ "$SKIP_PROFILE" != "1" ]]; then
  echo ">>> Starting Experiment B — Profile Sweep"
  bash scripts/exp_profile_sweep.sh
  echo ""
else
  echo ">>> SKIPPED Experiment B (SKIP_PROFILE=1)"
fi

# ── Experiment A ────────────────────────────────────────────────────
if [[ "$SKIP_NOISE" != "1" ]]; then
  echo ">>> Starting Experiment A — Noise Stress Test"
  bash scripts/exp_noise_stress.sh
  echo ""
else
  echo ">>> SKIPPED Experiment A (SKIP_NOISE=1)"
fi

# ── Experiment D ────────────────────────────────────────────────────
if [[ "$SKIP_COMBINED" != "1" ]]; then
  echo ">>> Starting Experiment D — Combined Stress"
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
echo "   C: ADP/logs/exp_regularization_features/"
echo "   B: ADP/logs/exp_profile_sweep/"
echo "   A: ADP/logs/exp_noise_stress/"
echo "   D: ADP/logs/exp_noise_profile_combined/"
echo "============================================================"
