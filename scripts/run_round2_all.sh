#!/usr/bin/env bash
# ============================================================================
# Round 2 — Full Experiment Orchestrator
#
# Runs all corrected experiments in order:
#   1. B2: Corrected daily-repeating baseline (small + medium)
#   2.  I: NR-honest features (small + medium + cross-size)
#   3.  J: Weekly-repeating (small + medium + cross-size)
#
# Environment variables you can set:
#   WORKERS=16        — parallel workers for data collection
#   DP_TIME_LIMIT=120 — per-solve DP timeout (seconds)
#   RESUME=1          — skip experiments whose CSV already exists
#   SKIP_B2=1         — skip repeating baseline (if already run)
#   SKIP_I=1          — skip NR-honest
#   SKIP_J=1          — skip weekly
#
# Usage:
#   bash scripts/run_round2_all.sh              # run everything
#   SKIP_B2=1 bash scripts/run_round2_all.sh    # skip baseline, run I + J only
#   WORKERS=32 bash scripts/run_round2_all.sh   # use 32 workers
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.."

mkdir -p ADP/logs/logs/rigourous

SKIP_B2="${SKIP_B2:-0}"
SKIP_I="${SKIP_I:-0}"
SKIP_J="${SKIP_J:-0}"

echo "========================================================================"
echo " ROUND 2 — Corrected Experiments"
echo "   WORKERS_SMALL=${WORKERS_SMALL:-64}  WORKERS_MEDIUM=${WORKERS_MEDIUM:-16}  RESUME=${RESUME:-1}  DP_TIME_LIMIT=${DP_TIME_LIMIT:-120}"
echo "   Phases: B2=$( [[ $SKIP_B2 == 1 ]] && echo SKIP || echo RUN )"
echo "           I=$( [[ $SKIP_I == 1 ]] && echo SKIP || echo RUN )"
echo "           J=$( [[ $SKIP_J == 1 ]] && echo SKIP || echo RUN )"
echo "========================================================================"

T0=$SECONDS

# ── Phase B2: Corrected repeating baseline ────────────────────────────
if [[ "$SKIP_B2" != "1" ]]; then
  echo ""
  echo "================================================================"
  echo " Phase B2: Corrected daily-repeating (small + medium)"
  echo "================================================================"
  bash scripts/exp_repeating_v2.sh
fi

# ── Phase I: NR-honest features ──────────────────────────────────────
if [[ "$SKIP_I" != "1" ]]; then
  echo ""
  echo "================================================================"
  echo " Phase I: Non-Repeating with NR-honest features"
  echo "================================================================"
  bash scripts/exp_nr_honest.sh
fi

# ── Phase J: Weekly repeating ─────────────────────────────────────────
if [[ "$SKIP_J" != "1" ]]; then
  echo ""
  echo "================================================================"
  echo " Phase J: Weekly-Repeating (H_cycle=140)"
  echo "================================================================"
  bash scripts/exp_weekly_repeating.sh
fi

ELAPSED=$(( SECONDS - T0 ))
echo ""
echo "========================================================================"
echo " ROUND 2 COMPLETE — total wall time: ${ELAPSED}s"
echo " Results:"
echo "   B2: ADP/logs/exp_repeating_v2/"
echo "    I: ADP/logs/exp_nr_honest/"
echo "    J: ADP/logs/exp_weekly/"
echo "========================================================================"
