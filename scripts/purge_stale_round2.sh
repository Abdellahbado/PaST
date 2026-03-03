#!/usr/bin/env bash
# ============================================================================
# Purge ALL round-2 results so everything re-runs with corrected labeling.
#
# Small instances: now use --label-mode subproblem (dp_time_limit=120s)
# Medium instances: keep --label-mode optimal_path + topup
#
# Run this BEFORE run_round2_all.sh to force a clean re-run.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Purging stale round-2 results..."

# Experiment B2: repeating baseline
echo "  exp_repeating_v2:"
rm -vf ADP/logs/exp_repeating_v2/B2_*.csv  ADP/logs/exp_repeating_v2/B2_*.log 2>/dev/null || true
rm -vf ADP/models/exp_repeating_v2/vhat_B2_*.npz 2>/dev/null || true

# Experiment I: NR-honest
echo "  exp_nr_honest:"
rm -vf ADP/logs/exp_nr_honest/I1_*.csv  ADP/logs/exp_nr_honest/I1_*.log 2>/dev/null || true
rm -vf ADP/logs/exp_nr_honest/I2_*.csv  ADP/logs/exp_nr_honest/I2_*.log 2>/dev/null || true
rm -vf ADP/logs/exp_nr_honest/I3_*.csv  ADP/logs/exp_nr_honest/I3_*.log 2>/dev/null || true
rm -vf ADP/models/exp_nr_honest/vhat_I1_*.npz 2>/dev/null || true

# Experiment J: weekly
echo "  exp_weekly:"
rm -vf ADP/logs/exp_weekly/J1_*.csv  ADP/logs/exp_weekly/J1_*.log 2>/dev/null || true
rm -vf ADP/logs/exp_weekly/J2_*.csv  ADP/logs/exp_weekly/J2_*.log 2>/dev/null || true
rm -vf ADP/logs/exp_weekly/J3_*.csv  ADP/logs/exp_weekly/J3_*.log 2>/dev/null || true
rm -vf ADP/models/exp_weekly/vhat_J1_*.npz 2>/dev/null || true

echo ""
echo "Done.  Now run:  bash scripts/run_round2_all.sh"
