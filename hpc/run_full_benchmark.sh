#!/usr/bin/env bash
###############################################################################
# run_full_benchmark.sh — Master script: builds & runs everything.
#
# This is the "one script to rule them all". Run this on the HPC node.
# It calls the individual scripts in order.
#
# Usage:
#   bash hpc/run_full_benchmark.sh [--skip-paper]  # skip paper solver
#   bash hpc/run_full_benchmark.sh                  # run everything
#
# Time estimate: ~2-6 hours depending on hardware + time limit.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKIP_PAPER=false
TIME_LIMIT=600  # seconds per instance (paper uses 600s = 10min)

for arg in "$@"; do
    case $arg in
        --skip-paper) SKIP_PAPER=true ;;
        --time-limit=*) TIME_LIMIT="${arg#*=}" ;;
    esac
done

echo "============================================================"
echo " PaST HPC Full Benchmark Suite"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Root: $ROOT"
echo " Time limit: ${TIME_LIMIT}s per instance"
echo " Skip paper solver: $SKIP_PAPER"
echo "============================================================"
echo ""

cd "$ROOT"

# ── Step 0: Dependencies ─────────────────────────────────────────────────
echo ""
echo ">>> Step 0: Checking dependencies..."
bash hpc/00_install_deps.sh

# ── Step 1: Build our solver ─────────────────────────────────────────────
echo ""
echo ">>> Step 1: Building our C++ solver..."
bash hpc/01_build_our_solver.sh

# ── Step 2: Build paper solver ───────────────────────────────────────────
if [ "$SKIP_PAPER" = false ]; then
    echo ""
    echo ">>> Step 2: Building paper's C# solver..."
    bash hpc/02_build_paper_solver.sh
else
    echo ""
    echo ">>> Step 2: SKIPPED (--skip-paper)"
fi

# ── Step 3: Run our solver ───────────────────────────────────────────────
echo ""
echo ">>> Step 3: Running our solver on all 1517+ instances..."
python3 hpc/03_run_our_solver.py \
    --section all \
    --time-limit "$TIME_LIMIT" \
    --output-dir hpc/results_ours

# ── Step 4: Run paper solver ─────────────────────────────────────────────
if [ "$SKIP_PAPER" = false ]; then
    echo ""
    echo ">>> Step 4: Running paper's solver on all instances..."
    python3 hpc/04_run_paper_solver.py \
        --section all \
        --time-limit "$TIME_LIMIT" \
        --output-dir hpc/results_paper \
        --method single
else
    echo ""
    echo ">>> Step 4: SKIPPED (--skip-paper)"
fi

# ── Step 5: Analysis ─────────────────────────────────────────────────────
echo ""
echo ">>> Step 5: Analyzing results..."
python3 hpc/05_analyze_results.py \
    --ours hpc/results_ours \
    --paper hpc/results_paper \
    --output hpc/analysis

echo ""
echo "============================================================"
echo " BENCHMARK COMPLETE"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo ""
echo "Results:"
echo "  Our solver:     hpc/results_ours/all_results.csv"
echo "  Paper solver:   hpc/results_paper/all_results.csv"
echo "  Analysis:       hpc/analysis/analysis_report.txt"
echo "  Logs:           hpc/results_ours/run_log.txt"
echo "                  hpc/results_paper/run_log.txt"
echo ""
echo "Quick check:"
echo "  grep 'OPEN GAPS' hpc/results_ours/run_log.txt"
echo "  grep 'REGRESSION' hpc/analysis/analysis_report.txt"
echo ""
