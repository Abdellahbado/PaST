#!/bin/bash
###############################################################################
# SLURM job script — Our solver only (quick regression check)
#
# Submit: sbatch hpc/slurm_ours_only.sbatch
#
# This runs just our C++ solver on all 1517+ instances.
# Typically finishes in < 1 hour for easy instances, a few hours max.
###############################################################################
#SBATCH --job-name=past-ours
#SBATCH --output=hpc/slurm_logs/ours_%j.out
#SBATCH --error=hpc/slurm_logs/ours_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
# #SBATCH --account=YOUR_ACCOUNT   # uncomment & edit

set -euo pipefail

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

mkdir -p hpc/slurm_logs

# Build
bash hpc/01_build_our_solver.sh

# Run (our solver only)
python3 hpc/03_run_our_solver.py \
    --section all \
    --time-limit 600 \
    --output-dir hpc/results_ours

# Quick analysis (paper results may not exist — that's fine)
python3 hpc/05_analyze_results.py \
    --ours hpc/results_ours \
    --output hpc/analysis || true

echo ""
echo "End: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
