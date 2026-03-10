#!/bin/bash
###############################################################################
# SLURM job script — Full benchmark (both solvers, all instances)
#
# Submit: sbatch hpc/slurm_full.sbatch
#
# Adjust partition, account, QOS, and resource requests for your cluster.
###############################################################################
#SBATCH --job-name=past-bench
#SBATCH --output=hpc/slurm_logs/full_%j.out
#SBATCH --error=hpc/slurm_logs/full_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
# #SBATCH --account=YOUR_ACCOUNT   # uncomment & edit
# #SBATCH --qos=normal             # uncomment & edit

set -euo pipefail

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# Navigate to project root
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
ROOT="$PWD"

# Create log directory
mkdir -p hpc/slurm_logs

# Load modules (adjust for your cluster)
# module load cmake gcc python dotnet  # uncomment & edit

# Run the full benchmark
bash hpc/run_full_benchmark.sh --time-limit=600

echo ""
echo "End: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
