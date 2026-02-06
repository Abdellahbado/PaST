#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# SLURM job-array script for NeuroLS training
# Launches 9 variants: 3 action spaces × 3 price modes
#
# Usage:
#   sbatch scripts/slurm_neurols_array.sh
#
# Or launch a single variant:
#   sbatch --array=0 scripts/slurm_neurols_array.sh   # AA_none
#   sbatch --array=8 scripts/slurm_neurols_array.sh   # AANP_full (base)
# ═══════════════════════════════════════════════════════════════════

#SBATCH --job-name=neurols
#SBATCH --output=logs/slurm/neurols_%A_%a.out
#SBATCH --error=logs/slurm/neurols_%A_%a.err
#SBATCH --partition=gpu             # Adjust to your cluster's GPU partition
#SBATCH --gres=gpu:1                # 1 GPU per variant
#SBATCH --cpus-per-task=16          # 16 cores
#SBATCH --mem=32G
#SBATCH --time=48:00:00             # 48 hours wall-time
#SBATCH --array=0-8                 # 9 variants

set -euo pipefail

# ── Module setup (adapt to your cluster) ──────────────────────────
# module load python/3.10 cuda/12.1  # uncomment and adapt
# source $HOME/venvs/past/bin/activate  # uncomment if using venv
# conda activate past                   # uncomment if using conda

# ── Thread configuration ──────────────────────────────────────────
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4

# ── Navigate to project root ─────────────────────────────────────
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# ── Create log directory ──────────────────────────────────────────
mkdir -p logs/slurm

# ── Map SLURM_ARRAY_TASK_ID → config ─────────────────────────────
CONFIGS=(
    "configs/neurols_AA_none.yaml"       # 0
    "configs/neurols_AA_zprice.yaml"     # 1
    "configs/neurols_AA_full.yaml"       # 2
    "configs/neurols_AAN_none.yaml"      # 3
    "configs/neurols_AAN_zprice.yaml"    # 4
    "configs/neurols_AAN_full.yaml"      # 5
    "configs/neurols_AANP_none.yaml"     # 6
    "configs/neurols_AANP_zprice.yaml"   # 7
    "configs/neurols_base.yaml"          # 8 = AANP_full
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "════════════════════════════════════════════════════════════"
echo " NeuroLS Training — Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo " Config: $CONFIG"
echo " Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Date: $(date)"
echo "════════════════════════════════════════════════════════════"

# ── Verify GPU access ────────────────────────────────────────────
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# ── Run training ──────────────────────────────────────────────────
python -m PaST.neurols.train \
    --config "$CONFIG" \
    --device cuda \
    --seed 42

echo "Training completed at $(date)"
