#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Run all NeuroLS variants sequentially (for non-SLURM setups)
#
# Usage:
#   bash scripts/run_neurols_all.sh              # all 9 variants
#   bash scripts/run_neurols_all.sh AANP_full    # single variant
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")/.."

# Some setups mount the *package directory* at ~/PaST (i.e., this cwd contains
# neurols/, config.py, etc.), while others use a repo root that contains PaST/.
# Make imports work in both cases.
if [[ -d "PaST" && -f "PaST/__init__.py" ]]; then
    # Repo root layout: ./PaST is importable from cwd
    TRAIN_MODULE="PaST.neurols.train"
elif [[ -f "__init__.py" && -d "neurols" ]]; then
    # Package-dir layout: need parent on PYTHONPATH to import PaST.*
    export PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}"
    TRAIN_MODULE="PaST.neurols.train"
elif [[ -d "neurols" ]]; then
    # Fallback: treat cwd as a flat project containing neurols/
    TRAIN_MODULE="neurols.train"
else
    echo "Could not locate NeuroLS package layout from $(pwd)" >&2
    exit 1
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

# Optionally activate a conda environment: CONDA_ENV=new-ml-env bash scripts/...
if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

declare -A CONFIGS
CONFIGS=(
    ["AA_none"]="configs/neurols_AA_none.yaml"
    ["AA_zprice"]="configs/neurols_AA_zprice.yaml"
    ["AA_full"]="configs/neurols_AA_full.yaml"
    ["AAN_none"]="configs/neurols_AAN_none.yaml"
    ["AAN_zprice"]="configs/neurols_AAN_zprice.yaml"
    ["AAN_full"]="configs/neurols_AAN_full.yaml"
    ["AANP_none"]="configs/neurols_AANP_none.yaml"
    ["AANP_zprice"]="configs/neurols_AANP_zprice.yaml"
    ["AANP_full"]="configs/neurols_base.yaml"
)

# If argument given, run just that variant
if [[ $# -gt 0 ]]; then
    KEY="$1"
    if [[ -z "${CONFIGS[$KEY]+x}" ]]; then
        echo "Unknown variant: $KEY"
        echo "Available: ${!CONFIGS[@]}"
        exit 1
    fi
    echo "Running variant: $KEY"
    python -m "$TRAIN_MODULE" --config "${CONFIGS[$KEY]}" --device "$DEVICE" --seed "$SEED"
    exit 0
fi

# Run all variants
for KEY in AA_none AA_zprice AA_full AAN_none AAN_zprice AAN_full AANP_none AANP_zprice AANP_full; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " Starting: $KEY  ($(date))"
    echo "════════════════════════════════════════════════════════════"
    python -m "$TRAIN_MODULE" --config "${CONFIGS[$KEY]}" --device "$DEVICE" --seed "$SEED"
    echo " Finished: $KEY  ($(date))"
done

echo ""
echo "All 9 variants completed."
