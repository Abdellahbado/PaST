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

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

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
    python -m PaST.neurols.train --config "${CONFIGS[$KEY]}" --device "$DEVICE" --seed "$SEED"
    exit 0
fi

# Run all variants
for KEY in AA_none AA_zprice AA_full AAN_none AAN_zprice AAN_full AANP_none AANP_zprice AANP_full; do
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " Starting: $KEY  ($(date))"
    echo "════════════════════════════════════════════════════════════"
    python -m PaST.neurols.train --config "${CONFIGS[$KEY]}" --device "$DEVICE" --seed "$SEED"
    echo " Finished: $KEY  ($(date))"
done

echo ""
echo "All 9 variants completed."
