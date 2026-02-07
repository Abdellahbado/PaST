#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Run all NeuroLS variants — with optional cross-variant parallelism
#
# Usage:
#   bash scripts/run_neurols_all.sh              # all 9, parallel
#   bash scripts/run_neurols_all.sh AANP_full    # single variant
#   MAX_PARALLEL=1 bash scripts/run_neurols_all.sh  # sequential
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")/.."

# ── locate package ────────────────────────────────────────────────
if [[ -d "PaST" && -f "PaST/__init__.py" ]]; then
    TRAIN_MODULE="PaST.neurols.train"
elif [[ -f "__init__.py" && -d "neurols" ]]; then
    export PYTHONPATH="$(pwd)/..:${PYTHONPATH:-}"
    TRAIN_MODULE="PaST.neurols.train"
elif [[ -d "neurols" ]]; then
    TRAIN_MODULE="neurols.train"
else
    echo "Could not locate NeuroLS package layout from $(pwd)" >&2
    exit 1
fi

# ── defaults ──────────────────────────────────────────────────────
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
# How many per-variant episode-parallel workers (0 = auto-detect)
N_PARALLEL="${N_PARALLEL:-0}"
# How many variants to run concurrently (default 3 for GPU sharing)
MAX_PARALLEL="${MAX_PARALLEL:-3}"

# Optionally activate a conda environment
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
    ["AANP_full"]="configs/neurols_AANP_full.yaml"
    # Tripartite-A: same AANP actions, tripartite graph encoder
    ["triA_none"]="configs/neurols_triA_AANP_none.yaml"
    ["triA_zprice"]="configs/neurols_triA_AANP_zprice.yaml"
    ["triA_full"]="configs/neurols_triA_AANP_full.yaml"
    # Tripartite-B: AANPD (learned destroy), tripartite graph encoder
    ["triB_full"]="configs/neurols_triB_AANPD_full.yaml"
)

# ── helper: run one variant ──────────────────────────────────────
run_variant() {
    local KEY="$1"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " Starting: $KEY  ($(date))"
    echo "════════════════════════════════════════════════════════════"
    python -m "$TRAIN_MODULE" \
        --config "${CONFIGS[$KEY]}" \
        --device "$DEVICE" \
        --seed "$SEED" \
        --n-parallel "$N_PARALLEL" \
        2>&1 | sed "s/^/[$KEY] /"
    echo " Finished: $KEY  ($(date))"
}

# ── single variant mode ──────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    KEY="$1"
    if [[ -z "${CONFIGS[$KEY]+x}" ]]; then
        echo "Unknown variant: $KEY"
        echo "Available: ${!CONFIGS[@]}"
        exit 1
    fi
    echo "Running single variant: $KEY"
    run_variant "$KEY"
    exit 0
fi

# ── run all variants in parallel batches ──────────────────────────
ALL_KEYS=(AA_none AA_zprice AA_full AAN_none AAN_zprice AAN_full AANP_none AANP_zprice AANP_full triA_none triA_zprice triA_full triB_full)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Running ${#ALL_KEYS[@]} variants, up to $MAX_PARALLEL at a time      ║"
echo "║  Per-variant workers: ${N_PARALLEL} (0=auto)                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Use xargs for controlled parallelism (works on both Linux and macOS)
printf '%s\n' "${ALL_KEYS[@]}" | xargs -I {} -P "$MAX_PARALLEL" bash -c "$(declare -f run_variant); $(declare -p CONFIGS TRAIN_MODULE DEVICE SEED N_PARALLEL); run_variant {}"

echo ""
echo "All ${#ALL_KEYS[@]} variants completed."
