#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Run ONLY the tripartite NeuroLS variants (A + B)
#
# Uses the same seed, device, and parallelism settings as the
# bipartite run for a fair comparison.
#
# Usage:
#   N_PARALLEL=8 MAX_PARALLEL=2 bash scripts/run_neurols_tripartite.sh
#   bash scripts/run_neurols_tripartite.sh triA_full   # single variant
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

# ── defaults (match bipartite run) ────────────────────────────────
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
N_PARALLEL="${N_PARALLEL:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

declare -A CONFIGS
CONFIGS=(
    ["triA_none"]="configs/neurols_triA_AANP_none.yaml"
    ["triA_zprice"]="configs/neurols_triA_AANP_zprice.yaml"
    ["triA_full"]="configs/neurols_triA_AANP_full.yaml"
    ["triB_full"]="configs/neurols_triB_AANPD_full.yaml"
)

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
    for KEY in "$@"; do
        if [[ -z "${CONFIGS[$KEY]+x}" ]]; then
            echo "Unknown variant: $KEY"
            echo "Available: ${!CONFIGS[@]}"
            exit 1
        fi
    done
    echo "Running ${#} variant(s): $*"
    printf '%s\n' "$@" | xargs -I {} -P "$MAX_PARALLEL" \
        bash -c "$(declare -f run_variant); $(declare -p CONFIGS TRAIN_MODULE DEVICE SEED N_PARALLEL); run_variant {}"
    exit 0
fi

# ── run all tripartite variants ───────────────────────────────────
ALL_KEYS=(triA_none triA_zprice triA_full triB_full)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Tripartite variants: ${#ALL_KEYS[@]}  (up to $MAX_PARALLEL parallel)     ║"
echo "║  Per-variant workers: ${N_PARALLEL} (0=auto)                         ║"
echo "║  Seed: ${SEED}  Device: ${DEVICE}                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

printf '%s\n' "${ALL_KEYS[@]}" | xargs -I {} -P "$MAX_PARALLEL" \
    bash -c "$(declare -f run_variant); $(declare -p CONFIGS TRAIN_MODULE DEVICE SEED N_PARALLEL); run_variant {}"

echo ""
echo "All ${#ALL_KEYS[@]} tripartite variants completed."
