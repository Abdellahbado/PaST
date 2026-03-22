#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BAB_DIR="$ROOT/data/green-scheduling-bab"
PAPER_ROOT="$BAB_DIR/Iirc.EnergyStatesAndCostsScheduling"
PAPER_DATA="$PAPER_ROOT/data"
DATASETS_DIR="$PAPER_DATA/datasets"
TARBALL="$ROOT/data/paper_datasets.tar.gz"

echo "============================================================"
echo " PaST Benchmark Data Setup"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

if [ ! -d "$BAB_DIR/.git" ]; then
    echo "Cloning upstream benchmark repo into $BAB_DIR"
    git clone --depth 1 https://github.com/CTU-IIG/green-scheduling-bab.git "$BAB_DIR"
else
    echo "Benchmark repo already present at $BAB_DIR"
fi

mkdir -p "$DATASETS_DIR"

if [ ! -f "$TARBALL" ]; then
    echo "ERROR: missing dataset tarball at $TARBALL"
    echo "This file is required to install the benchmark instances used by our runners."
    exit 1
fi

dataset_count=$(find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
if [ "$dataset_count" = "0" ]; then
    echo "Extracting benchmark datasets from $TARBALL"
    tar -xzf "$TARBALL" -C "$DATASETS_DIR"
else
    echo "Datasets already installed in $DATASETS_DIR ($dataset_count directories)"
fi

echo ""
echo "--- Verifying benedikt2025b_groups against corrected regeneration ---"
python3 "$ROOT/scripts/regenerate_instances.py"

CORRECTED_DIR="$DATASETS_DIR/benedikt2025b_groups_corrected"
GROUPS_DIR="$DATASETS_DIR/benedikt2025b_groups"
if [ -d "$CORRECTED_DIR" ]; then
    echo "Installing corrected benedikt2025b_groups dataset"
    rm -rf "$GROUPS_DIR"
    mv "$CORRECTED_DIR" "$GROUPS_DIR"
else
    echo "benedikt2025b_groups already matches the corrected regeneration"
fi

echo ""
echo "Benchmark data ready."
echo "Dataset root: $DATASETS_DIR"
