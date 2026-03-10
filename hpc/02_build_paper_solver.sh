#!/usr/bin/env bash
###############################################################################
# 02_build_paper_solver.sh — Build the paper's C# B&B-SPACES solver
#
# Requires: .NET 8 SDK
# Usage:    bash hpc/02_build_paper_solver.sh
#
# This script is idempotent: it will re-clone the paper repo if missing,
# patch sources for .NET 8 / no-Gurobi, and build from clean state.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BAB_DIR="$ROOT/data/green-scheduling-bab"
PAPER_ROOT="$BAB_DIR/Iirc.EnergyStatesAndCostsScheduling"

echo "============================================================"
echo " Building Paper's C# Solver (B&B-SPACES)"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

if ! command -v dotnet &>/dev/null; then
    echo "ERROR: dotnet SDK not found. Install .NET 8 SDK first."
    echo "  wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh"
    echo "  chmod +x dotnet-install.sh && ./dotnet-install.sh --channel 8.0"
    echo "  export PATH=\$HOME/.dotnet:\$PATH"
    exit 1
fi

echo ".NET version: $(dotnet --version)"
echo ""

# ── Clone the paper repo if it's missing ────────────────────────────────────
if [ ! -d "$BAB_DIR" ]; then
    echo "Paper repo not found at $BAB_DIR — cloning..."
    git clone https://github.com/CTU-IIG/green-scheduling-bab.git "$BAB_DIR"
    echo ""
fi

# ── Remove Gurobi (not needed for B&B experiments) ─────────────────────────
GUROBI_DIR="$BAB_DIR/Iirc.Utils.Gurobi"
if [ -d "$GUROBI_DIR" ]; then
    echo "Removing Iirc.Utils.Gurobi directory..."
    rm -rf "$GUROBI_DIR"
fi

echo "Stripping Gurobi references from .csproj files..."
find "$BAB_DIR" -name '*.csproj' -exec \
    sed -i.bak '/Iirc\.Utils\.Gurobi/d' {} \;
find "$BAB_DIR" -name '*.csproj.bak' -delete 2>/dev/null || true

SHARED_SOLVERS="$PAPER_ROOT/Iirc.EnergyStatesAndCostsScheduling.Shared/Solvers"
for f in BaseMilpSolver.cs IlpRef.cs; do
    if [ -f "$SHARED_SOLVERS/$f" ]; then
        echo "  Removing $f (requires Gurobi)..."
        rm -f "$SHARED_SOLVERS/$f"
    fi
done

# ── Retarget net6.0 → net8.0 ───────────────────────────────────────────────
echo "Retargeting projects from net6.0 to net8.0..."
find "$BAB_DIR" -name '*.csproj' -exec \
    sed -i.bak 's|<TargetFramework>net6\.0</TargetFramework>|<TargetFramework>net8.0</TargetFramework>|g' {} \;
find "$BAB_DIR" -name '*.csproj.bak' -delete 2>/dev/null || true

# ── Enable BinaryFormatter at runtime (.NET 8 blocks it by default) ─────────
# The paper's code uses BinaryFormatter for serialization. .NET 8 blocks it.
# We inject the EnableUnsafeBinaryFormatterSerialization property into .csproj
# files so it gets baked into runtimeconfig.json.
echo "Enabling BinaryFormatter in .csproj files (required by JsonInputWriter)..."
for csproj in $(find "$BAB_DIR" -name '*.csproj'); do
    if ! grep -q 'EnableUnsafeBinaryFormatterSerialization' "$csproj"; then
        sed -i.bak 's|<TargetFramework>net8\.0</TargetFramework>|<TargetFramework>net8.0</TargetFramework>\n    <EnableUnsafeBinaryFormatterSerialization>true</EnableUnsafeBinaryFormatterSerialization>|g' "$csproj"
        rm -f "${csproj}.bak"
    fi
done

# ── Clean stale build caches ────────────────────────────────────────────────
echo "Cleaning stale build caches..."
find "$BAB_DIR" -type d \( -name obj -o -name bin \) -exec rm -rf {} + 2>/dev/null || true

# ── Build ───────────────────────────────────────────────────────────────────
# /p:NoWarn=SYSLIB0011 — BinaryFormatter became a hard error in .NET 8;
# the library uses it for deep-cloning but B&B never hits that path.
echo ""
echo "--- Building Experiments project ---"
cd "$PAPER_ROOT"
dotnet build \
    Iirc.EnergyStatesAndCostsScheduling.Experiments/Iirc.EnergyStatesAndCostsScheduling.Experiments.csproj \
    -c Release \
    /p:NoWarn=SYSLIB0011 \
    2>&1

# ── Install pre-generated datasets ───────────────────────────────────────────
# The paper's DatasetGenerators have a NullReferenceException bug in the
# Benedikt2020b generator when HopsCount > 0 (benedikt2025b_drops prescription).
# The datasets are NOT stored in the paper's git repo (gitignored), so we ship
# the pre-generated datasets as a tarball in our repo and extract them here.
DATA_DIR="$PAPER_ROOT/data"
DATASETS_DIR="$DATA_DIR/datasets"
DATASETS_TARBALL="$ROOT/data/paper_datasets.tar.gz"

if [ ! -d "$DATASETS_DIR" ] || [ -z "$(ls -A "$DATASETS_DIR" 2>/dev/null)" ]; then
    echo ""
    echo "--- Installing pre-generated datasets ---"
    mkdir -p "$DATASETS_DIR"
    if [ -f "$DATASETS_TARBALL" ]; then
        tar xzf "$DATASETS_TARBALL" -C "$DATASETS_DIR"
        echo "  Extracted $(ls "$DATASETS_DIR" | wc -l) datasets from tarball"
    else
        echo "ERROR: Datasets tarball not found at $DATASETS_TARBALL"
        echo "       This file should be in the PaST repo under data/paper_datasets.tar.gz"
        exit 1
    fi
else
    echo "Datasets directory already populated ($(ls "$DATASETS_DIR" | wc -l) datasets)"
fi

# ── Verify ──────────────────────────────────────────────────────────────────
EXPERIMENTS_DLL="$PAPER_ROOT/Iirc.EnergyStatesAndCostsScheduling.Experiments/bin/Release/net8.0/Iirc.EnergyStatesAndCostsScheduling.Experiments.dll"

echo ""
if [ -f "$EXPERIMENTS_DLL" ]; then
    echo "OK: $EXPERIMENTS_DLL"
else
    echo "ERROR: Experiments binary not found at expected path."
    exit 1
fi

echo ""
echo "============================================================"
echo " Paper solver build complete."
echo "============================================================"
echo ""
echo "Next step: python3 hpc/03_run_our_solver.py"
