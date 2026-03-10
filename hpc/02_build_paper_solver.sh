#!/usr/bin/env bash
###############################################################################
# 02_build_paper_solver.sh — Build the paper's C# B&B-SPACES solver
#
# Requires: .NET 8 SDK, Gurobi libs, CPLEX libs (for C++ B&B binary)
# Usage:    bash hpc/02_build_paper_solver.sh
#
# The paper's solver has two parts:
#   1. C# Experiments runner — orchestrates solving, reads/writes results
#   2. C++ BranchAndBoundJob binary — the actual B&B-SPACES solver
#      (pre-built Linux ELF in the repo, needs libgurobi.so + libcplex2212.so)
#
# This script:
#   - Clones the paper repo if missing
#   - Patches for .NET 8 / no-Gurobi (C# side only — removes Gurobi C# wrapper)
#   - Builds the C# Experiments project
#   - Places the pre-built C++ binary where C# expects it (cpp/bin/)
#   - Installs pre-generated datasets from tarball (repo doesn't include them)
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

# ── Remove Gurobi C# wrapper (not needed — we use pre-built C++ binary) ────
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
        echo "  Removing $f (requires Gurobi C# bindings)..."
        rm -f "$SHARED_SOLVERS/$f"
    fi
done

# ── Retarget net6.0 → net8.0 ───────────────────────────────────────────────
echo "Retargeting projects from net6.0 to net8.0..."
find "$BAB_DIR" -name '*.csproj' -exec \
    sed -i.bak 's|<TargetFramework>net6\.0</TargetFramework>|<TargetFramework>net8.0</TargetFramework>|g' {} \;
find "$BAB_DIR" -name '*.csproj.bak' -delete 2>/dev/null || true

# ── Enable BinaryFormatter at runtime (.NET 8 blocks it by default) ─────────
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

# ── Build C# Experiments project ────────────────────────────────────────────
echo ""
echo "--- Building Experiments project ---"
cd "$PAPER_ROOT"
dotnet build \
    Iirc.EnergyStatesAndCostsScheduling.Experiments/Iirc.EnergyStatesAndCostsScheduling.Experiments.csproj \
    -c Release \
    /p:NoWarn=SYSLIB0011 \
    2>&1

# ── Place pre-built C++ B&B binary where C# expects it ─────────────────────
# BaseCppSolver.cs looks for: {DLL_DIR}/cpp/bin/BranchAndBoundJob
# The paper repo ships pre-built Linux ELF binaries in Shared/cpp/
CPP_SRC="$PAPER_ROOT/Iirc.EnergyStatesAndCostsScheduling.Shared/cpp"
DLL_DIR="$PAPER_ROOT/Iirc.EnergyStatesAndCostsScheduling.Experiments/bin/Release/net8.0"
CPP_BIN_TARGET="$DLL_DIR/cpp/bin"

echo ""
echo "--- Installing C++ B&B-SPACES binary ---"
mkdir -p "$CPP_BIN_TARGET"
for solver in BranchAndBoundJob ConstructiveHeuristic GeneticAlgorithm; do
    if [ -f "$CPP_SRC/$solver" ]; then
        cp "$CPP_SRC/$solver" "$CPP_BIN_TARGET/$solver"
        chmod +x "$CPP_BIN_TARGET/$solver"
        echo "  Installed $solver"
    fi
done

# Verify the B&B binary is in place
if [ ! -f "$CPP_BIN_TARGET/BranchAndBoundJob" ]; then
    echo "ERROR: BranchAndBoundJob binary not found in paper repo."
    exit 1
fi

# Check if the binary can actually run (needs libgurobi.so + libcplex2212.so)
echo ""
echo "--- Checking C++ binary runtime dependencies ---"
if command -v ldd &>/dev/null; then
    MISSING_LIBS=$(ldd "$CPP_BIN_TARGET/BranchAndBoundJob" 2>&1 | grep "not found" || true)
    if [ -n "$MISSING_LIBS" ]; then
        echo "WARNING: Missing shared libraries for BranchAndBoundJob:"
        echo "$MISSING_LIBS"
        echo ""
        echo "The C++ B&B solver needs libgurobi.so and libcplex2212.so at runtime."
        echo "Try: module load gurobi cplex    (or set LD_LIBRARY_PATH)"
        echo "The solver will crash at runtime if these are not available."
    else
        echo "  All shared libraries found."
    fi
else
    echo "  ldd not available; skipping dependency check."
fi

# ── Install pre-generated datasets ───────────────────────────────────────────
# Datasets are NOT in the paper's git repo. The DatasetGenerators have a bug
# in Benedikt2020b when HopsCount > 0 (NullReferenceException). We ship
# pre-generated datasets as a tarball (1.7MB, 11 datasets, 1434 instances).
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

# ── Final verification ──────────────────────────────────────────────────────
EXPERIMENTS_DLL="$DLL_DIR/Iirc.EnergyStatesAndCostsScheduling.Experiments.dll"

echo ""
echo "--- Verification ---"
PASS=true
for check_file in \
    "$EXPERIMENTS_DLL" \
    "$CPP_BIN_TARGET/BranchAndBoundJob" \
    "$DATASETS_DIR/aghelinejad2017a_1/0.json" \
    "$DATASETS_DIR/benedikt2025b_drops/0.json"; do
    if [ -f "$check_file" ]; then
        echo "  OK: $(basename "$(dirname "$check_file")")/$(basename "$check_file")"
    else
        echo "  MISSING: $check_file"
        PASS=false
    fi
done

echo "  Datasets: $(ls "$DATASETS_DIR" | wc -l) directories"
echo "  Instances: $(find "$DATASETS_DIR" -name '*.json' | wc -l) files"

if [ "$PASS" = false ]; then
    echo ""
    echo "ERROR: Some files are missing. Build incomplete."
    exit 1
fi

echo ""
echo "============================================================"
echo " Paper solver build complete."
echo "============================================================"
echo ""
echo "Next step: python3 hpc/03_run_our_solver.py"
