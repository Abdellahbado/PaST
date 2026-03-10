#!/usr/bin/env bash
###############################################################################
# 02_build_paper_solver.sh — Build the paper's C# B&B-SPACES solver
#
# Requires: .NET 8 SDK
# Usage:    bash hpc/02_build_paper_solver.sh
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPER_ROOT="$ROOT/data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling"

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

# ── Surgically remove ALL Gurobi references from the codebase ──────────────
# The Shared project has a ProjectReference to Iirc.Utils.Gurobi AND two .cs
# files (BaseMilpSolver.cs, IlpRef.cs) that "using Iirc.Utils.Gurobi".
# None of these are needed for the pure B&B experiments.

GUROBI_DIR="$ROOT/data/green-scheduling-bab/Iirc.Utils.Gurobi"
if [ -d "$GUROBI_DIR" ]; then
    echo "Removing Iirc.Utils.Gurobi directory..."
    rm -rf "$GUROBI_DIR"
fi

# Strip Gurobi ProjectReference from EVERY .csproj that has one
echo "Stripping Gurobi references from .csproj files..."
find "$ROOT/data/green-scheduling-bab" -name '*.csproj' -exec \
    sed -i.bak '/Iirc\.Utils\.Gurobi/d' {} \;
find "$ROOT/data/green-scheduling-bab" -name '*.csproj.bak' -delete 2>/dev/null || true

# Remove the two Shared source files that "using Iirc.Utils.Gurobi"
SHARED_SOLVERS="$PAPER_ROOT/Iirc.EnergyStatesAndCostsScheduling.Shared/Solvers"
for f in BaseMilpSolver.cs IlpRef.cs; do
    if [ -f "$SHARED_SOLVERS/$f" ]; then
        echo "  Removing $f (requires Gurobi, not needed for B&B)..."
        rm -f "$SHARED_SOLVERS/$f"
    fi
done

# ── Retarget net6.0 → net8.0 (HPC only has .NET 8 runtime) ─────────────────
echo "Retargeting projects from net6.0 to net8.0..."
find "$ROOT/data/green-scheduling-bab" -name '*.csproj' -exec \
    sed -i.bak 's|<TargetFramework>net6\.0</TargetFramework>|<TargetFramework>net8.0</TargetFramework>|g' {} \;
find "$ROOT/data/green-scheduling-bab" -name '*.csproj.bak' -delete 2>/dev/null || true

# Clean stale build caches that may reference removed projects
echo "Cleaning stale build caches..."
find "$ROOT/data/green-scheduling-bab" -type d -name obj -exec rm -rf {} + 2>/dev/null || true
find "$ROOT/data/green-scheduling-bab" -type d -name bin -exec rm -rf {} + 2>/dev/null || true

# Build only the Experiments project (pure B&B, no Gurobi dependency).
echo "--- Building Experiments project ---"
cd "$PAPER_ROOT"
dotnet build Iirc.EnergyStatesAndCostsScheduling.Experiments/Iirc.EnergyStatesAndCostsScheduling.Experiments.csproj -c Release 2>&1

# Verify build
EXPERIMENTS=$(find . -path "*/Experiments/bin/Release/net8.0/Iirc.EnergyStatesAndCostsScheduling.Experiments.dll" | head -1)

echo ""
echo "--- Verifying Experiments build ---"
if [ -n "$EXPERIMENTS" ]; then
    echo "Experiments binary: $EXPERIMENTS"
else
    echo "WARNING: Experiments binary not found at expected path, but build reported success."
fi

echo ""
echo "============================================================"
echo " Paper solver build complete (Experiments project only)."
echo " Note: Iirc.Utils.Gurobi / SolverCli skipped — no Gurobi"
echo " license on this machine; Experiments does not need Gurobi."
echo "============================================================"
echo ""
echo "Next step: python3 hpc/03_run_our_solver.py"
