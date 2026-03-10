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

# ── Remove Iirc.Utils.Gurobi (requires Gurobi license, not needed for B&B) ──
# Even though Experiments.csproj doesn't reference it, stale NuGet caches or
# implicit solution-level resolution can pull it in and cause build failures.
GUROBI_DIR="$ROOT/data/green-scheduling-bab/Iirc.Utils.Gurobi"
if [ -d "$GUROBI_DIR" ]; then
    echo "Removing Iirc.Utils.Gurobi (not needed for B&B experiments)..."
    rm -rf "$GUROBI_DIR"
fi

# Also strip the Gurobi ProjectReference from SolverCli (belt-and-suspenders)
SOLVERCLI="$PAPER_ROOT/Iirc.EnergyStatesAndCostsScheduling.SolverCli/Iirc.EnergyStatesAndCostsScheduling.SolverCli.csproj"
if [ -f "$SOLVERCLI" ]; then
    sed -i.bak '/Iirc\.Utils\.Gurobi/d' "$SOLVERCLI" && rm -f "${SOLVERCLI}.bak"
fi

# Clean stale build caches that may reference removed projects
echo "Cleaning stale build caches..."
find "$ROOT/data/green-scheduling-bab" -type d -name obj -exec rm -rf {} + 2>/dev/null || true
find "$ROOT/data/green-scheduling-bab" -type d -name bin -exec rm -rf {} + 2>/dev/null || true

# Build only the Experiments project (pure B&B, no Gurobi dependency).
echo "--- Building Experiments project ---"
cd "$PAPER_ROOT"
dotnet build Iirc.EnergyStatesAndCostsScheduling.Experiments/Iirc.EnergyStatesAndCostsScheduling.Experiments.csproj -c Release 2>&1

# Verify build (net6.0 or net8.0 depending on the project's TargetFramework)
EXPERIMENTS=$(find . -path "*/Experiments/bin/Release/*/Iirc.EnergyStatesAndCostsScheduling.Experiments.dll" | head -1)

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
