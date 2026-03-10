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

# Build only the Experiments project (it has no Gurobi dependency).
# We skip the full .sln build because Iirc.Utils.Gurobi requires a licensed
# Gurobi installation that is not present on the HPC.  The runner script
# (04_run_paper_solver.py) uses the Experiments project exclusively.
echo "--- Building Experiments project ---"
cd "$PAPER_ROOT"
dotnet build Iirc.EnergyStatesAndCostsScheduling.Experiments/Iirc.EnergyStatesAndCostsScheduling.Experiments.csproj -c Release 2>&1

# Verify build
EXPERIMENTS=$(find . -path "*/Experiments/bin/Release/net8.0/Iirc.EnergyStatesAndCostsScheduling.Experiments" -o -path "*/Experiments/bin/Release/net8.0/Iirc.EnergyStatesAndCostsScheduling.Experiments.dll" | head -1)

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
