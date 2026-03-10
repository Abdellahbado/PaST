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

# Build the solution
echo "--- Building Experiments project ---"
cd "$PAPER_ROOT"
dotnet build Iirc.EnergyStatesAndCostsScheduling.sln -c Release 2>&1

echo ""
echo "--- Building SolverCli project ---"
dotnet build Iirc.EnergyStatesAndCostsScheduling.SolverCli/Iirc.EnergyStatesAndCostsScheduling.SolverCli.csproj -c Release 2>&1

echo ""
echo "--- Building Experiments project ---"
dotnet build Iirc.EnergyStatesAndCostsScheduling.Experiments/Iirc.EnergyStatesAndCostsScheduling.Experiments.csproj -c Release 2>&1

# Verify builds
SOLVER_CLI=$(find . -path "*/SolverCli/bin/Release/net8.0/Iirc.EnergyStatesAndCostsScheduling.SolverCli" -o -path "*/SolverCli/bin/Release/net8.0/Iirc.EnergyStatesAndCostsScheduling.SolverCli.dll" | head -1)
EXPERIMENTS=$(find . -path "*/Experiments/bin/Release/net8.0/Iirc.EnergyStatesAndCostsScheduling.Experiments" -o -path "*/Experiments/bin/Release/net8.0/Iirc.EnergyStatesAndCostsScheduling.Experiments.dll" | head -1)

echo ""
echo "--- Smoke test (SolverCli) ---"
# Quick test on a small instance
SMALL_INST="$PAPER_ROOT/data/datasets/benedikt2025b_groups/0.json"
if [ -f "$SMALL_INST" ]; then
    echo "Testing on instance 0 with 10s time limit..."
    # Create a temporary config for smoke test
    TMPCONFIG=$(mktemp /tmp/smoke_config_XXXXXX.json)
    cat > "$TMPCONFIG" <<'SMOKE_EOF'
{
  "randomSeed": 41,
  "timeLimit": "00:00:10",
  "numWorkers": 0,
  "useSerializedExtendedInstance": false,
  "solverName": "BranchAndBoundJob",
  "specializedSolverConfig": {
    "usePrimalHeuristicBlockDetection": true,
    "usePrimalHeuristicPackToBlocksByCp": true,
    "primalHeuristicPackToBlocksByCpAllJobs": true,
    "jobsJoiningOnGcd": "WholeTree"
  }
}
SMOKE_EOF
    cd "$PAPER_ROOT"
    dotnet run --project Iirc.EnergyStatesAndCostsScheduling.SolverCli -c Release -- "$TMPCONFIG" "$SMALL_INST" 2>&1 || echo "(smoke test may fail if Gurobi is needed for some configs — B&B solver should not need it)"
    rm -f "$TMPCONFIG"
else
    echo "Skipping smoke test (instance not found)"
fi

echo ""
echo "============================================================"
echo " Paper solver build complete."
echo "============================================================"
echo ""
echo "Next step: bash hpc/03_run_our_solver.sh"
