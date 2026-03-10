#!/usr/bin/env bash
###############################################################################
# 02_build_paper_solver.sh — Build the paper's C# B&B-SPACES solver
#
# Requires: .NET 8 SDK, g++ (for C++ compilation), cmake
# Usage:    bash hpc/02_build_paper_solver.sh
#
# The paper's solver has two parts:
#   1. C# Experiments runner — orchestrates solving, reads/writes results
#   2. C++ BranchAndBoundJob binary — the actual B&B-SPACES solver
#
# The pre-built C++ binary needs Gurobi + CPLEX (commercial) at runtime.
# Since these are not available on HPC, we patch the source to stub out
# the two optional heuristics (BlockFinding → Gurobi, PackToBlocksByCp → CPLEX)
# and compile from source with just g++ + OpenMP.
#
# This script:
#   - Clones the paper repo if missing
#   - Patches for .NET 8 / no-Gurobi (C# side only — removes Gurobi C# wrapper)
#   - Builds the C# Experiments project
#   - Patches C++ source to remove Gurobi/CPLEX deps, compiles from source
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

# ── Build C++ B&B-SPACES binaries from source (no Gurobi/CPLEX) ─────────────
# The pre-built binaries in the repo need libgurobi.so + libcplex2212.so +
# a Gurobi license at runtime.  The HPC has none of these.
#
# Gurobi & CPLEX are used for two OPTIONAL primal heuristics:
#   - BlockFinding (Gurobi MIP) — guarded by mBlockFinding config
#   - PackToBlocksByCp (CPLEX CP Optimizer) — guarded by usePrimalHeuristicPackToBlocksByCp
#
# The core B&B algorithm doesn't need either.  We patch the source to
# stub out these two heuristics, then compile with g++ + OpenMP only.
CPP_SRC_DIR="$PAPER_ROOT/Iirc.EnergyStatesAndCostsScheduling.Shared/cpp"
DLL_DIR="$PAPER_ROOT/Iirc.EnergyStatesAndCostsScheduling.Experiments/bin/Release/net8.0"
CPP_BIN_TARGET="$DLL_DIR/cpp/bin"

echo ""
echo "--- Patching C++ source (removing Gurobi/CPLEX dependencies) ---"

# 1) Stub BlockFinding.h — remove Gurobi header and GRBEnv member
cat > "$CPP_SRC_DIR/src/algorithms/BlockFinding.h" << 'STUBEOF'
#ifndef ENERGYSTATESANDCOSTSSCHEDULING_BLOCKFINDING_H
#define ENERGYSTATESANDCOSTSSCHEDULING_BLOCKFINDING_H
#include <map>
#include "../datastructs/Block.h"
#include "../datastructs/FixedPermCostComputation.h"
namespace escs {
    class BlockFinding {
    public:
        enum BlockFindingStrategy { MinimizeLengthDifference = 0 };
        vector<int> mAssignments;
        bool mSolutionSameAsBlocks;
        BlockFinding() : mSolutionSameAsBlocks(true) {}
        void solve(BlockFindingStrategy, const Instance&, const vector<Block>&,
                   optional<chrono::milliseconds>) {
            mAssignments.clear();
            mSolutionSameAsBlocks = true;
        }
    };
}
#endif
STUBEOF
echo "  Stubbed BlockFinding.h (no Gurobi)"

# 2) Stub BlockFinding.cpp — empty (all logic is now inline in header)
cat > "$CPP_SRC_DIR/src/algorithms/BlockFinding.cpp" << 'STUBEOF'
#include "BlockFinding.h"
STUBEOF
echo "  Stubbed BlockFinding.cpp"

# 3) Stub PackToBlocksByCp.cpp — returns false (no CPLEX)
cat > "$CPP_SRC_DIR/src/algorithms/PackToBlocksByCp.cpp" << 'STUBEOF'
#include "PackToBlocksByCp.h"
namespace escs {
    PackToBlocksByCp::PackToBlocksByCp() {}
    bool PackToBlocksByCp::solve(
            const vector<Block>&, const vector<int>&,
            optional<chrono::milliseconds>) {
        mPermStartTimes.clear();
        mPermProcTimes.clear();
        return false;
    }
}
STUBEOF
echo "  Stubbed PackToBlocksByCp.cpp (no CPLEX)"

# 4) Stub PackToBlocksByCp.h — remove <ilcp/cp.h> include
cat > "$CPP_SRC_DIR/src/algorithms/PackToBlocksByCp.h" << 'STUBEOF'
#ifndef ENERGYSTATESANDCOSTSSCHEDULING_PACKTOBLOCKSBYCP_H
#define ENERGYSTATESANDCOSTSSCHEDULING_PACKTOBLOCKSBYCP_H
#include <map>
#include "../datastructs/Block.h"
#include "../datastructs/FixedPermCostComputation.h"
namespace escs {
    class PackToBlocksByCp {
    public:
        vector<int> mPermProcTimes;
        vector<int> mPermStartTimes;
        PackToBlocksByCp();
        bool solve(const vector<Block> &blocks, const vector<int> &procTimes,
                   optional<chrono::milliseconds> timeLimit);
    };
}
#endif
STUBEOF
echo "  Stubbed PackToBlocksByCp.h (no CPLEX)"

# 5) Patch BranchAndBoundJob.h — remove #include <gurobi_c++.h> and GRBEnv member
sed -i.bak '/#include <gurobi_c++.h>/d' "$CPP_SRC_DIR/src/solvers/BranchAndBoundJob.h"
sed -i.bak '/const GRBEnv mEnv;/d' "$CPP_SRC_DIR/src/solvers/BranchAndBoundJob.h"
rm -f "$CPP_SRC_DIR/src/solvers/BranchAndBoundJob.h.bak"
echo "  Patched BranchAndBoundJob.h (removed GRBEnv)"

# 6) Patch BranchAndBoundJob.cpp — remove gurobi include, fix BlockFinding call
sed -i.bak '/#include <gurobi_c++.h>/d' "$CPP_SRC_DIR/src/solvers/BranchAndBoundJob.cpp"
sed -i.bak 's/BlockFinding blockFinding(mEnv)/BlockFinding blockFinding()/g' "$CPP_SRC_DIR/src/solvers/BranchAndBoundJob.cpp"
rm -f "$CPP_SRC_DIR/src/solvers/BranchAndBoundJob.cpp.bak"
echo "  Patched BranchAndBoundJob.cpp (removed Gurobi refs)"

# 7) Patch ConstructiveHeuristic.cpp — remove CPLEX include
sed -i.bak 's|#include <ilcp/cp.h>|// #include <ilcp/cp.h>  // stubbed out|' "$CPP_SRC_DIR/src/solvers/ConstructiveHeuristic.cpp"
rm -f "$CPP_SRC_DIR/src/solvers/ConstructiveHeuristic.cpp.bak"
echo "  Patched ConstructiveHeuristic.cpp (removed CPLEX include)"

# 8) Write new CMakeLists.txt — no Gurobi, no CPLEX, just g++ + OpenMP
cat > "$CPP_SRC_DIR/CMakeLists.txt" << 'CMAKEOF'
cmake_minimum_required(VERSION 3.10)
project(EnergyStatesAndCostsScheduling)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra")
IF(CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
ENDIF()

set(LIB_SRC
    src/datastructs/FixedPermCostComputation.cpp
    src/datastructs/GcdOfValues.cpp
    src/algorithms/PackToBlocksByCp.cpp
    src/algorithms/BlockFinding.cpp
    src/input/Instance.cpp
    src/input/Job.cpp
    src/input/Interval.cpp
    src/input/readers/CppInputReader.cpp
    src/output/Result.cpp
    src/solvers/SolverConfig.cpp
    src/utils/Stopwatch.cpp
)

add_executable(BranchAndBoundJob src/solvers/BranchAndBoundJob.cpp ${LIB_SRC})
add_executable(ConstructiveHeuristic src/solvers/ConstructiveHeuristic.cpp ${LIB_SRC})
add_executable(GeneticAlgorithm src/solvers/GeneticAlgorithm.cpp ${LIB_SRC})

target_link_libraries(BranchAndBoundJob ${CMAKE_DL_LIBS})
target_link_libraries(ConstructiveHeuristic ${CMAKE_DL_LIBS})
target_link_libraries(GeneticAlgorithm ${CMAKE_DL_LIBS})
CMAKEOF
echo "  Wrote new CMakeLists.txt (no Gurobi/CPLEX)"

# 9) Compile from source
echo ""
echo "--- Compiling C++ solvers from source ---"

# Remove stale CMakeCache.txt from repo (points to original author's paths)
rm -f "$CPP_SRC_DIR/CMakeCache.txt"

CPP_BUILD_DIR="$CPP_SRC_DIR/build_nogurobi"
rm -rf "$CPP_BUILD_DIR"
mkdir -p "$CPP_BUILD_DIR"
cd "$CPP_BUILD_DIR"

cmake "$CPP_SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    2>&1

make -j"$(nproc 2>/dev/null || echo 4)" 2>&1

# 10) Install compiled binaries where C# expects them
mkdir -p "$CPP_BIN_TARGET"
for solver in BranchAndBoundJob ConstructiveHeuristic GeneticAlgorithm; do
    SRC_BIN="$CPP_SRC_DIR/bin/$solver"
    if [ -f "$SRC_BIN" ]; then
        cp "$SRC_BIN" "$CPP_BIN_TARGET/$solver"
        chmod +x "$CPP_BIN_TARGET/$solver"
        echo "  Installed $solver (compiled from source)"
    else
        echo "  WARNING: $solver not produced by build"
    fi
done

cd "$PAPER_ROOT"

if [ ! -f "$CPP_BIN_TARGET/BranchAndBoundJob" ]; then
    echo "ERROR: BranchAndBoundJob failed to compile."
    exit 1
fi

# Verify no missing libraries
echo ""
echo "--- Checking C++ binary runtime dependencies ---"
if command -v ldd &>/dev/null; then
    MISSING_LIBS=$(ldd "$CPP_BIN_TARGET/BranchAndBoundJob" 2>&1 | grep "not found" || true)
    if [ -n "$MISSING_LIBS" ]; then
        echo "WARNING: Missing shared libraries:"
        echo "$MISSING_LIBS"
    else
        echo "  All shared libraries resolved — no Gurobi/CPLEX needed."
    fi
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
