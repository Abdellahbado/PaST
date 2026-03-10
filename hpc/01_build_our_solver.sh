#!/usr/bin/env bash
###############################################################################
# 01_build_our_solver.sh — Build the C++ relaxed DP solver
#
# Usage:  bash hpc/01_build_our_solver.sh
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo " Building Our C++ Solver (stateful_compare)"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

cd "$ROOT/solvers/cpp"
mkdir -p build
cd build

# Clean rebuild to ensure consistency
rm -rf CMakeCache.txt CMakeFiles/

# Configure with Release flags
cmake -DCMAKE_BUILD_TYPE=Release .. 2>&1

# Build with available cores
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo ""
echo "Building with $NPROC threads..."
make -j"$NPROC" stateful_compare 2>&1

# Verify binary
BINARY="$ROOT/solvers/cpp/build/stateful_compare"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    exit 1
fi

echo ""
echo "--- Smoke test ---"
"$BINARY" paper-example

echo ""
echo "--- Binary info ---"
ls -lh "$BINARY"
file "$BINARY" 2>/dev/null || true

echo ""
echo "============================================================"
echo " Build successful: $BINARY"
echo "============================================================"
echo ""
echo "Next step: bash hpc/02_build_paper_solver.sh"
