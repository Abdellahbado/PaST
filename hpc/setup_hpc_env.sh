#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo " PaST HPC Study Setup"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

bash "$ROOT/hpc/00_install_deps.sh"

echo ""
echo "--- Installing free Python backends for relaxed-block packing ---"
python3 -m pip install --upgrade --target "$ROOT/vendor/constraint_py" python-constraint
python3 -m pip install --upgrade --target "$ROOT/vendor/ortools_py" ortools
python3 -m pip install --upgrade --target "$ROOT/vendor/z3_py" z3-solver

echo ""
echo "--- Installing analysis packages ---"
python3 -m pip install --user --upgrade pandas matplotlib

echo ""
echo "Setup complete."
echo "Next step: bash hpc/build_solver.sh"
