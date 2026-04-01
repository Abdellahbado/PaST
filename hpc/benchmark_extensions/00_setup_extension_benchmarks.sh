#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 hpc/benchmark_extensions/build_extension_suites.py --suite all

echo "Formal benchmark-extension suites are ready under:"
echo "  $ROOT/data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/datasets/"
