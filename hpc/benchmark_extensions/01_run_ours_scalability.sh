#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-$ROOT/hpc/results_studies/benchmark_extensions}"
mkdir -p "$OUT_DIR"

python3 "$ROOT/hpc/benchmark_extensions/run_our_extension_suite.py" \
  --suite scalability_large_n \
  --out "$OUT_DIR/ours_scalability_large_n.csv" \
  --solver-timeout 600 \
  --batch-size 4
