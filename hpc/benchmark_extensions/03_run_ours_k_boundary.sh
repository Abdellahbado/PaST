#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-$ROOT/hpc/results_studies/benchmark_extensions}"
mkdir -p "$OUT_DIR"

python3 "$ROOT/hpc/benchmark_extensions/run_our_extension_suite.py" \
  --suite k_boundary \
  --out "$OUT_DIR/ours_k_boundary.csv" \
  --solver-timeout 600 \
  --exact-time-limit 20 \
  --batch-size 2
