#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-$ROOT/hpc/results_studies/benchmark_extensions}"
mkdir -p "$OUT_DIR"

python3 "$ROOT/hpc/benchmark_extensions/run_paper_extension_suite.py" \
  --suite k_boundary \
  --time-limit 600 \
  --output "$OUT_DIR/paper_k_boundary.csv"
