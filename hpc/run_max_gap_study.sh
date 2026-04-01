#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT/hpc/studies/max_gap_robustness.py" \
  --target table2 \
  --output-dir "$ROOT/hpc/results_studies/max_gap_table2" \
  "$@"
