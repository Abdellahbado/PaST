#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT/hpc/studies/structure_hardness.py" \
  --target table2 \
  --output-dir "$ROOT/hpc/results_studies/structure_hardness_table2" \
  "$@"
