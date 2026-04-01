#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT/hpc/studies/certification_contribution.py" \
  --suite scalability_large_n \
  --output-dir "$ROOT/hpc/results_studies/certification_scalability" \
  "$@"
