#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT/hpc/studies/g_sweep.py" \
  --section 2 \
  --output-dir "$ROOT/hpc/results_studies/g_sweep" \
  "$@"
