#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT/hpc/studies/spaces_ablation.py" \
  --section 2 \
  --output-dir "$ROOT/hpc/results_studies/spaces_ablation" \
  "$@"
