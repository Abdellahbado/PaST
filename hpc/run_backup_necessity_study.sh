#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT/hpc/studies/backup_necessity.py" \
  --output-dir "$ROOT/hpc/results_studies/backup_necessity" \
  "$@"
