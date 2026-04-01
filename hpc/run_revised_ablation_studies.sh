#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_BASE="$ROOT/hpc/results_studies"

bash "$ROOT/hpc/run_relaxation_quality.sh" --output-dir "$OUTPUT_BASE/revised_relaxation_quality"
bash "$ROOT/hpc/run_max_gap_study.sh" --output-dir "$OUTPUT_BASE/max_gap_table2"
bash "$ROOT/hpc/run_certification_study.sh" --output-dir "$OUTPUT_BASE/certification_scalability"
bash "$ROOT/hpc/run_backup_necessity_study.sh" --output-dir "$OUTPUT_BASE/backup_necessity"
bash "$ROOT/hpc/run_structure_hardness_study.sh" --output-dir "$OUTPUT_BASE/structure_hardness_table2"
