#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_BASE="$ROOT/hpc/results_studies"

log() {
  echo "[$(date +%H:%M:%S)] $*"
}

study_done() {
  local out_dir="$1"
  [[ -f "$out_dir/report.md" ]]
}

run_if_needed() {
  local label="$1"
  local out_dir="$2"
  shift 2
  if study_done "$out_dir"; then
    log "Skipping $label (existing report found at $out_dir/report.md)"
    return 0
  fi
  "$@"
}

ensure_extension_benchmarks() {
  local data_root="$ROOT/data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/datasets"
  if [[ ! -d "$data_root/paperext_scalability_large_n_202604" || ! -d "$data_root/paperext_backup_realistic_202604" || ! -d "$data_root/paperext_k_boundary_202604" ]]; then
    log "Formal extension datasets missing; running setup once."
    bash "$ROOT/hpc/benchmark_extensions/00_setup_extension_benchmarks.sh"
  fi
}

run_if_needed "Relaxation Quality Study" \
  "$OUTPUT_BASE/revised_relaxation_quality" \
  bash "$ROOT/hpc/run_relaxation_quality.sh" --output-dir "$OUTPUT_BASE/revised_relaxation_quality"

run_if_needed "Max-Gap Robustness Study" \
  "$OUTPUT_BASE/max_gap_table2" \
  bash "$ROOT/hpc/run_max_gap_study.sh" --output-dir "$OUTPUT_BASE/max_gap_table2"

ensure_extension_benchmarks

run_if_needed "Certification Contribution Study" \
  "$OUTPUT_BASE/certification_scalability" \
  bash "$ROOT/hpc/run_certification_study.sh" --output-dir "$OUTPUT_BASE/certification_scalability"

run_if_needed "Backup Necessity Study" \
  "$OUTPUT_BASE/backup_necessity" \
  bash "$ROOT/hpc/run_backup_necessity_study.sh" --output-dir "$OUTPUT_BASE/backup_necessity"

run_if_needed "Structure Hardness Study" \
  "$OUTPUT_BASE/structure_hardness_table2" \
  bash "$ROOT/hpc/run_structure_hardness_study.sh" --output-dir "$OUTPUT_BASE/structure_hardness_table2"
