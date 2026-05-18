#!/usr/bin/env bash
set -euo pipefail

# Paper rerun wrapper for the stateful-DP single-machine TOU experiments.
# Run from the repository root on the HPC login/compute node.
#
# Usage:
#   bash hpc/run_stateful_dp_paper_experiments.sh build
#   bash hpc/run_stateful_dp_paper_experiments.sh original-benchmark
#   bash hpc/run_stateful_dp_paper_experiments.sh paper-groups
#   bash hpc/run_stateful_dp_paper_experiments.sh k4-generator
#   bash hpc/run_stateful_dp_paper_experiments.sh k2-routing
#   bash hpc/run_stateful_dp_paper_experiments.sh easy-k
#   bash hpc/run_stateful_dp_paper_experiments.sh hard-k
#   bash hpc/run_stateful_dp_paper_experiments.sh hard-k-cert
#   bash hpc/run_stateful_dp_paper_experiments.sh main
#
# Notes:
# - "main" runs the final paper-facing reruns, not every failed diagnostic plan.
# - Some tasks are intentionally long. Submit individual modes as separate Slurm
#   jobs if your cluster has wall-time limits.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUILD_JOBS="${BUILD_JOBS:-1}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

build_solver() {
  log "Configuring/building stateful_compare"
  cmake -S solvers/cpp -B solvers/cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build solvers/cpp/build --target stateful_compare -j"${BUILD_JOBS}"
}

original_benchmark() {
  log "Original/corrected benchmark rerun"
  bash hpc/setup_benchmark_data.sh
  bash hpc/run_revised_ablation_studies.sh
}

paper_groups() {
  log "Paper-group large-n extension rerun"
  "${PYTHON_BIN}" research/k_vs_arithmetic_axes_20260412/run_plan05_paper_groups_extension.py \
    --phase n \
    --resume \
    --n-seeds 0,1 \
    --phase2-n-values 300,400,500,600,750,1000 \
    --phase2-n-large 1500,2500,3500,5000,6000,7000,8000,10000 \
    --time-limit-n 900 \
    --time-limit-n-large 1800 \
    --force-extend-families g12345678910,g3567,g37,g810,g246810,g12357,g24
}

k4_generator() {
  log "K=4 generator / energy-core rerun"
  "${PYTHON_BIN}" research/k_vs_arithmetic_axes_20260412/run_plan10_k4_generator_compare.py
}

k2_routing() {
  log "K=2 corrected routing rerun (g37/g810 context)"
  "${PYTHON_BIN}" research/k_vs_arithmetic_axes_20260412/run_plan13_two_track_recovery.py
}

easy_k() {
  log "Easy contiguous-unit K scaling rerun"
  "${PYTHON_BIN}" research/k_vs_arithmetic_axes_20260412/run_plan28_easy_k_scaling.py
  "${PYTHON_BIN}" research/k_vs_arithmetic_axes_20260412/run_plan28_k40_optional.py || {
    log "Optional K=40 runner failed or skipped; inspect output before using K=40 in paper."
  }
}

hard_k() {
  log "Hard irregular fixed-n K-axis boundary rerun"
  "${PYTHON_BIN}" research/k_vs_arithmetic_axes_20260412/run_plan18_k_boundary_refine_n1000.py
}

hard_k_cert() {
  log "Hard K10/K12 certified anytime rerun"
  "${PYTHON_BIN}" research/k_vs_arithmetic_axes_20260412/run_plan33_cert_anytime.py --phase all --time-limit 3000
}

case "${1:-}" in
  build)
    build_solver
    ;;
  original-benchmark)
    build_solver
    original_benchmark
    ;;
  paper-groups)
    build_solver
    paper_groups
    ;;
  k4-generator)
    build_solver
    k4_generator
    ;;
  k2-routing)
    build_solver
    k2_routing
    ;;
  easy-k)
    build_solver
    easy_k
    ;;
  hard-k)
    build_solver
    hard_k
    ;;
  hard-k-cert)
    build_solver
    hard_k_cert
    ;;
  main)
    build_solver
    original_benchmark
    paper_groups
    k4_generator
    k2_routing
    easy_k
    hard_k
    hard_k_cert
    ;;
  *)
    cat <<'EOF'
Usage: bash hpc/run_stateful_dp_paper_experiments.sh <mode>

Modes:
  build               Build solvers/cpp/build/stateful_compare only.
  original-benchmark  Corrected original benchmark / component ablation rerun.
  paper-groups        Paper job-groups large-n extension.
  k4-generator        K=4 energy-core generator speedup rerun.
  k2-routing          Corrected K=2 routing rerun.
  easy-k              Easy contiguous-unit K scaling rerun.
  hard-k              Hard irregular K-axis boundary rerun.
  hard-k-cert         PLAN33 hard K10/K12 certified anytime rerun.
  main                Run all final paper-facing modes sequentially.

Recommended HPC use:
  Submit each mode separately as a Slurm job unless the wall-time limit is very large.
EOF
    exit 2
    ;;
esac

