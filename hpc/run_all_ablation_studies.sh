#!/usr/bin/env bash
# ============================================================================
#  Unified Ablation Studies Runner
#  
#  Runs all 5 ablation studies from the implementation plan:
#    1. Packing Hierarchy (UB Contribution)
#    2. Relaxation Quality (LB Contribution)
#    3. G Sweep (SPACES Bandwidth Sensitivity)
#    4. SPACES Storage (Banded vs Full)
#    5. Pipeline Cascade (already answered by existing data)
#
#  Usage:
#    bash hpc/run_all_ablation_studies.sh [OPTIONS]
#
#  Options:
#    --section SECTION    Dataset section: 2 (default), 1, all, table1, table2, fig9
#    --max-instances N    Limit instances (0 = all, default)
#    --time-limit SEC     Per-instance time limit (default: 600)
#    --studies LIST       Comma-separated: 1,2,3,4 or "all" (default: all)
#    --parallel           Run independent studies in background (studies 2-4)
#    --output-dir DIR     Base output directory
#    --dry-run            Print commands without executing
#    --help               Show this help
#
#  HPC Setup (run once before studies):
#    bash hpc/setup_hpc_env.sh
#    bash hpc/setup_benchmark_data.sh
#    cmake --build solvers/cpp/build --target stateful_compare -j4
#
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
SECTION="2"
MAX_INSTANCES=0
TIME_LIMIT=600
STUDIES="all"
PARALLEL=false
OUTPUT_BASE="$ROOT/hpc/results_studies"
DRY_RUN=false
SOLVER="$ROOT/solvers/cpp/build/stateful_compare"
IS_CHILD="${PAST_ABLATION_CHILD:-0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

usage() {
    head -45 "$0" | tail -35
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --section)      SECTION="$2"; shift 2 ;;
        --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
        --time-limit)   TIME_LIMIT="$2"; shift 2 ;;
        --studies)      STUDIES="$2"; shift 2 ;;
        --parallel)     PARALLEL=true; shift ;;
        --output-dir)   OUTPUT_BASE="$2"; shift 2 ;;
        --solver)       SOLVER="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --help|-h)      usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

# Validate solver exists
if [[ ! -x "$SOLVER" ]]; then
    log_error "Solver not found: $SOLVER"
    log_info "Build with: cmake --build solvers/cpp/build --target stateful_compare -j4"
    exit 1
fi

# Build study list
if [[ "$STUDIES" == "all" ]]; then
    STUDY_LIST=(1 2 3 4)
else
    IFS=',' read -ra STUDY_LIST <<< "$STUDIES"
fi

# Common args for all studies
COMMON_ARGS=(
    --section "$SECTION"
    --time-limit "$TIME_LIMIT"
    --solver "$SOLVER"
)
if [[ $MAX_INSTANCES -gt 0 ]]; then
    COMMON_ARGS+=(--max-instances "$MAX_INSTANCES")
fi

mkdir -p "$OUTPUT_BASE"

# Track background PIDs
declare -a BG_PIDS=()
declare -a BG_NAMES=()

run_cmd() {
    local name="$1"
    shift
    if $DRY_RUN; then
        log_info "[DRY-RUN] $name:"
        echo "  $*"
    else
        log_info "Running: $name"
        "$@"
    fi
}

run_bg() {
    local name="$1"
    shift
    if $DRY_RUN; then
        log_info "[DRY-RUN] $name (would run in background):"
        echo "  $*"
    else
        log_info "Starting background: $name"
        "$@" &
        BG_PIDS+=($!)
        BG_NAMES+=("$name")
    fi
}

wait_all_bg() {
    if [[ ${#BG_PIDS[@]} -eq 0 ]]; then
        return 0
    fi
    log_info "Waiting for ${#BG_PIDS[@]} background jobs..."
    local failed=0
    for i in "${!BG_PIDS[@]}"; do
        if wait "${BG_PIDS[$i]}"; then
            log_ok "${BG_NAMES[$i]} completed"
        else
            log_error "${BG_NAMES[$i]} failed"
            ((failed++)) || true
        fi
    done
    BG_PIDS=()
    BG_NAMES=()
    return $failed
}

# ============================================================================
# Study 1: Packing Hierarchy (UB Contribution)
# ============================================================================
run_study_1() {
    log_info "============================================"
    log_info "Study 1: Packing Hierarchy (UB Contribution)"
    log_info "============================================"
    
    local outdir="$OUTPUT_BASE/study1_packing_hierarchy"
    mkdir -p "$outdir"
    
    # Config: step1_only_default (Step 1 heuristic packing only)
    run_cmd "Study 1: step1_only_default" \
        python3 "$ROOT/hpc/studies/component_ablation.py" \
        "${COMMON_ARGS[@]}" \
        --config step1_only_default \
        --output-dir "$outdir/step1_only_default"
    
    # Note: full_blockdp_only_pack is already run and available at:
    # hpc/studies/full_blockdp_only.csv
    log_info "Note: full_blockdp_only_pack results at: hpc/studies/full_blockdp_only.csv"
    
    log_ok "Study 1 complete. Results at: $outdir"
}

# ============================================================================
# Study 2: Relaxation Quality (LB Contribution)
# ============================================================================
run_study_2() {
    log_info "============================================"
    log_info "Study 2: Relaxation Quality (LB Contribution)"
    log_info "============================================"
    
    local outdir="$OUTPUT_BASE/study2_relaxation_quality"
    mkdir -p "$outdir"
    
    run_cmd "Study 2: relaxation quality" \
        python3 "$ROOT/hpc/studies/relaxation_quality.py" \
        "${COMMON_ARGS[@]}" \
        --output-dir "$outdir"
    
    log_ok "Study 2 complete. Results at: $outdir"
}

# ============================================================================
# Study 3: G Sweep (SPACES Bandwidth Sensitivity)
# ============================================================================
run_study_3() {
    log_info "============================================"
    log_info "Study 3: G Sweep (SPACES Bandwidth)"
    log_info "============================================"
    
    local outdir="$OUTPUT_BASE/study3_g_sweep"
    mkdir -p "$outdir"
    
    run_cmd "Study 3: G sweep" \
        python3 "$ROOT/hpc/studies/g_sweep.py" \
        "${COMMON_ARGS[@]}" \
        --output-dir "$outdir"
    
    log_ok "Study 3 complete. Results at: $outdir"
}

# ============================================================================
# Study 4: SPACES Storage (Banded vs Full)
# ============================================================================
run_study_4() {
    log_info "============================================"
    log_info "Study 4: SPACES Storage (Banded vs Full)"
    log_info "============================================"
    
    local outdir="$OUTPUT_BASE/study4_spaces_ablation"
    mkdir -p "$outdir"
    
    run_cmd "Study 4: SPACES ablation" \
        python3 "$ROOT/hpc/studies/spaces_ablation.py" \
        "${COMMON_ARGS[@]}" \
        --output-dir "$outdir"
    
    log_ok "Study 4 complete. Results at: $outdir"
}

# ============================================================================
# Study 5: Pipeline Cascade
# ============================================================================
report_study_5() {
    log_info "============================================"
    log_info "Study 5: Pipeline Cascade (pre-existing data)"
    log_info "============================================"
    
    local outdir="$OUTPUT_BASE/study5_pipeline_cascade"
    mkdir -p "$outdir"
    local existing="$ROOT/hpc/studies/full_blockdp_only.csv"
    if [[ -f "$existing" ]]; then
        log_ok "Study 5 data exists at: $existing"
        log_info "This study uses existing data - no new run needed."
        log_info "Key finding: ALL 560 instances solved at fwd_relax (Phase 1)."
        cp "$existing" "$outdir/full_blockdp_only.csv"
        log_info "Snapshot copied to: $outdir/full_blockdp_only.csv"
    else
        log_warn "Expected file not found: $existing"
        log_info "Run Study 1 first to generate the data."
    fi
}

# ============================================================================
# Main execution
# ============================================================================
log_info "============================================"
log_info "Ablation Studies Runner"
log_info "============================================"
log_info "Section: $SECTION"
log_info "Max instances: $([ $MAX_INSTANCES -eq 0 ] && echo 'all' || echo $MAX_INSTANCES)"
log_info "Time limit: ${TIME_LIMIT}s"
log_info "Studies: ${STUDY_LIST[*]}"
log_info "Output: $OUTPUT_BASE"
log_info "Parallel: $PARALLEL"
log_info "Solver: $SOLVER"
echo

# Study 1 always runs first (others may depend on it)
if [[ " ${STUDY_LIST[*]} " =~ " 1 " ]]; then
    run_study_1
fi

# Studies 2, 3, 4 can run in parallel
if $PARALLEL; then
    for study in "${STUDY_LIST[@]}"; do
        case $study in
            2)
                run_bg "Study 2" env PAST_ABLATION_CHILD=1 bash "$ROOT/hpc/run_all_ablation_studies.sh" \
                    --section "$SECTION" \
                    --time-limit "$TIME_LIMIT" \
                    --studies 2 \
                    --output-dir "$OUTPUT_BASE" \
                    --solver "$SOLVER" \
                    $([[ $MAX_INSTANCES -gt 0 ]] && printf -- '--max-instances %s' "$MAX_INSTANCES") \
                    $($DRY_RUN && printf -- '--dry-run')
                ;;
            3)
                run_bg "Study 3" env PAST_ABLATION_CHILD=1 bash "$ROOT/hpc/run_all_ablation_studies.sh" \
                    --section "$SECTION" \
                    --time-limit "$TIME_LIMIT" \
                    --studies 3 \
                    --output-dir "$OUTPUT_BASE" \
                    --solver "$SOLVER" \
                    $([[ $MAX_INSTANCES -gt 0 ]] && printf -- '--max-instances %s' "$MAX_INSTANCES") \
                    $($DRY_RUN && printf -- '--dry-run')
                ;;
            4)
                run_bg "Study 4" env PAST_ABLATION_CHILD=1 bash "$ROOT/hpc/run_all_ablation_studies.sh" \
                    --section "$SECTION" \
                    --time-limit "$TIME_LIMIT" \
                    --studies 4 \
                    --output-dir "$OUTPUT_BASE" \
                    --solver "$SOLVER" \
                    $([[ $MAX_INSTANCES -gt 0 ]] && printf -- '--max-instances %s' "$MAX_INSTANCES") \
                    $($DRY_RUN && printf -- '--dry-run')
                ;;
        esac
    done
    wait_all_bg
else
    for study in "${STUDY_LIST[@]}"; do
        case $study in
            2) run_study_2 ;;
            3) run_study_3 ;;
            4) run_study_4 ;;
        esac
    done
fi

if [[ "$IS_CHILD" != "1" ]]; then
    # Study 5 just reports existing data
    report_study_5
fi

# ============================================================================
# Summary
# ============================================================================
if [[ "$IS_CHILD" != "1" ]]; then
    echo
    log_info "============================================"
    log_info "All studies complete!"
    log_info "============================================"
    log_info "Results are in: $OUTPUT_BASE"
    echo
    log_info "Study outputs:"
    for study in "${STUDY_LIST[@]}"; do
        case $study in
            1) echo "  Study 1 (Packing): $OUTPUT_BASE/study1_packing_hierarchy/" ;;
            2) echo "  Study 2 (Relaxation): $OUTPUT_BASE/study2_relaxation_quality/" ;;
            3) echo "  Study 3 (G Sweep): $OUTPUT_BASE/study3_g_sweep/" ;;
            4) echo "  Study 4 (SPACES): $OUTPUT_BASE/study4_spaces_ablation/" ;;
        esac
    done
    echo "  Study 5 (Pipeline): $OUTPUT_BASE/study5_pipeline_cascade/full_blockdp_only.csv"
fi
