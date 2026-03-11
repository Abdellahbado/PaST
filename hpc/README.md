# HPC Benchmark Suite — PaST 1||TEC Solver

Full regression + head-to-head comparison on all **1,517+ instances** from
Benedikt et al. (2025), plus 84 hard synthetic instances.

## Quick Start

```bash
# One-command full run (both solvers, all instances):
bash hpc/run_full_benchmark.sh

# Or just our solver (regression check):
bash hpc/run_full_benchmark.sh --skip-paper

# On SLURM:
sbatch hpc/slurm_full.sbatch       # both solvers
sbatch hpc/slurm_ours_only.sbatch  # our solver only
```

## Prerequisites

| Dependency | Version | Needed For |
|------------|---------|------------|
| C++17 compiler (g++ / clang++) | ≥ GCC 9 | Our solver |
| CMake | ≥ 3.16 | Our solver build |
| Python 3 | ≥ 3.8 | Runner scripts |
| .NET SDK | 8.0 | Paper's C# solver (optional) |

Run `bash hpc/00_install_deps.sh` to check/install all dependencies.

## Scripts

All scripts are in `hpc/` and numbered in execution order:

| # | Script | Purpose |
|---|--------|---------|
| 00 | `00_install_deps.sh` | Check & install dependencies |
| 01 | `01_build_our_solver.sh` | CMake Release build of our C++ solver |
| 02 | `02_build_paper_solver.sh` | `dotnet build` of paper's C# solver |
| 03 | `03_run_our_solver.py` | Run our solver on all 1,517+ instances |
| 04 | `04_run_paper_solver.py` | Run paper's solver on all instances |
| 05 | `05_analyze_results.py` | Aggregate, compare, generate tables & plots |
| 06 | `06_ablation_study.py` | Ablation: Banded SPACES vs Bound-and-Refine |
| 07 | `07_analyze_ablation.py` | Analyze ablation results, generate LaTeX + plots |
| — | `run_full_benchmark.sh` | Master wrapper: runs 00–05 in sequence |
| — | `slurm_full.sbatch` | SLURM job: both solvers |
| — | `slurm_ours_only.sbatch` | SLURM job: our solver only |

## Instance Breakdown

| Section | Dataset(s) | # Instances | Machine Type |
|---------|-----------|-------------|-------------|
| Table 1 (§5.1) | 6 datasets (3, 5, 6, 7, 8, 9, 10 jobs) | 72 | NOSBY + TWOSBY |
| Table 2 (§5.2) | `benedikt2025b_groups` | 560 | TWOSBY |
| Figure 9 (§5.3) | `benedikt2025_groups` | 560 | TWOSBY |
| Drops ablation | `benedikt2025b_drops` | 240 | TWOSBY |
| GCD analysis | `benedikt2025b_gcd` | 1 | NOSBY |
| Hard synthetic | Generated on-the-fly | 84 | NOSBY + TWOSBY |
| **Total** | | **1,517** | |

## Running Individual Sections

```bash
# Build first
bash hpc/01_build_our_solver.sh

# Run only Table 1 instances
python3 hpc/03_run_our_solver.py --section 1 --output-dir hpc/results_ours

# Run only Table 2
python3 hpc/03_run_our_solver.py --section 2 --output-dir hpc/results_ours

# Run only Figure 9
python3 hpc/03_run_our_solver.py --section fig9 --output-dir hpc/results_ours

# Run only hard synthetic
python3 hpc/03_run_our_solver.py --section synthetic --output-dir hpc/results_ours

# Available sections: all, 1, 2, fig9, drops, gcd, synthetic
```

## Output Structure

After a full run:

```
hpc/
├── results_ours/          # Our solver results (03_run_our_solver.py)
├── results_paper/         # Paper's solver results (04_run_paper_solver.py)
├── results_ablation/      # Ablation study (06_ablation_study.py)
│   ├── ablation_combined.csv
│   ├── ablation_{full,full_spaces,exact_only,baseline}.csv
│   ├── ablation_log.txt
│   └── analysis/
│       ├── ablation_report.txt
│       ├── ablation_tables.tex
│       ├── ablation_speedup.pdf
│       └── ablation_steps.pdf
├── analysis/              # Comparison analysis (05_analyze_results.py)
└── slurm_logs/
```

## Ablation Study

The ablation study isolates the contribution of each algorithmic component:

| Config | Banded SPACES | Bound-and-Refine | Tests |
|--------|:---:|:---:|---|
| A: Full method | ✓ | ✓ | Complete system |
| B: Full SPACES | ✗ | ✓ | Banded SPACES contribution |
| C: Exact only + banded | ✓ | ✗ | Bound-and-Refine contribution |
| D: Baseline | ✗ | ✗ | Neither optimization |

```bash
# Run all 4 configs on Table 2 + Figure 9 instances:
python3 hpc/06_ablation_study.py --section all --time-limit 600

# Run only on Table 2 instances:
python3 hpc/06_ablation_study.py --section 2

# Run a single config:
python3 hpc/06_ablation_study.py --config full

# Analyze results and generate paper tables + figures:
python3 hpc/07_analyze_ablation.py --input-dir hpc/results_ablation
```

**Comparison matrix:** A vs B = Banded SPACES speedup, A vs C = Bound-and-Refine speedup,
A vs D = combined speedup. Per-step timing breakdown shows where time is spent.

## CSV Format

All result CSVs share the same schema:

```
section,dataset,instance_id,n_jobs,horizon,ub,lb,gap_pct,feasible,is_optimal,timed_out,runtime_sec,peak_rss_kb
```

| Column | Description |
|--------|------------|
| `section` | table1, table2, fig9, drops, gcd, synthetic |
| `dataset` | Dataset name (e.g., `factored_high_10_20_TWOSBY`) |
| `instance_id` | Instance filename or index |
| `n_jobs` | Number of jobs in the instance |
| `horizon` | Time horizon (sum of processing times) |
| `ub` | Best upper bound found (objective) |
| `lb` | Best lower bound found |
| `gap_pct` | Optimality gap: (ub-lb)/ub × 100 |
| `feasible` | 1 = feasible solution found |
| `is_optimal` | 1 = proved optimal (gap = 0) |
| `timed_out` | 1 = hit time limit |
| `runtime_sec` | Wall-clock time (seconds) |
| `peak_rss_kb` | Peak resident memory (KB) |

## Time Limits

Default: **600 seconds** (10 minutes) per instance, matching the paper.

Override:
```bash
python3 hpc/03_run_our_solver.py --time-limit 300  # 5 minutes
bash hpc/run_full_benchmark.sh --time-limit=300
```

## What to Check After a Run

### 1. Regression (Critical)
```bash
# Should find NO lines with non-zero gap (except timeouts):
grep -E 'gap_pct' hpc/results_ours/all_results.csv | awk -F, '$8 > 0.0 && $11 == 0'
```
Or check the analysis report:
```bash
grep 'REGRESSION\|OPEN GAP' hpc/analysis/analysis_report.txt
```

### 2. Table 1 Cost Validation
The runner validates Table 1 costs against known paper values automatically.
Check the log for `COST MISMATCH` warnings:
```bash
grep 'COST MISMATCH' hpc/results_ours/run_log.txt
```

### 3. Timing Comparison
```bash
# Quick summary
head -50 hpc/analysis/analysis_report.txt
```

### 4. Speedup per Section
The analysis script produces per-section and per-group timing breakdowns,
including geometric mean speedup ratios when both solvers' results are available.

## SLURM Notes

Before submitting:

1. **Edit partition name**: Replace `compute` in `.sbatch` files with your cluster's partition
2. **Edit account**: Uncomment and set `--account=YOUR_ACCOUNT`
3. **Load modules**: Uncomment/add module loads (e.g., `module load cmake gcc dotnet`)
4. **Create log directory**: `mkdir -p hpc/slurm_logs`
5. **Walltime**: Full run (both solvers, 1517 instances × 600s limit) worst case ~10h.
   In practice with our solver it should be < 1h; paper's solver may take longer.

## Solver Details

### Our Solver (C++ DP)
- Binary: `solvers/cpp/build/stateful_compare`
- Mode: `solve-stdin` (reads JSONL from stdin, outputs CSV)
- Build: CMake Release with `-O3 -DNDEBUG -march=native -ffast-math`
- Algorithm: 7-step cascade (relaxed DP → heuristic → local search → backward LB
  → two-class LB → exact multiset DP → sparse exact DP)

### Paper's Solver (C# B&B)
- Solution: `data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/`
- Projects: `SolverCli` (single instance) and `Experiments` (batch)
- Algorithm: Branch-and-bound with SPACES preprocessing + ILP
- Requires: .NET 8 SDK
