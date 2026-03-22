# HPC Benchmark Suite — PaST 1||TEC Solver

Full regression + head-to-head comparison on all **1,517+ instances** from
Benedikt et al. (2025), plus dedicated study runners for:

- component ablation
- banded-vs-full SPACES
- `G` parameter sweeps

## Quick Start

```bash
# 1. Prepare environment
bash hpc/setup_hpc_env.sh

# 2. Install benchmark data
bash hpc/setup_benchmark_data.sh

# 3. Build our solver
bash hpc/build_solver.sh

# 4. Run one study
bash hpc/run_component_ablation.sh
bash hpc/run_spaces_ablation.sh
bash hpc/run_g_sweep.sh

# 5. Or run the full regression benchmark
bash hpc/run_full_benchmark.sh --skip-paper
```

## Prerequisites

| Dependency | Version | Needed For |
|------------|---------|------------|
| C++17 compiler (g++ / clang++) | ≥ GCC 9 | Our solver |
| CMake | ≥ 3.16 | Our solver build |
| Python 3 | ≥ 3.8 | Runner scripts |
| .NET SDK | 8.0 | Paper's C# solver (optional) |

Run `bash hpc/setup_hpc_env.sh` for the recommended setup path.

## Scripts

### Main setup / benchmark scripts

| Script | Purpose |
|---|---|
| `setup_hpc_env.sh` | Install/check dependencies and free Python backends |
| `setup_benchmark_data.sh` | Clone upstream benchmark repo and install datasets from tarball |
| `build_solver.sh` | Build our C++ solver |
| `00_install_deps.sh` | Lower-level dependency checker |
| `01_build_our_solver.sh` | Lower-level C++ build script |
| `02_build_paper_solver.sh` | Build the paper's solver |
| `03_run_our_solver.py` | Run our solver on the main benchmark |
| `04_run_paper_solver.py` | Run the paper's solver on the main benchmark |
| `05_analyze_results.py` | Aggregate and compare benchmark results |
| `run_full_benchmark.sh` | Master wrapper for the full regression run |

### Study runners

| Script | Purpose |
|---|---|
| `run_component_ablation.sh` | Exact bin-packing + smart-reconstruction study |
| `run_spaces_ablation.sh` | Banded SPACES vs full SPACES |
| `run_g_sweep.sh` | `G` parameter sweep around the automatic value |
| `studies/component_ablation.py` | Study implementation |
| `studies/spaces_ablation.py` | Study implementation |
| `studies/g_sweep.py` | Study implementation |
| `studies/common.py` | Shared study utilities |

### Deprecated study scripts

Older study runners live in [deprecated/README.md](/Users/mac/Documents/Study/PFE/PaST/hpc/deprecated/README.md).

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
# Prepare benchmark data and build first
bash hpc/setup_benchmark_data.sh
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
├── results_studies/
│   ├── component_ablation/
│   │   ├── combined.csv
│   │   ├── report.md
│   │   └── run.log
│   ├── spaces_ablation/
│   │   ├── combined.csv
│   │   ├── report.md
│   │   └── run.log
│   └── g_sweep/
│       ├── combined.csv
│       ├── report.md
│       └── run.log
├── analysis/              # Comparison analysis (05_analyze_results.py)
├── deprecated/            # Older study runners kept for reference
└── slurm_logs/
```

## Study Commands

```bash
# Component ablation
bash hpc/run_component_ablation.sh

# Banded SPACES vs full SPACES
bash hpc/run_spaces_ablation.sh

# G sweep (safe values only)
bash hpc/run_g_sweep.sh

# G sweep with exploratory below-auto values
bash hpc/run_g_sweep.sh --include-unsafe
```

Default study outputs:

```text
hpc/results_studies/component_ablation/
hpc/results_studies/spaces_ablation/
hpc/results_studies/g_sweep/
```

Each directory contains:

- `combined.csv`
- one CSV per configuration
- `report.md`
- `run.log`

## Benchmark Data Layout

Our benchmark and study runners expect the paper datasets at:

```text
data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/datasets/
```

A fresh upstream clone does not include those dataset directories. Use:

```bash
bash hpc/setup_benchmark_data.sh
```

This script:

- clones `CTU-IIG/green-scheduling-bab` if missing
- extracts the benchmark datasets from `data/paper_datasets.tar.gz`
- verifies that `benedikt2025b_groups` matches the corrected local regeneration

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
- SPACES controls:
  - `PAST_MAX_GAP_OVERRIDE=full|auto|<int>`
  - `PAST_MAX_GAP_SCALE=<float>`

### Paper's Solver (C# B&B)
- Solution: `data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/`
- Projects: `SolverCli` (single instance) and `Experiments` (batch)
- Algorithm: Branch-and-bound with SPACES preprocessing + ILP
- Requires: .NET 8 SDK
