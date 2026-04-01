# Ablation Studies - HPC Execution Guide

## Quick Start

### 1. Setup (run once)
```bash
# SSH to HPC
ssh user@hpc.example.com

# Clone/update repository
git clone https://github.com/your/PaST.git
cd PaST

# Setup environment and build
bash hpc/setup_hpc_env.sh
bash hpc/setup_benchmark_data.sh
cmake --build solvers/cpp/build --target stateful_compare -j4
```

### 2. Run All Studies (Unified Script)
```bash
# Full run (all 560 instances, ~45 min total)
bash hpc/run_all_ablation_studies.sh --section 2 --studies all

# Or submit as SLURM job
sbatch hpc/slurm_ablation_studies.sbatch
```

### 3. Run Individual Studies
```bash
# Study 1: Packing Hierarchy (~15 min)
bash hpc/run_component_ablation.sh --section 2 --config step1_only_default

# Study 2: Relaxation Quality (~5 min)
bash hpc/run_relaxation_quality.sh --section 2

# Study 3: G Sweep (~15 min)
bash hpc/run_g_sweep.sh --section 2

# Study 4: SPACES Ablation (~10 min)
bash hpc/run_spaces_ablation.sh --section 2

# Study 5: Pipeline Cascade - uses existing data
cat hpc/studies/full_blockdp_only.csv

# Revised journal-facing studies
bash hpc/run_max_gap_study.sh
bash hpc/run_certification_study.sh
bash hpc/run_backup_necessity_study.sh
bash hpc/run_structure_hardness_study.sh
```

---

## Unified Script Options

```bash
bash hpc/run_all_ablation_studies.sh [OPTIONS]

Options:
  --section SECTION    Dataset section: 2 (default), 1, all, table1, table2, fig9
  --max-instances N    Limit instances (0 = all, default)
  --time-limit SEC     Per-instance time limit (default: 600)
  --studies LIST       Comma-separated: 1,2,3,4 or "all" (default: all)
  --parallel           Run studies 2-4 in background (faster on multi-core)
  --output-dir DIR     Base output directory
  --dry-run            Print commands without executing
```

### Examples
```bash
# Quick test with 10 instances
bash hpc/run_all_ablation_studies.sh --max-instances 10 --dry-run

# Run only studies 2 and 3
bash hpc/run_all_ablation_studies.sh --studies 2,3

# Run studies 2-4 in parallel
bash hpc/run_all_ablation_studies.sh --studies 2,3,4 --parallel

# Custom output directory
bash hpc/run_all_ablation_studies.sh --output-dir results/ablation_$(date +%Y%m%d)
```

---

## SLURM Submission

### Default submission
```bash
sbatch hpc/slurm_ablation_studies.sbatch
```

### With resource overrides
```bash
sbatch --time=02:00:00 --mem=16G hpc/slurm_ablation_studies.sbatch
```

### Check job status
```bash
squeue -u $USER
```

### Cancel job
```bash
scancel <job_id>
```

---

## Output Files

After running, results are organized as:

```
hpc/results_studies/
├── study1_packing_hierarchy/
│   └── step1_only_default/
│       ├── step1_only_default.csv
│       ├── report.md
│       └── run.log
├── study2_relaxation_quality/
│   ├── combined.csv
│   ├── report.md
│   └── run.log
├── study3_g_sweep/
│   ├── g_auto.csv
│   ├── g_1p25x.csv
│   ├── g_1p50x.csv
│   ├── g_2p00x.csv
│   ├── g_full.csv
│   ├── combined.csv
│   ├── report.md
│   └── run.log
├── study4_spaces_ablation/
│   ├── banded_auto.csv
│   ├── full_spaces.csv
│   ├── combined.csv
│   ├── report.md
│   └── run.log
└── run_YYYYMMDD_HHMMSS/  (timestamped if using SLURM)
    └── ...
```

---

## Study Descriptions

| Study | Question | Key Output |
|-------|----------|------------|
| 1. Packing Hierarchy | What does block DP contribute beyond Step 1 heuristic packing? | Instances solved at Phase 1, max time reduction |
| 2. Relaxation Quality | How tight is semigroup vs unit/GCD? | Gap percentages, strict improvement counts |
| 3. G Sweep | How sensitive is performance to G parameter? | Runtime vs G, mismatches vs full |
| 4. SPACES Storage | Does banded storage lose anything? | Speedup, same optimal values |
| 5. Pipeline Cascade | Are intermediate phases needed? | Step reached counts (all fwd_relax with block DP) |
| Max-gap robustness | Is the sharpened auto-gap safe and faster? | Speedup vs full, mismatch count |
| Certification contribution | Which certification path closes the scalable hard cases? | Phase and submethod counts |
| Backup necessity | When does `R_feas` matter beyond semigroup? | Packability rescue and LB improvements |
| Structure hardness | Which processing-time structures are hardest? | Runtime and stage by signature |

---

## Troubleshooting

### Solver not found
```bash
cmake --build solvers/cpp/build --target stateful_compare -j4
```

### Dataset not found
```bash
bash hpc/setup_benchmark_data.sh
```

### Module load errors (HPC)
Edit the `#SBATCH` section of `slurm_ablation_studies.sbatch` to match your HPC modules:
```bash
module load python/3.10
module load gcc/12.0
module load cmake/3.24
```

### Memory issues
Increase memory in SLURM script:
```bash
#SBATCH --mem=64G
```

Or limit instances:
```bash
bash hpc/run_all_ablation_studies.sh --max-instances 100
```
