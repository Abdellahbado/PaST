# Stateful DP HPC Rerun README

This file is the practical entry point for rerunning the stateful-DP single-machine TOU paper experiments on HPC.

Canonical method/result map:

- `research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md`

Main wrapper:

- `hpc/run_stateful_dp_paper_experiments.sh`

Main solver binary:

- `solvers/cpp/build/stateful_compare`

Build:

```bash
bash hpc/run_stateful_dp_paper_experiments.sh build
```

## Clone Only This Branch On HPC

```bash
git clone --branch codex/stateful-dp-hpc-repro-20260518 --single-branch https://github.com/Abdellahbado/PaST.git PaST-stateful-dp-hpc
cd PaST-stateful-dp-hpc
```

If the repository is already cloned:

```bash
git fetch origin codex/stateful-dp-hpc-repro-20260518
git checkout codex/stateful-dp-hpc-repro-20260518
git pull --ff-only
```

## Recommended HPC Execution

Do not run everything in one interactive session unless the wall-time limit is very large. Prefer one Slurm job per mode:

```bash
bash hpc/run_stateful_dp_paper_experiments.sh original-benchmark
bash hpc/run_stateful_dp_paper_experiments.sh paper-groups
bash hpc/run_stateful_dp_paper_experiments.sh k4-generator
bash hpc/run_stateful_dp_paper_experiments.sh k2-routing
bash hpc/run_stateful_dp_paper_experiments.sh easy-k
bash hpc/run_stateful_dp_paper_experiments.sh hard-k
bash hpc/run_stateful_dp_paper_experiments.sh hard-k-cert
```

One-shot sequential run:

```bash
bash hpc/run_stateful_dp_paper_experiments.sh main
```

## What Each Mode Produces

| Mode | Purpose | Main outputs |
|---|---|---|
| `original-benchmark` | Corrected original benchmark / ablation rerun | `hpc/results_studies/...` |
| `paper-groups` | Large-`n` paper job-group extension | `research/k_vs_arithmetic_axes_20260412/csv/plan05/` |
| `k4-generator` | K=4 energy-core generator validation | `research/k_vs_arithmetic_axes_20260412/csv/plan10/` |
| `k2-routing` | Corrected K=2 routing / g37 evidence | `research/k_vs_arithmetic_axes_20260412/csv/plan13/` |
| `easy-k` | Easy contiguous-unit K scaling | `research/k_vs_arithmetic_axes_20260412/csv/plan30/` |
| `hard-k` | Hard irregular K-axis boundary | `research/k_vs_arithmetic_axes_20260412/csv/plan18/` |
| `hard-k-cert` | PLAN33 certified anytime K10/K12 panel | `research/k_vs_arithmetic_axes_20260412/csv/plan33/` |

## Paper-Use Rules

- Use HPC-generated CSVs for final runtime tables.
- Local CSVs are provenance/design evidence only unless explicitly rerun on HPC.
- Do not cite PLAN32B parallel UB as valid. It solved the wrong model.
- Do not cite old misrouted `g37` rows as method failures. Use corrected K=2 routing evidence.
- Treat profile-repair beam and survivor-policy variants as heuristic/profile-realization components unless exact profile realization proves closure.
- PLAN33 hard-K rows provide certified finite gaps, not exact closure.

## Responsible Code

Core solver:

- `solvers/cpp/stateful_compare.cpp`
- `solvers/cpp/stateful_dp_solver.cpp`
- `solvers/cpp/stateful_dp_solver.hpp`

Benchmark regeneration / original benchmark:

- `scripts/regenerate_instances.py`
- `hpc/setup_benchmark_data.sh`
- `hpc/run_revised_ablation_studies.sh`
- `hpc/studies/component_ablation.py`
- `hpc/studies/spaces_ablation.py`

Stateful-DP paper reruns:

- `research/k_vs_arithmetic_axes_20260412/run_plan05_paper_groups_extension.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan10_k4_generator_compare.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan13_two_track_recovery.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan18_k_boundary_refine_n1000.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan28_easy_k_scaling.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan28_k40_optional.py`
- `research/k_vs_arithmetic_axes_20260412/run_plan33_cert_anytime.py`

## Minimal Sanity Checks After HPC Runs

```bash
python3 - <<'PY'
from pathlib import Path
for p in [
    'research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv',
    'research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_generator_compare.csv',
    'research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_g37_k2_reroute.csv',
    'research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_n1000_raw.csv',
    'research/k_vs_arithmetic_axes_20260412/csv/plan30/PLAN30_easy_k_scaling_raw.csv',
    'research/k_vs_arithmetic_axes_20260412/csv/plan33/PLAN33_cert_anytime_raw.csv',
]:
    path = Path(p)
    print(f'{p}: exists={path.exists()} size={path.stat().st_size if path.exists() else 0}')
PY
```

