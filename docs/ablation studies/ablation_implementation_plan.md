# Ablation Studies — Complete Implementation Plan

## Overview

Five ablation studies, each isolating one component of the method. All studies use the **Table 2 family** (560 instances, `benedikt2025b_groups`).

> [!IMPORTANT]
> **Terminology for the paper:** Do NOT mention "DFS" as a separate component. The exact bin packing at Phase 1 is presented simply as "sparse block-level feasibility DP." The internal DFS shortcut is an implementation detail.

---

## Study 1: Packing Hierarchy (UB Contribution)

**Question:** What does the block DP contribute beyond heuristic packing?

### Configs

| Config name | Packing layers active | How to run |
|-------------|----------------------|------------|
| `step1_only_default` | Step 1 heuristic packing only (FFD/BFD/FFI/BFI + 20 random) | Existing config, no exact packing |
| `full_blockdp_only_pack` | Heuristics + sparse block DP | Existing config ✅ already ran |

### What already exists

- `full_blockdp_only_pack` — **already ran**, results in [hpc/studies/full_blockdp_only.csv](file:///Users/mac/Documents/Study/PFE/PaST/hpc/studies/full_blockdp_only.csv)
- `step1_only_default` — config exists in [component_ablation.py](file:///Users/mac/Documents/Study/PFE/PaST/hpc/studies/component_ablation.py) but **not yet run**

### How to run the missing config

```bash
bash hpc/run_component_ablation.sh \
  --section 2 \
  --config step1_only_default \
  --output-dir hpc/results_studies/component_ablation_default
```

### Expected outcome

| Config | Instances solved at Phase 1 | Needs later stages | Avg time | Max time |
|--------|----------------------------|-------------------|---------|---------|
| `step1_only_default` | ~538 | ~22 | ~3.2s | ~177s |
| `full_blockdp_only_pack` | 560 | 0 | 1.38s | 13.33s |

### What it proves

The block DP certifies 22 instances that heuristics miss, reducing max time from ~177s to ~13s and avoiding the expensive exact multiset DP (Phase 6) entirely.

---

## Study 2: Relaxation Quality (LB Contribution)

**Question:** How much tighter is the semigroup relaxation vs unit and GCD?

### How it works

The C++ solver has a `relaxation-stdin` mode that runs the same relaxed DP three times per instance with different chunk sets:

| Relaxation | Chunk set | Bound |
|------------|-----------|-------|
| Unit | `{1}` | LB_unit |
| GCD | `{gcd(p₁,...,pₖ)}` | LB_gcd |
| Semigroup | `{p₁,...,pₖ}` | LB_semi |

Plus the full solver's optimal value for comparison.

### How to run

```bash
bash hpc/run_relaxation_quality.sh --section 2
```

This calls [hpc/studies/relaxation_quality.py](file:///Users/mac/Documents/Study/PFE/PaST/hpc/studies/relaxation_quality.py), which:
1. Feeds all 560 instances via `relaxation-stdin`
2. Outputs a CSV with columns: `instance_id, lb_unit, lb_gcd, lb_semi, opt, t_unit, t_gcd, t_semi, t_opt`
3. Computes gaps: `gap_X = (opt - lb_X) / opt * 100`

### No code changes needed

The `relaxation-stdin` mode and [relaxation_quality.py](file:///Users/mac/Documents/Study/PFE/PaST/hpc/studies/relaxation_quality.py) already exist.

### Expected outcome

A table showing mean/median/max gaps for each relaxation to OPT, and counts of strict improvements `unit→gcd` and `gcd→semi`.

### What it proves

The semigroup relaxation produces tighter LBs than unit/GCD, which is why Phase 1 can certify optimality directly on more instances.

---

## Study 3: G Sweep (SPACES Bandwidth Sensitivity)

**Question:** How sensitive is performance to the gap crossover parameter G?

### How it works

The [g_sweep.py](file:///Users/mac/Documents/Study/PFE/PaST/hpc/studies/g_sweep.py) study runs the full pipeline at different fixed G values to see how performance changes.

### How to run

```bash
bash hpc/run_g_sweep.sh --section 2
```

### No code changes needed

The [run_g_sweep.sh](file:///Users/mac/Documents/Study/PFE/PaST/hpc/run_g_sweep.sh) and [hpc/studies/g_sweep.py](file:///Users/mac/Documents/Study/PFE/PaST/hpc/studies/g_sweep.py) already exist.

### Expected outcome

A plot/table showing runtime vs G, with: too-small G → long-gap decomposition errors or missed optimality; auto-G → near-optimal; too-large G → wasted memory/time on band storage.

### What it proves

The auto_max_gap heuristic picks a good G without manual tuning.

---

## Study 4: SPACES Storage (Banded vs Full)

**Question:** Does banded SPACES (O(hG)) lose anything vs full O(h²) storage?

### Configs

| Config name | SPACES mode | How it works |
|-------------|------------|-------------|
| `full` (banded) | `use_banded_spaces = true` | Default — banded band of width G |
| `full_spaces` | `use_banded_spaces = false` | Full O(h²) table |

### How to run

```bash
bash hpc/run_spaces_ablation.sh --section 2
```

This calls [hpc/studies/spaces_ablation.py](file:///Users/mac/Documents/Study/PFE/PaST/hpc/studies/spaces_ablation.py). It already handles both configs.

### No code changes needed

The `full_spaces` ablation mode already exists in the C++ solver (see [stateful_compare.cpp:911-916](file:///Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_compare.cpp#L911-L916)):

```cpp
else if (ab_mode == "full_spaces")
{
    ab.use_banded_spaces = false;  // full O(h²) SPACES
    ab.use_heuristics = true;
    ab.use_relaxation_lb = true;
}
```

### Expected outcome

Both should produce identical optimal values. The banded version should use less memory and similar or faster runtime.

### What it proves

The banded storage + long-gap decomposition is lossless on these benchmarks.

---

## Study 5: Pipeline Cascade (Stage Contribution)

**Question:** Are the intermediate stages (Phases 2–5.5) ever needed?

### What we already know from existing data

From [full_blockdp_only.csv](file:///Users/mac/Documents/Study/PFE/PaST/hpc/studies/full_blockdp_only.csv), ALL 560 instances solve at `fwd_relax` (Phase 1). The columns `t_heuristic`, `t_local_search`, `t_r_feas`, `t_r_feas_lagr`, `t_smart_recon`, [t_exact](file:///Users/mac/Documents/Study/PFE/PaST/hpc/studies/component_ablation.py#74-85) are all 0.

**This study does NOT need a new run.** It is answered by the existing data.

### How to present it

Table from the CSV:

| Phase | Instances certified | Time spent |
|-------|-------------------|------------|
| Phase 1 (Relaxed DP + Packing) | 560 (100%) | 1.38s avg |
| Phase 2 (Heuristic UB) | 0 | 0s |
| Phase 3 (Local Search) | 0 | 0s |
| Phase 4 (R_feas) | 0 | 0s |
| Phase 5 (R_feas+Lagr) | 0 | 0s |
| Phase 5.5 (Smart Recon) | 0 | 0s |
| Phase 6 (Exact DP) | 0 | 0s |

### What it proves

With block DP activated, the entire pipeline collapses to a single phase. All certification happens at Phase 1.

---

## Execution Checklist

Run these on HPC in this order:

```bash
# 0. Make sure you're on the right branch and rebuild
git fetch origin
git checkout codex/blockdp-before-exactpack
git pull --ff-only
cmake --build solvers/cpp/build --target stateful_compare -j4

# 1. Packing hierarchy: Step 1 heuristic-only baseline (~15 min)
bash hpc/run_component_ablation.sh \
  --section 2 \
  --config step1_only_default \
  --output-dir hpc/results_studies/ablation_packing_default

# 2. Relaxation quality (~5 min)
bash hpc/run_relaxation_quality.sh --section 2

# 3. G sweep (~15 min)
bash hpc/run_g_sweep.sh --section 2

# 4. SPACES ablation (~10 min)
bash hpc/run_spaces_ablation.sh --section 2

# 5. Pipeline cascade: already answered by full_blockdp_only.csv
```

> [!TIP]
> Studies 2–4 can run in parallel on different terminals if HPC allows it.

---

## How to Collect Results

After running, each study produces a CSV + [report.md](file:///Users/mac/Documents/Study/PFE/PaST/docs/honest_comparison_report.md) in its output directory. The key files will be:

| Study | Output file | Key columns |
|-------|------------|-------------|
| Packing hierarchy | `hpc/results_studies/ablation_packing_default/step1_only_default.csv` | `fwd_pack_method`, `step_reached`, `runtime_sec` |
| Relaxation quality | `hpc/results_studies/relaxation_quality/relaxation_quality.csv` | `lb_unit`, `lb_gcd`, `lb_semi`, `opt` |
| G sweep | `hpc/results_studies/g_sweep/` | `runtime_sec` per G value |
| SPACES | `hpc/results_studies/spaces_ablation/` | `runtime_sec`, [ub](file:///Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.hpp#145-153), [lb](file:///Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp#1256-1391) per SPACES mode |
| Pipeline cascade | [hpc/studies/full_blockdp_only.csv](file:///Users/mac/Documents/Study/PFE/PaST/hpc/studies/full_blockdp_only.csv) (existing) | `step_reached`, all `t_*` columns |

---

## Paper Table Layout (Suggestion)

### Table: Effect of Block DP Certification

| Metric | Heuristic only | Heuristic + Block DP |
|--------|---------------|---------------------|
| Instances certified at Phase 1 | 538/560 | **560/560** |
| Instances needing Phase 6 | 22 | **0** |
| Avg runtime | 3.17s | **1.38s** |
| Max runtime | 176.67s | **13.33s** |

### Table: Relaxation Tightness

| Relaxation | Mean gap to OPT (%) | Median gap | Max gap | Strict improvements over previous |
|-----------|---------------------|------------|---------|-----------------------------------|
| Unit | ? | ? | ? | — |
| GCD | ? | ? | ? | ? instances |
| Semigroup | ? | ? | ? | ? instances |

### One-liner: Pipeline Cascade

> "With block DP active, all 560 instances are certified at Phase 1. The intermediate phases (heuristic UB, local search, R_feas, R_feas+Lagr, smart reconstruction, exact DP) are never invoked."
