# Benchmark Extensions For The Paper

This directory contains the formal benchmark-extension harness for the paper.

The goal is to keep the benchmark definition explicit and reusable for:

- our solver now,
- the paper's solver later,
- and HPC runs without ad hoc command reconstruction.

## Suites

Three formal suites are defined.

### `scalability_large_n`

Purpose:

- large-`n` extension of the paper-hard `{8,10}` family,
- used to show scalability of the default production method.

Policy for our solver:

- run the default production path only.

### `backup_realistic`

Purpose:

- small realistic bounded-count suite where semigroup is sometimes enough and
  sometimes repaired by `R_feas`.

Policy for our solver:

- report only semigroup and `R_feas`,
- do not run the other backup variants in the paper-facing harness.

### `k_boundary`

Purpose:

- increasing-`K` benchmark to probe whether many distinct job sizes create a
  real practical boundary.

Policy for our solver:

- run the default production path,
- and record only semigroup and `R_feas` from the hierarchy side.

## Setup

Build the formal suites:

```bash
bash hpc/benchmark_extensions/00_setup_extension_benchmarks.sh
```

This creates the named datasets under:

```text
data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling/data/datasets/
```

with names:

- `paperext_scalability_large_n_202604`
- `paperext_backup_realistic_202604`
- `paperext_k_boundary_202604`

## Run Our Solver

After pulling new code on the HPC, rebuild `stateful_compare` before running the
extension suites:

```bash
bash hpc/01_build_our_solver.sh
```

If the runner reports that modes such as `relax-pack-stdin` or
`relax-hierarchy-stdin` are missing, the built solver is older than the Python
wrapper and needs to be rebuilt.

```bash
bash hpc/benchmark_extensions/01_run_ours_scalability.sh
bash hpc/benchmark_extensions/02_run_ours_backup.sh
bash hpc/benchmark_extensions/03_run_ours_k_boundary.sh
```

Outputs go by default to:

```text
hpc/results_studies/benchmark_extensions/
```

## Run The Paper Solver

Later, the same suites can be run on the paper solver:

```bash
bash hpc/benchmark_extensions/04_run_paper_scalability.sh
bash hpc/benchmark_extensions/05_run_paper_backup.sh
bash hpc/benchmark_extensions/06_run_paper_k_boundary.sh
```

These wrappers call:

- [run_paper_extension_suite.py](/Users/mac/Documents/Study/PFE/PaST/hpc/benchmark_extensions/run_paper_extension_suite.py)

which uses the paper's `Experiments` runner on the formal dataset directory.

## Notes

- The paper-facing harness intentionally excludes the extra backup variants
  (`R_lagr`, partial variants, adaptive variants).
- Those remain available in the exploratory scripts, but they are not part of
  the formal benchmark package described here.
- The backup suite is intentionally small and explanatory; it is not meant to
  replace the larger exploratory stress benchmark.
- The default wrappers use single-instance batches for the scalability and
  `K`-boundary suites so one hard row does not block an entire subprocess.
- If an instance exceeds the external subprocess cap, the runner records an
  `external_timeout` row and continues instead of aborting the whole suite.
