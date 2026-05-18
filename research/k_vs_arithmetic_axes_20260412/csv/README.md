# CSV Layout

This folder groups experiment CSV artifacts by workstream instead of keeping a
flat research root.

## Subfolders

- [baseline](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/baseline)
  Early baseline grid slices.

- [two_axis_grid](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid)
  Main two-axis experiment outputs and validation tables.

- [plan03b](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/plan03b)
  Step-3 / Step-4 diagnostics cleanup and small beam-weight sweeps.

- [plan03c](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/plan03c)
  Step-3 family unification experiments (exact mode vs beam mode behavior).

- [plan03d](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/plan03d)
  Exact-vs-beam selector experiments, calibration scans, and fallback probes.

- [plan04c](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/plan04c)
  Exact-DP matrix experiments and incumbent-quality studies.

- [plan05](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/plan05)
  Paper-groups extension study outputs.

- `plan10`
  K=4 energy-core generator policy and runtime comparison.

- `plan13`
  `{1,...,10}` / `g37` recovery diagnostics, including corrected K=2 reroute.

- `plan14`
  Dense-unit Step-2 fastpath artifacts for `{1,...,10}`.

- `plan16`, `plan17`, `plan18`
  Fixed-`n=1000` K-axis scaling and hard-irregular boundary studies.

- `plan30`
  Final easy-vs-hard K-scaling story, including easy `{1,...,K}` exact through
  `K=40`.

- `plan31`
  Family-aware Step-3 survivor diagnostics.

- `plan32c`, `plan33`
  Hard K12 validity recovery and certified anytime hard-K prepass. PLAN33 is
  the current recommended hard K10/K12 paper-facing source.

## Paper/HPC Reproducibility

Use
`/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PAPER_HPC_REPRODUCIBILITY_MAP.md`
to map result claims to the responsible runner scripts, solver functions, and
environment toggles before rerunning on HPC.

## Naming rule

The original CSV filenames were preserved so that earlier notes and logs remain
easy to map to the generated artifacts. Only the directory structure changed.
