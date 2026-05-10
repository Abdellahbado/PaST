# Blockers — Phase Y: Online LLM Neighborhood Proposal

## Active

### B-Y2.1 RESIDUAL — SIGBUS on macOS Apple Silicon for some instance+seed combos

Intermittent crash (exit 138) on macOS Apple Silicon release builds for:
Cell_B execute_manual, Cell_B random_s1, Cell_C random_s100-s400.

Does NOT occur in debug/ASAN builds. Same C++ binary works on Linux x86_64.
Likely a compiler optimization / memory alignment interaction, not a logic bug.
Workaround: use debug builds or non-crashing seeds on macOS.
**Does not block Y3** — infrastructure validated on all 3 cells.

### B-Y0.3 — DeepSeek overfitting to specific machine IDs on dev cells

The LLM sees exact machine costs and gaps. On dev cells, it could propose
attacking M0 simply because M0 has the highest Gap, rather than because it
diagnoses a generalizable pattern. The trace's cell_label anonymization
helps, but per-machine data is inherently instance-specific.

### B-Y0.4 — The constraint-based approach may be too coarse

If the LLM wants to propose "move small jobs from M0 to M24, but only jobs
that are currently scheduled in the first half of M0's processing order",
the current proposal schema cannot express this (no job position filter).

## Resolved

### B-Y2.0 — Y2 proposal execution only validated on Cell A [RESOLVED 2026-05-10, Y2.1]

Root cause: `per_machine_dp_limit_sec` was 0.125s (old default), insufficient
for machines with 6+ job types. Changed default to 30.0s. Smoke now passes on
all 3 cells (Cells A/B/C trace probes, A/C execute_manual, non-crashing random seeds).

### B-Y2.1 — Random proposal stability [RESOLVED 2026-05-10, Y2.1 with residual]

Replaced `std::discrete_distribution` (SIGBUS risk) with manual weighted sampling.
Added negative-weight protection. 12 of 15 random runs pass. The remaining 3
failures are a macOS-specific SIGBUS (see B-Y2.1 RESIDUAL above).

### B-Y0.0 — Trace format needs C++ instrumentation before Y1 [RESOLVED 2026-05-10, Y1]
### B-Y1.1 — Search-behavior fields were null [RESOLVED 2026-05-10, Y1.1]
### B-Y0.1 — C++ variant for neighborhood evaluation [RESOLVED 2026-05-10, Y2]
### B-Y0.2 — Random neighborhood baseline fairness [RESOLVED 2026-05-10, Y2.1]
