# Phase F Configuration Probe Readiness

Date: 2026-04-19

## Requirement checklist

- [x] Read active thread context (`ACTIVE.md`, prior summary, plan file)
- [x] Read required truth sources (`dp_solver.hpp/.cpp`, `parallel_heuristic_compare.cpp`, `parallel_f2_cp_sat.py`, paper text excerpt, paper results dirs)
- [x] Create new iteration folder and initialize memory files
- [x] Write explicit design note before coding
- [x] Implement bounded regime-1 probe pipeline end-to-end
- [x] Run primary target `46/77`
- [x] Compare TEC against exact, paper EHS, and one-shot baselines
- [x] Save machine-readable artifacts in `temp/`
- [x] Update ACTIVE/LOG and iteration summaries

## Primary success criteria status

1. Master implemented correctly: **yes** (class machine-count equalities + type coverage equalities + integer class/config variables)
2. Model solves `46/77`: **yes** (`OPTIMAL`)
3. TEC matches exact optimum: **yes** (`103`)

## Scope compliance

- Stayed in regime 1 and started at `46/77`.
- Only after success, tested two additional epsilon points on instance 46 (`73`, `120`).
- Did not launch 61/90 or full-front campaign.
- Did not rewrite DP core.
