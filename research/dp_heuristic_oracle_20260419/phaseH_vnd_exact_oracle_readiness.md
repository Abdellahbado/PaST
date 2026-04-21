# Phase H VND-Exact-Oracle Readiness

Date: 2026-04-20

## Requirement checklist

- [x] Read required thread state and prior phase docs
- [x] Read required method sources (`VND`, `Exact and heuristic`)
- [x] Create new Phase H iteration and switch `ACTIVE.md`
- [x] Write short Phase H design note before final reporting
- [x] Implement bounded VND-inspired prototype with required neighborhoods
- [x] Use bounded screening + exact-DP-on-touched-machines rule
- [x] Use deterministic bounded run controls and caching
- [x] Run main target only (`61/347`)
- [x] Save machine-readable artifacts in `temp/phaseH_vnd_exact_oracle/`
- [x] Report required metrics and baseline comparisons

## Scope compliance

- single-point viability test only (`61/347`)
- no full EOA, no epsilon oscillation loop
- no full frontier
- no broad instance sweep
- no decomposition-branch reopening

## Outcome status

- quality signal: **positive** vs current same-epsilon baselines (`6944` vs `7053` relocate-only)
- mechanism signal: **weak** in this run (no accepted VND move)

## Decision posture

- conditionally continue with one more bounded Phase H step focused on producing move-level acceptance signal at the same point.
- if next bounded pass still shows no accepted improving moves, stop Phase H despite current quality gain.
