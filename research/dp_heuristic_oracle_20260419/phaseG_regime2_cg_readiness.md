# Phase G Regime-2 CG Readiness

Date: 2026-04-20

## Requirement checklist

- [x] Read required thread and Phase F documents
- [x] Read required code truth sources
- [x] Read required paper/result truth sources
- [x] Create new iteration for Phase G and update active branch
- [x] Write reduced-cost validation design note first
- [x] Implement one restricted regime-2 probe for `61/347`
- [x] Save machine-readable artifacts in `temp/phaseG_regime2_cg/`
- [x] Report required metrics and comparisons
- [x] Fix duplicate-stop logic so loop searches best non-duplicate improving column before stopping
- [x] Run one corrected bounded rerun on `61/347`
- [x] Record follow-up design note

## Scope compliance

- single-instance regime-2 probe only (`61/347`)
- no full frontier
- no broad instance sweep
- no branch-and-price claim
- explicit note that reduced-cost mapping is only approximately complete in this bounded probe

## Outcome status

- corrected probe produced feasible restricted-master solution (`TEC=7040`, optimal in restricted master)
- corrected loop added genuinely new columns (`259 -> 271`) and no longer stopped on duplicate
- LP bound and TEC stayed unchanged from pre-fix run
- result still improves over one-shot heuristic baselines but remains far from EHS/reference

## Continuation decision

- **Not ready to continue as-is**: duplicate-stop bug was fixed, but corrected rerun did not yield material quality gain.
- Branch should be stopped in its current bounded form unless we commit to a new bounded mechanism specifically targeting LP-quality movement (not just more columns).
