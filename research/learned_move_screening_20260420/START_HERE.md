# Start Here

This file is the fast entry surface for this research thread.

Use this read order:

1. `ACTIVE.md`
2. the iteration `SUMMARY.md` pointed to by `ACTIVE.md`
3. `OVERVIEW.md`
4. `reference/phaseM_benchmark_role_note.md`

Read `reference/PLAN_supervised_move_ranking.md` only if you need the original staged plan.

Read `history/phase_reports/` only if you need detailed phase-by-phase evidence or older design/results/readiness notes.

## Current thread shape

- `ACTIVE.md`
  - points to the current active iteration
- `OVERVIEW.md`
  - stable thread purpose and inherited facts
- `LITERATURE.md`
  - reusable literature notes
- `LOG.md`
  - compact chronological checkpoints
- `reference/`
  - stable policy / protocol notes worth reusing
- `history/phase_reports/`
  - archived phase-specific markdown that should not clutter the main surface
- `iterations/`
  - iteration-local working memory

## Current protocol facts

- benchmark-derived phases `L1` to `L2.5` are development-only
- synthetic train/val is the clean learning-data path
- benchmark `61-90` is primary test-only later
- benchmark `1-60` is secondary robustness-only later
- current active work is Phase P full synthetic train/val freeze, which produced the training-ready synthetic dataset for the next offline model-training stage

## When to go deeper

- Need active next step:
  - use `ACTIVE.md` and the current iteration `SUMMARY.md`
- Need protocol boundaries:
  - use `reference/phaseM_benchmark_role_note.md`
- Need original staged intent:
  - use `reference/PLAN_supervised_move_ranking.md`
- Need old evidence or forensic detail:
  - use `history/phase_reports/`
