# PLAN 12 — Deep Cleanup and Method-Provenance Registry

## Objective

Do a deeper organizational cleanup of the
`research/k_vs_arithmetic_axes_20260412` thread so that:

1. current paper-facing results are easy to find,
2. stale or superseded material is clearly separated,
3. accepted solver results are stored in a structured way,
4. each solved instance clearly points to the solver path and code entrypoint
   that solved it,
5. and future readers do not need to reconstruct method provenance from scratch.

This is not a solver-improvement task.

It is a:

- cleanup,
- current-vs-archive separation,
- provenance,
- and paper-organization task.

## Why this is needed

The thread now contains:

- accepted K=4 recovery and generator results,
- accepted paper-group frontier results,
- older contradictory or superseded notes,
- and a mix of current and historical narrative inside large docs.

That makes it too easy to lose track of:

- which artifact is the source of truth,
- which method actually solved a given row,
- and which code path should be cited in a paper or supervisor note.

The specific failure to prevent is:

> redoing old work because the result exists but the exact solving method is not
> recorded clearly enough.

## Main principle

Do not delete accepted results.

Do not flatten chronology into one giant rewritten story.

Instead:

1. preserve accepted artifacts,
2. preserve historical evidence,
3. create a clean current-facing layer,
4. and add a method-provenance registry that makes solver attribution explicit.

## Required outputs

At the end of this task, the thread should have three clear layers.

### A. Current-facing layer

A small set of files that answers:

- what is currently solved,
- what is the current accepted method per regime,
- what are the current paper-group frontiers,
- what are the current blockers.

### B. Provenance layer

A structured table that answers for each accepted row:

- which solver path solved it,
- which deciding step solved it,
- which runtime artifact is the source of truth,
- which environment/toggles mattered,
- and which code function(s) should be cited as the concrete implementation path.

### C. Historical archive layer

Older detailed logs and superseded diagnostics remain available, but they are
clearly framed as archive/history, not the primary current narrative.

## Required execution order

### Phase A — Inventory and classify the current thread surface

Inventory the thread and classify files into:

1. current truth source,
2. current-facing summary,
3. accepted milestone evidence,
4. historical archive,
5. stale or superseded narrative,
6. temporary scratch.

Do this for:

- docs in the thread root,
- `csv/plan05`,
- `csv/plan10`,
- `csv/plan11`,
- cleanup/archive folders,
- implementation plans if they are still actively referenced.

### Phase B — Create a current-facing results index

Create a small current-facing document, for example:

- `research/k_vs_arithmetic_axes_20260412/CURRENT_RESULTS_INDEX.md`

This file should be the first place a future reader checks.

It must clearly state:

1. current accepted K=4 package,
2. current paper-group frontiers,
3. current source-of-truth artifacts,
4. current unresolved blockers,
5. where to find detailed provenance.

This file should be short, stable, and current-facing.

### Phase C — Create a structured method-provenance registry

Create a structured provenance artifact, for example:

- `research/k_vs_arithmetic_axes_20260412/csv/CURRENT_METHOD_PROVENANCE.csv`
- and optionally a matching markdown note:
  - `research/k_vs_arithmetic_axes_20260412/METHOD_PROVENANCE.md`

This is the most important part of the task.

For each accepted or benchmark-significant solved row, record:

- logical row id / case id
- family / family_id
- `K`
- `n`
- `lambda`
- seed
- solved / exact / finite-gap / unresolved
- deciding step
- pack method string from artifact
- selector policy / selector decision if relevant
- accepted solver package label
- source artifact path
- source row identity inside that artifact if needed
- code file path
- function name(s)
- short solver-path description
- important env toggles if non-default
- whether this row is:
  - current accepted benchmark evidence,
  - historical continuity evidence,
  - or archive-only evidence

### Phase D — Use real code entrypoints, not vague names

Where possible, the provenance registry should point to concrete code entrypoints
such as:

- `compute_relaxed_completion_table(...)`
- `generate_energy_core_patterns(...)`
- `block_repair_energy_core_ub(...)`
- `block_repair_feasible_beam_ub(...)`
- `block_repair_profile_repair_beam_ub(...)`
- `profile_realization_dp_exact` candidate path
- `step1_exact_guided` workflow entry in `stateful_compare.cpp`

Do not rely only on labels like:

- "energy core"
- "beam"
- "exact"

Those are too lossy.

The registry must make it possible to answer:

> which concrete function path solved this row?

### Phase E — Separate current docs from historical docs

Without destroying chronology, make the thread easier to read by separating:

- current-facing summaries,
- from large chronological or mixed archive notes.

Allowed approaches:

- add a short current-facing index and link older material as archive,
- add explicit “current” vs “historical” headings,
- move clearly archival notes under an archive subfolder if references are fixed.

Do not do a destructive rewrite of all history.

### Phase F — Make the structure paper-friendly

Organize current result surfaces so that paper writing becomes easier.

At minimum, the current-facing layer should support:

1. paper-family frontier table,
2. accepted K=4 method statement,
3. frontier/blocker summary,
4. method provenance lookup.

If helpful, create a dedicated paper-facing note, for example:

- `research/k_vs_arithmetic_axes_20260412/PAPER_RESULTS_READY.md`

This note should summarize the current benchmark story without requiring a
reader to parse the full historical log.

## Constraints

- No solver algorithm changes.
- No new experiments.
- No silent deletion of accepted artifacts.
- No loss of historical evidence that still matters.
- Be conservative with moves; if a file may matter, archive it instead of
  deleting it.

## Required artifacts to preserve

These must remain intact and easy to locate:

- plan05 source-of-truth CSVs and summary,
- plan10 accepted K=4 artifacts,
- plan11 frontier artifacts,
- current docs (`LOG.md`, `RESULTS.md`, `BLOCKERS.md`,
  `METHOD_BOUNDARIES.md`, `ENERGY_CORE_FORTIFICATION_NOTE.md`),
- implementation plans.

## Suggested deliverables

Create:

- `research/k_vs_arithmetic_axes_20260412/CURRENT_RESULTS_INDEX.md`
- `research/k_vs_arithmetic_axes_20260412/csv/CURRENT_METHOD_PROVENANCE.csv`
- `research/k_vs_arithmetic_axes_20260412/METHOD_PROVENANCE.md`
- optionally `research/k_vs_arithmetic_axes_20260412/PAPER_RESULTS_READY.md`
- and a cleanup/provenance note such as:
  - `research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/cleanup/CLEANUP_AND_PROVENANCE_20260420.md`

## Success criteria

This task succeeds only if:

1. the thread has a clean current-facing entrypoint,
2. current paper-facing results are easy to find,
3. accepted rows have explicit method provenance,
4. provenance points to concrete code entrypoints or function names,
5. current vs archive material is clearer than before,
6. and future work will not need to rediscover which method solved an instance.
