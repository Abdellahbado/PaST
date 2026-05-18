# PLAN 14 — Dense-Unit Large-K Step-2 Scaling Diagnosis

This plan targets the large dense unit-containing paper group:

- `g12345678910 = {1,2,3,4,5,6,7,8,9,10}`

and prepares the method for the supervisor-requested direction:

- scaling dense unit-containing groups toward `{1,...,20}`.

This is not a broad Step-3 redesign. The evidence says `{1..10}` closes by
Step 2 when it completes. The immediate problem is that, at `n=5000`, the
generic pipeline times out or is memory-killed before it returns a usable
incumbent / closure row.

## Current Evidence

From the accepted paper-group ledger:

| family | n | status | runtime | deciding step | method |
|---|---:|---|---:|---|---|
| `{1..10}` | 1500 | exact | `352.2711s` | Step 2 | `ffd` |
| `{1..10}` | 2500 | exact | `623.8650s` | Step 2 | `ffd` |
| `{1..10}` | 3500 | exact | `1002.4226s` | Step 2 | `ffd` |
| `{1..10}` | 5000 | timeout | `~900-1020s` | none | no result emitted |
| `{1..10}` | 6000/7000 | timeout/kill | `~1378-1921s` | none | no result emitted |

From PLAN13 memory-safe diagnostics:

- baseline energy-core at `n=5000`, seeds `0/1`: external timeout,
- additive reroute probes: memory-limit kill near strict cap,
- no feasible incumbent, no `UB/LB`, no finite gap emitted.

Interpretation:

- `{1..10}` is not currently a finite-gap failure.
- It is a runtime / memory / no-checkpoint-output failure.
- The rows that finish close at Step 2 by `ffd`, not Step 3.

## Working Hypothesis

`{1..10}` is an easy arithmetic family trapped in an expensive general-purpose
pipeline.

The likely problem is before or around Step 2:

1. Step-1 / profile recovery may be too expensive at large `K`.
2. Step-2 realization may be expanding jobs or block structures too literally.
3. The pipeline may allocate Step-3/selector metadata even though Step 2 is the
   correct closure mechanism.
4. External timeout / memory kill prevents partial incumbent output, so the
   archive records `ub=-1`, `lb=-1`, `gap=nan`.

## Objective

Diagnose exactly where `{1..10} n=5000` spends time and memory, then implement
the smallest additive fix that gets `{1..10}` to at least:

- `n=5000`, and ideally
- `n=6000`,

without changing the accepted baseline silently.

The long-term design target is a dense-unit fast path suitable for testing
`{1,...,20}`.

## Hard Constraints

- Do not rewrite the accepted baseline in place.
- Do not change the accepted K=4 package.
- Do not start column generation.
- Do not tune energy-core first for this task.
- Do not use broad Step-3 exact/beam redesign as the first response.
- Any new behavior must be an explicit experiment or toggle.
- Every result must record runtime, memory, step reached, and whether an
  incumbent was emitted.

## Phase A — Instrumented Diagnosis

Add or enable fine-grained instrumentation for `{1..10}` rows.

Required measurements:

- time in Step 1 / profile recovery,
- time building merged blocks,
- time before first Step-2 candidate,
- time inside Step-2 `ffd`,
- whether Step 2 is reached,
- whether Step 2 produces a candidate UB,
- current LB at that point,
- peak RSS,
- allocation / memory-limit kill stage if applicable.

Run:

- `g12345678910`, `n=3500`, seed `0` as a known exact control,
- `g12345678910`, `n=5000`, seeds `0/1`,
- optionally `n=4000` or `4500` if needed to locate the transition.

Output:

- `csv/plan14/PLAN14_g12345678910_diagnosis.csv`

## Phase B — Checkpoint Incumbent Output

Before attempting deeper optimization, ensure the solver or wrapper can emit a
usable partial row when killed or timed out.

Required output on timeout / memory kill:

- best UB found so far, if any,
- current LB, if known,
- best method label,
- phase/stage reached,
- elapsed phase timings,
- peak RSS,
- whether Step 2 was reached.

This phase may not improve exactness, but it must prevent blind `ub=-1/lb=-1`
records when a useful incumbent exists internally.

Output:

- `csv/plan14/PLAN14_g12345678910_checkpoint_probe.csv`

## Phase C — Dense-Unit Step-2 Fast-Path Experiment

Implement an additive experimental path, for example:

- `PAST_DENSE_UNIT_STEP2_FASTPATH=1`

Trigger conditions:

- job sizes are contiguous,
- minimum processing time is `1`,
- `K >= 8` or another explicit threshold,
- target family is dense unit-containing, e.g. `{1..10}` or future `{1..20}`.

Behavior:

1. recover only the minimum profile/lower-bound information required for Step-2
   closure;
2. run Step-2 `ffd` / direct realization as early as possible;
3. if `UB=LB`, return immediately;
4. avoid constructing Step-3 candidate pools, exact profile DP, beam metadata,
   or energy-core structures unless Step 2 fails.

This experiment should answer:

- can we close `{1..10} n=5000` by exiting earlier on the Step-2 path?
- does it reduce memory?
- does it preserve the known `n=3500` exact row?

Output:

- `csv/plan14/PLAN14_g12345678910_fastpath_compare.csv`

## Phase D — Count-Based FFD / Direct Realization Experiment

If Phase C still spends too much time or memory in Step 2, implement a
count-based realization experiment.

Goal:

- avoid building large vectors of individual jobs when the instance can be
  represented as counts:
  - `count[1], count[2], ..., count[K]`.

Candidate toggle:

- `PAST_COUNT_BASED_FFD=1`

Requirements:

- compare against existing Step-2 `ffd`,
- preserve exactness on `n=3500`,
- test `n=5000`,
- record memory and runtime.

Output:

- can be included in `PLAN14_g12345678910_fastpath_compare.csv`, with variant
  label `exp_count_based_ffd`.

## Phase E — Future `{1..20}` Smoke Test

Only after `{1..10} n=5000` is diagnosed and preferably closed:

- run a small smoke test for `{1,...,20}` at modest `n`, e.g. `n=1000` and
  `n=2000`.

Purpose:

- check that the dense-unit fast path generalizes,
- not to claim final `{1..20}` frontier yet.

Output:

- `csv/plan14/PLAN14_dense_unit_1_20_smoke.csv`

## Required Documentation Updates

Update:

- `LOG.md`
- `RESULTS.md`
- `BLOCKERS.md`
- `METHOD_BOUNDARIES.md`
- `PAPER_RESULTS_READY.md`
- `PRESENTATION_RESULTS_SUMMARY.md`

Document clearly:

- whether `{1..10}` failure was Step-1, Step-2, memory, or no-output timeout,
- whether an incumbent exists on failed rows,
- whether the dense-unit fast path is accepted or remains experimental,
- whether `{1..20}` smoke tests were attempted.

## Success Criteria

This task succeeds if it produces one of these outcomes:

1. `{1..10} n=5000` closes exactly with a clearly labeled additive fast path;
2. or `{1..10} n=5000` still fails, but the failure stage is precisely known
   and the run emits meaningful incumbent/gap diagnostics instead of
   `ub=-1/lb=-1`;
3. and in either case, the path toward `{1..20}` is clarified.

Preferred success:

- `g12345678910 n=5000`, seeds `0/1`, exact at Step 2,
- with lower memory than the PLAN13 probes,
- and with the existing `n=3500` result preserved.

## What Not To Do

- Do not make dense-unit fast path the default without comparison.
- Do not hide experimental behavior inside the baseline.
- Do not optimize energy-core for this family first.
- Do not run broad paper-group campaigns during this task.
- Do not claim `{1..20}` scaling from a single smoke test.

