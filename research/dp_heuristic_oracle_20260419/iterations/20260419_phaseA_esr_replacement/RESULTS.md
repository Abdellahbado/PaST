# Results

## Phase A implementation

Added fixed-`epsilon` comparison driver:

- `solvers/cpp/parallel_heuristic_compare.cpp`

Build wiring updated in:

- `solvers/cpp/CMakeLists.txt`

## Tested rows

- `46 / 120`
- `61 / 350`
- `64 / 77`
- `90 / 82` (optional fourth, included)

## Main outcome

With assignment held fixed (LPT-style greedy insertion under `epsilon`):

- `greedy_dp` improved TEC over `greedy_esr` on all tested rows.

TEC totals:

- `46/120`: ESR `118`, DP `112`
- `61/350`: ESR `7263`, DP `7053`
- `64/77`: ESR `30640`, DP `30598`
- `90/82`: ESR `53300`, DP `53294`

## Exact-reference comparison

CP-SAT fixed-`epsilon` references (`solvers/parallel_f2_cp_sat.py`):

- `46/120`: `103`
- `61/350`: `6642`
- `64/77`: `30580`
- `90/82`: `53294`

Exact relative gap `(TEC_variant - TEC_exact) / TEC_exact`:

- `46/120`: ESR `0.145631`, DP `0.087379`
- `61/350`: ESR `0.093496`, DP `0.061879`
- `64/77`: ESR `0.001962`, DP `0.000589`
- `90/82`: ESR `0.000113`, DP `0.000000`

DP improves exact-gap on every tested row.

## Assignment-conditioned lower bound (corrected)

Correction note:

- the previous LB diagnostic was invalid because multiplicities were dropped in `relaxed_machine_lb(...)`.
- this was fixed; the code now enforces safe per-machine LB values.

Reporting policy correction:

- do not treat `greedy_esr` assignment-conditioned LB as a validated safe bound for phase decisions.
- in this iteration, ESR LB values are diagnostic only; phase decision is based on TEC and exact-gap improvements.

Current rule:

- use relaxed-DP LB on the actual assigned multiset when valid
- otherwise use safe slot-based fallback LB
- if any computed LB exceeds a known feasible machine cost, replace by safe fallback

Schedule-level assignment-conditioned LB:

- `46/120`: `107`
- `61/350`: `6781`
- `64/77`: `30604` (ESR), `30598` (DP)
- `90/82`: `53294`

Interpretation remains diagnostic only (not a global bound, and not a gating signal for ESR).

## Runtime / memory

Heuristic runs stayed well below the 16 GB cap (process RSS in MB-scale).

## Decision

Phase A success criterion is met:

- DP machine optimization gives a clear quality improvement over ESR on the tested subset.

## Phase B extension (same driver)

Added variant:

- `dp_guided_assignment_dp`

Compared against `greedy_dp`:

- `46/120`: `154` vs `112` (worse)
- `61/350`: `10402` vs `7053` (worse)
- `64/77`: `30580` vs `30598` (better)
- `90/82`: `53294` vs `53294` (tie)

No clear additional gain from DP-guided assignment on this subset.
