# PLAN 03F — Restore `K=2` Profile Repair And Add An Exact/MMKP Selector

Date: 2026-04-15

Status: NEW

## 1. Purpose

This plan fixes a real regression in Step 3.

The current unified Step-3 story is good in theory:

- exact profile realization when tractable,
- beam profile realization otherwise.

But in code, the current `profile_repair_beam` path explicitly skips `K<=2`,
which means the old hard two-type `{8,10}` repair behavior is no longer being
used. As a result, the current paper-group extension campaign is not
reproducing the previously validated low-memory `K=2` wins.

This plan restores the missing `K=2` exact repair capability and makes the
exact-vs-beam decision explicit for `K>=4`.

## 2. Core Design

Step 3 should become one family:

- **profile-realization DP**
  - exact mode for small/tractable cases
  - beam mode for large/intractable cases

With this concrete policy:

- `K=2`
  - default to a dedicated exact repair mode
  - low-memory, profile-based, block-by-block
- `K>=4`
  - try exact/MMKP-style profile realization only when tractable
  - otherwise use beam

This is the cleanest reconciliation of:

- old `block_repair_dp`,
- old `block_repair_mmkp`,
- current beam/profile-repair code,
- and the fixed-block DP family story.

## 3. What We Already Know

From earlier validated archive results:

### `K=2` / `{8,10}`

Two distinct Step-1/profile-repair success regimes existed:

- moderate large `n`:
  - `fwd_relax:block_dp_exact`
- harder large `n` tail:
  - `fwd_relax:block_repair_dp`

Representative evidence:

- [docs/scalability_gap_cap_20260331/scalability_gap_cap_findings.md](/Users/mac/Documents/Study/PFE/PaST/docs/scalability_gap_cap_20260331/scalability_gap_cap_findings.md)
- [research/large_k_large_n_attempt_20260409/RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/large_k_large_n_attempt_20260409/RESULTS.md)

### `K=4` / `3567_plus`

The earlier success pattern was:

- Step 1 incumbent from `block_repair_energy_core`
- then `step1_exact_guided` closes

Representative evidence:

- [research/large_k_large_n_attempt_20260409/RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/large_k_large_n_attempt_20260409/RESULTS.md)

### Current regression

In current code:

- `block_repair_profile_repair_beam_ub(...)` returns `kInf` for `K<=2`
- current Plan 05 runs therefore do not exercise a true Step-3 repair path on
  `{8,10}`
- some rows fall into exact with no incumbent and come back unresolved

## 4. Main Objective

Restore a practical, low-memory Step-3 path for `K=2` and make Step-3 mode
selection structural for `K>=4`.

Success means:

1. `K=2` no longer bypasses Step 3.
2. `{8,10}` large-`n` rows again use a real Step-3 profile-repair mode.
3. `K>=4` exact/MMKP mode is attempted only when tractable.
4. Beam remains the scalable fallback, not the only multitype path.

## 5. Implementation Tasks

### Task A — Audit current Step-3 exact and beam modes

Inspect:

- [stateful_dp_solver.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp)
- current exact fixed-block/profile mode
- current beam/profile mode
- current selector logic

Produce a short technical note in `LOG.md` stating:

- what the current exact profile mode really solves,
- why `K=2` is currently skipped,
- how exact mode and beam mode differ operationally.

### Task B — Restore `K=2` exact profile repair

Implement a dedicated exact profile-repair mode for `K=2`.

Requirements:

- use the recovered blocks/profile
- work block by block on counts/patterns
- remain low-memory
- do not rely on global exact DP to do the profile-repair job

Acceptable forms:

- recreated specialized two-type DP
- or a clean two-type specialization of the current exact profile-realization
  framework

But it must be:

- exact on the recovered profile
- clearly part of Step 3
- usable as the default exact mode for `K=2`

### Task C — Make exact/MMKP profile mode explicit for `K>=4`

For `K>=4`, exact profile realization should exist as the exact mode of Step 3.

This can reuse the current exact profile/fixed-block machinery if that is
already the right core.

The important point is not the name; it is the role:

- exact profile realization when tractable
- beam otherwise

If an MMKP/configuration-selection view is the cleanest explanation, document it
that way in the archive.

### Task D — Implement a structural selector

The selector must decide whether Step 3 uses:

- exact profile mode
- or beam mode

The selector must not depend only on `K`.

At minimum it should use:

- merged block count
- profile state-space estimate
- total pattern/composition estimate
- max patterns per block
- arithmetic hardness alarm if available

Target policy:

- `K=2`
  - exact mode by default unless a clear safety threshold rejects it
- `K>=4`
  - exact mode only when tractable
  - beam otherwise

The selector decision and reason must be logged in CSV output.

### Task E — Keep Step 3 one family

Do not turn this into a branch zoo.

The final explanation must stay:

> Step 3 is profile-realization DP. We use exact mode when the recovered-profile
> frontier is tractable, and beam mode otherwise.

The `K=2` special case is a specialization of exact profile realization, not a
separate theory branch.

## 6. What Must Not Happen

Do not:

- reintroduce exact-L2 into the mainline
- reintroduce Lagrangian as a co-equal default branch
- solve this by increasing memory limits
- use the global exact DP as the first rescue for rows that Step 3 should solve
- add local search

## 7. Validation Set

### Mandatory `K=2` rows

Run at least:

- `{8,10}`, `n=500`
- `{8,10}`, `n=600`
- `{8,10}`, `n=750`
- `{8,10}`, `n=1000`

If those recover:

- `{8,10}`, `n=1500`
- `{8,10}`, `n=2500`
- `{8,10}`, `n=3500`
- `{8,10}`, `n=5000`

### Mandatory `K=4` row

At least one representative `3567_plus` or equivalent previously validated
frontier row where Step 1 incumbent quality mattered.

### Mandatory `K=6` row

At least one `K=6` row where the selector should choose beam, to confirm the
new policy does not try exact mode too broadly.

## 8. Required Measurements

For each validation row record:

- Step-3 mode chosen
- selector reason
- runtime
- UB
- LB
- gap
- deciding step
- whether exact DP was needed
- whether the result matches earlier archive expectations

## 9. Success Criteria

This plan succeeds only if:

1. `K=2` is no longer skipped by Step 3.
2. `{8,10}` again has a real Step-3 repair path.
3. the exact-vs-beam decision is explicit and structural.
4. the Step-3 story remains unified.
5. the benchmark path becomes low-memory again on the paper two-type family.

## 10. Deliverables

Update:

- [LOG.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/LOG.md)
- [RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md)
- [BLOCKERS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md)
- [METHOD_BOUNDARIES.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md)
- [EXPERT_GUIDANCE.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/EXPERT_GUIDANCE.md)

Return only when:

- the missing `K=2` exact profile-repair capability is restored,
- the selector is implemented,
- the validation rows are rerun,
- and the new Step-3 policy is stated in one compact paragraph.
