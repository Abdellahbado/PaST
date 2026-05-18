# Plan 01: Recover Level-2 Baseline and Finish the Phase-1 Two-Axis Grid

## Status

Prepared on: `2026-04-12`

Archive context:

- [PROBLEM.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PROBLEM.md)
- [PLAN.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/PLAN.md)
- [EXPERT_GUIDANCE.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/EXPERT_GUIDANCE.md)
- [RESULTS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md)
- [BLOCKERS.md](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md)

Primary code/files:

- [stateful_dp_solver.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp)
- [stateful_compare.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_compare.cpp)
- [run_two_axis_grid.py](/Users/mac/Documents/Study/PFE/PaST/scripts/run_two_axis_grid.py)

---

## Objective

Execute the next coding cycle in a way that is:

- faithful to the two-axis framework,
- consistent with the expert guidance,
- and disciplined enough that the result can be trusted as a new baseline.

This plan has **two goals**:

1. recover the best known clean Level-2 baseline,
2. finish the missing high-value phase-1 grid cells under that cleaned policy.

This plan is intentionally **not** the place to implement dynamic pricing,
arc-flow, or a new Level-2 method. It is the last structured baseline-and-
evidence pass before any heavier redesign.

---

## Core diagnosis this plan assumes

These points are already supported by the archive and should be treated as the
starting assumptions for this plan:

1. **The two-axis framing is valid.**
   - difficulty is not monotone in `K`
   - arithmetic structure is a separate axis

2. **Level 3 separation was a real win and should remain in place.**
   - do not remove the exact per-block multiset DP evaluation
   - do not revert to the old global ascending/descending surrogate

3. **The main remaining structural bottleneck is Level 2.**
   - the winning incumbents on the open rows still come from
     `block_repair_feasible_beam`
   - the Lagrangian branch is currently regressed relative to an earlier
     validated state

4. **The post-Lagrangian beam combination is not the right default.**
   - the solver policy has already been cleaned so hidden beam rescues are no
     longer default behavior
   - keep that cleanup

---

## Non-goals

Do **not** do these in this plan unless the plan explicitly reaches the
fallback stage that asks for them:

1. do not implement dynamic pricing
2. do not implement arc-flow
3. do not redesign the whole Level-2 assignment layer
4. do not retune many solver branches at once
5. do not replace the new grid runner with ad hoc commands

This plan is about:

- recovery,
- controlled validation,
- and only then a narrow decision on what comes next.

---

## Main deliverables

By the end of this plan, the coder should produce:

1. a restored and validated Level-2 baseline, or a documented proof that it
   cannot be recovered cheaply from the current branch
2. a completed phase-1 grid slice including the missing high-value irregular
   high-`K` cells
3. updated archive files:
   - `LOG.md`
   - `RESULTS.md`
   - `BLOCKERS.md`
4. a single explicit decision:
   - either “the current Level-2 baseline is adequate”
   - or “move next to dynamic pricing / volume-algorithm style recovery”

---

## Phase A. Protect the current cleaned policy

### Purpose

Preserve the current policy cleanup before making any other changes.

### Required checks

Confirm in [stateful_dp_solver.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp):

1. `PAST_BLOCK_REPAIR_LAGR_USE_SEEDED_BEAM` defaults to `0`
2. `PAST_BLOCK_REPAIR_LAGR_BEAM_POLISH_MAX_MERGED` defaults to `0`

### Rule

Do not re-enable hidden beam help by default during this plan.

If you need seeded beam or beam polish for a diagnostic, use environment
overrides only and record them explicitly in `LOG.md`.

### Success criterion

The baseline policy remains interpretable:

- Lagrangian path is genuinely Lagrangian-first
- feasible beam path is genuinely a separate comparison branch

---

## Phase B. Recover the best validated Lagrangian baseline

### Purpose

Recover the earlier corrected-Lagrangian regime that achieved approximately:

- `2345711 n=1000`: gap `0.0129%`
- `2345711 n=1500`: gap `0.0079%`
- `2345711 n=2500`: gap `0.0070%`

with `block_repair_lagrangian_assign` active as the incumbent method.

### Why this matters

Right now, the archive’s best clean story is still compromised by Level-2
regression:

- current default policy is cleaner,
- but the Lagrangian branch is weaker than its best validated earlier state.

If this recovery succeeds, later comparisons against beam or future pricing
will be fair.

### Exact tasks

1. Read the relevant validated history in:
   - `research/large_k_large_n_attempt_20260409/LOG.md`
   - `research/large_k_large_n_attempt_20260409/RESULTS.md`
   - `research/large_k_large_n_attempt_20260409/BLOCKERS.md`

2. Extract the corrected-Lagrangian characteristics that were present in the
    validated cycle. In particular verify whether the best cycle depended on:
    - dual-gap-scaled subgradient steps
    - alpha halving on stagnation
    - adaptive regularization by merged-profile size
    - adaptive `repair_l1`
    - final repair from best near-feasible iterate
    - widened merged-block gate
    - local improver settings

3. Compare that against the current implementation of:
    - `block_repair_lagrangian_assign_ub(...)`

4. Recover the best clean configuration **without** reintroducing hidden beam
    defaults.

### Critical context from recent diagnostics

The latest tracing showed that the Lagrangian converges well (best_l1 = 23
on n=1000, very close to feasible). The failure is NOT in the dual search —
it is in the **repair handoff**: converting a near-feasible assignment
(23-unit L1 residual) into a fully feasible one. Recovery efforts should
focus on the repair step, not the dual trajectory.

Also verify that the **proxy/exact split** is correctly active:

- The Lagrangian dual loop should use the old corrected block-energy PROXY
  for assignment guidance during search
- Final candidate scoring should use the new exact per-block Level 3
  evaluator

The Level 3 change inadvertently made the dual loop optimize the exact
evaluator, which broke search guidance. The coder who implemented the split
confirmed it builds cleanly. Verify this split is still in place before
attempting any recovery.

### Important restriction

Do not recover the baseline by silently reintroducing the beam as a fallback
inside the Lagrangian path.

The target is:

- a cleaner Lagrangian baseline,
- not another hybrid that masks the regression.

### Required validation rows

At minimum rerun:

- `2345711 n=1000`
- `2345711 n=1500`
- `2345711 n=2500`

using:

- `step1_exact_guided`
- and, when needed, `solve-stdin`

### Success criterion

One of these two outcomes must be achieved and documented:

#### Preferred

The Lagrangian branch is restored near its earlier validated quality, with
Level 3 still active.

#### Acceptable fallback

The coder demonstrates clearly that the earlier behavior cannot be recovered
cleanly from the current code path without reintroducing hidden hybrids or
undoing other justified improvements.

In that fallback case, the archive must state that the current beam-owned
Level-2 baseline is the stable baseline for the next phase.

---

## Phase C. Finish the missing phase-1 grid cells

### Purpose

Complete the highest-value missing two-axis cells under the cleaned policy.

### Use this tool, not ad hoc commands

Use:

- [run_two_axis_grid.py](/Users/mac/Documents/Study/PFE/PaST/scripts/run_two_axis_grid.py)

Do not switch back to one-off Python snippets unless the runner itself is
broken and you first document why.

### Before running

Confirm the runner still treats `n` as **total jobs**, not jobs-per-type.

### Required cells

These are the minimum required cells for this plan:

1. `hard_k4_irregular = {3,5,7,11}`, `n=1000`
2. `hard_k8_irregular = {3,5,7,11,13,17,19,23}`, `n=1000`
3. `hard_k10_irregular = {2,3,5,7,11,13,17,19,23,29}`, `n=1000`

If runtime allows, also add:

4. `medium_k6_dense`, another seed
5. `hard_k6_2345711`, another seed

### Why these cells

They answer the most important unresolved question:

- do arithmetic hardness and larger `K` compound at scale,
- or do more generators partly compensate for irregular arithmetic?

### IMPORTANT: Known segfault risk

The previous coder observed **segfaults** on `hard_k8_irregular n=1000`
and `hard_k10_irregular n=1000`. The same families at `n=300` were fine.
This is likely a code bug (buffer overflow in Level 3 DP state arrays or
pattern-pool growth), NOT a fundamental algorithmic boundary.

**If segfaults occur:**

1. Rebuild with `-fsanitize=address` (ASAN) to get a precise stack trace
2. Fix the bounds issue before recording results
3. Document the fix in `LOG.md`
4. Do NOT skip these cells — they are the most important experiment

Also note: a first-cut dynamic pricing implementation was added behind
the flag `PAST_BLOCK_REPAIR_LAGR_DYNAMIC_PRICING`. It is disabled by
default. Do NOT enable it in this plan — it is unstable and caused the
previous crashes. If you encounter it in the code, leave it off.

### Output location

Write the next consolidated CSV into the current archive, for example:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase2.csv`

### Wall-clock control

Keep an external wall timeout per cell in the runner.

If a cell does not finish cleanly:

- record the timeout or crash,
- keep the partial row if valid,
- and write the failure mode into `LOG.md` and `BLOCKERS.md`.

Do not omit the row silently.

### Success criterion

At the end of this phase, the archive should contain a clean statement about
the irregular high-`K`, larger-`n` cell:

- exact at Step 1
- finite-gap but not exact
- timeout
- or crash

Any of these is useful if recorded honestly.

---

## Phase D. Compare current generated rows against prior benchmark anchors

### Purpose

Avoid mixing conclusions from new generated instances with conclusions from the
previous benchmark rows without saying so.

### Instance generation caveat

The grid runner (`run_two_axis_grid.py` line 140) generates jobs by picking
types with **uniform probability**: `rng.choice(lengths)`. For irregular
families with very different lengths (e.g., 2 vs 29), this creates different
total work profiles compared to the old benchmark instances. The new and old
instances are NOT directly comparable without an explicit bridging check.

### Required task

Take at least one representative row from each of:

1. old hard six-type benchmark family
2. old medium six-type benchmark family

and rerun them under the current solver/policy so the archive can say:

- whether the new two-axis driver agrees with the older benchmark story
- and whether the Level 3 gains still hold on the original anchors

### Why this matters

The paper should not accidentally compare:

- old benchmark rows on one policy
- against new generated rows on another policy

without a bridging statement.

### Success criterion

The archive contains at least one direct apples-to-apples bridge result between:

- earlier benchmark anchors
- and the current two-axis phase-1 driver results

---

## Phase E. Add one small diagnostic to the runner

### Purpose

Implement the smallest extra diagnostics suggested by the expert that can
help interpret why Step 1 closes or fails.

### Recommended diagnostics

#### 1. Residual-core proxy (from existing CSV output)

After parsing solver output, record:

- active incumbent method (`fwd_pack_method` / `winner_detail`)
- merged block count (`fwd_merged_block_count`)
- whether the winner was a trivial `ffd` closure or a repaired-profile
  incumbent

#### 2. Three-gap decomposition (if feasible from existing output)

The remaining gap decomposes into:

- **Relaxation gap**: LB vs true optimal (currently ≈ 0)
- **Configuration-pool gap**: the optimal assignment needs a pattern NOT
  in the filtered pool
- **Search gap**: the optimal pattern IS in the pool but the search
  didn't find the right combination

To distinguish pool vs search gap: after a run, check whether the beam's
winning patterns are a SUBSET of the patterns available to the Lagrangian.
If the solver CSV already reports `fwd_pack_method`, this may be inferable.
Otherwise note it as a future instrumentation task.

#### 3. Block-boundary separability check (one-time diagnostic)

On ONE representative hard row, compare the sum of per-block Level 3
costs against the cost of the global `solve_fixed_sequence` on the same
assignment. If they match, the Level 3 separability is correct. If they
differ, the boundary handling needs fixing. This is a one-time check,
not a permanent diagnostic.

#### 4. Block capacity distribution

When reporting arithmetic descriptors, also record the distribution of
recovered block capacities (min, max, mean). The hardness triplet is
(length set, block capacities, bounded multiplicities), not just
length set alone.

### Important restriction

Do not add a large new solver instrumentation branch in this plan.

If the desired diagnostic is not already exposed by CSV output:

- record that as a future instrumentation task,
- do not derail this plan to build a new tracing subsystem.

---

## Decision rules at the end of the plan

At the end of Phases A–E, the coder must choose exactly one of the following
next-step recommendations and write it explicitly into `BLOCKERS.md` and
`LOG.md`.

### Recommendation 1. Baseline is adequate, move to paper-facing experiments

Choose this only if:

- the recovered or current Level-2 baseline is stable,
- the missing high-value grid cells are completed,
- and the evidence is already sufficient to support the two-axis paper story.

### Recommendation 2. Level 2 still needs one final algorithmic escalation

Choose this if:

- beam still dominates Level 2 on the important open rows,
- and the new cross-cells suggest the remaining frontier is genuinely
  assignment-driven.

In that case, the next plan should be:

- dynamic pricing inside the Lagrangian loop

not generic tuning.

### Recommendation 3. Robustness bug first

Choose this if:

- the missing cross-cells expose crashes or policy-instability that prevent
  trustworthy comparison.

In that case, the next plan should be:

- robustness/stability repair,
- not algorithmic escalation.

---

## Fallback ladder if recovery fails

If Phase B fails to recover the earlier clean Lagrangian baseline, the coder
must not improvise broadly. Use this fallback order:

1. keep the cleaned policy as baseline
2. finish the missing grid cells anyway
3. compare beam-owned Level-2 outcomes across the two-axis matrix
4. only then recommend dynamic pricing as the next algorithmic plan

This prevents the work from falling back into reactive patching.

---

## Exact files expected to change in this plan

Likely:

- `solvers/cpp/stateful_dp_solver.cpp`
- `scripts/run_two_axis_grid.py`
- `research/k_vs_arithmetic_axes_20260412/LOG.md`
- `research/k_vs_arithmetic_axes_20260412/RESULTS.md`
- `research/k_vs_arithmetic_axes_20260412/BLOCKERS.md`

Possible but optional:

- `solvers/cpp/stateful_dp_solver.hpp`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase2.csv`

Files that should probably **not** change in this plan unless strongly
justified:

- `IDEAS.md`
- `PROBLEM.md`
- `LITERATURE.md`

Those are framing files, not execution files, and this plan is about
execution.

---

## Reporting format for the coder using this plan

At the end of the work, the coder should report:

1. which baseline was used
2. whether Lagrangian recovery succeeded
3. the exact irregular `K=4`, `K=8`, `K=10` `n=1000` results
4. whether the axes appear to compound at larger `n`
5. the recommended next plan:
   - paper experiments
   - dynamic pricing
   - or robustness repair

The report should be written into the archive first, then summarized outside
it.
