# DP Heuristic Oracle Plan

Date: 2026-04-19

## Scope

This thread tests a heuristic use of the single-machine DP pipeline for BPMSTP.

The goal is not to produce a new exact solver. The goal is to test whether the DP can be the central engine of a **heuristic machine optimizer and evaluator** that competes with the paper's heuristic direction.

This plan is intentionally narrow. It starts from the smallest experiment that can falsify the idea quickly.

## Why this thread exists

The previous DP-centered exact directions already gave useful evidence:

- DP is strong on machine-local structure
- DP is weak as a global exact proof engine for the full parallel assignment search
- CP-SAT dominates the exact proof side

So the next realistic use of DP is:

- not global exact proof
- but machine-level heuristic optimization inside a parallel heuristic

## What we reuse from the paper

We reuse the paper's decomposition of the heuristic into separable roles:

1. a machine assignment step
2. a cross-machine improvement step
3. a machine-level retiming / optimization step

We specifically keep in mind:

- A-SGH assignment history is probably useful
- R-ES provides nontrivial cross-machine neighborhood improvement
- ESR is the weakest component, because it only optimizes timing while preserving the machine sequence

## What we do not reuse immediately

We do **not** start by reimplementing full EHS.

Reasons:

- A-SGH and R-ES add complexity that can hide the real effect of the DP
- we first need to know whether replacing ESR with our DP actually changes quality in a meaningful way
- if that first answer is weak, then the broader heuristic direction is probably not worth continuing

## Central hypothesis

For a fixed `epsilon` and a fixed machine assignment:

- paper-style ESR gives one machine-optimal cost under a sequence-preserving restriction
- our DP gives a globally optimal or near-optimal single-machine schedule for the assigned job multiset

Therefore:

- replacing ESR with our DP should reduce TEC for at least some instances and `epsilon` values

This is a hypothesis, not an assumption.

## Lower-bound interpretation

For any fixed assignment of jobs to machines:

```text
assignment_LB = sum_h relaxed_LB(machine_h_jobs)
```

is a valid lower bound for that **assignment-conditioned** machine optimization problem.

This is useful for:

- diagnosing whether the machine schedules are already near-optimal for the current assignment
- deciding whether more improvement must come from reassignment rather than retiming

This is **not** a global optimality gap for the full parallel problem.

## Implementation strategy

The work is split into three phases.

### Phase A: isolate ESR replacement

Goal:

- test whether our machine DP improves TEC over paper-style ESR when assignment is held fixed

Protocol:

1. Use a simple assignment baseline:
   - LPT-ordered greedy assignment
   - no split-location machinery in the first pass
2. For each machine assignment produced by that baseline:
   - Variant A: optimize each machine with paper-style ESR
   - Variant B: optimize each machine with our exact single-machine DP
3. Compare total TEC
4. Also compute assignment-conditioned lower bounds from our relaxed DP
5. Compare both heuristic variants against exact fixed-`epsilon` values on a small subset

Why this is the right first step:

- it isolates the machine-level effect
- it directly tests the core technical claim
- it is cheap to implement

### Phase B: DP-guided assignment scoring

Only start this if Phase A shows meaningful improvement.

Goal:

- test whether relaxed DP machine scores improve assignment quality

Protocol:

1. Keep the same fixed-`epsilon` subset
2. Add a new assignment variant:
   - when choosing the target machine for a job, use a fast relaxed-DP machine cost estimate instead of raw slot cost
3. Compare:
   - Variant B: simple assignment + DP machine optimization
   - Variant C: DP-guided assignment + DP machine optimization

This phase tests whether the relaxation signal is useful in assignment, without falling back into exact assignment enumeration.

### Phase C: exact-quality validation

Goal:

- measure heuristic quality using exact fixed-`epsilon` values already available from the CP-SAT branch

Protocol:

For each tested instance / `epsilon`:

- compute `gap_ESR = (TEC_ESR - TEC_exact) / TEC_exact`
- compute `gap_DP = (TEC_DP - TEC_exact) / TEC_exact`
- compute `gap_DP_assign = (TEC_DP_assign - TEC_exact) / TEC_exact` if Phase B is reached

This is more useful than heuristic-vs-heuristic comparison alone.

## Proposed test subset

Use a small but representative fixed-`epsilon` subset.

Recommended first subset:

- `46 / 120`
- `61 / 350`
- `64 / 77`

Rationale:

- `46 / 120` is a medium hard slice
- `61 / 350` is a large hard slice
- `64 / 77` is a large slice with exact benchmark value already known and a relatively tight structure

If a fourth row is needed:

- `90 / 82`

## What success means

### Phase A success

At least one of the following:

- DP machine optimization gives a clear TEC improvement over ESR on most tested rows
- or exact-gap results improve materially on most tested rows

### Phase A failure

Any of the following:

- ESR and our DP give almost identical TEC on the tested subset
- improvements are tiny and inconsistent
- runtime blows up enough that the heuristic becomes impractical before quality gain is clear

If Phase A fails, stop this branch.

### Phase B success

- DP-guided assignment yields a clear additional gain beyond simple assignment + DP optimization

### Phase B failure

- assignment scoring adds cost and complexity but little quality gain

If Phase B fails, do not escalate to full EHS replacement.

## Engineering constraints

- keep everything below 16 GB RSS
- use existing C++ single-machine DP components
- do not heavily rewrite the DP core
- prefer thin orchestration and comparison drivers
- do not implement the full `epsilon`-frontier loop yet
- do not reintroduce exact global assignment search

## Recommended code reuse

### Reuse from our side

- single-machine exact DP
- relaxed DP modes:
  - semigroup
  - feasible
  - partial
- machine schedule extraction and cost computation helpers where applicable

### Reuse from paper logic

- ESR as the baseline machine optimizer
- LPT ordering as a simple first assignment baseline
- later, if Phase A succeeds, assignment-history ideas can be added

### Do not reuse immediately

- full exact proof machinery from the failed DP-guided branch
- full configuration-master / pattern-master paths
- full A-SGH / R-ES replication before fixed-`epsilon` evidence exists

## Deliverables for this thread

1. A Phase A implementation and report
2. A decision note:
   - continue to Phase B
   - or stop the branch
3. If Phase B is reached:
   - a second report comparing assignment strategies

## Final recommendation

This direction is worth trying because it is narrow, testable, and uses the DP where the previous evidence says it is strongest.

It should be treated as a heuristic-quality experiment, not as an exact-method continuation.
