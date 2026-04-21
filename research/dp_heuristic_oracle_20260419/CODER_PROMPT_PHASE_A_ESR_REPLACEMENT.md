# Coder Prompt: Phase A ESR Replacement Test

You are working in:

- `/Users/mac/Documents/Study/PFE/PaST`

This task belongs to the research thread:

- `research/dp_heuristic_oracle_20260419/`

Read first:

- `research/dp_heuristic_oracle_20260419/DP_HEURISTIC_ORACLE_PLAN.md`
- `research/dp_heuristic_oracle_20260419/OVERVIEW.md`
- `research/dp_heuristic_oracle_20260419/LITERATURE.md`
- `Papers/Exact and heuristic.txt`

## Goal

Do the minimum viable experiment for the new heuristic-DP direction.

We are **not** building a full EHS replacement yet.

We are testing one narrow hypothesis:

- if machine assignment is held fixed, does replacing paper-style ESR with our single-machine DP improve TEC meaningfully?

## Hard constraints

- keep all heavy runs below **16 GB RSS**
- heavy runs one at a time
- do not heavily rewrite the DP core
- do not implement the full `epsilon`-frontier loop
- do not implement full A-SGH or full R-ES in this phase
- do not revive any exact DP-guided proof branch

## What to implement

Create a new C++ comparison driver for fixed-`epsilon` heuristic evaluation.

Suggested file:

- `solvers/cpp/parallel_heuristic_compare.cpp`

Add build wiring in:

- `solvers/cpp/CMakeLists.txt`

### The driver must support at least these variants

1. `greedy_esr`
- simple LPT-style assignment baseline
- paper-style ESR per machine

2. `greedy_dp`
- same assignment baseline
- our exact single-machine DP per machine

Optional if easy and clean:

3. `greedy_relaxed_pack`
- same assignment baseline
- relaxed DP + recovery/pack per machine

## Assignment baseline for Phase A

Keep assignment deliberately simple:

- process jobs in non-increasing processing time order
- assign each job to a feasible machine using a simple greedy rule

Recommended first rule:

- assign to the machine that gives the smallest immediate energy insertion cost while keeping load within `epsilon`

Do not implement split-locations in this first pass.
Do not try to reproduce full SGH exactly yet.

This phase is about machine optimization quality, not assignment sophistication.

## ESR baseline

Implement the paper's ESR logic as the baseline machine optimizer.

Paper source:

- `Papers/Exact and heuristic.txt`
- Section 5.3.3, Eq. (27)

Important:

- ESR preserves the original machine job sequence
- do not “improve” ESR beyond what is needed for a faithful baseline

## DP machine optimizer

For `greedy_dp`, after assignment:

- take the multiset of jobs on each machine
- run our existing exact single-machine DP
- compute the machine-optimal TEC for that assigned job set

Use existing C++ DP code. Prefer wrappers and orchestration over core rewrites.

## Lower-bound diagnostic

For each machine, also compute a relaxed DP lower bound if available.

Report:

- `machine_exact_cost`
- `machine_relaxed_lb`

And at the schedule level:

- `TEC_total`
- `assignment_conditioned_LB = sum(machine_relaxed_lb)`

Important:

- describe this as an **assignment-conditioned lower bound**
- do not present it as a global optimality gap

## Test instances and epsilons

Use the corrected benchmark root only:

- `/Users/mac/Documents/Study/PFE/PaST/temp/paper_exact_repo/instances`

Run at least:

- `46 / 120`
- `61 / 350`
- `64 / 77`

Optional fourth:

- `90 / 82`

## Exact reference values

Use the existing CP-SAT exact branch only as ground truth for evaluation.

For each tested row, compare the heuristic TEC against the exact fixed-`epsilon` TEC.

## What to measure

For each instance / `epsilon` / variant, record:

- runtime
- max RSS
- total TEC
- assignment-conditioned LB
- number of machines
- per-machine job counts
- per-machine exact cost
- per-machine relaxed LB
- exact reference TEC
- exact relative gap:
  - `(TEC_variant - TEC_exact) / TEC_exact`

## Deliverables

Create:

1. `research/dp_heuristic_oracle_20260419/phaseA_esr_replacement_results.md`
2. `research/dp_heuristic_oracle_20260419/phaseA_esr_replacement_readiness.md`

Also update:

- `research/dp_heuristic_oracle_20260419/LOG.md`
- `research/dp_heuristic_oracle_20260419/iterations/20260419_phaseA_esr_replacement/RESULTS.md`
- `research/dp_heuristic_oracle_20260419/iterations/20260419_phaseA_esr_replacement/SUMMARY.md`

## Required contents of the results report

- exact commands used
- files changed
- benchmark root used
- runtime and RSS
- TEC comparison: `greedy_esr` vs `greedy_dp`
- exact-gap comparison against CP-SAT optimum
- whether the DP gives a clear quality improvement

## Decision rule

The phase succeeds only if:

- `greedy_dp` clearly improves TEC over `greedy_esr` on most tested rows

If not, say so clearly and recommend stopping the branch.

Do not move to DP-guided assignment in this phase.
