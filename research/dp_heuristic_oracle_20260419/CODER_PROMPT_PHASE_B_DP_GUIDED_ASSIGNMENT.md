# Coder Prompt: Phase B DP-Guided Assignment

Only use this prompt if Phase A succeeded clearly.

You are working in:

- `/Users/mac/Documents/Study/PFE/PaST`

This task belongs to:

- `research/dp_heuristic_oracle_20260419/`

Read first:

- `research/dp_heuristic_oracle_20260419/DP_HEURISTIC_ORACLE_PLAN.md`
- `research/dp_heuristic_oracle_20260419/phaseA_esr_replacement_results.md`
- `research/dp_heuristic_oracle_20260419/phaseA_esr_replacement_readiness.md`

## Goal

Test whether relaxed-DP machine scores improve assignment quality beyond the simple greedy assignment used in Phase A.

This is still a fixed-`epsilon` experiment.
Do not implement the full frontier loop yet.

## Hard constraints

- keep runs below **16 GB RSS**
- heavy runs one at a time
- do not heavily rewrite the DP core
- do not implement full A-SGH or R-ES in this phase
- do not turn this into an exact assignment search

## What to implement

Extend the Phase A comparison driver.

Add a new variant:

- `dp_guided_assignment_dp`

Meaning:

1. process jobs in non-increasing processing time order
2. for each candidate target machine, estimate the post-insertion machine quality using a fast relaxed-DP score
3. assign the job to the machine with the best estimated score subject to feasibility under `epsilon`
4. after assignment, optimize each machine with our exact single-machine DP

Keep the existing Phase A variants for comparison:

- `greedy_esr`
- `greedy_dp`

## Relaxed score guidance

Use existing relaxed DP machinery only as a fast assignment score.

Allowed modes to test:

- semigroup
- feasible
- partial

Start with the cheapest mode that gives a stable signal.

Do not call expensive exact DP for every candidate machine insertion unless you prove it is still practical.

## Test rows

Use the same subset as Phase A unless the Phase A report already recommends a narrower set:

- `46 / 120`
- `61 / 350`
- `64 / 77`

Optional:

- `90 / 82`

## What to measure

For each instance / `epsilon` / variant:

- runtime
- max RSS
- TEC
- assignment-conditioned LB
- exact reference TEC
- exact relative gap

Also report:

- whether assignment changed materially relative to Phase A
- whether DP-guided assignment improves over `greedy_dp`

## Deliverables

Create:

1. `research/dp_heuristic_oracle_20260419/phaseB_dp_guided_assignment_results.md`
2. `research/dp_heuristic_oracle_20260419/phaseB_dp_guided_assignment_readiness.md`

Update:

- `research/dp_heuristic_oracle_20260419/LOG.md`
- `research/dp_heuristic_oracle_20260419/iterations/20260419_phaseA_esr_replacement/RESULTS.md`
- `research/dp_heuristic_oracle_20260419/iterations/20260419_phaseA_esr_replacement/SUMMARY.md`

## Decision rule

Phase B succeeds only if:

- the DP-guided assignment variant gives a clear additional quality gain over `greedy_dp`

If not:

- stop the branch at “DP is useful as machine optimizer, but not as assignment guide”

Do not escalate to full EHS replication unless the gain is clear.
