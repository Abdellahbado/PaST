# PLAN 10 — K=4 Early-Pruning and Speedup Pass

## Objective

Reduce runtime of the recovered `K=4` `energy_core + direct + step1_exact_guided`
path by attacking **early-stage candidate explosion**, while preserving exact
closure on the active `K=4` regime.

This is **not** a new-method plan.
This is a **continuity-safe speedup plan**.

The hard gate remains:

- all active `K=4` rows must still close exactly,
- then we optimize runtime,
- not the other way around.

## Why this plan now

The recent continuity work changed the picture.

Current evidence says the recovered hard `K=4` path is real again, but too slow,
and the dominant cost is **not** the exact-core finish.

From
[/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/archive_20260420/plan08_tmp/tmp_plan09_phaseA_continuity_baseline.csv](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/archive_20260420/plan08_tmp/tmp_plan09_phaseA_continuity_baseline.csv):

- `3567_plus n=3500 s=0`
  - exact
  - runtime `793.1761s`
  - `fwd_ec_time_pattern_generation = 199.5565`
  - `fwd_ec_time_phase1 = 292.5139`
  - `fwd_ec_time_exact_core = 2.2552`
- `3567_plus n=3500 s=1`
  - exact
  - runtime `418.7627s`
  - `fwd_ec_time_pattern_generation = 23.0162`
  - `fwd_ec_time_phase1 = 127.4128`
  - `fwd_ec_time_exact_core = 2.7389`
- `3567_plus n=5000 s=0`
  - older persisted baseline still finite-gap
  - runtime `1251.3767s`
  - `fwd_ec_time_pattern_generation = 302.8410`
  - `fwd_ec_time_phase1 = 443.6514`
  - `fwd_ec_time_exact_core = 26.4096`

From
[/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/plan08/PLAN08_energy_core_campaign.csv](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/plan08/PLAN08_energy_core_campaign.csv),
the hard paper rows show the same pattern:

- `g3567 n=2500 s=0`
  - `pattern_generation = 1035.3163`
  - `phase1 = 1067.0112`
  - `exact_core = 4.4361`
- `g3567 n=3500 s=0`
  - `pattern_generation = 1428.2783`
  - `phase1 = 1464.2137`
  - `exact_core = 8.1752`
- `g3567 n=5000 s=0`
  - `pattern_generation = 2091.7944`
  - `phase1 = 2198.5417`
  - `exact_core = 11.5908`

Interpretation:

- the exact-core solve is **not** the main bottleneck,
- the main bottleneck is **early candidate generation and early frontier
  retention**,
- so the most likely speedups come from:
  - generating fewer useless candidates,
  - trimming candidate/frontier structures earlier,
  - and doing so without losing the winning K=4 path.

## Starting baseline

All speedup work in this plan must start from the current continuity-safe
rollback-style package, not from the broader PLAN_08 fortified package.

The starting package is:

- `PAST_RELAXED_BINPACK_SOLVER=energy_core`
- `PAST_BLOCK_REPAIR_COMPLETION_MODE=direct`
- `PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS=500000000`
- `PAST_BLOCK_REPAIR_EC_STRONGER_CENTER=0`
- `PAST_BLOCK_REPAIR_EC_DIVERSIFY=0`
- `PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA=0`
- `PAST_BLOCK_REPAIR_EC_TWO_PHASE=0`
- `PAST_BLOCK_REPAIR_EG_STATE_KEEP=60000`

If the continuity-safe package changes later, update this plan before
continuing.

## Files to inspect and edit

- [/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.cpp)
- [/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.hpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_dp_solver.hpp)
- [/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_compare.cpp](/Users/mac/Documents/Study/PFE/PaST/solvers/cpp/stateful_compare.cpp)

Relevant code zones:

- pattern generation:
  - `generate_energy_core_patterns(...)`
  - around `stateful_dp_solver.cpp:2400-2675`
- phase-1 feasible beam:
  - around `stateful_dp_solver.cpp:3628-3887`
- exact-core:
  - around `stateful_dp_solver.cpp:2943-3216`
  - included only for validation, not as the first optimization target

## Main hypothesis

The best near-term speedup path is:

1. preserve the recovered K=4 package,
2. optimize the parts that currently do too much work before exact-core begins,
3. start with **same-output or near-same-output** speedups,
4. only then try bounded pruning/ranking changes.

This means the priority order is:

1. exact-preserving early-stage speedups
2. continuity-safe ranking/pruning improvements
3. only then wider heuristic changes

## Phase A — Freeze and persist the continuity-safe baseline

Before any speedup change:

1. reproduce and persist the current continuity-safe baseline on:
   - `3567_plus n=3500,5000`, seeds `0,1`
   - `g3567 n=2500,3500,5000`, seeds `0,1`, `lambda=1.3`
2. write an auditable baseline artifact, for example:
   - `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_baseline.csv`
3. include at minimum:
   - exact / not exact
   - runtime
   - UB
   - LB
   - gap
   - `diag_step3_decided`
   - `diag_step4_decided`
   - `fwd_ec_time_completion`
   - `fwd_ec_time_pattern_generation`
   - `fwd_ec_time_phase1`
   - `fwd_ec_time_exact_core`
   - `fwd_ec_generated_patterns_total`
   - `fwd_ec_retained_patterns_total`
   - `fwd_ec_delta_used`

Do not optimize from stale or mixed baselines.

## Phase B — Exact-preserving early-stage speedups

These are the first changes to try because they are the lowest-risk.

### B1. Bounded top-k maintenance inside pattern generation

Current pattern generation still grows oversized buckets and then sorts/trims
them. That is likely wasting time.

Required change:

- in `generate_energy_core_patterns(...)`,
- replace append-then-full-sort behavior with bounded top-k maintenance where
  possible,
- keep the same ranking criterion as the current code,
- and preserve the same retained set whenever practical.

Concrete targets:

- DP-generator per-work buckets
- DFS-generator per-work buckets
- final `flat` trimming when only the top `global_keep` matters

Preferred implementation style:

- use bounded heaps or partial selection (`nth_element`) instead of full sort
  when only a prefix is kept,
- sort only the retained prefix if ordering is needed later.

This is the most likely safe speedup.

### B2. Earlier exact duplicate elimination

Current exact duplicate suppression is late and string-heavy.

Required change:

- drop exact duplicate partial states earlier where the semantics are clearly
  unchanged,
- especially in pattern generation and other candidate-building stages,
- while keeping the current “exact duplicate only” policy.

Examples of acceptable changes:

- exact duplicate suppression on identical partial count states before bucket
  growth,
- cheaper exact-key representations instead of repeated string churn,
- maintaining only the best representative for the same exact partial state.

Do not introduce approximate dominance in this step.

### B3. Partial selection instead of full sort in phase-1 beam

Current phase-1 beam sorts the entire `next_layer` before resizing.
That is a likely avoidable cost on hard rows.

Required change:

- when `next_layer.size() > layer_width`,
- use partial selection so only the retained prefix is fully ordered.

Goal:

- preserve the same retained beam set,
- reduce time spent sorting large temporary layers.

### B4. Cheap profiling of key construction cost

If needed, measure whether:

- `pattern_counts_key(...)`
- `count_key_from_counts(...)`

are materially hot on hard rows.

Only if confirmed hot, replace them with cheaper fixed-width keys in the early
stages.

Do not do speculative key rewrites without evidence.

## Phase C — Continuity-safe earlier pruning and ranking

Only start this phase if Phase B is implemented and exact closure is still
preserved.

### C1. Improve phase-1 pattern ordering, not just phase-1 width

The feasible beam already uses:

- `pattern_pref_rank`
- `pattern_local_rank`
- center / feasibility / arithmetic pressure

The likely better next move is:

- improve which patterns are considered early,
- so discrepancy pruning removes weak patterns sooner,
- rather than only widening or shrinking width.

Allowed work:

- retune local pattern ranking to better reflect scarce-type pressure and suffix
  feasibility,
- keep diversification off for hard K=4 unless explicitly revalidated.

### C2. Better partial ranking in pattern generation

If Phase B is not enough, improve the ranking used inside the generator while
keeping the same pool size budget.

Allowed direction:

- rank partial states using the current score plus a cheap optimistic residual
  term for remaining types,
- so the retained `per_work_keep` states are more informative earlier.

This is allowed only if continuity rows remain exact.

### C3. Width retuning only after C1/C2

Changing `state_keep` or phase-1 width rules is allowed only after:

- same-output speedups are done,
- pattern ordering is improved,
- and exactness on continuity rows is preserved.

Reason:

- width changes are more likely to silently break closure than the Phase B
  changes.

## What not to do

- do not reactivate the broader PLAN_08 feature package on hard K=4
- do not jump to full column generation or branch-and-price
- do not broaden this task to K>4 transfer work
- do not accept finite-gap K=4 rows as a successful speedup
- do not chase exact-core micro-optimizations first
- do not change three ranking/pruning dimensions at once without ablation

## Required experiment matrix

### Continuity gate

Always test first on:

- `3567_plus n=3500 s=0`
- `3567_plus n=3500 s=1`
- `3567_plus n=5000 s=0`
- `3567_plus n=5000 s=1`

Any package that loses exactness on any of these is disqualified.

### Hard paper rows

Then test:

- `g3567 n=2500 s=0,1`
- `g3567 n=3500 s=0,1`
- `g3567 n=5000 s=0,1`

If hard paper rows remain open, keep going.

### Optional easy control rows

Use only if needed to confirm no broad regression:

- `g3567 n=1000 s=0,1`
- `g3567 n=1500 s=0,1`

## Success criteria

This plan succeeds only if:

1. the continuity-safe K=4 package remains exact on the required continuity
   rows,
2. runtime drops materially on the hard recovered rows,
3. the main reduction comes from:
   - `fwd_ec_time_pattern_generation`, and/or
   - `fwd_ec_time_phase1`,
   not from unrelated exact-core drift,
4. and the hard `g3567` rows are not made worse.

Preferred order of wins:

1. exact closure preserved
2. lower runtime on `3567_plus n=5000`
3. lower runtime on hard `g3567`
4. smaller memory footprint

## Deliverables

Create:

- `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_baseline.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_ablation.csv`

Update:

- `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/LOG.md`
- `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/RESULTS.md`
- `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/BLOCKERS.md`
- `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md`
- `/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/archive_20260421/markdown/k4_history/ENERGY_CORE_FORTIFICATION_NOTE.md`

If useful, create:

- `K4_SPEEDUP_NOTE.md`

## Final decision rule

The final recommended package should be the simplest one that:

- keeps all active K=4 rows exact,
- reduces early-stage runtime materially,
- and does not depend on broad speculative tuning.

If same-output speedups already give a meaningful win, stop there.
Do not force riskier pruning changes just because they are available.
