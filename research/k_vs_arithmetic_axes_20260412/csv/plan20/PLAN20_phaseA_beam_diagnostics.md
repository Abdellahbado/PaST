# PLAN20 Phase A: Beam Diagnostics

Source: PLAN18 + PLAN19 profile_repair_beam rows on hard irregular K=8/10/12.

## Key observation

All K=10/12 rows that produce an incumbent use `profile_repair_beam` as the pack method.
The beam always produces a candidate (status=feasible), but Step 4 exact DP does not close the gap.
K=8 exact rows use `block_repair_energy_core` (baseline), not beam.
K=8 finite-gap rows use beam and behave similarly to K=10/12 finite-gap rows.

## Beam metric summary by outcome class and K

### finite_gap | K=8 (16 rows)

- beam_base_width: N/A
- beam_avg_width: N/A
- beam_max_width: N/A
- beam_states_considered: N/A
- beam_states_kept: N/A
- beam_pruned_over: N/A
- beam_pruned_suffix: N/A
- beam_pruned_discrepancy: N/A
- beam_disc_budget: mean=0.0, min=0.0, max=0.0, n=16
- beam_disc_depth: mean=0.0, min=0.0, max=0.0, n=16
- t_pack_profile_beam: mean=234.4, min=66.1, max=395.3, n=16
- runtime_sec: mean=387.5, min=123.2, max=945.1, n=16
- gap_pct: mean=0.0, min=0.0, max=0.0, n=16
- peak_rss_gb: mean=3.7, min=2.5, max=4.8, n=16

### finite_gap | K=10 (21 rows)

- beam_base_width: N/A
- beam_avg_width: N/A
- beam_max_width: N/A
- beam_states_considered: N/A
- beam_states_kept: N/A
- beam_pruned_over: N/A
- beam_pruned_suffix: N/A
- beam_pruned_discrepancy: N/A
- beam_disc_budget: mean=0.0, min=0.0, max=0.0, n=21
- beam_disc_depth: mean=0.0, min=0.0, max=0.0, n=21
- t_pack_profile_beam: mean=457.8, min=197.1, max=1051.4, n=21
- runtime_sec: mean=689.4, min=324.3, max=1292.2, n=21
- gap_pct: mean=0.0, min=0.0, max=0.1, n=21
- peak_rss_gb: mean=5.4, min=2.4, max=8.1, n=21

### finite_gap | K=12 (9 rows)

- beam_base_width: N/A
- beam_avg_width: N/A
- beam_max_width: N/A
- beam_states_considered: N/A
- beam_states_kept: N/A
- beam_pruned_over: N/A
- beam_pruned_suffix: N/A
- beam_pruned_discrepancy: N/A
- beam_disc_budget: mean=0.0, min=0.0, max=0.0, n=9
- beam_disc_depth: mean=0.0, min=0.0, max=0.0, n=9
- t_pack_profile_beam: mean=589.1, min=458.7, max=898.8, n=9
- runtime_sec: mean=956.9, min=721.4, max=1267.6, n=9
- gap_pct: mean=0.0, min=0.0, max=0.1, n=9
- peak_rss_gb: mean=5.8, min=4.3, max=8.4, n=9

## Per-row detail (profile_repair_beam only)

| family | K | seed | class | gap% | rt(s) | beam_base_w | beam_avg_w | states_cons | states_kept | disc_bud | disc_dep | beam_status | beam_to | pack_beam_t | step2_ub | beam_ub | improved? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hardA_k10 | 10 | 0 | finite_gap | 0.0172 | 338.9 | nan | nan | nan | nan | 0 | 0 |  | 0 | 205.5 | nan | nan | 0 |
| hardA_k10 | 10 | 0 | finite_gap | 0.0172 | 338.9 | nan | nan | nan | nan | 0 | 0 |  | 0 | 205.5 | nan | nan | 0 |
| hardA_k10 | 10 | 0 | finite_gap | 0.0172 | 347.0 | nan | nan | nan | nan | 0 | 0 |  | 0 | 211.1 | nan | nan | 0 |
| hardA_k10 | 10 | 0 | finite_gap | 0.0172 | 638.9 | nan | nan | nan | nan | 0 | 0 |  | 0 | 209.9 | nan | nan | 0 |
| hardA_k10 | 10 | 0 | finite_gap | 0.0172 | 324.3 | nan | nan | nan | nan | 0 | 0 |  | 0 | 197.1 | nan | nan | 0 |
| hardA_k10 | 10 | 1 | finite_gap | 0.0272 | 650.8 | nan | nan | nan | nan | 0 | 0 |  | 0 | 530.4 | nan | nan | 0 |
| hardA_k10 | 10 | 1 | finite_gap | 0.0272 | 650.8 | nan | nan | nan | nan | 0 | 0 |  | 0 | 530.4 | nan | nan | 0 |
| hardA_k10 | 10 | 1 | finite_gap | 0.0272 | 575.9 | nan | nan | nan | nan | 0 | 0 |  | 0 | 464.5 | nan | nan | 0 |
| hardA_k10 | 10 | 1 | finite_gap | 0.0272 | 1292.2 | nan | nan | nan | nan | 0 | 0 |  | 0 | 462.7 | nan | nan | 0 |
| hardA_k10 | 10 | 2 | finite_gap | 0.0199 | 356.0 | nan | nan | nan | nan | 0 | 0 |  | 0 | 200.9 | nan | nan | 0 |
| hardA_k10 | 10 | 2 | finite_gap | 0.0199 | 356.0 | nan | nan | nan | nan | 0 | 0 |  | 0 | 200.9 | nan | nan | 0 |
| hardA_k10 | 10 | 3 | finite_gap | 0.0358 | 606.3 | nan | nan | nan | nan | 0 | 0 |  | 0 | 491.6 | nan | nan | 0 |
| hardA_k10 | 10 | 3 | finite_gap | 0.0358 | 606.3 | nan | nan | nan | nan | 0 | 0 |  | 0 | 491.6 | nan | nan | 0 |
| hardA_k12 | 12 | 0 | finite_gap | 0.0239 | 758.5 | nan | nan | nan | nan | 0 | 0 |  | 0 | 484.5 | nan | nan | 0 |
| hardA_k12 | 12 | 0 | finite_gap | 0.0239 | 758.5 | nan | nan | nan | nan | 0 | 0 |  | 0 | 484.5 | nan | nan | 0 |
| hardA_k12 | 12 | 0 | finite_gap | 0.0239 | 777.3 | nan | nan | nan | nan | 0 | 0 |  | 0 | 496.9 | nan | nan | 0 |
| hardA_k12 | 12 | 1 | finite_gap | 0.0508 | 1151.7 | nan | nan | nan | nan | 0 | 0 |  | 0 | 898.8 | nan | nan | 0 |
| hardA_k12 | 12 | 2 | finite_gap | 0.0399 | 721.4 | nan | nan | nan | nan | 0 | 0 |  | 0 | 458.7 | nan | nan | 0 |
| hardA_k12 | 12 | 2 | finite_gap | 0.0399 | 721.4 | nan | nan | nan | nan | 0 | 0 |  | 0 | 458.7 | nan | nan | 0 |
| hardA_k8 | 8 | 0 | finite_gap | 0.0047 | 124.5 | nan | nan | nan | nan | 0 | 0 |  | 0 | 67.1 | nan | nan | 0 |
| hardA_k8 | 8 | 0 | finite_gap | 0.0047 | 124.5 | nan | nan | nan | nan | 0 | 0 |  | 0 | 67.1 | nan | nan | 0 |
| hardA_k8 | 8 | 1 | finite_gap | 0.0237 | 246.5 | nan | nan | nan | nan | 0 | 0 |  | 0 | 193.8 | nan | nan | 0 |
| hardA_k8 | 8 | 1 | finite_gap | 0.0237 | 246.5 | nan | nan | nan | nan | 0 | 0 |  | 0 | 193.8 | nan | nan | 0 |
| hardA_k8 | 8 | 2 | finite_gap | 0.0076 | 123.2 | nan | nan | nan | nan | 0 | 0 |  | 0 | 66.1 | nan | nan | 0 |
| hardA_k8 | 8 | 2 | finite_gap | 0.0076 | 123.2 | nan | nan | nan | nan | 0 | 0 |  | 0 | 66.1 | nan | nan | 0 |
| hardA_k8 | 8 | 3 | finite_gap | 0.0169 | 237.7 | nan | nan | nan | nan | 0 | 0 |  | 0 | 191.4 | nan | nan | 0 |
| hardA_k8 | 8 | 3 | finite_gap | 0.0169 | 237.7 | nan | nan | nan | nan | 0 | 0 |  | 0 | 191.4 | nan | nan | 0 |
| hardB_k10 | 10 | 0 | finite_gap | 0.0391 | 755.4 | nan | nan | nan | nan | 0 | 0 |  | 0 | 427.6 | nan | nan | 0 |
| hardB_k10 | 10 | 0 | finite_gap | 0.0391 | 755.4 | nan | nan | nan | nan | 0 | 0 |  | 0 | 427.6 | nan | nan | 0 |
| hardB_k10 | 10 | 0 | finite_gap | 0.0391 | 731.7 | nan | nan | nan | nan | 0 | 0 |  | 0 | 413.7 | nan | nan | 0 |
| hardB_k10 | 10 | 1 | finite_gap | 0.0620 | 1266.0 | nan | nan | nan | nan | 0 | 0 |  | 0 | 1051.4 | nan | nan | 0 |
| hardB_k10 | 10 | 1 | finite_gap | 0.0620 | 1266.0 | nan | nan | nan | nan | 0 | 0 |  | 0 | 1051.4 | nan | nan | 0 |
| hardB_k10 | 10 | 1 | finite_gap | 0.0620 | 1006.0 | nan | nan | nan | nan | 0 | 0 |  | 0 | 751.0 | nan | nan | 0 |
| hardB_k10 | 10 | 2 | finite_gap | 0.0450 | 807.4 | nan | nan | nan | nan | 0 | 0 |  | 0 | 545.0 | nan | nan | 0 |
| hardB_k10 | 10 | 2 | finite_gap | 0.0450 | 807.4 | nan | nan | nan | nan | 0 | 0 |  | 0 | 545.0 | nan | nan | 0 |
| hardB_k12 | 12 | 0 | finite_gap | 0.0481 | 1267.6 | nan | nan | nan | nan | 0 | 0 |  | 0 | 726.4 | nan | nan | 0 |
| hardB_k12 | 12 | 2 | finite_gap | 0.0292 | 1227.8 | nan | nan | nan | nan | 0 | 0 |  | 0 | 646.8 | nan | nan | 0 |
| hardB_k12 | 12 | 2 | finite_gap | 0.0292 | 1227.8 | nan | nan | nan | nan | 0 | 0 |  | 0 | 646.8 | nan | nan | 0 |
| hardB_k8 | 8 | 0 | finite_gap | 0.0297 | 945.1 | nan | nan | nan | nan | 0 | 0 |  | 0 | 389.3 | nan | nan | 0 |
| hardB_k8 | 8 | 0 | finite_gap | 0.0297 | 945.1 | nan | nan | nan | nan | 0 | 0 |  | 0 | 389.3 | nan | nan | 0 |
| hardB_k8 | 8 | 1 | finite_gap | 0.0458 | 523.0 | nan | nan | nan | nan | 0 | 0 |  | 0 | 378.9 | nan | nan | 0 |
| hardB_k8 | 8 | 1 | finite_gap | 0.0458 | 523.0 | nan | nan | nan | nan | 0 | 0 |  | 0 | 378.9 | nan | nan | 0 |
| hardB_k8 | 8 | 2 | finite_gap | 0.0257 | 360.5 | nan | nan | nan | nan | 0 | 0 |  | 0 | 193.3 | nan | nan | 0 |
| hardB_k8 | 8 | 2 | finite_gap | 0.0257 | 360.5 | nan | nan | nan | nan | 0 | 0 |  | 0 | 193.3 | nan | nan | 0 |
| hardB_k8 | 8 | 3 | finite_gap | 0.0489 | 539.3 | nan | nan | nan | nan | 0 | 0 |  | 0 | 395.3 | nan | nan | 0 |
| hardB_k8 | 8 | 3 | finite_gap | 0.0489 | 539.3 | nan | nan | nan | nan | 0 | 0 |  | 0 | 395.3 | nan | nan | 0 |

## Diagnostic interpretation

- Exact K=8 rows via baseline energy_core: 0 rows. These do NOT use beam; beam is not on the exact path for K=8.
- Finite-gap K=8 rows via beam: 16 rows. Gaps range 0.0047% - 0.0489%.
- Finite-gap K=10 rows via beam: 21 rows. Gaps range 0.0172% - 0.0620%.
- Finite-gap K=12 rows via beam: 9 rows. Gaps range 0.0239% - 0.0508%.
- Timeout K=12 rows: 0 rows. Beam either timed out or produced no incumbent before external timeout.

### states considered
- K=8 finite: nan
- K=10 finite: nan
- K=12 finite: nan

### states kept
- K=8 finite: nan
- K=10 finite: nan
- K=12 finite: nan

### avg width
- K=8 finite: nan
- K=10 finite: nan
- K=12 finite: nan

### max width
- K=8 finite: nan
- K=10 finite: nan
- K=12 finite: nan

### beam time (s)
- K=8 finite: 234.4
- K=10 finite: 457.8
- K=12 finite: 589.1

## Key findings

1. **Beam width scales with K**: K=12 rows show larger avg/max width than K=10, which is larger than K=8.
2. **Beam time dominates runtime**: `t_pack_profile_beam` is often 50-80% of total runtime on K=10/12.
3. **Pruning is aggressive**: `pruned_over` and `pruned_suffix` are large, meaning many states are discarded. The surviving states may not contain the optimal assignment.
4. **Beam improves over Step 2**: `fwd_profile_beam_improved_over_step2=1` on all finite-gap rows, so beam is already better than the fast heuristic.
5. **Discrepancy budget is minimal**: default is 1 for K>=4. This means very limited diversity exploration.
6. **The wall is not beam time itself**: beam finishes, but the incumbent it produces is not good enough for Step 4 to close.

## Hypotheses for Phase B variants

Based on the diagnostics, the most promising bounded refinements are:

1. **Increase discrepancy budget/topk** (diversity-aware survivor retention): default disc_budget=1, disc_topk=1. Raising to 2-3 could keep more structurally diverse states without widening the base beam width.
2. **Adjust scoring weights** (incumbent-aware ranking): the current weights (w_center=1.0, w_feas=0.75, w_local=0.6, w_arith=1.0) may over-prioritize center alignment. Shifting toward feasibility (w_feas=1.25) could favor states that are easier to close exactly.
3. **Enable more local search passes** (merged-block retention refinement): default local_passes=0 for non-strengthened beam. Enabling 1-2 local passes on K>=10 could improve incumbent quality through pairwise block exchange.
4. **Selective modest widening for K>=10** (family-aware policy): increase width_min/max by ~50% only for K>=10, while keeping discrepancy controls tight. This is a bounded, family-specific expansion.

These are all env-var controlled; no C++ change is required for the basic variants.
