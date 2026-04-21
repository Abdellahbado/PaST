# Phase H VND-Exact-Oracle Design

Date: 2026-04-20

## 1) VND parts borrowed from the paper

- fixed-epsilon local-search setting (single `epsilon` subproblem)
- neighborhood family and order: `swap_intra -> swap_inter -> insert_inter`
- first-improvement VND flow (restart from first neighborhood after acceptance)

## 2) What is not borrowed yet

- no epsilon oscillation
- no full EOA loop / perturbation cycles
- no Pareto-front generation
- no SGS-ES replication or full paper metaheuristic stack

## 3) How exact DP is used

- exact per-machine TEC evaluation for touched machine(s) in candidate acceptance
- exact DP caching by machine multiset + rate to reduce repeated calls
- no changes to DP core implementation

## 4) Initial solution(s)

- bounded deterministic multistart: randomized LPT assignment with fixed seeds (small RCL), best-start kept
- if all randomized starts fail, fallback to deterministic greedy LPT seed

## 5) Neighborhoods in this bounded pass

1. `swap_intra`: swap two jobs inside one machine sequence
2. `swap_inter`: swap one job between two machines
3. `insert_inter`: move one job from one machine to another

## 6) Screening / caching

- feasibility filter first (`load <= epsilon`)
- lower-bound screening (slot-based LB delta) for inter-machine candidate ranking
- shortlist + capped exact-DP evaluations per neighborhood pass
- exact-DP cache hit/miss counters recorded

## 7) Continue signal criterion

Continue only if at least one holds at `61/347`:

1. material TEC gain vs `greedy_dp_local_search_relocate_only`
2. meaningfully reduced gap to paper/reference
3. clear move-level evidence that richer neighborhoods + exact DP improve quality
