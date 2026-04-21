# Phase J Insert Screening Redesign Design

Date: 2026-04-20

Why Phase H failed mechanistically:

- `vnd_exact_dp` accepted no neighborhood move at `61/347`
- only `6` exact local-search evaluations were executed, too narrow for the candidate space

What Phase I proved:

- under no-screen exact evaluation, improving `insert_inter` moves exist from the same start and improve `6944 -> 6920`

Analytical ideas considered:

1. **A - source-target dual pressure ranking** (source cost/gap + target load/headroom + job-size interaction)
2. **B - top-k per source machine** (preserve diversity instead of one global shortlist)
3. **C - two-stage screening** (permissive stage-1 then stricter rerank)
4. **D - incumbent-gap-aware source priority** (focus on high exact-cost / high exact-minus-LB machines)

Selected now:

- `vnd_exact_dp_insert_rank_v1`: A + D
- `vnd_exact_dp_insert_rank_diverse`: B + C + D

Why selected:

- directly targets observed failure (over-aggressive global pruning) while staying analytical, bounded, and insert-focused
- gives a controlled comparison between a simple dual-pressure rank and a diversity-preserving two-stage rank

Enough-signal criterion to continue:

- at least one redesign beats `6944` and approaches/matches `6920` under bounded exact-DP budget.
