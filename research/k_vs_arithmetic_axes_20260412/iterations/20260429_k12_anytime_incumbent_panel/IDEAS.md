# Ideas

## 1. Initial UB safety layer

Use existing sequence-based UB tools before long hard-K work:

- `compute_initial_ub`
- `solve_fixed_sequence`
- `local_search_ub`

The solver should never lose this incumbent. If the beam times out, return the initial UB.

## 2. Anytime beam checkpointing

If profile repair beam constructs any complete candidate, store it immediately. Do not wait until all beam work or Step 4 finishes.

## 3. Family-aware K12 beam policy

Use PLAN31 evidence:

- hardA-like families: try `uniform_mult2`
- hardB-like families: try `ambig_scoreband_mult2` or `late_ambig`

Judge by feasible UB coverage and gap, not exactness.

## 4. Fixed-K arithmetic panel

Test several K=12 families:

- easy unit-contiguous
- dense without 1
- structured even
- existing hardA
- existing hardB
- sparse large-gap

This separates K effects from arithmetic effects.

