# Ideas

Chosen probe architecture:

- restricted column pool (not full enumeration)
- LP restricted master for duals
- pricing-style search for negative reduced-cost columns
- bounded iteration loop
- integer restricted master solve for feasible TEC

Initial pool strategy:

1. zero pattern
2. machine multisets from greedy assignment at target epsilon
3. mono-type columns

Pricing mapping used:

- reduced cost `cost_c(a) - alpha_c - sum_k beta_k a_k`
- `solve_pricing_dp` with rewards=`beta`, sigma=`alpha_c` for non-empty columns

Caveat retained:

- empty-column handling is outside pricing and fixed in master pool.
