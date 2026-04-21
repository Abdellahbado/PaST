# Summary

This iteration starts from a focused post-Phase-D design-and-prototype pass.

Decision made:

- choose multi-start randomized assignment plus relocate-only local search as the next serious branch.

What was delivered in the pass:

- ranked design memo across three required directions
- one working prototype (`greedy_dp_local_search_relocate_multistart`)
- one sanity experiment on `61/345` showing strong gain vs one-shot baselines

Key evidence point:

- `61/345` TEC improved from `7085` (one-shot relocate) to `6960` (multistart prototype), while paper EHS is `6723`.

Immediate next steps:

- reduce runtime cost of multistart (adaptive stopping, lighter per-start budget)
- validate on additional required rows before promoting to default mode
