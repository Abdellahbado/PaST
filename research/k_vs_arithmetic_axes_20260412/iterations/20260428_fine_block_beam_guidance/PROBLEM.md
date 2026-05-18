# Problem

PLAN29 showed that replacing the fine recovered block sequence with adjacent coarsened views is not reliable. No coarsening view passed the hard K10 gate, and the likely causes are loss of price-profile fidelity, reduced count-allocation precision, and wider-block pattern growth.

The next hypothesis is narrower:

> Keep the original fine blocks for actual Step-3 beam transitions, but improve survivor selection using auxiliary scoring signals.

This iteration focuses on hard irregular `K=10`, fixed `n=1000`, `lambda=1.3`.

## What not to do

- Do not revive adjacent coarsening as the active beam layer.
- Do not use strict block-local realizability as the main diagnostic or repair.
- Do not restart Step-4 corridor or local-corridor work.
- Do not force exact fixed-block DP at `K=10/12`.
- Do not revive `smart_reconstruct(...)` as the main method.

## Core question

Can we obtain a better Step-3 beam incumbent by choosing better survivor states, while preserving the fine-block sequence?

