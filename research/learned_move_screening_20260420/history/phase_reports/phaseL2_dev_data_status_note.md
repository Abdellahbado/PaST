# Development Data Status Note

Date: 2026-04-20

- Stage L1 and Stage L1.5 datasets in this thread are designated development-only.
- They are valid for feasibility probing (feature/target sanity, ranking learnability checks).
- They are not valid for final benchmark generalization claims, because rows are collected from benchmark contexts and from a policy-induced candidate stream.

Current risks to keep explicit in all L2 reporting:

1. potential benchmark leakage if reused as final evaluation data
2. dense-labeling selection bias in candidate distribution
3. context imbalance (`64/79` contributes heavily)
4. target-semantics ambiguity risk (`improving` vs `accepted`) even if currently aligned

Therefore:

- Stage L2 results are interpreted only as development-stage learnability evidence.
- Any final claim requires a new data protocol with non-benchmark training corpus and strict hold-out design.
