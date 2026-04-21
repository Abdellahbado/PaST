# Summary

Phase K is a bounded final non-ML efficiency pass over Phase J insert-screening redesign at `61/347`.

What changed:

- Implemented two insert-focused variants in `parallel_heuristic_compare.cpp`:
  - `vnd_exact_dp_insert_rank_diverse_trimmed`
  - `vnd_exact_dp_insert_rank_diverse_budgeted`
- Changes are purely analytical/bounded screening refinements (pool caps, per-target quotas, staged budget), with exact DP acceptance unchanged.

Observed outcome:

- Both variants keep best TEC at `6884` (tie with Phase J best).
- Screening volume reduced materially:
  - trimmed: `29160 -> 19416`
  - budgeted: `29160 -> 22440`
- Exact LS evals reduced (`17 -> 15`).
- Runtime/RSS not materially improved on this point.

Decision signal:

- This pass validates cleaner and tighter screening structure while preserving quality.
- Non-ML headroom now looks limited; the recommended next step is pivot to learned move screening/ranking on top of the current insert-focused exact-DP base.
