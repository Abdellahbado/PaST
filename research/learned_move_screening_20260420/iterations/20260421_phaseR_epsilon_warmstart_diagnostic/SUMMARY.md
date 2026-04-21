# Summary

Phase Q showed that synthetic-only learned move ranking produces only modest gains over the strongest handcrafted screening baseline, so the next branch pivots back to the main algorithmic hypothesis.

Current branch:

- test epsilon warm-starting on benchmark instance `61`,
- compare warm-started VND against fresh-start VND on the same descending epsilon ladder,
- measure whether cumulative cross-epsilon continuity improves TEC enough to justify a full warm-start sweep branch.

Grounding:

- Phase I proved improving insert-inter moves exist on `paper_61` at `epsilon=347` (`6944 -> 6920`).
- Phase J/K improved the same instance to `6884` with `vnd_exact_dp_insert_rank_diverse`, but the overall heuristic still trails stronger literature baselines.
- Phase Q learned ranking gains were real but too small to justify continuing that line as the main paper contribution.

Success condition for this branch:

- warm-started VND is consistently equal or better than fresh-start VND across the chosen epsilon ladder, with a measurable cumulative advantage near the hard region around `347`.

Exact next step:

- implement and run a bounded warm-start vs fresh-start diagnostic on instance `61`, then decide whether to scale to a full epsilon-sweep branch.
