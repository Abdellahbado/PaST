# Problem

Phase E question:

- should the next serious branch prioritize assignment diversification (multi-start randomized construction) rather than immediate full history-repair hardening?

Working hypothesis:

- a strong share of the remaining quality gap comes from assignment structure, and bounded randomized starts can recover better basins before relocate-only DP cleanup.

Scope of this branch:

- keep existing DP evaluator and relocate-only local search,
- add bounded randomized assignment starts,
- compare to one-shot baselines and paper EHS references at same epsilon.
