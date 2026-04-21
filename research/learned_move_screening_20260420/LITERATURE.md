# Literature

## Core heuristic context

### Gaggero, Paolucci, Ronco (2023)

Artifact in repo:

- `Papers/Exact and heuristic.txt`

Relevant takeaways:

- EHS is strong because of assignment history and neighborhood search, not only machine-level retiming
- ESR is sequence-preserving, while our single-machine DP optimizes over the whole job multiset

## Learning-for-optimization inspiration

### Hijazi, Ozaltin, Uzsoy (2024)

Artifact in repo:

- `Papers/arXiv 2410.15601.txt`

Relevant takeaways:

- learning is inserted at one expensive subproblem location
- the exact dynamic program remains as fallback / verifier
- the learning component is trained offline and used to accelerate search, not replace correctness

What transfers to this thread:

- the useful pattern is:
  - exact optimizer + learned shortcut + exact fallback
- the direct analogue here is:
  - learned move ranking + exact DP verification of selected moves

### ADP paper

Artifact in repo:

- `Papers/ADP.txt`

Current decision for this thread:

- do not pursue ADP / value-function approximation first
- it is less aligned with the current bottleneck than move ranking

## Practical conclusion for this thread

The first learning target should be:

- supervised ranking or screening of `insert_inter` moves

Not:

- RL
- transformers
- end-to-end schedule prediction
- full assignment prediction

The baseline method to augment is the handcrafted insert-focused exact-DP heuristic from:

- `research/dp_heuristic_oracle_20260419/phaseJ_insert_screening_redesign_results.md`
- `research/dp_heuristic_oracle_20260419/phaseK_insert_efficiency_pass_results.md`
