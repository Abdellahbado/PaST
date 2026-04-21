# Literature

## Core paper

### Gaggero, Paolucci, Ronco (2023)

Main artifact in repo:

- `Papers/Exact and heuristic.txt`

What matters for this thread:

- Formulation 2 (F2) is the paper's compact exact model for the fixed-horizon problem
- the exact algorithm uses the `epsilon`-constraint style loop over downsized instances `D(Khat)`
- the heuristic EHS has three relevant components:
  - A-SGH: assignment with history across decreasing `Khat`
  - R-ES: cross-machine neighborhood improvement
  - ESR: exact single-machine rescheduler

Important exact wording from the paper:

- ESR minimizes the machine energy cost **while preserving the original processing sequence**
- this makes ESR a sequence-preserving retiming DP, not a full single-machine schedule optimizer

Why this matters:

- our single-machine DP is much stronger than ESR at the machine level
- but the paper's heuristic strength also comes from A-SGH and R-ES, not only ESR

## Practical conclusion for this thread

The right comparison is not:

- DP vs CP-SAT exact proof

The right comparison is:

- paper-style heuristic machine optimization vs DP-based machine optimization

This thread therefore starts with a narrow question:

- if assignment is held fixed, does replacing ESR with our DP materially improve TEC?

Only if the answer is yes do we proceed to DP-guided assignment experiments.
