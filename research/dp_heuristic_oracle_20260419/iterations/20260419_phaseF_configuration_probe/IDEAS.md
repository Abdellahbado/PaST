# Ideas

Chosen method for this probe:

- full configuration enumeration for one machine under fixed `epsilon`
- exact pricing of each configuration per rate class using `solve_sparse_dp`
- integer class-count master with exact machine-count and type-coverage equalities

Implementation split selected:

1. C++ (`solvers/cpp/phaseF_config_probe.cpp`)
   - instance loading
   - type extraction and rate-class grouping
   - full configuration enumeration
   - per-class DP pricing
   - artifact export (CSV + JSON)

2. Python (`solvers/phaseF_config_master_probe.py`)
   - OR-Tools CP-SAT integer master
   - solution extraction and summary artifacts

Why this split:

- keeps DP pricing near existing C++ core,
- keeps master model concise and auditable in Python,
- avoids any DP-core rewrite.

Column-generation note:

- `solve_pricing_dp` was inspected but not promoted to true CG in this task; reduced-cost mapping details are not yet fully validated for exact branch-and-price behavior.
