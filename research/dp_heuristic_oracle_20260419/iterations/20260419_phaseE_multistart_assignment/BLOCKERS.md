# Blockers

Current blockers:

- runtime overhead scales with number of starts; current prototype is much slower than one-shot variants on instance `61`.
- assignment randomization is shallow (single RCL policy, fixed starts count); may leave quality on table.

Mitigation candidates:

- adaptive stop when no better TEC appears after several starts,
- tune starts and RCL size per instance scale,
- optionally warm-start from prior best assignment across nearby epsilons.
