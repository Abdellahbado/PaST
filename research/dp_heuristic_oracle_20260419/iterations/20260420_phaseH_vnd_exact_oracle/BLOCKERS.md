# Blockers

Observed blockers in first bounded pass:

- VND neighborhoods did not find an improving accepted move at `61/347` under current screening/evaluation budget.
- Inter-neighborhood candidate counts are large, and strict exact-DP acceptance makes improvements sparse.
- Current intra-machine swap implementation changes sequence but not assignment; benefit may be limited under DP re-optimization for this instance.

Implication:

- branch has signal via final TEC quality, but move-level evidence is weak.
- next bounded step should target move acceptance quality (not just more candidate enumeration).
