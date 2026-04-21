# Blockers

## Active blockers

1. It is not yet proven that warm-start continuity, rather than another hidden factor, is the missing lever behind the gap to stronger epsilon-sweep heuristics.
2. The current repo does not yet expose a clean harness for comparing warm-start and fresh-start VND on the same epsilon ladder with identical accounting.

## Diagnostic discipline

- Do not claim warm-start benefit before a paired warm-start vs fresh-start table exists.
- Do not broaden to multiple instances before instance `61` gives a clear directional result.
- Do not mix in learning changes, new neighborhoods, or assignment-construction changes in this branch.
