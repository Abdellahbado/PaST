# Blockers

Current blockers:

- repair step can become infeasible on tight transitions (`64: 78 -> 77`, `90: 83 -> 82`)
- displaced-job removal policy is still simplistic and may damage recoverability
- repair candidate scoring is machine-local and may miss sequence/interaction effects needed for robust continuity

Impact:

- prevents complete history chains on two of four tested instances
- limits any claim of broad superiority over one-shot relocate baseline

Near-term mitigation:

- improve displacement policy (avoid over-removal / choose removal sets closer to exact overload)
- add bounded fallback reinsertion attempt order when first-choice reinsertion fails
