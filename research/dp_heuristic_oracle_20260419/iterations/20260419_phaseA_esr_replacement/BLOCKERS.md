# Blockers

Known risks before implementation:

1. Replacing ESR with exact DP may improve machine schedules only marginally if the assignment baseline already induces near-optimal machine sequences.
2. Even if machine-level improvement is clear, that may not be enough to beat the paper's full heuristic, because A-SGH and R-ES also contribute real value.
3. Assignment-conditioned lower bounds are diagnostically useful, but they are not global optimality certificates.
