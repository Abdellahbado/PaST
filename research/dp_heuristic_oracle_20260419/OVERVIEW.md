# Overview

This research thread studies a heuristic use of the single-machine DP pipeline for the BPMSTP benchmark in:

- `temp/paper_exact_repo/instances`

The purpose of this thread is not to replace the exact compact F2 / CP-SAT solver. The purpose is to test whether the DP can play a central role in a **heuristic** method that competes with the paper's EHS heuristic on solution quality.

Problem view:

- objective 1: minimize makespan `Cmax`
- objective 2: minimize total energy cost
- operational view for this thread:
  - fix a makespan cap `epsilon`
  - build a heuristic schedule for that `epsilon`
  - compare heuristic quality against:
    - the paper-style heuristic baseline
    - exact fixed-`epsilon` values already available from the CP-SAT branch on a subset

Stable facts at thread start:

- the DP-guided exact proof branch was not competitive as a primary exact line
- the compact fixed-`epsilon` CP-SAT solver remains the strongest exact production method
- the 2023 paper's heuristic EHS is strong because of:
  - assignment history (A-SGH)
  - cross-machine local search (R-ES)
  - per-machine retiming (ESR)
- ESR is weaker than our single-machine DP because ESR preserves the job processing order on each machine

Thread hypothesis:

- replacing or strengthening the **machine-level heuristic oracle** is more realistic than trying to use DP as the exact global proof engine
- the first thing to test is whether replacing ESR with our DP improves fixed-`epsilon` heuristic quality

Current active branch:

- `iterations/20260419_phaseA_esr_replacement/`
