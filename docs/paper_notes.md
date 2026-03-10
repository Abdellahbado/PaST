# Paper Notes: arXiv 2506.10405
# "Green Scheduling with Time-of-Use Tariffs and Machine States"
# Benedikt, Módos, Šůcha (2025)

## Problem Definition
- **1||TEC**: single machine, non-preemptive, minimize Total Energy Cost
- Machine has states {off, proc, idle} (NOSBY) or {off, proc, idle, sby1, sby2} (TWOSBY)
- Transitions have duration & power consumption
- Energy prices vary per time slot (TOU tariffs)
- Jobs: each has processing time pj, must run contiguously

## Machine Configs (from paper + GitHub data)

### NOSBY (3-state)
- off→proc: T=2, P=5
- proc→off: T=1, P=1
- proc→idle: T=0, P=0 (instantaneous)
- idle→proc: T=0, P=0 (instantaneous)
- proc self-loop: T=1, P=4
- idle self-loop: T=1, P=2
- off self-loop: T=1, P=0

### TWOSBY (5-state)
- OffOnTime=[4,3,2], OffOnPower=[15,13,12]
- OnOffTime=[1,1,1], OnOffPower=[2,2,2]
- OnPower=10, IdlePower=8, OffPower=[0,2,4]
- 3 "off" levels: full-off (P=0), standby1 (P=2), standby2 (P=4)
- Faster startup from higher standby levels, at cost of standby power

## Experimental Sections

### §5.1 — Table 1 (24 instances)
- 12 NOSBY + 12 TWOSBY
- n ∈ {150, 170, 190}, λ ∈ {1.3, 1.6, 1.9, 2.2}
- pj ~ U[1,5], costs ~ U[1,10], seed=42
- Horizon h = λ × (T(off,proc) + Σpj + T(proc,off))
- All solved optimally by B&B-SPACES
- NOSBY B&B-SPACES: avg 1.3s algo + 7.9s preprocessing = 9.2s total
- **Paper explicitly says U[1,5] instances are EASY** (§5.1, line 2031-2033)

### §5.2 — Table 2 (560 instances)
- TWOSBY only
- n ∈ {50, 100, 150, 200}
- 7 processing time groups: {1-10}, {1,2,3,5,7}, {2,4,6,8,10}, {2,4}, {3,5,6,7}, {3,7}, {8,10}
- **Real OTE Czech electricity prices** (not synthetic U[1,10])
- 20 instances per (n, proc_group) combination
- B&B-SPACES fails on 10 instances — all in {8,10} group, n∈{150,200}
- Key insight: "{8,10} are two of the larger processing times; thus, the relaxation procedure has *larger relaxation gap on average*"
- These instances are **NOT in the GitHub repo**

### §5.3 — Figure 9 (480 instances)
- Additional harder processing time groups: {9,10}, {8,9,10}, {8,9}, {7,8,9,10}, {7,8}, {10}, {7,9}, {1,2,10}
- n ∈ {100, 150, 200}, 20 instances each
- Tests the boundary of where B&B-SPACES fails
- Also **NOT in the GitHub repo**

## Known Optimal Costs — Table 1 NOSBY
| idx | n   | λ   | Optimal |
|-----|-----|-----|---------|
| 0   | 150 | 1.3 | 8582    |
| 1   | 150 | 1.6 | 8409    |
| 2   | 150 | 1.9 | 8132    |
| 3   | 150 | 2.2 | 8078    |
| 4   | 170 | 1.3 | 10068   |
| 5   | 170 | 1.6 | 9820    |
| 6   | 170 | 1.9 | 9637    |
| 7   | 170 | 2.2 | 9620    |
| 8   | 190 | 1.3 | 12008   |
| 9   | 190 | 1.6 | 11758   |
| 10  | 190 | 1.9 | 11611   |
| 11  | 190 | 2.2 | 11465   |

## Paper's B&B-SPACES Times (seconds)
### NOSBY: algo + P-P preprocessing
| idx | BB algo | P-P    | Total |
|-----|---------|--------|-------|
| 0   | 0.6     | 1.0    | 1.6   |
| 1   | 0.9     | 2.9    | 3.8   |
| 2   | 0.8     | 5.5    | 6.3   |
| 3   | 1.1     | 9.1    | 10.2  |
| 4   | 0.7     | 2.3    | 3.0   |
| 5   | 0.8     | 4.6    | 5.4   |
| 6   | 1.7     | 9.3    | 11.0  |
| 7   | 2.4     | 13.4   | 15.8  |
| 8   | 0.7     | 3.9    | 4.6   |
| 9   | 1.1     | 6.9    | 8.0   |
| 10  | 2.4     | 13.3   | 15.7  |
| 11  | 2.0     | 22.7   | 24.7  |

## Algorithms

### SPACES Preprocessing (paper's C#, our: C++ banded)
- Computes c_star[t_end, t_start] = optimal switching cost between proc intervals
- Also c_start[t] = cost from off to proc at t, c_end[t] = cost from proc to off at t
- Complexity: O(h² × |S| × (|S| + log h + log |S|))
- Our banded version: O(h × max_gap × |S|²), where max_gap ≈ 58 for NOSBY/U[1,10]

### B&B-SPACES (Branch & Bound)
- Bin-packing structure: partition jobs into proc intervals, then pack
- Variable ordering by processing time length type
- Branching: try placing each interval, with dominance pruning

### Our Approach (Relaxed DP)
- State = (t_end, remaining_work): drop per-type constraint, track only total work
- Forward + backward + two-class LB refinements
- Bin-packing UB from relaxed schedule (FFD)
- Heuristic UB + local search

## Paper Hardware
- Intel Xeon Silver 4110 @ 2.10 GHz, 100 GB RAM
- B&B-SPACES: C++, SPACES preprocessing: C#, ILP: Python + Gurobi

## GitHub Data: CTU-IIG/EnergyStatesAndCostsSchedulingData
- `datasets/benedikt2020a_large_nosby/` — 12 instances (Table 1 NOSBY)
- `datasets/benedikt2020a_large_twosby/` — 12 instances (Table 1 TWOSBY)
- `datasets/benedikt2020a_medium_nosby/` — 12 instances (n=30,60,90)
- `datasets/benedikt2020a_medium_twosby/` — 12 instances (n=30,60,90)
- `datasets/aghelinejad2017a_1/` — 12 instances (n=5-60, some infeasible under NOSBY config)
- `results/` — ILP-REF and ILP-SPACES solutions (NOT B&B-SPACES)
- Instance format: JSON with Jobs, EnergyCosts, OffOnTime, OnOffTime, etc.
