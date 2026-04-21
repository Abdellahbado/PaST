# Plan: Configuration Decomposition with DP Pricing — Feasibility Probe

Date: 2026-04-19

## Purpose

Test whether a rate-class configuration master with DP pricing is viable as a new method for fixed-epsilon BPMSTP.

This is a feasibility probe, not yet a paper claim.

## Verified Structural Facts

### Rate-class counts across all 90 instances

Verified from the paper instance files used by our current loader:

- jobs: `temp/paper_exact_repo/instances/Data_p*.txt`
- prices: `temp/paper_exact_repo/instances/Data_c*.txt`
- machine rates: `temp/paper_exact_repo/instances/Data_e*.txt`

| Instance range | Machines | Rate classes | Job types |
|---|---|---|---|
| 1–30 | 3–7 | 2–5 | 3–5 |
| 31–60 | 8–25 | 3–5 | 3–4 |
| 61–90 | 25–40 | 5–6 | 12 |

- Rate classes: min=2, max=6 across all 90 instances
- Job types: min=3, max=12
- Distribution: 38 instances have ≤3 classes, 50 have ≤4, all ≤6

### Configuration space probe (completed)

Tested using naive recursive enumeration in Python:

| Instance | eps | Rate cls | Job types | Configs/class | Total | Time |
|---|---|---|---|---|---|---|
| 1 | 50 | 2 | 4 | 36 | 72 | <1ms |
| 46 | 77 | 3 | 4 | 4,536 | 13,608 | 3ms |
| 46 | 120 | 3 | 4 | 4,536 | 13,608 | 3ms |

Instances 61 (12 job types) and 90 (12 job types) did **not complete** enumeration within ~3 minutes in Python. This strongly suggests the configuration space explodes for 12 job types with large totals.

### Implication

The benchmark naturally splits into two regimes:

1. **Instances 1–60** (3–5 job types): full configuration enumeration is clearly feasible. Configs per class are in the low thousands.
2. **Instances 61–90** (12 job types): full enumeration is likely infeasible. Column generation or restricted enumeration is required.

This split is structurally important for planning. The probe should start in regime 1.

---

## What the Method Would Be

### Fixed-epsilon subproblem view

Given: instance data + makespan cap `epsilon`.
Find: assignment of jobs to machines minimizing total energy cost.

### Decomposition

1. **Group machines by rate class** (≤6 classes, verified)
2. **Define a configuration** for one machine: a type-count vector `(c_1, ..., c_K)` with `Σ c_k × L_k ≤ epsilon` and `0 ≤ c_k ≤ n_k`
3. **Price each configuration** using the exact single-machine DP:
   - `cost(config, rate, prices, epsilon) = rate × solve_sparse_dp(lengths, counts, prefix, epsilon)`
4. **Formulate a set-partitioning master**:
   - select one configuration per machine
   - all machines in the same class use configurations from the same pool
   - coverage: `Σ (machines using config q) × c_k(q) = n_k` for each job type k
   - minimize: `Σ cost(selected configs)`

### Master IP structure

Variables: `x_{c,q}` = number of machines in class c using configuration q, integer ≥ 0.

Constraints:
- `Σ_q x_{c,q} = m_c` (each class c has exactly m_c machines)
- `Σ_{c,q} x_{c,q} × c_k(q) = n_k` (all jobs of type k assigned)
- `x_{c,q} ≥ 0`, integer

Objective: minimize `Σ_{c,q} x_{c,q} × cost(c, q)`

This is a small IP when the configuration pool is small (regime 1). For regime 2, column generation is needed.

### Where the DP is pivotal

- The DP computes `cost(c, q)` exactly for every configuration
- In column generation, the DP pricing function (`solve_pricing_dp`) generates new columns
- No other component can replace the DP in this role — it is the subproblem solver

---

## Experiment Plan

### Experiment 1: Full enumeration on instance 46 (regime 1)

**Goal**: Establish that the configuration master + DP pricing produces correct and competitive TEC on a small instance.

**Steps**:
1. Enumerate all feasible configs for each rate class (expect ~4,500 per class, ~13,600 total)
2. Price each config using `solve_sparse_dp` in C++
3. Formulate and solve the master IP (e.g., using OR-Tools CP-SAT or a simple IP solver)
4. Compare the resulting TEC against:
   - our exact CP-SAT result (known: 103 at eps=77)
   - paper EHS result at the same epsilon (`103` at `eps=77`; also note EHS reaches `103` in 6/10 runs at `eps=73`)
   - the same-epsilon one-shot baselines:
     - `greedy_dp = 118` at `46/77`
     - `greedy_dp_local_search_relocate_only = 109` at `46/77`

**Success criterion**: TEC matches the exact optimum. This would confirm the method is exact for this instance.

**Expected outcome**: With 13,608 configs and 3 classes, the master IP should be small. Actual pricing time must be measured rather than assumed.

### Experiment 2: Selected epsilon values on instance 46

**Goal**: Verify that the method remains correct away from a single epsilon, without committing yet to a full-front run.

**Steps**:
1. Run Experiment 1 at a small set of diagnostic epsilon values, for example `77`, `73`, and one looser point
2. Compare the resulting Pareto front against:
   - the paper's reference near-optimal front
   - the paper's EHS front

**Success criterion**: The configuration method matches exact or reference values at the tested epsilon points.

### Experiment 3: Test on a 12-type instance with restricted enumeration

**Goal**: Determine whether the method extends to regime 2 (instances 61–90) with column generation.

**Steps**:
1. Pick instance 61, epsilon=350
2. Attempt restricted enumeration (limit max jobs per machine, or limit per-type counts)
3. If enumeration is still too large, implement column generation:
   - Start with a small set of initial configs (e.g., from greedy assignment)
   - Solve the LP relaxation of the master
   - Use `solve_pricing_dp` to find the most negative reduced-cost configuration
   - Add it to the master and re-solve
   - Repeat until no negative reduced-cost column exists
4. Round the LP solution or solve the restricted master IP

**Success criterion**: Produce a feasible master/pricing loop and quantify the gap against:
- paper EHS at the same epsilon
- our exact reference at the same epsilon when available

This experiment is about viability, not yet about beating EHS.

### Experiment 4 (optional): Compare against EHS on a wider instance set

Only if Experiments 1–3 succeed. Run on instances 31–60 (regime 1, medium size) to build a broader comparison base.

---

## Existing Infrastructure to Reuse

| Component | Location | Role |
|---|---|---|
| `solve_sparse_dp` | `solvers/cpp/dp_solver.hpp:189` | Price each configuration exactly |
| `solve_pricing_dp` | `solvers/cpp/dp_solver.hpp:196` | Column generation subproblem |
| Rate-class aggregation | `solvers/parallel_f2_cp_sat.py` | Rate-class identification logic |
| Instance loader | `solvers/cpp/parallel_heuristic_compare.cpp` | C++ parser for paper instances |
| CP-SAT IP solver | OR-Tools (Python) | Candidate master IP solver for regime 1 |

## What Needs to Be Built

1. **Configuration enumerator** (C++ or Python): enumerate feasible type-count vectors for given capacity
2. **Pricing harness**: call `solve_sparse_dp` for each configuration (C++ preferred)
3. **Master IP formulation**: set-partitioning IP (Python + OR-Tools CP-SAT or SCIP)
4. **Column generation loop** (for regime 2): LP relaxation + pricing + iteration
5. **Comparison harness**: load EHS results, compute gaps

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Config space explosion on regime 2 | Cannot enumerate | Column generation with `solve_pricing_dp` |
| Column generation doesn't converge | Slow or no solution | Use greedy-seeded restricted master; fall back to heuristic |
| LP-IP gap in master | Suboptimal rounding | Use branch-and-price (heavier) or best-round heuristic |
| DP pricing too slow per config | Bottleneck at scale | Measure on regime 1 first; batch and cache where possible |

## What This Plan Does NOT Claim

- This is not yet a paper contribution
- "Novel for BPMSTP" is a hypothesis to verify against literature
- The method may not beat EHS on all instances
- The configuration approach may require different treatment for regime 1 vs regime 2

## Decision Point

After Experiment 1:
- If TEC = exact: method is correct, proceed to Experiments 2–3
- If TEC > exact: debug
- If pricing takes too long: re-evaluate

After Experiment 3:
- If column generation works and TEC is competitive: this is a viable paper direction
- If column generation stalls or TEC is poor: reconsider architecture

## Scope guard

This probe is intentionally staged:

1. prove the regime-1 exact master works on a small instance,
2. test a few additional epsilon points on the same instance,
3. only then attempt a 12-type regime-2 probe.

Do not jump directly to a full-front or full-benchmark claim from this plan alone.
