# Presentation Results Summary

This document summarizes the current experimental results for the paper job-size
groups in a presentation-friendly form. It is intentionally neutral: accepted
results, in-progress/additive results, and unresolved blockers are separated.

## PLAN14 update note (2026-04-22)

A focused dense-unit large-`K` diagnosis/recovery pass (PLAN14) was completed on:

- `g12345678910 = {1,2,3,4,5,6,7,8,9,10}`.

Key outcome:

- baseline still times out at `n=5000` in tested windows,
- but additive dense-unit Step-2 fast-path closes `n=5000` exactly on seeds `0/1`
  (Step 2, `UB=LB`),
- indicating the prior wall is mainly generic pipeline/runtime behavior around
  Step-2 reach/return, not intrinsic Step-2 hardness on this family.

## Scope and Data Sources

The experiments summarized here use the paper job-size groups at `lambda=1.3`.
Unless noted otherwise, rows come from the current accepted pipeline:

- workflow: `ablation-stdin step1_exact_guided`
- binary: `solvers/cpp/build/stateful_compare`
- primary ledger: `csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- PLAN11 extension ledger: `csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`
- K=4 generator evidence: `csv/plan10/PLAN10_k4_generator_compare.csv`
- current provenance registry: `csv/CURRENT_METHOD_PROVENANCE.csv`
- HPC/code provenance map: `PAPER_HPC_REPRODUCIBILITY_MAP.md`

Important caveat:

- Local timings are laptop timings. Use this document for presentation
  structure and method provenance; regenerate final paper runtimes on HPC.
- `g37 = {3,7}` is now corrected: the old ledger only reached `n=600`
  because later rows were misrouted. PLAN13 reroute evidence closes tested rows
  through `n=5000` using Step-3 exact profile realization.

## Pipeline Overview

The current solver pipeline is a staged exact-guided workflow.

| Step | Role | Typical interpretation in results |
|---|---|---|
| Step 1 | Recover a relaxed profile / lower-bound structure. | Establishes the profile that later realization steps try to certify. |
| Step 2 | Fast direct realization using simple packing, most often `ffd`. | If `UB=LB`, the row closes immediately without Step 3. Many arithmetic-friendly groups close here. |
| Step 3 | Block/profile repair and exact profile realization. | Main nontrivial closure layer. Different Step-3 methods are used depending on the family structure. |
| Step 4 | Global exact fallback / diagnostic exact mode. | Used as a fallback; current paper-facing successes should not rely on failed Step-4 entries as evidence of closure. |

## Main Step-3 Methods Used

Step 3 is not one method. It is a family of realization/repair paths selected
by group structure and solver policy.

| Step-3 method label | Main function path | Where it matters | Explanation |
|---|---|---|---|
| `profile_realization_dp_exact` | `solve_relaxed_dp_with_binpack(...)` -> `profile_realization_dp_exact` candidate path in `stateful_dp_solver.cpp` | K=2 rows such as `g810` and corrected `g37` | This is the intended exact profile-realization route for small-`K` cases. It tries to certify the relaxed profile directly rather than using the global Step-4 fallback. |
| `block_repair_energy_core` | `compute_relaxed_completion_table(...)` -> `generate_energy_core_patterns(...)` -> `block_repair_energy_core_ub(...)` in `stateful_dp_solver.cpp` | Hard K=4 rows, especially `g3567` | Generates a structured pattern pool around the energy core and uses direct completion to close the profile. The accepted K=4 policy uses the DP-style generator and disables K=4 signature dedup by default. |
| `block_repair_feasible_beam_ub` / profile repair beam | `block_repair_feasible_beam_ub(...)`, `block_repair_profile_repair_beam_ub(...)` | Diagnostic / historical selector contexts | Beam-style repair is useful diagnostically, but the current accepted frontiers are not mainly claimed through this path. |
| `block_repair_exact_level2_ub` | `block_repair_exact_level2_ub(...)` | Archive-only exact-L2 diagnostics | This is retained as historical evidence, not the current mainline policy. |

Workflow entry and top-level implementation anchors:

- `stateful_compare.cpp`: `ablation-stdin step1_exact_guided`
- `stateful_dp_solver.cpp`: `solve_relaxed_dp_with_binpack(...)`
- K=4 energy-core path: `compute_relaxed_completion_table(...)`,
  `generate_energy_core_patterns(...)`, `block_repair_energy_core_ub(...)`
- K=2 exact profile path: `profile_realization_dp_exact` candidate path

## Accepted Paper-Group Frontiers

The table below gives the current accepted frontier per paper group. Runtime is
reported at the accepted frontier row. When both seeds were available at the
frontier, both runtimes are shown.

| Family | K | Current accepted exact frontier | Runtime at frontier | Deciding step | Method that closed the frontier | Next observed regime |
|---|---:|---:|---|---|---|---|
| `g24 = {2,4}` | 2 | `n=10000` | seed 0: `697.3714s`; seed 1: `591.2132s` | Step 2 | `ffd` | No break observed through `n=10000`. |
| `g37 = {3,7}` | 2 | corrected evidence: `n=5000` | seed 0: `282.5078s`; seed 1: `237.3366s` | Step 3 | `profile_realization_dp_exact` | Old ledger beyond `n=600` was misrouted; PLAN13 reroute closes tested rows through `n=5000`. |
| `g810 = {8,10}` | 2 | `n=5000` | seed 0: `361.8299s` | Step 3 | `profile_realization_dp_exact` | `std::length_error` crash from `n=6000`. |
| `g3567 = {3,5,6,7}` | 4 | `n=6000` | seed 0: `706.2664s`; seed 1: `583.6889s` | Step 3 | `block_repair_energy_core` | Timeout/kill at `n=7000`; `std::length_error` at `n=8000`. |
| `g12357 = {1,2,3,5,7}` | 5 | `n=8000` | seed 0: `1172.9227s`; seed 1: `1045.7509s` | Step 2 | `ffd` | Timeout at `n=10000`. |
| `g246810 = {2,4,6,8,10}` | 5 | `n=6000` | seed 0: `1017.5724s`; seed 1: `974.8643s` | Step 2 | `ffd` | `std::length_error` from `n=7000`. |
| `g12345678910 = {1,...,10}` | 10 | baseline accepted: `n=3500` | seed 0: `1002.4226s` (accepted ledger) | Step 2 | `ffd` | **PLAN14 additive**: exact at `n=5000` (seeds `0/1`) via dense-unit fast-path Step 2; baseline path remains timeout-limited at `n=5000` in tested windows. |

## Interpretation by Group

### Easy Step-2 Closures

Several groups close through Step 2 using `ffd`. These rows are important
because they show that, for some arithmetic structures, the relaxed profile and
simple realization are already aligned.

| Group | Accepted frontier | Interpretation |
|---|---:|---|
| `g24 = {2,4}` | `n=10000` | Very stable Step-2 behavior; no break through tested range. |
| `g12357 = {1,2,3,5,7}` | `n=8000` | Dense/has-one arithmetic remains favorable; timeout begins at larger `n`. |
| `g246810 = {2,4,6,8,10}` | `n=6000` | Step-2 closes frontier, but larger rows hit robustness failure rather than finite-gap behavior. |
| `g12345678910 = {1,...,10}` | baseline accepted: `n=3500`; PLAN14 additive: exact `n=5000` on seeds `0/1` | Baseline generic path still times out at `n=5000`, but additive dense-unit fast-path recovers exact Step-2 closure, supporting “easy arithmetic family trapped in expensive generic pipeline” interpretation. |

### Step-3 Exact Profile Realization

`g37 = {3,7}` and `g810 = {8,10}` are the clean K=2 examples of Step-3 exact
profile realization:

- corrected `g37` tested frontier: `n=5000`, seeds `0/1`
- accepted `g810` frontier: `n=5000`
- deciding step: Step 3
- method: `profile_realization_dp_exact`
- `g37` reroute runtime at `n=5000`: seed 0 `282.5078s`, seed 1 `237.3366s`
- `g810` runtime at accepted frontier: `361.8299s` for seed 0
- next observed `g810` blocker: `std::length_error` from `n=6000`

This result supports the small-`K` exact-profile realization component of the
pipeline.

### Step-3 Energy-Core Repair

`g3567 = {3,5,6,7}` is the main accepted K=4 Step-3 result:

- accepted frontier: `n=6000`
- deciding step: Step 3
- method: `block_repair_energy_core`
- runtime at accepted frontier:
  - seed 0: `706.2664s`
  - seed 1: `583.6889s`
- next observed blocker:
  - timeout/kill at `n=7000`
  - `std::length_error` at `n=8000`

The K=4 improvement came from changing the generator policy:

- DP-style pattern generator active for K=4
- K=4 signature-dedup disabled by default

Measured PLAN10 effect on required rows:

| Scope | Baseline mean runtime | Final K=4 generator mean runtime | Change |
|---|---:|---:|---:|
| hard `g3567` rows | `1083.240s` | `250.839s` | `-76.8%` |
| continuity rows | `415.663s` | `294.919s` | `-29.0%` |
| all required K=4 rows | `816.209s` | `268.471s` | `-67.1%` |

The main measured reduction was in pattern generation time.

## `g37 = {3,7}` Correction

`g37` should no longer be presented as open at `n=1000` or limited to `n=600`.

Correct interpretation:

- old unresolved rows used `selector_reason=non_mainline_solver`;
- they did not test the intended K=2 Step-3 exact route;
- PLAN13 reroute uses `selector_decision=exact`,
  `selector_reason=k2_exact_default`, `step3_mode=exact`;
- all tested rows through `n=5000` close with `UB=LB`;
- responsible method: `profile_realization_dp_exact`.

Presentation wording:

> `g37` was not a true method failure. It was a routing failure. Under the
> intended K=2 Step-3 exact profile-realization path, tested rows close through
> `n=5000`.

## Current Failure Modes

For several unsolved high-`n` rows, the current artifacts do not contain a
meaningful incumbent or finite optimality gap. The common recorded pattern is:

- `feasible=0`
- `ub=-1`
- `lb=-1`
- `gap=nan`

This means the run did not return a usable solution record before timeout,
external kill, or crash.

| Group | First important failure after accepted frontier | Recorded behavior |
|---|---:|---|
| `g3567` | `n=7000` | timeout/kill; no incumbent or gap emitted |
| `g3567` | `n=8000` | `std::length_error`; no incumbent or gap emitted |
| `g810` | `n=6000` | `std::length_error`; no incumbent or gap emitted |
| `g12357` | `n=10000` | timeout; no incumbent or gap emitted |
| `g246810` | `n=7000` | `std::length_error`; no incumbent or gap emitted |
| `g12345678910` | baseline `n=5000` | timeout in baseline window; PLAN14 checkpoint probes record failure stage + peak RSS; additive fast-path variants emit exact `UB=LB` rows at `n=5000` |
| `g37` | beyond corrected `n=5000` | not yet tested under corrected K=2 route |

The immediate interpretation is that many remaining failures are not currently
"small gap" failures. They are runtime, routing, or robustness failures.

## Results Suitable for Presentation

The safest presentation claims at this stage are:

1. The pipeline solves several paper groups to large `n` at `lambda=1.3`.
2. Many favorable arithmetic groups close at Step 2 through `ffd`.
3. Step 3 contributes two distinct successful mechanisms:
   - exact profile realization for small-`K` cases such as `g810`;
   - energy-core block repair for hard K=4 cases such as `g3567`.
4. The K=4 generator specialization materially improves runtime while
   preserving exactness on the active K=4 gate.
5. Remaining high-`n` unresolved rows are mostly limited by timeout or
   crash/robustness, not by large reported finite optimality gaps.
6. For hard K10/K12 fixed-`n` rows, PLAN33 gives certified finite gaps rather
   than exact closure; this should be presented separately from exact results.

## Recommended Slide Structure

Suggested presentation sequence:

1. Problem and benchmark families.
2. Four-step solver pipeline.
3. Current accepted frontier table.
4. Step-2 success cases.
5. Step-3 exact-profile realization case: `g810`.
6. Step-3 energy-core case: `g3567`.
7. K=4 generator speedup result.
8. Current blockers and in-progress cases:
   - baseline/runtime path for dense large-`K` (`g12345678910`) vs additive recovery
   - hard K10/K12 exact closure vs PLAN33 certified finite gaps
   - high-`n length_error` rows
9. Next experimental priorities.

## Next Priorities

For the paper-facing experimental program, the next useful work is:

1. regenerate the final selected result tables on HPC;
2. preserve and document PLAN14 dense-unit fast-path as additive evidence
   (do not silently replace baseline policy);
3. improve timeout/kill checkpoint output quality so baseline failures are less
   likely to end as `ub=-1/lb=-1` when an incumbent exists;
4. add robustness guards for `std::length_error` high-`n` rows if time allows;
5. prepare final paper tables from HPC outputs, using
   `PAPER_HPC_REPRODUCIBILITY_MAP.md` as the code/provenance guide.
