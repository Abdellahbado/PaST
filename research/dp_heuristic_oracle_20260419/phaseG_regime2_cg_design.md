# Phase G Regime-2 CG Probe Design

Date: 2026-04-19

## 1) Restricted master used in this probe

At fixed instance and fixed `epsilon`, with a restricted column pool `Q_r`:

- variables: `x_{c,q} >= 0` for rate class `c` and column/config `q in Q_r`
- LP relaxation for dual extraction, then integer restricted master for feasible TEC

Constraints:

- class machine counts: `sum_{q in Q_r} x_{c,q} = m_c` for each class `c`
- job-type coverage: `sum_c sum_{q in Q_r} a_{kq} x_{c,q} = n_k` for each type `k`

Objective:

- minimize `sum_c sum_{q in Q_r} cost_{c,q} x_{c,q}`

## 2) LP duals used

For LP relaxation:

- `alpha_c` dual for class-machine equality `sum_q x_{c,q} = m_c`
- `beta_k` dual for type-coverage equality `sum_{c,q} a_{kq} x_{c,q} = n_k`

## 3) Reduced cost of a new configuration

For a candidate configuration `a = (a_k)` evaluated for class `c`:

- `rc_c(a) = cost_c(a) - alpha_c - sum_k beta_k * a_k`

Negative reduced cost indicates an improving LP column.

## 4) Does `solve_pricing_dp` match this exactly?

For non-empty patterns, yes in intended form:

- set rewards `r_k = beta_k`
- set `sigma = alpha_c`
- pricing objective becomes `cost_c(a) - sum_k beta_k a_k - alpha_c`

However, `solve_pricing_dp` excludes the empty pattern by construction (`jd > 0`).

## 5) Adjustment/approximation used here

Probe handling:

- keep an explicit zero column in the master from initialization,
- run pricing only for non-empty columns,
- accept this as a bounded probe approximation around exact CG behavior,
- do not claim full branch-and-price exactness.

Also, pricing is run as a bounded best-first mode through existing `pricing_compare` tooling, not a full provably exhaustive negative-column enumeration in this phase.

## 6) Initial column set

Restricted pool starts from:

1. per-machine type-count patterns from a greedy LPT assignment at target `epsilon`,
2. zero pattern,
3. mono-type patterns (`only type k`, bounded by capacity and type total).

Each column is exactly priced per class using `solve_sparse_dp`.

## 7) Termination condition

Iterate LP master + pricing until one of:

- no new negative-reduced-cost non-empty column is found for any class,
- max iteration cap reached,
- pricing time budget reached.

Then solve integer restricted master on final pool and report best feasible TEC.
