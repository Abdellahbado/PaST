# Phase F Configuration Probe Design

Date: 2026-04-19

## 1) Master formulation to implement (regime 1)

Fixed instance and fixed `epsilon`.

Sets:

- job types `k in K` (distinct processing lengths)
- rate classes `c in C` (unique machine rates)
- feasible single-machine configurations `q in Q`

Parameters:

- `n_k`: number of jobs of type `k`
- `m_c`: number of machines in rate class `c`
- `a_{kq}`: number of jobs of type `k` in configuration `q`
- `cost_{cq}`: TEC of assigning configuration `q` to one machine in class `c`

Decision variables:

- `x_{cq} in Z_{>=0}` = number of class-`c` machines using configuration `q`

Model:

- machine counts per class: `sum_q x_{cq} = m_c` for all `c`
- exact type coverage: `sum_c sum_q a_{kq} x_{cq} = n_k` for all `k`
- objective: minimize `sum_c sum_q cost_{cq} x_{cq}`

This is the exact regime-1 integer master for the enumerated configuration pool.

## 2) What a configuration is in code

A configuration is an integer vector of per-type counts:

- `counts[q] = (a_{1q}, ..., a_{|K|q})`

with feasibility conditions:

- `0 <= a_{kq} <= n_k`
- `sum_k a_{kq} * length_k <= epsilon`

In this probe, `Q` is the **full enumeration** of all such vectors.

## 3) Rate-class representation

Rate classes are built by sorting machine rates and grouping equal values.

Stored as:

- `class_rates[c]`: numeric rate value
- `class_machine_counts[c] = m_c`: machines in that class

## 4) Configuration cost from `solve_sparse_dp`

For each `(c, q)`, build prefix costs with class rate scaling:

- `prefix_c[t+1] = prefix_c[t] + class_rates[c] * prices[t]`

Then call:

- `solve_sparse_dp(lengths, counts[q], prefix_c, epsilon)`

The returned optimal single-machine cost is `cost_{cq}`.

## 5) Full enumeration in regime 1?

Yes. This probe uses full configuration enumeration (no approximation, no restricted pool) for instance `46` first.

## 6) Does `solve_pricing_dp` exactly match the true reduced-cost subproblem?

Not yet confirmed as exact for full branch-and-price use without extra care.

## 7) Column-generation stance for this task

For this task, true column generation is postponed.

Reason:

- `solve_pricing_dp` currently enforces a non-empty selected pattern (`jd > 0`), while a complete master/pricing mapping needs explicit handling of zero-pattern behavior under class machine-count duals.
- additional reduced-cost sign conventions and dual mapping details are not yet validated end-to-end in this repository.

Therefore this Phase F task is deliberately a **full-enumeration regime-1 probe** only.
