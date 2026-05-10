# Phase C: LLM-Guided Adversarial Benchmark Design for EHS

## Core Hypothesis

> An LLM can design structured BPMSTP instance families that expose weaknesses
> of the EHS pipeline more efficiently than random instance generation or
> simple human parameter sweeps.

## Motivation

All 9 EHS improvement surfaces are proven closed (B6.2-B6.17). The EHS
reconstruction is faithful (97-98% of published HV at convergence). Every
branch that tried to improve EHS — single-hook patches, VND, portfolio
control, restart ensembles, warm-started G-LNS — reached saturation or
closure.

This branch flips the question. Instead of asking "can an LLM improve EHS?",
we ask "can an LLM find the instances where EHS struggles?" This is a
benchmark-design contribution, not a solver-improvement one.

An LLM that understands EHS's mechanism hierarchy (SGH constructive strength,
A-SGH history reuse, R-ES reinsertion bottlenecks, ES-local search
limitations) might design instance families that deliberately stress
specific mechanisms. This would be an adversarial benchmark design task
analogous to adversarial example generation in ML, but for combinatorial
optimization.

## Target Solver

- **Primary**: EHS faithful reconstruction (`glns/paper_heuristics.py`)
  - Default EHS: `run_ehs()` with all defaults
  - eps: `run_ehs(eps_ordering="expensive_source_first")` — best at ≥120s
  - fast_eps: `run_ehs(fast_mode=True, eps_ordering="expensive_source_first")` — best at ≤60s

## Comparator / Diagnostic Solvers

- **Long-budget EHS**: Same config, 300-600s (near-converged baseline)
- **DP-centered heuristic** (if available for comparison)
- **Exact/DP diagnostic** (one-machine oracle where feasible)

## Non-Goals

- NOT beating EHS as a solver
- NOT changing EHS implementation
- NOT claiming SOTA
- NOT proposing new heuristics
- NOT generating a huge benchmark by brute force
- NOT counting trivial size-only hardness as success

## Key Constraints

- Each family must target a specific EHS failure **mechanism**, not just "make it large"
- Random baselines must use the same legal schema and same family count
- No DeepSeek calls until the family schema and baseline generators exist
- Smoke pilot (C3) before any full campaign
- Gate (C4) before any further investment
