# Phase C Experiment Protocol

## Overview

This is NOT a solver-improvement branch. The LLM proposes instance-family
hypotheses, not moves/policies/assignments/neighborhoods.

## Target

EHS faithful reconstruction (`glns/paper_heuristics.py`).

## Rules

- Published instances 61-90 remain test-only for EHS; do not use for
  benchmark design iteration.
- All families must target a specific EHS failure mechanism.
- No DeepSeek calls until family schema and baseline generators exist.
- Random baselines must use same schema and same family count.
- Trivial size-only hardness does not count as LLM success.

## Phase C0: Protocol Definition ✅

Define the research question, hypotheses, and evaluation framework.

## Phase C1: Family Schema ✅

Define `families/family_schema.json` with all required parameters, validity
constraints, and rejection conditions.

## Phase C2: Baseline Generators

Implement three generator classes:

### C2.1 LLM-designed families
- DeepSeek V4 Pro receives:
  - EHS mechanism summary (from B6 closure evidence)
  - Prior EHS failure/closure evidence
  - family_schema documentation
  - 8-10 specific mechanism targets to stress
- Outputs 8-10 family specs in schema JSON format

### C2.2 Random families
- Same schema
- Uniformly sample legal parameter ranges
- Same number of families as LLM
- Rejection: regenerate if any validity constraint violated

### C2.3 Human/simple sweeps
- Fixed parameter sweeps: n/m/T/tightness/price_volatility
- Same number of families as LLM
- Documented design rationale

## Phase C3: Smoke Pilot

### Setup
- 2 LLM families + 2 random families + 2 human sweep families
- 3 instances per family (total: 18 instances)
- Run EHS at 2 budgets: **short** (60s) and **long** (300s)
- If feasible, run one diagnostic comparator (exact DP per machine, or
  long-budget EHS 600s)

### Metrics per instance
- Feasibility rate
- EHS short/long HV gap
- TEC gap at matched Cmax (energetic comparison)
- HV gap if Pareto front available
- Runtime/convergence behavior
- Number of khats explored
- Front count

### Diagnostic yield definition
An instance is **high-yield** if ANY of:
- Short-vs-long EHS HV gap ≥ 5%
- Matched-Cmax TEC gap ≥ 2% against diagnostic comparator
- EHS fails to reach meaningful Cmax/front point under short budget
- Seed variance in HV ≥ 3% across 3 seeds
- Documented mechanism failure confirmed by logs
- EHS produces ≤ 1 front point at 60s (near-zero coverage)

### Family-level yield
A family is **high-yield** if ≥ 33% of its instances are high-yield.

## Phase C4: Decision Gate

After smoke:

| Condition | Action |
|-----------|--------|
| LLM yield > random/human by clear margin (≥2×) | Proceed to full campaign (C5) |
| Random matches or exceeds LLM yield | Stop LLM-critical claim. Report benchmark stress testing only. |
| Generated instances are invalid/unrealistic | Repair schema, rerun one smoke |
| 0/6 families produce useful stress (yield=0) | Stop branch. Report null result. |
| LLM yield > human but random ≈ LLM | Equivocal. Do not claim LLM advantage. |

## Phase C5 (Conditional): Full Campaign

- If C4 gate passes: 8-10 LLM families, 8-10 random families, 8-10 human families
- 5 instances per family
- Full EHS evaluation at 3 budgets
- Statistical comparison of yield rates

## Artifacts

```
C1: families/family_schema.json
C2: families/llm_families.json, families/random_families.json, families/human_sweep_families.json
C3: eval/c3_smoke_raw.csv, eval/c3_smoke_summary.csv, notes/c3_smoke_decision.md
C4: notes/c4_gate_decision.md
```
