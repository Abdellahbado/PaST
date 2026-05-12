# C4 Literature Baseline Decision

**Date**: 2026-05-11
**Literature generators**: `wang_mls_generator` (Wang 2018, medium-scale, capped T≤200), `anghinolfi_vls_capped` (Anghinolfi 2021 VLS, capped n/T)
**Budgets**: 30s / 90s

## Naming Revision

The previous `human` / `human_sweep` arm was generated inside our agent/coder
research workflow. It is now renamed to **`agent_manual_sweep`** (internal control).
It is NOT presented as the main independent baseline:

- **Main external baselines**: random schema generator + literature-derived generators
- **Internal control**: `agent_manual_sweep` (reported separately)

The family file `families/human_sweep_families.json` is preserved as-is.
Only textual references are updated.

## Literature Generators

### `wang_mls_generator`
- Source: Wang et al. (2018) J Cleaner Production 193; Gaggero et al. (2023) EJOR 311, instances 31-60
- Parameters: n∈{30,60,100,150}, m∈{8,16,25}, T∈{100,200(capped from 300)}
- p_j ~ U[1,4], e ∈ {1,2,3}, c_t ∈ {1,2,3,4}
- Discrete rates and prices (faithful to published benchmark)

### `anghinolfi_vls_capped`
- Source: Anghinolfi et al. (2021); Gaggero et al. (2023), instances 61-90
- Parameters CAPPED for Phase C: n∈[60,150], m∈[8,25], T∈[100,200]
- p_j ~ U[1,12], e_h ~ U[1,6], c_t ~ U[1,8]
- Continuous uniform rates and prices (faithful to published ranges)
- Explicitly labeled as CAPPED; NOT the original VLS benchmark

## Main Comparison

| Arm | N | Feasible | Evaluable | HY | HY% | Mean Δfs | TO (short) |
|-----|--|----------|-----------|----|-----|----------|------------|
| **LLM Call2** | **20** | **20** | **20** | **20** | **100%** | **20.4** | **20/20** |
| Literature (Wang+Anghi) | 40 | 40 | 40 | 35 | 88% | 2.4 | 34/40 |
| Random (schema) | 20 | 15 | 15 | 14 | 93% | 2.6 | 15/15 |
| agent_manual_sweep (internal) | 20 | 20 | 20 | 20 | 100% | 38.2 | 20/20 |

### Literature breakdown
| Generator | N | Eval | HY | Rate | Mean Δfs |
|-----------|--|------|----|------|----------|
| Wang (medium) | 20 | 20 | 15 | 75% | 4.2 |
| Anghinolfi (capped) | 20 | 20 | 20 | 100% | 0.6 |

## Gate: STRONG

LLM Call2 beats literature generator on HY rate (100% vs 88%) and mean Δfs (20.4 vs 2.4).
LLM Call2 also beats random on both metrics.

## Interpretation

### Why literature baselines perform modestly
- Wang medium-scale instances have p_j~U[1,4] (very small), which SHOULD produce many khats.
  But the discrete machine rates {1,2,3} and prices {1,2,3,4} create narrow energy landscapes
  where each khat adds few unique front points. Result: Δfs=0 to +21, mean=4.2.
- Anghinolfi capped has p_j~U[1,12] (larger jobs), making khat step size larger.
  Fewer khats → Δfs=0 to +4, mean=0.6. But still 100% HY because every instance gets ≥2
  points (just barely).
- Both generators produce valid, feasible instances but the energy-cost-makespan tradeoff
  space is relatively narrow compared to LLM-designed families.

### Why LLM Call2 does better
- LLM families combine tight epsilon + uniform small jobs (structural front-growth) with
  mechanism-specific stress (heterogeneous/step rates, dual-peak TOU, etc.)
- These create richer energy-cost landscapes with more distinct tradeoff points
- The mechanism stress amplifies front growth beyond what literature generators produce

### Why agent_manual_sweep still leads raw Δfs
- loose_epsilon human family: Δfs +50 to +116 via many khat iterations
- This is a structural property of uniform p_j=(1,10) + coarse epsilon
- The sweep was crafted by an agent/coder who understands the metric, not by an
  independent human expert

## Paper-Safe Wording

> We compare against random generation and literature-derived benchmark generators
> (Wang 2018 medium-scale, Anghinolfi 2021 VLS capped). We also report an internal
> agent-crafted sweep as a strong stress-control baseline.

## Decision Questions

1. **Does LLM beat literature baseline?** Yes. 100% HY vs 88%, mean Δfs 20.4 vs 2.4.
2. **Does LLM beat random?** Yes. 100% vs 93%, mean Δfs 20.4 vs 2.6.
3. **Is the revised framing cleaner?** Yes. Literature baselines are externally defined,
   random is a proper strawman, and agent_manual_sweep is an internal control.
4. **Strong enough for thesis?** Yes — STRONG gate on external baselines.
   The LLM advantage is clear on both yield rate and magnitude.
