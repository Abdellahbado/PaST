# Phase C Summary

**Status**: C4-Lit complete. Gate STRONG on literature baselines.

**Branch**: LLM-guided adversarial benchmark design for EHS.

## C4-Lit: Literature Baseline + Framing Revision (2026-05-11)

**Naming revision**: `human` → `agent_manual_sweep` (internal control).
Main baselines: random schema + literature-derived generators.

### Main comparison (literature baseline)

| Arm | HY% | Mean Δfs |
|-----|-----|----------|
| **LLM Call2** | **100%** | **20.4** |
| Literature (Wang+Anghi) | 88% | 2.4 |
| Random (schema) | 93% | 2.6 |

| Arm (internal) | HY% | Mean Δfs |
|----------------|-----|----------|
| agent_manual_sweep | 100% | 38.2 |

**Gate: STRONG** — LLM beats literature on both HY rate and Δfs magnitude.

Literature baseline details:
- Wang medium-scale (p_j~U[1,4]): 75% HY, mean Δfs=4.2 — narrow energy landscape
- Anghinolfi VLS capped (p_j~U[1,12]): 100% HY, mean Δfs=0.6 — larger jobs, fewer khats

## C4 Validation (frozen families)

LLM 100% HY vs agent_manual_sweep 100% HY (MODERATE).
LLM M2 (asgh_lock_in): 5/5 mechanism-confirmed.

## C3-L2 Counter-Sweep

DeepSeek Call 2: STRONG on development (6 instances).

## Key Artifacts
- `families/literature_generators.json` — Wang + Anghinolfi generator specs
- `families/human_sweep_families.json` — preserved as archival (renamed textually)
- `eval/c4_literature_baseline_raw.csv` — 80 EHS runs
- `notes/c4_literature_baseline_decision.md` — gate decision
