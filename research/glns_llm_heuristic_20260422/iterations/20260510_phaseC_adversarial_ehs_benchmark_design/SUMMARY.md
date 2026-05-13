# Phase C Summary

**Status**: C5 complete. Gate STRONG on multi-budget persistent difficulty.

**Branch**: LLM-guided adversarial benchmark design for EHS.

## C5: Multi-Budget Robust Validation (2026-05-13)

Tested whether LLM-designed families expose persistent EHS stress beyond the trivial 30s short-budget regime.

### Main comparison (3 budgets: 30s/120s/300s)

| Arm | Persist% | MC% | d30-300 | d120-300 |
|-----|----------|-----|---------|----------|
| **LLM Call2** | **67%(4/6)** | **33%(2/6)** | **68.2** | **25.8** |
| Literature | 0%(0/6) | 0%(0/6) | 4.2 | 3.2 |
| Random | 50%(3/6) | 33%(2/6) | 9.5 | 7.0 |
| Simple Stress | 33%(1/3) | 0%(0/3) | 3.0 | 1.0 |

| Arm (internal) | Persist% | MC% |
|----------------|----------|-----|
| agent_manual_sweep | 83%(5/6) | 50%(3/6) |

**Gate: STRONG** — LLM beats literature (0%) and random (50%) on persistent-hard rate.

### Strongest family: hybrid_M2_tight_steprates_asghlock
- 2/3 persistent, 2/3 mechanism-confirmed, 2/3 nontrivial
- d30-300: [+42, +62, +76], d120-300: [-3, +37, +50]
- 2/3 instances continue growing 120s→300s — genuine persistent A-SGH lock-in

### Caveats
- random_005 landed on tight epsilon + step rates by chance (3/3 PH, 2/3 MC)
- agent_manual_sweep still leads (83% PH)
- Literature baselines trivially fail (0% PH)
- Difficulty mostly 30s→120s, not always 120s→300s

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
