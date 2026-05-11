# Phase C Summary

**Status**: C4 validation complete. Gate MODERATE.

**Branch**: LLM-guided adversarial benchmark design for EHS.

## C3-L2 Counter-Sweep (2026-05-10)

DeepSeek Call 2: hybrid families combining human-sweep front-growth with mechanism stress.
**Gate: STRONG** (100% HY, Δfs_mean=19.5 > human=15.2, 6 instances).

## C4 Validation (2026-05-11)

Fresh instances, 4× families, 5 per family = 60 total.

| Arm | HY% | Mean Δfs | MC% | NT-MC-HY% |
|-----|-----|----------|-----|-----------|
| **LLM L2** | **100%** | **20.4** | **25%** | **25%** |
| Human sweep | 100% | 38.2 | 50% | 50% |
| Random | 93% | 2.6 | 0% | 0% |

**Gate: MODERATE** — LLM ties human on HY rate but loses on NT-MC-HY.

### What transferred
- HY rate held at 100% across 4× more families (2→4 families, 6→20 instances)
- LLM M2 (asgh_lock_in): 5/5 NT-MC-HY — only family with perfect mechanism confirmation
- Generation quality: 100% vs Random 75%

### What didn't
- LLM M1, M3, M4 got 0/5 mechanism confirmation — subtle mechanisms at 30s budget
- Human loose_epsilon Δfs +50 to +116 swamps LLM raw Δfs — structural epsilon advantage
- All 60 instances timed out at 30s — budget too short for mechanism discrimination

### Decision
- Transfer confirmed on HY rate but mechanism specificity weakened
- LLM shows qualitative advantage (1 family with perfect mechanism match vs 2 for human)
- Budget re-calibration (longer short budget?) needed for better mechanism discrimination
- MODERATE gate: strong enough for exploratory thesis chapter, but C5 full campaign is optional

## Key Artifacts
- C3-L2: `families/llm_counter_sweep_families.json`, `eval/c3_l2_counter_sweep_raw.csv`
- C4: `eval/c4_validation_raw.csv`, `notes/c4_validation_decision.md`
- Scripts: `_c4_validation.py`, `_c4_eval_sequential.py`, `_c4_metrics_decision.py`
