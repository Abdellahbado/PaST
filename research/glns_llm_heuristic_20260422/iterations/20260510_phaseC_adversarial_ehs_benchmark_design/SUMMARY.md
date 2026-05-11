# Phase C Summary

**Status**: C3-L2 counter-sweep complete. Gate STRONG.

**Branch**: LLM-guided adversarial benchmark design for EHS.

## C3-Regular Repaired (for reference)

Human sweep won (100% yield) > LLM Call1 (83%) on front-size growth metric.
Human advantage: uniform p_j=(1,10) + tight/loose epsilon → many khats → large Δfs.

## C3-L2 Counter-Sweep: DeepSeek Call 2

### Method
- Showed DeepSeek the C3-Regular failure and asked it to design hybrid families
  combining human-sweep front-growth ingredients with mechanism-specific stress
- 5 families generated (output truncated at 16000 tokens, 3 missing)
- Evaluated 2 best families (hybrid_M1: first_khat_dominance, hybrid_M2: asgh_lock_in)

### Results

| Arm | Instances | High-Yield | Rate | Mean Δfs | Median Δfs |
|-----|----------|------------|------|----------|------------|
| **LLM Call2 (L2)** | **6** | **6** | **100%** | **19.5** | **10** |
| human (C3) | 6 | 6 | 100% | 15.2 | 13 |
| LLM Call1 (C3) | 6 | 5 | 83% | 2.8 | 3 |
| random (C3) | 6 | 4 | 67% | 8.5 | 5 |

### Gate: STRONG
LLM Call2 ≥ human sweep on yield rate (100%) AND beats on mean front growth (19.5 > 15.2).

### Mechanism Confirmation
- M1 hybrid (tight epsilon + heterogeneous rates): 3/3 HIGH, Δfs +7 to +66
- M2 hybrid (tight epsilon + step machine rates): 3/3 HIGH, Δfs +9 to +17
- Both hybrid families successfully combined sweep-layer front growth with mechanism-specific stress

## Key Artifacts
- `responses/call2_counter_sweep_raw.md` — DeepSeek Call 2 response
- `families/llm_counter_sweep_families.json` — 5 valid family specs
- `eval/c3_l2_counter_sweep_raw.csv` — 12 EHS runs
- `notes/c3_l2_counter_sweep_decision.md` — gate decision
- `notes/c3_l2_deepseek_counter_sweep_evidence.md` — evidence pack sent to DeepSeek

## Next
- C4 gate decision: full campaign justified?
