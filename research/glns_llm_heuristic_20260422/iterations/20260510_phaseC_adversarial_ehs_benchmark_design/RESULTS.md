# Phase C Results

## C0-C1: Protocol + Schema — COMPLETED 2026-05-10

## C2: Baseline Generators + LLM Prompt — COMPLETED 2026-05-10

## C3: Smoke Pilot — COMPLETED 2026-05-10

### C3A: DeepSeek LLM Families ✅
- 8/8 families valid, $0.01 cost
- Each targets a distinct EHS failure mechanism

### C3B: Family Selection ✅
- 2 LLM + 2 random + 2 human families selected

### C3-Smoke (original, truncated) ⚠️
- 8/36 EHS runs completed. Human and most random not evaluated.
- Discovered EHS time-limit bug.

### EHS Time-Limit Fix ✅
- Added cooperative deadline checks in `split_greedy_heuristic()` (every 5 jobs)
- Added pre-operation checks before exchange and ESR
- `run_ehs()` sets/clears deadline
- Overrun reduced from 5× budget to ≤40% for capped instances

### C3-Scalability (Track B) ✅
- LLM `first_khat_dominance_giant` family (n=800+) correctly identified non-interruptible SGH construction
- Documented in `notes/c3_scalability_first_khat_diagnostic.md`

### C3-Regular (Track A) ✅
- 18 instances: LLM (6), random (6), human (6)
- n≤150, T≤200 caps
- Budgets: 30s / 90s

**LLM arm (6/6 evaluable, 5/6 high-yield)**:
- `asgh_trajectory_conflict`: 3/3 high-yield — fs growth +3, +6, +6
- `es_local_optima_trap_extreme_rates`: 2/3 high-yield — fs growth +3, 1→2 (very_slow)

**Random arm (1/6 evaluable)**:
- random_000: 3/3 infeasible (bimodal jobs too large for T≤200)
- random_001: 1 evaluable (very_slow_single_point), 2 incomplete

**Human arm (0/6 evaluable)**:
- All 6 instances: eval incomplete (timeout before reaching them)

### Gate: INCONCLUSIVE
- Cannot compare arms when 11/18 non-LLM instances couldn't be evaluated
- LLM demonstrates strong mechanism targeting (5/6 high-yield, interpretable scaling)
- But fair comparison requires redesigning baseline arms for capped evaluation scale

## Key Artifacts
- `glns/paper_heuristics.py` — EHS time-limit fix (cooperative deadline)
- `families/llm_families.json` — 8 valid LLM family specs
- `families/random_families.json` — 8 random family specs
- `families/human_sweep_families.json` — 8 human family specs
- `eval/c3_regular_raw.csv` — 36 EHS eval rows
- `eval/c3_regular_summary.csv` — 18 instance-level summary rows
- `notes/c3_regular_decision.md` — C3-Regular gate decision
- `notes/c3_scalability_first_khat_diagnostic.md` — Scalability note
