# Phase C Results

## C0-C2: Protocol, Schema, Generators — COMPLETED 2026-05-10

## C3: Smoke Pilot — COMPLETED 2026-05-10

### C3-Scalability (Track B) ✅
- LLM `first_khat_dominance_giant` (n=800+) correctly identified non-interruptible SGH construction.
- Side diagnostic. Not counted in main yield metric.

### EHS Time-Limit Fix ✅
- Cooperative deadline checks in `glns/paper_heuristics.py`: every 5 jobs in SGH, at duration class boundaries, before post-SGH operations.
- Overrun ≤40% for capped instances.

### C3-Regular Repaired (Track A) ✅
- 18 instances (6 per arm), n≤150, T≤200, 30s/90s budgets
- All 18 instances feasible and evaluable (100% generation quality)
- 36/36 EHS runs completed

**Adversarial Yield Results:**

| Arm | High-Yield | Rate | Mechanism Match |
|-----|-----------|------|-----------------|
| human | 6/6 | 100% | Tight/loose epsilon trivially produce front growth |
| llm | 5/6 | 83% | A-SGH lock-in (3/3 HIGH), ES tension (2/3 HIGH) |
| random | 4/6 | 67% | Small instances saturate quickly (2 saturated) |

**Gate: FAIL** — Human sweep families (100%) beat LLM (83%) on adversarial yield.

Human advantage is structural: uniform p_j=(1,10) processing times allow many khat iterations, producing large Δfs (+7 to +33). This is a property of the sweep design rather than adversarial insight.

LLM families show mechanism-aware targeting (A-SGH lock-in confirmed on 3/3 instances, ES tension on 2/3), but the raw front-size-growth metric favors simple small-instance designs.

Random families at 67% confirm the metric is sensitive to instance size/structure, not mechanism specificity.

## Artifacts

### Code
- `glns/paper_heuristics.py` — EHS cooperative deadline fix
- `scripts/_c3_repaired.py` — C3-Regular repaired pipeline
- `scripts/phaseC_adversarial_family_generation.py` — family generation + validation
- `scripts/phaseC3_smoke_pilot.py` — full C3 pipeline

### Data
- `eval/c3_regular_repaired_raw.csv` — 36 EHS runs
- `eval/c3_regular_repaired_summary.csv` — 18 instance-level summary
- `families/llm_families.json` — 8 LLM family specs
- `families/random_families.json` — 8 random family specs
- `families/human_sweep_families.json` — 8 human family specs

### Notes
- `notes/c3_regular_repaired_decision.md` — Gate decision
- `notes/c3_scalability_first_khat_diagnostic.md` — Scalability note
- `notes/c3_regular_decision.md` — Original C3-Regular decision (pre-repair)

### Generated Instances
- `generated_instances/c3_regular/` — original 18 instances
- `generated_instances/c3_repair/` — repaired 12 baseline instances
