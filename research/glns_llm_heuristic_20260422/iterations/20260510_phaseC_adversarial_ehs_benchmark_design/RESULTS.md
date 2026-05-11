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

## C3-L2: DeepSeek Counter-Sweep (Call 2) — COMPLETED 2026-05-10

- Second DeepSeek call with C3-Regular failure feedback
- 5/8 families complete (truncated at 16000 tokens)
- Evaluated 2 best families: hybrid_M1 (first_khat_dominance) + hybrid_M2 (asgh_lock_in)

**Results (6 instances, 30s/90s):**

| Arm | Instances | High-Yield | Rate | Mean Δfs | Median Δfs |
|-----|----------|------------|------|----------|------------|
| **LLM Call2** | **6** | **6** | **100%** | **19.5** | **10** |
| human (C3) | 6 | 6 | 100% | 15.2 | 13 |
| LLM Call1 (C3) | 6 | 5 | 83% | 2.8 | 3 |
| random (C3) | 6 | 4 | 67% | 8.5 | 5 |

**Gate: STRONG** — LLM Call2 ≥ human on yield rate (100%) AND beats on mean Δfs (19.5 > 15.2).

LLM learned to combine human-sweep front-growth ingredients with mechanism-specific stress.
Total cost: $0.0178.

### Artifacts
- `responses/call2_counter_sweep_raw.md` — Call 2 response
- `families/llm_counter_sweep_families.json` — 5 valid family specs
- `eval/c3_l2_counter_sweep_raw.csv` — 12 EHS runs
- `notes/c3_l2_counter_sweep_decision.md`
- `notes/c3_l2_deepseek_counter_sweep_evidence.md`

## C4: Validation — COMPLETED 2026-05-11

- Validation on fresh instances (no C3/C3-L2 seed reuse)
- 12 families × 5 = 60 instances, n≤150, T≤200, 30s/90s
- 120 EHS runs (2.6h total)

**Results (60 instances):**

| Arm | N | Feasible | Evaluable | HY | HY% | Mean Δfs | MC% | NT-MC-HY% |
|-----|---|----------|-----------|----|-----|----------|-----|-----------|
| **LLM L2** | **20** | **20** | **20** | **20** | **100%** | **20.4** | **25%** | **25%** |
| Human sweep | 20 | 20 | 20 | 20 | 100% | 38.2 | 50% | 50% |
| Random | 20 | 15 | 15 | 14 | 93% | 2.6 | 0% | 0% |

**Gate: MODERATE** — LLM ties human on HY rate (100% vs 100%) but loses on NT-MC-HY (25% vs 50%).

### Key findings
- **LLM Call2 transfer confirmed**: HY rate held at 100% across 4× more families (from 2→4 families, 6→20 instances)
- **LLM M2 (asgh_lock_in)** confirmed mechanism on all 5/5 instances — the only family achieving 5/5 NT-MC-HY
- **LLM M1, M3, M4** had 0/5 mechanism confirmation — mechanisms too subtle for operational rules at 30s budget
- **Human loose_epsilon** produced massive Δfs (+50 to +116) via coarse epsilon → many khats → large growth — purely structural
- **Human many_machines_sparse and mixed_job_sizes** got 5/5 NT-MC-HY each — nontrivial mechanism families
- **Random** 75% generation quality (5 infeasible), 93% HY but tiny Δfs (mean=2.6) — saturating instances
- **All 60 instances timed out on short budget** — Δfs metric dominated by budget truncation

### Caveat
The 30s budget was too short for all arms. Δfs differences reflect epsilon granularity
(how many khats fit in the time window), not necessarily adversarial mechanism stress.

### Artifacts
- `notes/c4_validation_design.md` — design doc with mechanism rules
- `notes/c4_selected_families.json` — frozen family selection
- `generated_instances/c4_validation/` — 60 instances
- `eval/c4_validation_raw.csv` — 120 EHS runs
- `eval/c4_validation_summary.csv` — 60 instance summaries
- `notes/c4_validation_decision.md` — gate decision
- `scripts/_c4_validation.py` — generation script
- `scripts/_c4_eval_sequential.py` — evaluation script
- `scripts/_c4_metrics_decision.py` — metrics + decision script
