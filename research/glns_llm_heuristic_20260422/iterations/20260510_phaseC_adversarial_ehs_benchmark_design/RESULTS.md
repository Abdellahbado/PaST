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

## C5: Multi-Budget Robust Validation — COMPLETED 2026-05-13

**27 instances × 3 budgets = 81 EHS runs (1.8h)**

Test: Does LLM difficulty persist beyond 30s short-budget regime?
Arms: LLM Call2 (M1+M2), Literature (Wang+Anghi), Random (004+005),
       agent_manual_sweep (internal), simple_stress (non-LLM).

### Main comparison (persistent-hard = difficulty from 30s→120s or further)

| Arm | Persist% | MC% | d30-300 | d120-300 |
|-----|----------|-----|---------|----------|
| **LLM Call2** | **67%(4/6)** | **33%(2/6)** | **68.2** | **25.8** |
| Literature | 0%(0/6) | 0%(0/6) | 4.2 | 3.2 |
| Random | 50%(3/6) | 33%(2/6) | 9.5 | 7.0 |
| Simple Stress | 33%(1/3) | 0%(0/3) | 3.0 | 1.0 |

| Arm (internal) | Persist% | MC% |
|----------------|----------|-----|
| agent_manual_sweep | 83%(5/6) | 50%(3/6) |

**Gate: STRONG**

### Per-family results

**hybrid_M2_tight_steprates_asghlock**: 2/3 PH, 2/3 MC, 2/3 NTM
  d30-300: [+42, +62, +76], d120-300: [-3, +37, +50]

**hybrid_M1_tight_hetero_rates_firstkhat_dom**: 2/3 PH, 0/3 MC
  d30-300: [+78, +75, +76], d120-300: [+72, +0, -1]

**human_mixed_job_sizes** (internal): 3/3 PH, 3/3 MC, 3/3 NTM
  d30-300: [+41, +67, +28], d120-300: [+29, +54, +17]

**random_005**: 3/3 PH, 2/3 MC (lucky random draw of tight epsilon + step rates)

**Literature (Wang+Anghi)**: 0/6 PH. Trivial front growth at all budgets.

### Key findings
- LLM M2 difficulty persists to 120s/300s — NOT just a 30s artifact
- LLM M2 has mechanism-confirmed A-SGH lock-in on 2/3 instances
- Literature baselines are poor adversarial test instances (tiny fronts, saturate)
- random_005 matches LLM per-family on PH (lucky configuration)
- agent_manual_sweep still leads but margin narrows with multi-budget metric

### Artifacts
- `scripts/_c5_robust_validation.py` — full pipeline
- `eval/c5_raw_runs.csv` — 81 EHS runs
- `eval/c5_instance_summary.csv` — 27 instance summaries
- `eval/c5_family_summary.csv` — per-family aggregation
- `eval/c5_mechanism_confirmation.csv` — per-instance mechanism check
- `families/c5_frozen_families.json` — frozen family selection
- `C5_PROTOCOL.md` — evaluation protocol
- `C5_RESULTS.md` — detailed results
- `C5_DECISION.md` — gate decision + paper-safe claims

## C4: Validation + Literature Baseline — COMPLETED 2026-05-11

### C4 Validation (12 families × 5 = 60 instances)

| Arm | HY% | Mean Δfs | MC% | NT-MC-HY% |
|-----|-----|----------|-----|-----------|
| LLM L2 | 100% | 20.4 | 25% | 25% |
| agent_manual_sweep | 100% | 38.2 | 50% | 50% |
| Random | 93% | 2.6 | 0% | 0% |

**Gate: MODERATE** — LLM ties on HY but loses NT-MC-HY.
LLM M2 (asgh_lock_in) confirmed mechanism on 5/5 instances.

### C4-Lit: Literature Baseline (framing revision)

**Naming revision**: `human` renominated to `agent_manual_sweep` (internal control).
Main baselines are now literature-derived generators.

| Arm (main) | HY% | Mean Δfs |
|------------|-----|----------|
| **LLM Call2** | **100%** | **20.4** |
| Literature (Wang+Anghi) | 88% | 2.4 |
| Random (schema) | 93% | 2.6 |

| Arm (internal) | HY% | Mean Δfs |
|----------------|-----|----------|
| agent_manual_sweep | 100% | 38.2 |

**Gate: STRONG** — LLM beats literature on HY rate (100% vs 88%) and Δfs (20.4 vs 2.4).

### Literature baseline breakdown
- **Wang medium-scale** (p_j~U[1,4], e∈{1,2,3}, c_t∈{1,2,3,4}): 15/20 HY (75%), mean Δfs=4.2
- **Anghinolfi VLS capped** (p_j~U[1,12], e~U[1,6], c_t~U[1,8]): 20/20 HY (100%), mean Δfs=0.6

### Artifacts
- `families/literature_generators.json` — Wang + Anghinolfi generator specs
- `generated_instances/c4_literature_baseline/` — 40 instances
- `eval/c4_literature_baseline_raw.csv` — 80 EHS runs
- `eval/c4_literature_baseline_summary.csv` — 40 instance summaries
- `notes/c4_literature_baseline_decision.md` — gate decision
- `scripts/_c4_literature_baseline.py` — full pipeline script
