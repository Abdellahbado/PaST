# Results — Phase X

## X2 Smoke (2026-05-08)

Full smoke on 3 dev cells × 6 arms. All 18 runs feasible.

| Inst/Eps | Trimmed | LLM Exc | Random Exc | Score Esc | PhaseX Example | PhaseX Random |
|:--------:|--------:|--------:|----------:|---------:|:--------------:|:-------------:|
| 61/347 | 6884 | 6884 | 6870 | 6884 | 6884 | 6884 |
| 62/290 | 9687 | 9687 | 9687 | 9489 | **9484** | 9503 |
| 65/195 | 27031 | 27031 | 27031 | 26508 | **26508** | 26749 |

## X3 Random Campaign (2026-05-09)

20 random DSL policies on 3 cells. Baselines recomputed.

### Baseline reference (mean TEC across 3 cells)

| Arm | Mean TEC |
|-----|---------:|
| trimmed | 14534.0 |
| phaseS_llm_exception_lane | 14534.0 |
| phaseS_random_exception_lane | 14522.3 |
| phaseV_score_escape_sampler | 14292.0 |
| phaseX_example_policy | 14292.0 |

### Random policy summary

| Metric | Value |
|--------|------:|
| Policies evaluated | 20 |
| Failed / infeasible | 0 / 0 |
| Best mean TEC | 14254.3 (Δ example = -37.7) |
| Median mean TEC | 14362.0 (Δ example = +70.0) |
| Worst mean TEC | 14471.0 |
| Beat example on mean | 2/20 (10%) |
| Beat trimmed on mean | 20/20 (100%) |
| Beat example on ≥2 cells | 2/20 (10%) |
| Beat score_escape on ≥1 cell | 11/20 (55%) |

### Top 5 random policies by mean TEC

| Rank | ID | Mean TEC | Δ Trimmed | Δ Example | BeatEx | BeatSE | RegrTr |
|:----:|:--:|--------:|---------:|---------:|:------:|:------:|:------:|
| 1 | c000 | 14254.3 | -279.7 | -37.7 | 2 | 2 | 0 |
| 2 | c002 | 14278.7 | -255.3 | -13.3 | 2 | 2 | 0 |
| 3 | c001 | 14294.7 | -239.3 | +2.7 | 2 | 2 | 0 |
| 4 | c016 | 14312.7 | -221.3 | +20.7 | 1 | 1 | 1 |
| 5 | c015 | 14321.3 | -212.7 | +29.3 | 0 | 0 | 0 |

### Classification: CASE B

**DSL contains useful policies but search is noisy.**

- Random median (14362.0) is worse than example_policy (14292.0): the DSL
  is NOT trivially random-searchable.
- Best random policy (14254.3, c000) beats example by -37.7 on mean TEC
  and beats it on 2/3 cells individually: good policies exist.
- Implications for X4: interactive LLM should be compared against BOTH
  median random policy AND best random policy. The LLM must show it finds
  good policies more efficiently than brute-force random search.

Files:
- `eval/x3_random_campaign_raw.csv` — 75 rows (15 baseline + 60 random)
- `eval/x3_random_campaign_summary.csv` — per-cell metrics
- `eval/x3_random_campaign_aggregate.csv` — per-policy aggregate
- `policies/random_campaign/x3_campaign_000.json` … `019.json`
