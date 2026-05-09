# X5 — Proposed Held-Out Validation Cells

14 cells spanning small-to-large instances and tight-to-loose epsilon regimes.
Excluded: 61/347, 62/290, 65/195 (used in X2-X4 dev).

## Cell List

| Inst/Eps | Jobs | Role | Epsilon Regime | Rationale |
|----------|------|------|----------------|-----------|
| 69/210 | 500 | tight-large | Tight (≤210) | Tests tight epsilon on large instance |
| 79/205 | 500 | tight-large | Tight (≤210) | Same regime, different instance |
| 68/240 | 400 | tight-mid | Tight-mid (240) | Bridge between tight and medium |
| 78/255 | 400 | medium-mid | Medium (255) | Lower end of medium regime |
| 67/260 | 400 | medium-mid | Medium (260) | Core medium epsilon |
| 77/270 | 400 | medium-mid | Medium (270) | Core medium epsilon |
| 82/295 | 250 | medium-small | Medium (295) | Small instance, medium epsilon |
| 84/300 | 300 | medium-mid | Medium (300) | Typical medium instance |
| 66/300 | 350 | medium-mid | Medium (300) | Larger instance count |
| 85/300 | 350 | medium-mid | Medium (300) | Large instance, medium epsilon |
| 75/320 | 350 | loose-mid | Loose (320) | Upper medium approaching loose |
| 64/305 | 300 | loose-mid | Loose (305) | Transition from medium to loose |
| 83/335 | 300 | loose-mid | Loose (335) | Core loose regime |
| 73/350 | 300 | loose-mid | Loose (350) | Upper end of epsilon range |

## Regime Coverage

| Regime | Epsilon Range | Cells |
|--------|-------------|-------|
| Tight | 205-240 | 69/210, 79/205, 68/240 (3) |
| Medium | 255-300 | 78/255, 67/260, 77/270, 82/295, 84/300, 66/300, 85/300 (7) |
| Loose | 305-350 | 75/320, 64/305, 83/335, 73/350 (4) |

## Size Coverage

| Size | Job Count | Cells |
|------|-----------|-------|
| Small | 250-300 | 82/295 (1) |
| Medium | 300-350 | 84/300, 66/300, 85/300, 75/320, 64/305, 83/335, 73/350 (7) |
| Large | 400-500 | 69/210, 79/205, 68/240, 78/255, 67/260, 77/270 (6) |

## Validation Arms

For each (inst, eps) cell, run all baselines:

1. `vnd_exact_dp_insert_rank_diverse_trimmed` — strict trimmed baseline
2. `phaseS_random_exception_lane` — random exception lane (seed 0)
3. `phaseV_score_escape_sampler` — score escape
4. `phaseX_example_policy` — example policy (llm_score default)
5. **`phaseX_policy_json` with `policies/llm_interactive/x4_round_2.json`** — BEST LLM interactive policy
6. **Random best-of-5 baseline** — per cell: generate 5 fresh random policies (seeds 6000..6004), evaluate all 5 on that cell, keep best TEC

## Expected Runtime

Per cell × 6 arms × ~80s = ~8 min per cell. 14 cells × 8 min = ~112 min.
With baseline-only eval (no random best-of-5): ~5 min per cell, ~70 min.

## Precondition

X5 Task A shows WEAK signal (LLM at 20th percentile vs random best-of-5).
Only proceed with validation if the paper position justifies it despite weak signal.

## NOT RUN YET

Validation is proposed but NOT executed. The decision to run validation should
be made after X5 results are reviewed.
