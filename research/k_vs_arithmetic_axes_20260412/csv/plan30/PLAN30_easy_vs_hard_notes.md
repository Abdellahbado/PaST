# PLAN30: Easy-vs-hard fixed-n K-scaling story (implements PLAN_16)

## Scope

Fixed `n=1000`, `lambda=1.3`. Extend easy unit-contiguous families to `K=24`, `K=30`, and optionally `K=40`. Compare against existing hard irregular data from PLAN17/18.

## Families

- `easy_k24_unit`: `{1,2,...,24}`, seeds 0,1
- `easy_k30_unit`: `{1,2,...,30}`, seeds 0,1
- `easy_k40_unit`: `{1,2,...,40}`, seeds 0,1 (optional)

Variants per family: `baseline`, `dense_step2_fastpath`.

## Results

All 12 rows exact (0% gap), all memory-safe (peak RSS 1.7–4.8 GB, no kills, no timeouts).

| K | exact_rows | mean_rt_sec | max_rt_sec | mean_rss_gb | max_rss_gb | deciding_step |
|---|---|---|---|---|---|---|
| 24 | 4/4 | 364.3 | 403.3 | 3.65 | 4.69 | easy_step2_exact |
| 30 | 4/4 | 683.2 | 738.0 | 4.17 | 4.83 | easy_step2_exact |
| 40 | 4/4 | 1551.9 | 1719.9 | 2.63 | 3.10 | easy_step2_exact |

All rows close at Step 2 (`ffd`). Dense fastpath is active on fastpath variants but Step 2 closes regardless.

## Easy vs Hard comparison

| arithmetic_class | K | exact_rows | mean_rt_sec | boundary |
|---|---|---|---|---|
| easy_contiguous_unit | 24 | 4/4 | 364.3 | exact |
| easy_contiguous_unit | 30 | 4/4 | 683.2 | exact |
| easy_contiguous_unit | 40 | 4/4 | 1551.9 | exact |
| hard_irregular_A | 8 | 2/4 | 157.9 | finite_gap |
| hard_irregular_A | 10 | 0/4 | 488.0 | finite_gap |
| hard_irregular_A | 12 | 0/4 | 933.1 | timeout_no_incumbent |
| hard_irregular_B | 8 | 2/4 | 490.7 | finite_gap |
| hard_irregular_B | 10 | 0/4 | 950.2 | finite_gap |
| hard_irregular_B | 12 | 0/4 | 1188.5 | timeout_no_incumbent |

## Key finding

Easy contiguous-unit families remain exact through K=40 at n=1000, with runtime growing linearly-ish with K. Hard irregular families degrade around K=8-10. This sharpens the two-axis claim: K alone is not the hardness driver; arithmetic structure is.

## Decision

**A** — The easy-vs-hard K-scaling story is now sufficiently documented. No further K extension needed for the paper. K=24 and K=30 establish that easy families scale far beyond the hard boundary. K=40 is bonus evidence.

## Artifacts

- `csv/plan30/PLAN30_easy_k_scaling_raw.csv` (12 rows)
- `csv/plan30/PLAN30_easy_k_scaling_summary.csv`
- `csv/plan30/PLAN30_easy_vs_hard_k_boundary.csv`
