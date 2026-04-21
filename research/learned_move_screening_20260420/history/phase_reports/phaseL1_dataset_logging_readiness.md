# Phase L1 Dataset Logging Readiness

Date: 2026-04-20

## Stage L1 success criteria check

1. Two data streams clearly separated and correct

- Pass.
- Broad stream and exact-labeled stream are separate files with compatible `record_id` linkage.

2. Aggregate exact-labeled set has enough positives

- Pass for Stage L2 start.
- `112` exact-labeled rows with `27` improving positives (`24.11%`).

3. Features are cheap, stable, and aligned with decision point

- Pass.
- Features are logged at candidate-generation / exact-eval decision points directly inside insert-screen search flow.

## Risks to monitor in Stage L2

- Exact-labeled data volume is still modest; avoid over-parameterized models first.
- Seed-level variance is non-trivial; use split policy that avoids leakage and preserves seed diversity.
- Preserve analytical fallback ordering when integrating learned ranking later.

## Readiness decision

- Ready to proceed to Stage L2 offline supervised ranking on the Stage L1 exact-labeled dataset.
- Recommended first model family: boosted trees with top-k recovery metrics under fixed exact-eval budgets.
