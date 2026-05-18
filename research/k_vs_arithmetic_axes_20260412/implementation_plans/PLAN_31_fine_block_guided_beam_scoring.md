# PLAN 31 — Fine-Block Guided Beam Scoring and Family-Aware Survivor Selection

## Objective

Improve hard irregular `K=10` Step-3 incumbent quality without repeating failed block/coarsening/corridor directions.

The core idea is:

> Keep the original fine recovered blocks for the actual beam transitions, but use auxiliary signals to decide which beam states survive.

This differs from PLAN29:

- PLAN29 changed the block sequence by coarsening adjacent blocks.
- PLAN31 must **not** replace fine blocks.
- PLAN31 may use coarse or price-window information only as scoring/lookahead features for the beam.

## Required Read Order

Read these files before coding:

1. `research/k_vs_arithmetic_axes_20260412/ACTIVE.md`
2. `research/k_vs_arithmetic_axes_20260412/iterations/20260428_fine_block_beam_guidance/SUMMARY.md`
3. `research/k_vs_arithmetic_axes_20260412/csv/plan27/PLAN27_step3_adaptive_survivor_notes.md`
4. `research/k_vs_arithmetic_axes_20260412/csv/plan29/PLAN29_multiview_block_reconstruction_notes.md`
5. `research/k_vs_arithmetic_axes_20260412/csv/plan20/PLAN20_phaseA_beam_diagnostics.md`
6. `research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md`
7. Code:
   - `solvers/cpp/stateful_dp_solver.cpp`
   - `solvers/cpp/stateful_dp_solver.hpp`
   - `solvers/cpp/stateful_compare.cpp`
   - `research/k_vs_arithmetic_axes_20260412/run_plan27_gate_a.py`
   - `research/k_vs_arithmetic_axes_20260412/run_plan29_multiview_blocks.py`

## Existing Evidence to Respect

### Do not repeat these failed directions

| Direction | Evidence | Rule |
|---|---|---|
| Plain adjacent coarsening | PLAN29 fails; no view improves >= 4/8 | Do not replace fine blocks with coarse blocks |
| Strict block-local realizability | PLAN28 and PLAN26 fail; base path rejected at layer 0 | Do not use block-local schedulability as the main signal |
| Step-4 corridor | PLAN24/24B blocked by sparse skip and int64 overflow | Do not restart corridor exact DP |
| Local corridor | PLAN25/26 invalid under current block/path assumptions | Do not use local corridor as a method claim |
| Force exact fixed-block DP | PLAN19 hits `skipped_comp_est` | Do not force exact mode for K=10/12 |
| Stronger/wider beam only | PLAN19 `beam_plus` times out more | Do not simply widen the beam |
| Role-based survivors | PLAN23 fails and increases runtime | Do not revive role/quota policy as implemented |
| `smart_reconstruct` global count search | known poor scaling | Do not revive as main path |

### Useful positive signals

| Signal | Meaning |
|---|---|
| `profile_repair_beam` is the useful hard-K incumbent path | Keep it as the base method |
| `uniform_mult2` passes PLAN27 | Best validated global survivor policy so far |
| `ambig_scoreband_mult2` / `late_ambig` help hardB more than hardA | Family-aware policy may help |
| PLAN29 views sometimes improve individual hardB rows | Coarse views contain information, but should guide scoring only |
| `residual_aware` had zero effect while diagnostic stayed `default` | First verify/fix env/config plumbing before judging the idea |

## Hard Constraints

- Do not change accepted baseline defaults silently.
- All new behavior must be behind explicit env toggles.
- Do not delete or overwrite prior artifacts.
- One heavy solver row at a time.
- Memory cap: default 16 GB. If a row needs more, cap at 20 GB maximum and document why.
- Do not run broad sweeps. Use the fixed Gate A rows unless a gate passes.
- Do not promote a policy based on a single best seed.

## Target Rows

Primary Gate A:

- `hardA_k10`, seeds `0,1,2,3`, `n=1000`, `lambda=1.3`
- `hardB_k10`, seeds `0,1,2,3`, `n=1000`, `lambda=1.3`

Only if Gate A passes:

- optional `K=12` smoke on `hardA_k12` and `hardB_k12`, seeds `0,1`

## Phase 0 — Residual-Aware Plumbing Audit

Before adding a new method, verify that existing scoring env vars are actually read.

Problem from PLAN27:

- `residual_aware` produced zero effect.
- Diagnostic `score_policy` stayed `default`.
- This may be a plumbing/build/env bug.

Required work:

1. Find where `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY` should be read.
2. Confirm whether it reaches `block_repair_profile_repair_beam_ub`.
3. Add or fix CSV diagnostics so the row records:
   - `profile_beam_score_policy`
   - residual weight values
   - whether residual-aware scoring was active
4. Run one cheap smoke row:
   - `hardA_k10`, seed `0`, standard vs `residual_aware`
5. Stop Phase 0 only when diagnostics show the requested policy, not `default`.

If the policy still cannot be activated quickly, document the exact blocker and skip residual-aware variants. Do not spend more than one focused debugging pass on this.

## Phase 1 — Best-of-Existing Policy Oracle

Before coding new scoring, compute a small oracle table from existing PLAN27 and PLAN29 rows.

Purpose:

> Estimate how much improvement is possible if we select among already-tested policies per family or per row.

Use existing artifacts:

- `csv/plan27/PLAN27_step3_adaptive_survivor_compare.csv`
- `csv/plan29/PLAN29_multiview_block_reconstruction_compare.csv`

Produce:

- `csv/plan31/PLAN31_existing_policy_oracle.csv`
- `csv/plan31/PLAN31_existing_policy_oracle_notes.md`

Required oracle summaries:

1. best single global policy across all 8 K10 rows
2. best family-aware policy:
   - one policy for hardA
   - one policy for hardB
3. best per-row policy upper bound

If best family-aware improvement is negligible, still continue to Phase 2 because `uniform_mult2` already passed PLAN27; but mark expectations as modest.

## Phase 2 — Family-Aware Survivor Selector

Implement only runner-level or env-level family-aware selection if possible.

Candidate policy:

- hardA_k10: `uniform_mult2`
- hardB_k10: compare:
  - `standard_beam`
  - `ambig_scoreband_mult2`
  - `late_ambig`
  - `uniform_mult2`

Run Gate A.

Produce:

- `csv/plan31/PLAN31_family_aware_survivor_raw.csv`
- `csv/plan31/PLAN31_family_aware_survivor_compare.csv`
- `csv/plan31/PLAN31_family_aware_survivor_summary.csv`

Gate:

- at least `6/8` rows not worse than standard
- mean gap no worse than standard
- runtime increase <= `10%`
- no memory kills

If this passes, it becomes the main candidate. If it fails, keep the best policy as a diagnostic only.

## Phase 3 — Fine-Block Coarse-Lookahead Scoring

Only after Phase 0 and Phase 1.

Goal:

> Use coarse/price-window signals as a score term while keeping the original fine blocks as the actual beam layers.

Allowed implementation:

- Add an experimental score policy:
  - `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY=fine_plus_coarse_lookahead`
- It may compute cheap lookahead features over a small future window, such as:
  - residual count imbalance over the next `W` fine blocks
  - price variation penalty if the node commits too much work in a volatile local window
  - remaining work / count balance relative to relaxed profile prefix

Forbidden implementation:

- Do not replace blocks with coarsened blocks.
- Do not call block-local exact schedulability as a score.
- Do not enumerate a large coarse-pattern pool.

Recommended first variants:

| Variant | Env idea |
|---|---|
| `lookahead_light` | small weights, window 2-3 blocks |
| `lookahead_medium` | moderate weights, window 4-6 blocks |
| `lookahead_plus_uniform` | combine lookahead with `uniform_mult2` |

Gate A:

- compare against `standard_beam` and `uniform_mult2`
- improve at least `4/8` rows vs standard
- not worse at least `6/8` rows vs standard
- runtime increase <= `20%`
- no memory kills

Produce:

- `csv/plan31/PLAN31_fine_block_lookahead_raw.csv`
- `csv/plan31/PLAN31_fine_block_lookahead_compare.csv`
- `csv/plan31/PLAN31_fine_block_lookahead_summary.csv`

## Phase 4 — Decision Note

Create:

- `csv/plan31/PLAN31_fine_block_guided_beam_notes.md`

The note must answer:

1. Did residual-aware scoring actually activate after the audit?
2. What does the existing-policy oracle say?
3. Does family-aware survivor selection beat the current global policy?
4. Does coarse-lookahead scoring help without replacing fine blocks?
5. What should be promoted, kept as diagnostic, or abandoned?

Update docs:

- `research/k_vs_arithmetic_axes_20260412/LOG.md`
- `research/k_vs_arithmetic_axes_20260412/RESULTS.md`
- `research/k_vs_arithmetic_axes_20260412/BLOCKERS.md`
- `research/k_vs_arithmetic_axes_20260412/METHOD_BOUNDARIES.md`
- `research/k_vs_arithmetic_axes_20260412/CURRENT_RESULTS_INDEX.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260428_fine_block_beam_guidance/SUMMARY.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260428_fine_block_beam_guidance/RESULTS.md`
- `research/k_vs_arithmetic_axes_20260412/iterations/20260428_fine_block_beam_guidance/BLOCKERS.md`

## Final Decision Labels

Use exactly one:

- `A`: promote a policy as default candidate for hard K10 beam rows
- `B`: keep as additive optional policy only
- `C`: diagnostic value only; no method promotion
- `D`: abandon direction due to structural blocker
- `E`: implementation bug or incomplete evidence; rerun required before conclusion

