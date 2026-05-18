# Summary

## Status: Decision A

Family-aware survivor selection passes PLAN31 Phase 2 gate. Promoted as recommended K=10 hard irregular Step-3 beam policy.

## What was done

### Phase 0: Fixed residual-aware plumbing
Missing copy from `beam_diag` to pack result. Now works correctly.

### Phase 1: Built existing-policy oracle
From PLAN27 data, computed best global (uniform_mult2), best family-aware, and per-row upper bound.

### Phase 2: Family-aware survivor selection
Tested `family_aware_ambig` (hardA=uniform_mult2, hardB=ambig_scoreband_mult2) and `family_aware_late` (hardA=uniform_mult2, hardB=late_ambig). Both pass with 6/8 improved, 7/8 not worse.

### Phase 3: Coarse-lookahead
Single smoke test shows worse gap. Not promoted.

## Recommended policy

| Family | Policy | Env Vars |
|--------|--------|----------|
| hardA_k10 | uniform_mult2 | PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY=uniform, PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX=2 |
| hardB_k10 | ambig_scoreband_mult2 | PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY=ambig_scoreband, PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_MAX=2 |

Or globally: `uniform_mult2` (PLAN27 A) if family information is not available.

## Artifacts
- `csv/plan31/PLAN31_existing_policy_oracle.csv`
- `csv/plan31/PLAN31_existing_policy_oracle_notes.md`
- `csv/plan31/PLAN31_family_aware_survivor_raw.csv`
- `csv/plan31/PLAN31_family_aware_survivor_compare.csv`
- `csv/plan31/PLAN31_family_aware_survivor_summary.csv`
- `csv/plan31/PLAN31_fine_block_guided_beam_notes.md`
