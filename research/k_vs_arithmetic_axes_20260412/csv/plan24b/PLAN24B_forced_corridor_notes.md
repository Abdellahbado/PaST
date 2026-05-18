# PLAN24B Forced-Entry Corridor Exact DP - Notes

## Purpose
PLAN24 showed that beam-corridor exact DP never entered the search because `sparse_skip_theoretical` blocked it. PLAN24B tests whether the corridor can prune states if exact DP is forced to enter.

## Implementation
- New env var: `PAST_EXACT_CORRIDOR_FORCE_ENTRY=1` (off by default, experimental)
- When corridor is enabled AND force_entry is set, bypass `sparse_skip_theoretical` guardrail
- Time limit clamped by `PAST_EXACT_CORRIDOR_TIME_LIMIT` (default 300s)
- State limit guarded by `PAST_EXACT_CORRIDOR_MAX_STATES` (default 50M)
- New diagnostics: `stop_reason`, `corridor_force_entry`, `corridor_max_states`, `corridor_time_limit`
- Changes in `stateful_dp_solver.cpp`, `stateful_dp_solver.hpp`, `stateful_compare.cpp`

## Target rows
- hardA_k10 seed=0 (representative, strong beam-policy sensitivity)
- hardB_k10 seed=2 (one of the better ambig-scoreband cases)
- Why only 2 rows: This is a diagnostic, not a promotion run.

## Variants
- `standard_step4`: current behavior, no corridor, no force entry
- `forced_corridor_delta1_300s`: corridor enabled, delta=1, force entry, 300s exact limit
- `forced_corridor_delta2_300s`: corridor enabled, delta=2, force entry, 300s exact limit

## Results

### Did forced-entry corridor exact DP actually enter the search?
**No.** All forced-entry rows hit `sparse_skip_overflow` immediately after bypassing the theoretical guardrail. The mixed-radix encoding (single int64) overflows for K=10 at n=1000 because the product of (totals[i] + 1) ≈ 100^10 exceeds int64 range.

### Did it prune any states?
**No.** `exact_diag_corridor_pruned=0` for all rows. Zero states were generated, so zero were pruned.

### Did it improve UB/LB/gap?
**No.** All variants produce identical UB, LB, and gap_pct on both rows. Gaps equal standard (hardA_k10: 0.0273%, hardB_k10: 0.0450%).

### Did it reduce or worsen runtime?
**Neither.** Runtime is nearly identical across variants (~490-680s), dominated by the beam Step 3. The overflow check is O(K) and happens before any search.

### Did it stay under memory cap?
**Yes.** Max RSS ~7.7 GB, well under 16 GB cap.

### Is corridor exact worth continuing?
**No.** The fundamental blocker is the int64 state-space encoding: sparse exact DP cannot represent the state for K=10 at n=1000 regardless of guardrails. The corridor machinery is functional but cannot be tested because the encoding itself is too small.

## Decision
**D**: Corridor still cannot enter meaningfully; abandon corridor under current exact DP. The blocking issue is the int64 mixed-radix encoding overflow, not the theoretical bound guardrail. The sparse exact DP encoding is fundamentally limited to ~K=8 at n=1000 on hard irregular families. No amount of guardrail relaxation or corridor tuning can overcome this.

## Artifacts
- PLAN24B_forced_corridor_raw.csv: 6 rows (2 families × 3 variants)
- PLAN24B_forced_corridor_compare.csv: Side-by-side comparison
- PLAN24B_forced_corridor_notes.md: This file
