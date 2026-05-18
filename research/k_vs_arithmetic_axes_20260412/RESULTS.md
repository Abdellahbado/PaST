# Results

## 2026-04-30 — PLAN33 Certified Anytime Hard-K Prepass — Decision A

Artifacts:
- raw: `csv/plan33/PLAN33_cert_anytime_raw.csv` (24 rows, 12 plan32c + 12 plan33)
- compare: `csv/plan33/PLAN33_cert_anytime_compare.csv` (12 head-to-head)
- summary: `csv/plan33/PLAN33_cert_anytime_summary.csv` (14 metrics)
- notes: `csv/plan33/PLAN33_notes.md`

Key findings:
- **All 12 plan33 rows cert_stop=1, all gaps ≤ 0.0593%, all UB ≥ LB.**
- PLAN33 avg runtime 1396.61s vs PLAN32C 1527.11s (130.49s faster, with certified semigroup LB).
- Polish improved UB in all 12 rows.
- hardA_k12 s3 PLAN32C panel corrected from 159M (stale, 5 trials) to 133.5M (PLAN33: 5 trials + polish).
- hardB_k12 s3 also updated to PLAN33 values.
- PLAN33 is the recommended hard-K default for tested K10/K12 hard rows.
- Decision A for both K=10 and K=12.

---

## 2026-04-28 — PLAN30 easy-vs-hard fixed-n K-scaling story (`n=1000`, `lambda=1.3`, implements PLAN_16)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan30/PLAN30_easy_k_scaling_raw.csv`
- summary: `research/k_vs_arithmetic_axes_20260412/csv/plan30/PLAN30_easy_k_scaling_summary.csv`
- comparison: `research/k_vs_arithmetic_axes_20260412/csv/plan30/PLAN30_easy_vs_hard_k_boundary.csv`
- notes: `research/k_vs_arithmetic_axes_20260412/csv/plan30/PLAN30_easy_vs_hard_notes.md`

### Key findings

- Easy contiguous-unit families exact through `K=40` at fixed `n=1000`.
- `K=24`: mean runtime 364s, all 4/4 exact.
- `K=30`: mean runtime 683s, all 4/4 exact.
- `K=40`: mean runtime 1552s, all 4/4 exact.
- All rows close at Step 2 (`ffd`). Memory-safe (peak RSS 1.7–4.8 GB).
- Hard irregular exact closure degrades around `K=8–10`; PLAN33 later recovers
  hard K10/K12 as certified finite-gap rows rather than exact closures.
- Sharpens the two-axis claim: difficulty is driven by K × arithmetic interaction, not K alone.

### Decision

**A** — The easy-vs-hard K-scaling story is sufficiently documented for the paper.

---

## 2026-04-28 — PLAN27 Step-3 adaptive survivor policy (`n=1000`, `lambda=1.3`, K=10)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan27/PLAN27_step3_adaptive_survivor_raw.csv`
- compare: `research/k_vs_arithmetic_axes_20260412/csv/plan27/PLAN27_step3_adaptive_survivor_compare.csv`
- summary: `research/k_vs_arithmetic_axes_20260412/csv/plan27/PLAN27_step3_adaptive_survivor_summary.csv`
- notes: `research/k_vs_arithmetic_axes_20260412/csv/plan27/PLAN27_step3_adaptive_survivor_notes.md`

### Key findings

- `uniform_mult2` passes Gate A promotion: 6/8 rows not worse, mean gap 0.0343% vs standard 0.0345%, runtime reduced 14.3%.
- `late_ambig` and `late_residual_ambig` show real signal (5W/3L) but fail the 6/8 not-worse threshold.
- `residual_aware` has zero gap effect; blocked by unresolved env-var/read issue.
- Multiplicity policies are family-dependent: `uniform_mult2` helps hardA, `ambig_scoreband_mult2`/`late_ambig` help hardB.
- All rows memory-safe (peak RSS 4.4–9.2 GB, no kills).

### Decision

**A** — `uniform_mult2` is validated as the best global Step-3 survivor policy, with family-dependence noted.

---

## 2026-04-27 — PLAN26 validate PLAN25 local corridor + multi-idea queue (`n=1000`, `lambda=1.3`, K=10)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan26/PLAN26_multi_idea_raw.csv`
- compare: `research/k_vs_arithmetic_axes_20260412/csv/plan26/PLAN26_multi_idea_compare.csv`
- notes: `research/k_vs_arithmetic_axes_20260412/csv/plan26/PLAN26_multi_idea_notes.md`
- plan: `research/k_vs_arithmetic_axes_20260412/implementation_plans/PLAN26_beam_corridor_multi_idea_queue.md`

### A) Phase 0 — Correctness repairs

- Fixed `lb=ub` bug: local corridor no longer incorrectly sets `lb=ub` when improving UB.
- Added alignment/validation diagnostics to `LocalCorridorDiag`.
- Fixed `merged_blocks` propagation through `RecoveredBlockPackingResult` -> `RelaxedDPResult` in all three solver paths.
- Added base-path survival simulation in `beam_corridor_local_dp`.

### B) Phase 1 — Base-path survival validation (one planned row missing)

Target rows: hardA_k10 s0, hardB_k10 s2.
Variants: standard_step4, local_corridor_delta1, local_corridor_delta2.

Key findings:
- `block_count_mismatch=0` on all rows: block partition alignment is correct after fix.
- `target_offset_l1=0` on all rows: beam assigns all jobs.
- `base_path_survives=0` on ALL local corridor rows.
- Reject reason: `base_candidate_not_found_at_layer_0`.
- Root cause: `evaluate_profile_block_counts` returns `kInf` for base beam candidate on some blocks.
- Missing planned row: `hardB_k10 seed=2 local_corridor_delta2_300s` is not present in the raw artifact. Since `hardB_k10 seed=2 delta1` already fails at layer 0 and hardA fails for both deltas, the invalid-corridor conclusion is still supported, but Phase 1 is not a complete 2x2 delta grid.

Why: `generate_energy_core_patterns` generates patterns by work capacity, not schedulability. The beam validates the full global sequence, not individual blocks. Local corridor's block-by-block evaluation is fundamentally mismatched with the beam's global validation.

### C) Phase 2 & 3 — Cancelled

Blocked by Phase 1 finding. No candidate-set or multi-center variant can succeed if the base path itself is rejected.

### D) Phase 4 — Step-3 beam scoring variants (partial)

Available tested variant: `ambig_scoreband_mult2`. The requested new `residual_aware` and `late_ambig` policies were not implemented in this pass, so this is a partial fallback check rather than a complete Step-3 scoring study.
Rows: hardA_k10 seeds 0,2; hardB_k10 seeds 0,2.

Results:
- hardA_k10 s0: standard gap 0.0273% vs ambig 0.0299% — worse
- hardA_k10 s2: standard gap 0.0217% vs ambig 0.0306% — worse
- hardB_k10 s0: standard gap 0.0391% vs ambig 0.0375% — better
- hardB_k10 s2: standard gap 0.0450% vs ambig 0.0389% — better

Promotion check:
- Not worse on 2/4 rows (needs 3/4): FAILS.
- Improved on 2/4 rows (needs 2/4): PASSES.
- Overall: do not promote.

### E) Decision

**C** — Local corridor invalid due to block/path mismatch. Do not use until fundamentally redesigned. Existing `ambig_scoreband_mult2` remains non-promotable; the requested new Step-3 scoring policies remain untested.

---

## 2026-04-26 — PLAN25 local corridor exact DP (`n=1000`, `lambda=1.3`, K=10)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan25/PLAN25_local_corridor_dp_raw.csv`
- compare: `research/k_vs_arithmetic_axes_20260412/csv/plan25/PLAN25_local_corridor_dp_compare.csv`
- notes: `research/k_vs_arithmetic_axes_20260412/csv/plan25/PLAN25_local_corridor_dp_notes.md`

Scope:

- fixed `n=1000`, `lambda=1.3`
- hardA_k10 seed=0, hardB_k10 seed=2
- Variants: `standard_step4`, `local_corridor_delta1_300s`, `local_corridor_delta2_300s`
- Overall time limit 1200s, internal local corridor limit 300s via `PAST_BEAM_CORRIDOR_LOCAL_TIME_LIMIT`
- Memory cap 16 GB

### A) Implementation

- Added `beam_corridor_local_dp()` with local offset encoding: `id = Σ (offset[j] + delta) * (2*delta+1)^j`
- Avoids global mixed-radix int64 overflow by using per-layer `unordered_map<int, double>` states
- Candidate generation: perturb beam counts with single + pair moves, keep top 50 per block by cost
- Hard state cap: `PAST_BEAM_CORRIDOR_LOCAL_MAX_STATES` (default 5M)
- New diagnostics: 14 CSV fields (`local_corridor_*`)

### B) Results

- All rows valid, memory safe (peak RSS 4–8 GB)
- Local corridor runs successfully:
  - delta1: ~25s, 40–60k states_seen, 5–7k states_kept_max
  - delta2: ~52–72s, 7.6–14.1M states_seen, 1.5–2.0M states_kept_max
- Status consistently `infeasible_corridor`
- `best_ub = inf` on all local corridor rows; incumbent unchanged
- Exact sparse DP still `sparse_skip_theoretical` on all rows (0 states)
- Important caveat: this does not yet prove the corridor lacks a better completion. Because the beam base count vector is inserted as a candidate for every block, the base beam path should be explicitly checked. PLAN25 did not record whether that base path survives.

### C) Answer to key questions

1. Does local corridor avoid int64 overflow? **Yes** — offset encoding stays in 32-bit range.
2. Does it find improving solutions? **No in the recorded rows**, but the reason is not yet validated.
3. Does it scale safely? **Yes for the smoke rows** — memory and time are bounded.
4. Is it mechanically validated? **Not yet** — base-path survival, block alignment, and corridor-limited proof handling must be checked.
5. Is it worth enabling by default? **No** — no measurable benefit and correctness diagnostics are incomplete.

### D) Decision

**Decision: diagnostic hold** — Keep local corridor code in solver but disable by default. PLAN25 validates the offset representation against overflow, but it does not validate the corridor method. Next work must verify base-path survival and ensure corridor-limited UB improvements never set `lb=ub` unless there is a separate global proof.

---

## 2026-04-26 — PLAN24B forced-entry corridor exact DP diagnostic (`n=1000`, `lambda=1.3`, K=10)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan24b/PLAN24B_forced_corridor_raw.csv`
- compare: `research/k_vs_arithmetic_axes_20260412/csv/plan24b/PLAN24B_forced_corridor_compare.csv`
- notes: `research/k_vs_arithmetic_axes_20260412/csv/plan24b/PLAN24B_forced_corridor_notes.md`

Scope:

- fixed `n=1000`, `lambda=1.3`
- hardA_k10 seed=0, hardB_k10 seed=2 (diagnostic only, 2 rows)
- Variants: `standard_step4`, `forced_corridor_delta1_300s`, `forced_corridor_delta2_300s`
- Overall time limit 1200s, internal corridor limit 300s via `PAST_EXACT_CORRIDOR_TIME_LIMIT`
- K=12 not run, not all seeds, not delta3
- Memory cap 16 GB

### A) Implementation

- Added `PAST_EXACT_CORRIDOR_FORCE_ENTRY=1` (off by default, experimental)
- Bypasses `sparse_skip_theoretical` guardrail when corridor is active
- Clamps internal time to `PAST_EXACT_CORRIDOR_TIME_LIMIT`
- State limit: `PAST_EXACT_CORRIDOR_MAX_STATES` (default 50M)
- New diagnostics: `stop_reason`, `corridor_force_entry`, `corridor_max_states`, `corridor_time_limit`

### B) Results

- Forced-entry correctly bypasses theoretical guardrail (`force_entry=1`, `corridor_en=1`, correct deltas)
- **All forced rows hit `sparse_skip_overflow`**: int64 mixed-radix encoding overflows. Product of (totals[i] + 1) for K=10 at n=1000 exceeds int64.
- **Zero corridor pruning** (`exact_diag_corridor_pruned=0`): zero states generated, zero pruned
- Identical UB/LB/gap to standard on all rows
- Runtime identical (~490-680s, dominated by beam). Memory safe (max ~7.7 GB)

### C) Answer to key questions

1. Did forced-entry corridor enter the search? **No** — encoding overflow blocks it.
2. Did it prune states? **No** — no states generated.
3. Did it improve UB/LB/gap? **No** — identical to standard.
4. Did it reduce runtime? **No** — identical.
5. Did it stay under cap? **Yes.**
6. Is corridor worth continuing? **No** — encoding is the fundamental blocker.

### D) Decision

**Decision: D** — Corridor still cannot enter meaningfully; abandon corridor under current exact DP. The blocking issue is the int64 mixed-radix encoding overflow. The sparse exact DP encoding is fundamentally limited to ~K=8 at n=1000 on hard irregular families. No amount of guardrail relaxation or corridor tuning can overcome this.

---

## 2026-04-26 — PLAN24 beam-guided Step-4 exact corridor (`n=1000`, `lambda=1.3`, K=10)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_raw.csv`
- invalid (energy_core misroute): `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_invalid_energy_core_misroute_raw.csv`
- compare: `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_compare.csv`
- summary: `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_summary.csv`
- notes: `research/k_vs_arithmetic_axes_20260412/csv/plan24/PLAN24_beam_corridor_exact_notes.md`

Scope and controls:

- fixed `n=1000`, `lambda=1.3`
- hardA_k10 seeds 0-3, hardB_k10 seeds 0-3
- variants: `standard_step4`, `corridor_delta0`, `corridor_delta1`, `corridor_delta2`, `corridor_widen_0_1_2`
- K=12 probe skipped (no signal from K=10)
- memory guard `16 GB`, one heavy row at a time
- timeout `1200s`/row
- Baseline env: `PAST_RELAXED_BINPACK_SOLVER=profile_repair_beam`, `PAST_PROFILE_REALIZATION_SELECTOR_POLICY=auto_v1`

### A) Critical fix

Initial smoke run (11 rows) used `PAST_RELAXED_BINPACK_SOLVER=energy_core` which produced no beam incumbent (`ub=-1`, `fwd_pack_method=none`, `fwd_pack_outcome=failed`). Corrected to `profile_repair_beam + auto_v1`. Invalid rows preserved in `PLAN24_invalid_energy_core_misroute_raw.csv`.

### B) Corridor implementation

- C++ corridor machinery built cleanly: `ExactCorridor` struct, `set_exact_corridor()`, `clear_exact_corridor()`, pruning checks in dense/sparse exact DP.
- Beam chosen counts exposed from `block_repair_profile_repair_beam_ub` through `RecoveredBlockPackingResult` → `RelaxedDPResult`.
- Diagnostics wired into CSV: `exact_diag_corridor_enabled`, `exact_diag_corridor_delta`, `exact_diag_corridor_pruned`, `exact_diag_corridor_infeasible`.
- Missing diag initialization fixed (corridor fields were not copied to `g_last_exact_dp_diag`).

### C) Results

- 33 valid rows (4 smoke + 29 Phase B).
- All rows reach step4 with `fwd_pack_method=profile_repair_beam`, `beam_status=feasible`.
- Corridor correctly activated: `corridor_enabled=1` with correct deltas.
- **Zero corridor pruning**: `exact_diag_corridor_pruned=0` for all 24 corridor rows. No state was ever pruned by the corridor.
- **Zero corridor infeasibility**: `exact_diag_corridor_infeasible=0` for all rows.
- **Sparse exact DP skipped**: `exact_diag_states_reached=0`, `exact_diag_elapsed≈0` for all rows. Mode is `sparse_skip_theoretical`.
- **Identical UB/LB/gap**: All corridor variants produce exactly the same results as `standard_step4`.
- **No exact closure**: `is_optimal=0` for all rows.
- Comparable runtime: No significant overhead or benefit.
- Memory well under cap: Max RSS ~8.1 GB < 16 GB.

### D) Root cause

The sparse exact DP mode `sparse_skip_theoretical` means the solver determines the theoretical lower bound is already tight enough, so it skips the exact search entirely. This prevents the corridor from having any effect because no states are ever generated or pruned.

### E) Decision

**Decision: D** — No evidence that beam-guided exact corridor improves exact closure, gap, or runtime on hard irregular K=10 rows. The corridor machinery is functional but the sparse exact DP skips the search on these instances, making corridor pruning irrelevant. With zero pruning in 24 corridor rows across 8 instances, the approach is unlikely to yield benefits even if exact DP ran.

---

## 2026-04-25 — PLAN23 role-based survivor policy (`n=1000`, `lambda=1.3`, Gate 1 only)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_raw.csv`
- compare: `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_compare.csv`
- summary: `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_summary.csv`
- notes: `research/k_vs_arithmetic_axes_20260412/csv/plan23/PLAN23_role_based_beam_notes.md`

Scope and controls:

- fixed `n=1000`, `lambda=1.3`
- Gate 1 rows: hardA_k10 seeds 0,1,2; hardB_k10 seeds 0,2
- Gate 1 variants: `standard_beam`, `uniform_mult2`, `ambig_scoreband_mult2`, `role_mult3`, `role_mult3_feas`
- memory guard `12 GB`, one heavy row at a time
- timeout `1200s`/row
- Role policy env vars:
  - `PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY=role`
  - `PAST_PROFILE_REPAIR_BEAM_ROLE_MAX=3`
  - `PAST_PROFILE_REPAIR_BEAM_ROLE_SCORE_BAND=0.08`
  - `PAST_PROFILE_REPAIR_BEAM_ROLE_KEEP_FEAS=0` (role_mult3) or `1` (role_mult3_feas)

### A) Gate 1 evaluation

Gate 1 pass condition: one role variant must beat or tie standard on >= 4/5 rows, improve gap on >= 2/5 rows, and not increase mean runtime by > 20%.

Results:

| variant | wins | losses | ties | improved | rt_increase_pct |
|---|---|---|---|---|---|
| role_mult3 | 1 | 1 | 3 | 1 | +62.7% |
| role_mult3_feas | 1 | 1 | 3 | 1 | +55.5% |

Both role variants failed the pass condition:
- Only 1/5 wins (need >= 4/5 wins+ties)
- Only 1/5 improved gaps (need >= 2/5)
- Runtime increased by > 55% (need <= 20%)

Per-row gaps on Gate 1:

| family | seed | std_gap | role_mult3_gap | role_mult3_feas_gap |
|---|---|---|---|---|
| hardA_k10 | 0 | 0.0172 | 0.0172 | 0.0172 |
| hardA_k10 | 1 | 0.0272 | 0.0283 | 0.0283 |
| hardA_k10 | 2 | 0.0199 | 0.0199 | 0.0199 |
| hardB_k10 | 0 | 0.0391 | 0.0391 | 0.0391 |
| hardB_k10 | 2 | 0.0450 | 0.0440 | 0.0440 |

### B) Step 3 beam interpretation

- All finite-gap role rows have `beam_status=feasible` and `deciding_step=step4`.
- Step 3 produced an incumbent; Step 4 exact DP attempted certification but did not close any gap to zero.
- Beam candidate UB for role variants is sometimes slightly different from standard/uniform (e.g., hardA_k10 s0: beam UB 96,890,106 vs standard 96,890,106 — no material change).
- Gap differences are driven by Step 3 beam quality, not Step 4 success.

### C) Conclusion

Role-based survivor selection did not demonstrate stable improvement over standard beam or uniform multiplicity. It produced identical gaps on 3/5 Gate 1 rows, one loss, and one marginal win (hardB_k10 s2: 0.045% → 0.044%). Runtime increased substantially due to larger candidate pools per key.

**Decision: E** — Gate 1 failed. No survivor-policy change is validated; move next to beam-guided Step 4 certification.

---

## 2026-04-25 — PLAN22B correction pass: ambig_scoreband_mult2 Gate 2 validation (`n=1000`, `lambda=1.3`)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_raw.csv`
- compare: `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_compare.csv`
- summary: `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_summary.csv`
- notes: `research/k_vs_arithmetic_axes_20260412/csv/plan22b/PLAN22B_ambig_scoreband_validation_notes.md`

Scope: same as PLAN22 Gate 2, but specifically runs the missing `ambig_scoreband_mult2` rows.

### A) Gate 2 validation result for ambig_scoreband_mult2

Per-row outcomes (newly validated rows only):

| family | seed | ambig_gap | std_gap | early_gap | winner_vs_std |
|---|---|---|---|---|---|
| hardA_k10 | 2 | 0.0299 | 0.0199 | 0.0199 | standard |
| hardA_k10 | 3 | 0.0359 | 0.0358 | 0.0383 | standard |
| hardB_k10 | 0 | 0.0375 | 0.0391 | 0.0391 | variant |
| hardB_k10 | 1 | 0.0712 | 0.0620 | 0.0691 | standard |
| hardB_k10 | 2 | 0.0389 | 0.0450 | 0.0416 | variant |
| hardB_k10 | 3 | 0.0526 | 0.0514 | 0.0505 | standard |
| hardA_k12 | 0 | 0.0254 | 0.0239 | 0.0239 | standard |
| hardA_k12 | 1 | 0.0494 | 0.0508 | 0.0510 | variant |
| hardB_k12 | 0 | 0.0439 | 0.0481 | 0.0485 | variant |
| hardB_k12 | 1 | inf | inf | inf | tie |

Gate 2 score vs standard_beam: wins=4, losses=5, ties=1.

### B) Step 3 beam interpretation

- All finite-gap rows: `beam_status=feasible`, `deciding_step=step4`.
- Step 3 produced an incumbent on all finite-gap rows; Step 4 exact DP attempted certification but did not close any gap to zero.
- Gap differences between variants are driven by Step 3 beam quality, not Step 4 success.

### C) Corrected conclusion

`ambig_scoreband_mult2` produces the best single gap on the key anchor (hardA_k10 s=0: 0.0172% → 0.0094%) and helps on roughly half of K=10 seeds, but it does NOT reliably generalize beyond Gate 1 (4-5 on Gate 2 vs standard). Mean gap across all 14 rows is 0.0357%, slightly worse than standard (0.0355%) and uniform (0.0353%).

**Corrected Decision: E** — Use `ambig_scoreband_mult2` only as a K=10 quality-improvement candidate, not as a global policy.

---

## 2026-04-25 — PLAN22 adaptive node evaluation / survivor policy (`n=1000`, `lambda=1.3`)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_raw.csv`
- compare: `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_compare.csv`
- summary: `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_summary.csv`
- notes: `research/k_vs_arithmetic_axes_20260412/csv/plan22/PLAN22_adaptive_node_eval_notes.md`

Scope and controls:

- fixed `n=1000`, `lambda=1.3`
- hard irregular ladder A (`{2,3,5,7,11,...}`) and hard irregular ladder B (`{3,5,7,11,13,...}`)
- K = 8, 10, 12
- Gate 1 anchors: hardA_k10 seeds 0,1; hardA_k8 seeds 1,3
- Gate 2 expansion: hardA_k10 seeds 2,3; hardB_k10 seeds 0-3; hardA_k12 seeds 0,1; hardB_k12 seeds 0,1
- memory guard `12 GB`, one heavy row at a time
- timeout `1200s`/row
- variants: `standard_beam`, `uniform_mult2`, `uniform_mult3_control`, `early_mult2`, `ambig_scoreband_mult2`, `hybrid_mult2`

### A) Gate 1 evaluation

Gate 1 decision rule: continue only if one adaptive policy is not worse than standard on at least 3/4 rows and improves gap or runtime on at least 2/4 rows.

Results:

- `ambig_scoreband_mult2`: not worse 3/4, improved 3/4. Best single improvement: hardA_k10 seed 0 gap 0.0172% → 0.0094%.
- `hybrid_mult2`: not worse 2/4, improved 2/4. **Failed Gate 1.**
- `early_mult2`: not worse 3/4, improved 3/4 (all runtime, no gap improvement).
- `uniform_mult2`: not worse 2/4, improved 2/4. **Failed Gate 1.**
- `uniform_mult3_control`: not worse 2/4, improved 2/4. **Failed Gate 1.**

Both `ambig_scoreband_mult2` and `early_mult2` passed Gate 1.

### B) Gate 2 evaluation

Gate 2 ran `standard_beam`, `early_mult2`, and `uniform_mult2` (the script-selected best policy from Gate 1).

Aggregate across all Gate 2 rows:

- `early_mult2`: mean gap 0.0360%, mean runtime 660.2s
- `uniform_mult2`: mean gap 0.0353%, mean runtime 593.1s
- `standard_beam`: mean gap 0.0355%, mean runtime 700.6s

`early_mult2` is slightly faster than standard but does not improve mean gap. `uniform_mult2` is marginally better on mean gap but also seed-dependent.

### C) Policy comparison on tested rows

| policy | rows | mean_gap% | min_gap% | mean_rt(s) |
|---|---|---|---|---|
| standard_beam | 14 | 0.0355 | 0.0169 | 700.6 |
| uniform_mult2 | 14 | 0.0353 | 0.0172 | 593.1 |
| early_mult2 | 14 | 0.0360 | 0.0172 | 660.2 |
| ambig_scoreband_mult2 | 4 | 0.0199 | 0.0094 | 297.8 |
| hybrid_mult2 | 4 | 0.0204 | 0.0079 | 326.3 |
| uniform_mult3_control | 4 | 0.0216 | 0.0101 | 238.7 |

`ambig_scoreband_mult2` and `hybrid_mult2` show the best mean/min gaps on their Gate 1 rows, but `hybrid_mult2` failed Gate 1 due to degradation on K=8 seeds.

### D) Conclusion (superseded by PLAN22B)

The scoreband + diversity filter (`ambig_scoreband_mult2`) is the only policy that both passed Gate 1 and produced material gap improvements on the key anchor (hardA_k10 s=0). It directly addresses the PLAN20B seed-dependence problem by keeping only high-quality, meaningfully different representatives.

**Original Decision: B** — promote `ambig_scoreband_mult2` as the next main candidate.

**Correction (PLAN22B):** Gate 2 validation shows `ambig_scoreband_mult2` does not generalize reliably (4-5 vs standard on Gate 2). Mean gap is slightly worse than standard and uniform. The corrected decision is **E**: use `ambig_scoreband_mult2` only as a K=10 quality-improvement candidate, not as a global policy.

## 2026-04-24 — PLAN19 K=10/12 bounded redesign (`n=1000`, `lambda=1.3`, seeds `0/1`)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_redesign_raw.csv`
- best-of-variant: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_best_variant_summary.csv`
- compare: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_redesign_compare.csv`
- failure shift: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_failure_shift.csv`
- method notes: `research/k_vs_arithmetic_axes_20260412/csv/plan19/PLAN19_k10_k12_method_notes.md`

Scope and controls:

- fixed `n=1000`, `lambda=1.3`, seeds `0/1`
- hard irregular ladder A (`{2,3,5,7,11,...}`) and hard irregular ladder B (`{3,5,7,11,13,...}`)
- K = 10, 12
- memory guard `12 GB`, one heavy row at a time
- timeout `1200s`/row (external watchdog `1320s`)
- variants: `exp_exact_after_beam_300`, `exp_exact_after_beam_600`, `exp_force_exact_300`, `exp_beam_plus`

### A) Redesign 1: exact closure after beam

- `exact_after_beam_300` / `600`: did not recover exact closure on any seed.
- The C++ hook `PAST_PROFILE_REALIZATION_EXACT_AFTER_BEAM_ENABLE=1` did not visibly trigger; rows still show `selector_decision=beam` and `block_dp_status=skipped_selector`.
- `force_exact_300` with selector guardrails raised to 1e12 immediately hits `skipped_comp_est`, confirming that exact fixed-block DP state space / comp_est is astronomically large for these rows (B≈20, merged>16).

Interpretation: exact fixed-block DP is structurally infeasible for K=10/12 hard irregular rows under practical budgets. The boundary is not a selector calibration issue; it is a fundamental state-space explosion.

### B) Redesign 2: routing override

- Baseline `energy_core` on K>=10 hard irregular rows consistently produces no incumbent (selector bypass) and wastes 500-1200s.
- Skipping baseline and routing directly to `profile_repair_beam` saves 30-50% runtime with no quality loss.
- This override is justified and should be kept for K>=10 hard irregular rows.

### C) Redesign 3: stronger K=12 beam

- `beam_plus` timed out on 6/8 K=12 seeds with no incumbent.
- On the 2 seeds where it produced an incumbent, gaps were identical to standard reroute but runtime was longer.
- Interpretation: stronger beam does not help at K=12 under current budgets; it increases timeout rate without improving quality.

### D) Boundary conclusion

PLAN19 confirms and sharpens the PLAN18 boundary:

- K=10: no exact rows; finite-gap incumbents (~0.02-0.06%) are the practical ceiling.
- Historical PLAN19 K=12 status: mostly timeout/no-incumbent; occasional finite-gap incumbents when beam succeeds, before PLAN33 recovery.
- The exactness boundary is real and not bypassable by simple guardrail relaxation or beam strengthening within current budgets.

Current-use note:

- This PLAN19 statement is historical boundary/negative-redesign evidence.
- Current hard K10/K12 incumbent status is PLAN33: all tested rows have valid
  certified finite gaps <= 0.0593%.

Recommendation:
1. Accept the boundary: exact closure at K=10/12 on hard irregular families is infeasible under current fixed-block-DP budgets.
2. Keep the routing override for K>=10 hard irregular.
3. Do not pursue stronger beams for K=12.
4. If further closure is needed, consider alternative exact methods (MIP/SAT) or better Step-2 heuristics, not fixed-block DP expansion.


## 2026-04-24 — PLAN18 fixed-n K-boundary refinement (`n=1000`, `lambda=1.3`, seeds `0/1/2/3`)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_n1000_raw.csv`
- best-of-route: `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_best_of_route.csv`
- by-K summary: `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_summary_by_k.csv`
- failure signatures: `research/k_vs_arithmetic_axes_20260412/csv/plan18/PLAN18_k_boundary_refine_failure_signatures.csv`

Scope and controls:

- fixed `n=1000`, `lambda=1.3`, seeds `0/1/2/3`
- hard irregular ladder A (`{2,3,5,7,11,...}`) and hard irregular ladder B (`{3,5,7,11,13,...}`)
- K = 8, 10, 12
- memory guard lowered to `12 GB`, one heavy row at a time
- timeout `1200s`/row (external watchdog `1320s`)
- routes: baseline `energy_core` + additive `profile_repair_beam/auto_v1` reroute

### A) K = 8

- hardA_k8: 2/4 exact (seeds 0,2 via baseline Step 3); seeds 1,3 are finite-gap via reroute.
- hardB_k8: 2/4 exact (seeds 0,2 via baseline Step 3); seeds 1,3 are finite-gap via baseline Step 4 (small gaps ~0.005%).

Interpretation: K=8 is the last K where exact closure still occurs on irregular ladders, and it is already seed-dependent.

### B) K = 10

- hardA_k10: 0/4 exact. All seeds produce finite-gap incumbents via irregular_reroute (gaps 1.7%–3.6%). Baseline rows emit no incumbent.
- hardB_k10: 0/4 exact. Three seeds produce finite gaps via reroute (3.9%–6.2%); one seed times out in Step 3.

Interpretation: K=10 is consistently non-exact. The dominant behavior is finite-gap after Step 4 via the additive reroute.

### C) K = 12

- hardA_k12: 0/4 exact. Two seeds produce finite gaps via reroute (2.4%, 4.0%); two seeds time out.
- hardB_k12: 0/4 exact. One seed produces a finite gap via reroute (2.9%); three seeds time out with no incumbent.

Interpretation at the time: K=12 was mostly budget-limited under this 1200s/12GB cap. PLAN33 later superseded the no-incumbent status with certified finite-gap rows.

### D) Boundary conclusion

The refined practical boundary at fixed `n=1000` on hard irregular arithmetic ladders:

- exactness boundary: around K=8 (mixed, seed-dependent);
- finite-gap boundary: K=10 is the last K where finite incumbents are usually produced;
- historical budget-limited boundary: K=12 was mostly timeout/no-incumbent in PLAN18, before PLAN33 certified prepass recovery.

Dominant failure mode near the boundary:
- baseline route: `selector_bypass` (`non_mainline_solver`) leading to no incumbent;
- additive reroute: `finite_gap_after_step4` at K=10, shifting to `no_incumbent_timeout` at K=12.

This sharpens the prior PLAN17 conclusion: the first hard-K boundary is between K=8 and K=10, not at K=12.


## 2026-04-23 — PLAN17 fixed-n K-axis boundary study (`n=1000`, `lambda=1.3`, seeds `0/1`)

Artifacts:

- raw: `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_n1000_raw.csv`
- by-family summary: `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_n1000_summary_by_family.csv`
- by-K summary: `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_n1000_summary_by_k.csv`
- boundary classification: `research/k_vs_arithmetic_axes_20260412/csv/plan17/PLAN17_k_axis_boundary_classification.csv`

Scope and controls:

- fixed `n=1000`, `lambda=1.3`, seeds `0/1`
- three ladders: easy unit-contiguous, hard irregular A, hard irregular B
- memory guard `16 GB`, one heavy row at a time
- timeout `900s`/row (external watchdog `1020s`)
- route-policy recorded per row (`variant_label`, `route_policy`)

### A) Easy ladder (`{1..K}`)

Outcome:

- all tested K values (`2,4,6,8,10,12,16,20`) close exactly at Step 2 on both seeds.
- dense Step-2 fastpath variants (`K>=8`) keep exactness and improve runtime across this ladder.

Interpretation:

- at fixed `n=1000`, increasing K alone is not sufficient to make the pipeline hard when arithmetic is favorable (has `1`, contiguous, semigroup density at prefix 100 = 1.0).

### B) Hard irregular ladders A and B

Outcome pattern:

- `K=4`: exact closure (Step2/Step3 depending on seed/ladder).
- `K=6`: exact closure via Step 3 under the accepted route in both ladders and both seeds.
- `K=8`: first degradation appears (mixed exact vs finite-gap/unresolved by seed).
- `K=10`: no exact closure in this budget; finite-gap/unresolved rows dominate.
- `K=12,16,20`: mostly timeout-limited under the current budget, with a remaining finite-gap reroute row at `hardA_k12`.

Additive irregular reroute (`profile_repair_beam + auto_v1`):

- applied explicitly only when baseline bypass/no-incumbent behavior was observed;
- produced finite-gap incumbents on some `K=8..10` rows, but did not recover exact closure at higher K in the current time/memory budget.

### C) Boundary classification result

The corrected PLAN17 boundary table still places the first stable transition from exact behavior to hard behavior at approximately `K=8` on hard irregular arithmetic ladders.

Practical paper-facing conclusion from this pass:

- the first hard-K boundary at fixed `n=1000` is around `K=8` for irregular arithmetic families, while easy unit-contiguous families remain exact through `K=20`.
- therefore, K is not the primary difficulty axis by itself; difficulty is driven by K × arithmetic interaction.
- note on artifact hygiene: the corrected PLAN17 summaries are variant-separated; they no longer mix baseline and reroute rows in the same exact-count denominator.


## 2026-04-22 — PLAN_14 dense-unit large-K diagnosis and fast-path recovery (`g12345678910 = {1..10}`)

Artifacts:

- diagnosis:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_g12345678910_diagnosis.csv`
- timeout/memory checkpoint probe:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_g12345678910_checkpoint_probe.csv`
- additive fast-path / count-FFD comparison:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_g12345678910_fastpath_compare.csv`
- `{1..20}` smoke placeholder:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan14/PLAN14_dense_unit_1_20_smoke.csv`

Scope and constraints respected:

- baseline package kept unchanged and explicit.
- no silent baseline replacement.
- no energy-core retuning-first, no column generation, no global Step-3 redesign.
- new behavior is additive and toggle-gated.

### A) Instrumented diagnosis (baseline path)

Required control and transition rows were run on `{1..10}`:

- `n=3500, seed=0` (exact control):
  - exact closure at Step 2 (`ffd`), `UB=LB=182357062`, `runtime=504.2145s`.
  - key timings from row:
    - `t_fwd_relax=504.0244s`
    - `t_fwd_pack_profile_recovery=0.2443s`
    - `t_fwd_pack_to_first_candidate=4.3453s`
    - `t_fwd_pack_ffd_only=98.3031s`
  - peak RSS: `5.491 GB`.

- `n=4500, seed=0` baseline:
  - external timeout, no emitted incumbent.
  - peak RSS: `7.569 GB`.

- `n=5000, seed=0/1` baseline:
  - external timeout on both seeds, no emitted incumbent.
  - peak RSS:
    - seed 0: `5.973 GB`
    - seed 1: `6.829 GB`.

Diagnosis interpretation:

- the main wall is before returning a row under current baseline runtime window.
- completed large `{1..10}` rows still close in Step 2.
- this supports the working hypothesis: easy dense unit-containing family, but expensive generic pipeline at larger `n`.

### B) Timeout / memory checkpoint probe

`PLAN14_g12345678910_checkpoint_probe.csv` now records stage/memory metadata for failure rows and controls.

Observed probes:

- timeout probes (`n=5000`, seeds `0/1`, `time_limit=900s`):
  - both external timeout, no incumbent emitted (`ub=-1, lb=-1`).
- forced memory-kill probes (`n=5000`, seeds `0/1`, tight cap):
  - both `memory_limit_kill` rows, no incumbent emitted.
- fast-path control probes (`n=5000`, seeds `0/1`):
  - exact Step-2 closure with finite `UB/LB`, showing checkpoint table now contains informative successful controls beside failure-stage rows.

### C) Dense-unit Step-2 fast-path experiment (additive)

Implemented additive toggles and diagnostics in solver:

- `PAST_DENSE_UNIT_STEP2_FASTPATH=1`
- `PAST_DENSE_UNIT_FASTPATH_K_MIN=8`
- trigger condition uses dense unit-containing shape (`has_one`, contiguous sizes, large `K`).

Added run-level diagnostics include:

- `fwd_dense_unit_fastpath_active`
- `fwd_count_based_ffd_active`
- `fwd_step2_reached`
- `fwd_step2_produced_ub`
- `t_fwd_pack_merge_blocks`
- `t_fwd_pack_to_first_candidate`
- `t_fwd_pack_ffd_only`

Result summary (`PLAN14_g12345678910_fastpath_compare.csv`):

- `n=3500, seed=0`:
  - baseline exact Step 2: `565.3569s`
  - fastpath exact Step 2: `384.4468s` (exactness preserved, faster)

- `n=5000` baseline:
  - seed 0/1 timeout (`1200s` window), no incumbent row.

- `n=5000` fastpath:
  - seed 0 exact Step 2: `840.5427s`, `UB=LB=259936545`
  - seed 1 exact Step 2: `741.2518s`, `UB=LB=260947838`

So `{1..10}` now closes exactly at `n=5000` on both seeds via additive fast-path Step 2.

### D) Count-based FFD experiment (explicit additive variant)

Toggle used:

- `PAST_COUNT_BASED_FFD=1` (with fast-path on)

Rows:

- `n=5000`, seeds `0/1`, variant `fastpath_count_ffd`.

Outcome:

- exact on both seeds with same optimal values as fastpath `ffd`.
- runtime in this run set is slightly slower than non-count fastpath:
  - seed 0: `887.9283s`
  - seed 1: `749.7764s`
- still additive, exact, and useful as a memory-safe realization variant.

### E) `{1..20}` smoke status

`PLAN14_dense_unit_1_20_smoke.csv` was written as explicit skipped placeholders:

- both requested smoke rows (`n=1000`, `n=2000`) recorded as `status=skipped`
- reason: current run harness family-id map contains paper groups only; `{1..20}` family id is not yet wired in payload builder path.

This clarifies immediate next step toward `{1..20}`:
- add explicit `{1..20}` family wiring in the run harness/group map, then rerun smoke.

### Bottom-line technical conclusion for this phase

For `{1..10}`:

- baseline failure mode at `n=5000` is runtime-window termination with no emitted incumbent.
- completed large rows are Step-2 closures.
- additive dense-unit fast-path resolves the immediate scaling blocker at `n=5000` (seeds `0/1`) with exact Step-2 closure and lower runtime than timed-out baseline window.
- issue classification for original blocker: unnecessary downstream/general-pipeline overhead around Step-2 path selection/execution, not intrinsic Step-2 impossibility and not an energy-core-first deficiency.


## 2026-04-21 — PLAN_13 two-track recovery (`{1..10}` and `g37` K=2 reroute)

Artifacts:

- Track A table:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_easyfamily_g12345678910.csv`
- Track B reroute table:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_g37_k2_reroute.csv`
- additive comparisons:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_variant_compare.csv`
  - `research/k_vs_arithmetic_axes_20260412/csv/plan13/PLAN13_g37_k2_variant_compare.csv`

Run mode / baseline integrity:

- binary: `solvers/cpp/build/stateful_compare`
- workflow: `ablation-stdin step1_exact_guided`
- accepted package preserved (no baseline rewrite):
  - `PAST_RELAXED_BINPACK_SOLVER=energy_core`
  - `PAST_BLOCK_REPAIR_COMPLETION_MODE=direct`
  - `PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS=500000000`
  - `PAST_BLOCK_REPAIR_EC_STRONGER_CENTER=0`
  - `PAST_BLOCK_REPAIR_EC_DIVERSIFY=0`
  - `PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA=0`
  - `PAST_BLOCK_REPAIR_EC_TWO_PHASE=0`
  - `PAST_BLOCK_REPAIR_EG_STATE_KEEP=60000`

### A) Track A (`g12345678910={1..10}`) bounded recovery outcome

Rows executed at `n=5000`:

- baseline (`energy_core`), seeds `0,1`
- additive reroute (`profile_repair_beam` + `auto_v1`), seeds `0,1`
- additive incumbent-source probe (`PAST_EXACT_INCUMBENT_SOURCE=i0`), seed `0`

Result in this bounded run window:

- baseline rows hit external timeout at the configured run cap;
- additive reroute probes (`exp_mainline_reroute`, `exp_step2_incumbent_i0`)
  hit bounded memory-limit kill under strict memory-safe caps;
- therefore no measured extension to exact `n=5000` (or `n=6000`) was achieved
  in this pass.

Memory-safe diagnostics (Track A):

- `PLAN13_easyfamily_g12345678910.csv` now records `peak_rss_kb/peak_rss_gb`
  and `memory_killed`.
- observed peaks remained below enforced hard cap (`<=16 GB`) in all recorded
  rows.

Interpretation:

- this remains an unresolved runtime wall for the easy-family recovery track;
- no baseline policy change is justified from this evidence.

### B) Track B (`g37={3,7}`) K=2 reroute-and-diagnose outcome

Required rows rerun under intended K=2 Step-3 exact path:

- `n=750,1000,1500,2500,3500,5000`, seed `0`;
- seed `1` reruns for recovered rows.

Observed routing and closure:

- selector policy/decision/reason:
  - `auto_v1 / exact / k2_exact_default` on all reroute rows;
- Step-3 mode and method:
  - `step3_mode=exact`, `fwd_pack_method=profile_realization_dp_exact`;
- internal exact profile realization:
  - `fwd_block_dp_status=feasible`;
- final result:
  - all tested rows close exactly (`UB=LB`, `gap=0.0000%`), deciding step `step3`,
    no Step-4 exact fallback use (`diag_exact_dp_used=0`).

Failure-stage classification required by task:

- reroute rows classify as `recovered_or_unclassified` (no failure stage present)
  because exact K=2 Step-3 path is entered and closes.

Memory-safe diagnostics (Track B):

- reroute rows were run sequentially with strict process memory caps and RSS
  monitoring.
- one strict-cap artifact at `n=5000, seed=1` was rechecked under a still-safe
  higher cap (below `16 GB`) and closed exactly; archived final reroute table
  reflects the successful rerun.

### C) Corrected interpretation versus prior plan05/plan11 evidence

The prior unresolved `g37` rows in accepted ledgers were observed with:

- `selector_decision=not_applicable`
- `selector_reason=non_mainline_solver`
- `step3_mode=none`

So those rows were not testing the intended K=2 Step-3 exact profile-realization
route. Under proper reroute, `g37` is recovered through at least `n=5000` in
this campaign.

## 2026-04-20 — PLAN_11 paper-group frontier extension after K=4 generator fix

Artifacts:

- PLAN11 baseline extension table:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`
- PLAN11 additive experiment table:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan11/PAPER_GROUPS_PLAN11_variant_compare.csv`
- refreshed source ledger:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- refreshed summary note:
  - `research/k_vs_arithmetic_axes_20260412/PAPER_GROUPS_EXTENSION_SUMMARY.md`

Run mode and preserved baseline package:

- binary: `solvers/cpp/build/stateful_compare`
- workflow: `ablation-stdin step1_exact_guided`
- package kept unchanged:
  - `PAST_RELAXED_BINPACK_SOLVER=energy_core`
  - `PAST_BLOCK_REPAIR_COMPLETION_MODE=direct`
  - `PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS=500000000`
  - `PAST_BLOCK_REPAIR_EC_STRONGER_CENTER=0`
  - `PAST_BLOCK_REPAIR_EC_DIVERSIFY=0`
  - `PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA=0`
  - `PAST_BLOCK_REPAIR_EC_TWO_PHASE=0`
  - `PAST_BLOCK_REPAIR_EG_STATE_KEEP=60000`

### A) Source-of-truth cleanup (required first)

`PAPER_GROUPS_PLAN05_n_extension.csv` was cleaned before extension runs:

- removed duplicate refreshed `g3567` entries at `(n,seed)=
  (2500,0/1),(3500,0/1),(5000,0/1)`.
- post-clean check: zero duplicate logical keys by `(family_id,n,lambda,seed)`.

### B) Group-by-group frontier outcomes (lambda=1.3, seeds 0/1)

`g3567` (stress-first order):

- `n=6000`: exact on both seeds (Step 3, `block_repair_energy_core`).
- `n=7000`: both seeds timeout/kill (`returncode=-9`).
- `n=8000`: both seeds immediate crash with
  `libc++abi ... std::length_error: vector` (`returncode=-6`).

Easy-scalable pass:

- `g24`:
  - exact through `n=10000` (both seeds), Step 2 decided.
- `g12357`:
  - exact through `n=8000`, timeout at `n=10000` (both seeds).
- `g246810`:
  - exact at `n=6000`, then `std::length_error` crashes at `n=7000,8000`.
- `g12345678910`:
  - timeout at `n=6000,7000` (existing `n=5000` timeout remains).

Diagnosed difficult families:

- `g810`:
  - `std::length_error` crash starts at `n=6000`; same failure at `n=7000,8000`.
- `g37`:
  - exact only through `n=600`.
  - `n=750,1000`: sparse exact times out without closure.
  - `n=1500,2500,3500,5000`: Step 4 is entered, but exact fallback still fails
    (`ub=lb=-1`, `is_optimal=0`).
  - `n=6000,7000`: unresolved with Step-4 entry
    (`diag_exact_dp_used=1`, no finite UB/LB closure).

### C) Additive-only experiment (no promotion)

Experiment executed only on stalled group `g810`:

- baseline vs `exp_g810_force_beam`
  (`PAST_PROFILE_REALIZATION_SELECTOR_POLICY=force_beam`), rows:
  `n=7000,8000`, seeds `0,1`.

Outcome:

- both variants fail identically with `std::length_error` (`returncode=-6`).
- no evidence for improvement; baseline remains unchanged.

### D) Updated practical frontiers from this pass

- `g24`: exact frontier extended to `n=10000`.
- `g12357`: exact frontier extended to `n=8000`.
- `g3567`: exact frontier extended to `n=6000`; regime break at `n=7000`.
- `g246810`: exact frontier now `n=6000`, then runtime/robustness failure regime.
- `g12345678910`, `g810`, `g37`: no exact frontier extension in this PLAN11 pass.
  Current correction: PLAN13 reroute later closes `g37` through tested
  `n=5000` rows with Step-3 `profile_realization_dp_exact`, and PLAN14
  recovers `{1,...,10}` at `n=5000` with dense-unit Step-2 fastpath.

## 2026-04-19 — PLAN_10 K=4 generator policy decision (DP@K=4 + signature-dedup measurement)

Artifacts:

- Phase-A DP@K=4 run table:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_generator_dp4.csv`
- Full generator comparison table:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_generator_compare.csv`

Run mode:

- binary: `solvers/cpp/build/stateful_compare`
- workflow: `ablation-stdin step1_exact_guided`
- baseline env package preserved:
  - `PAST_RELAXED_BINPACK_SOLVER=energy_core`
  - `PAST_BLOCK_REPAIR_COMPLETION_MODE=direct`
  - `PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS=500000000`
  - `PAST_BLOCK_REPAIR_EC_STRONGER_CENTER=0`
  - `PAST_BLOCK_REPAIR_EC_DIVERSIFY=0`
  - `PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA=0`
  - `PAST_BLOCK_REPAIR_EC_TWO_PHASE=0`
  - `PAST_BLOCK_REPAIR_EG_STATE_KEEP=60000`

Additional tested variants:

- `dp4_generator`: `PAST_BLOCK_REPAIR_PATTERN_DP_K=4`
- `dp4_generator_dedup_off`: `PAST_BLOCK_REPAIR_PATTERN_DP_K=4` +
  `PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP=0`

Memory-safety protocol (all heavy rows):

- one heavy run at a time,
- active RSS monitoring,
- hard stop threshold `16.5 GB` RSS,
- memory-unsafe/crashed runs excluded from accepted comparisons.

### A) Exactness gate status (required active K=4 rows)

Required rows:

- continuity: `3567_plus n=3500,5000`, seeds `0,1`
- hard paper: `g3567 n=2500,3500,5000`, seeds `0,1`, `lambda=1.3`

Result:

- baseline generator: `10/10` exact,
- `dp4_generator`: `10/10` exact,
- `dp4_generator_dedup_off`: `10/10` exact,
- all rows remain Step-3 decided (`diag_step3_decided=1`,
  `diag_step4_decided=0`).

### B) Phase-A decision: should K=4 use DP-style pattern generator?

Comparison (`dp4_generator` vs baseline):

- hard `g3567` rows:
  - runtime mean: `1083.240s -> 250.839s` (`-76.8%`)
  - pattern-generation mean:
    `832.942s -> 2.594s` (`-99.7%`)
- continuity rows:
  - runtime mean: `415.663s -> 294.919s` (`-29.0%`)
  - pattern-generation mean:
    `132.285s -> 1.552s` (`-98.8%`)
- all required rows:
  - runtime mean: `816.209s -> 268.471s` (`-67.1%`)

Phase-A verdict:

- DP-style generator at `K=4` is a clear exactness-safe and runtime-positive win.

### C) Phase-C signature-dedup usefulness measurement

Measured by `dp4_generator` vs `dp4_generator_dedup_off` on required rows:

- generated/retained pattern totals are unchanged row-by-row,
- exactness unchanged (`10/10` exact in both),
- runtime favors dedup-off slightly in aggregate:
  - all rows mean runtime: `268.471s -> 262.620s` (`-2.2%`)
  - hard `g3567` mean runtime: `250.839s -> 241.048s` (`-3.9%`)

Interpretation:

- string-key signature dedup gives negligible removal benefit in this K=4 gate,
- it is small overhead in the now-fast DP@K=4 generator path.

### D) Final K=4 generator package selected

Selected policy:

- `K=4` uses DP-style generator by default
  (`PAST_BLOCK_REPAIR_PATTERN_DP_K` default now resolves to `4` for `K=4`).
- K=4 signature dedup disabled by default
  (`PAST_BLOCK_REPAIR_EC_SIGNATURE_DEDUP` default now resolves to `0` for `K=4`).
- non-K=4 defaults remain continuity-safe and unchanged (`DP_K=5`, dedup on).

Code location:

- `solvers/cpp/stateful_dp_solver.cpp`

Runtime objective status:

- materially improved on hard active K=4 rows while preserving exactness and
  memory safety.

### E) Paper-group source-of-truth refresh (K=4 no longer stale)

Updated artifacts after revalidation under current exact K=4 package:

- `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- `research/k_vs_arithmetic_axes_20260412/PAPER_GROUPS_EXTENSION_SUMMARY.md`

Current `g3567` status in source-of-truth summary:

- largest exact `n` at `lambda=1.3` is now `5000` (seeds `0,1`),
- old finite-gap/timeout-era K=4 boundary is superseded in paper-group summary.

## 2026-04-19 — PLAN_10 K=4 continuity-safe speedup baseline and pass-1 ablation

Artifacts:

- baseline:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_baseline.csv`
- post-pass:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_after_pass1.csv`
- combined ablation table:
  - `research/k_vs_arithmetic_axes_20260412/csv/plan10/PLAN10_k4_speedup_ablation.csv`

Run mode:

- binary: `solvers/cpp/build/stateful_compare`
- workflow: `ablation-stdin step1_exact_guided`
- baseline env package:
  - `PAST_RELAXED_BINPACK_SOLVER=energy_core`
  - `PAST_BLOCK_REPAIR_COMPLETION_MODE=direct`
  - `PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS=500000000`
  - `PAST_BLOCK_REPAIR_EC_STRONGER_CENTER=0`
  - `PAST_BLOCK_REPAIR_EC_DIVERSIFY=0`
  - `PAST_BLOCK_REPAIR_EC_ADAPTIVE_DELTA=0`
  - `PAST_BLOCK_REPAIR_EC_TWO_PHASE=0`
  - `PAST_BLOCK_REPAIR_EG_STATE_KEEP=60000`

Memory safety for all heavy runs:

- one instance at a time,
- per-process RSS monitoring,
- kill threshold `16.5 GB` RSS,
- memory-unsafe/crashed runs excluded from accepted results.

### A) Required K=4 baseline closure (clean plan10 baseline)

Required rows:

- continuity: `3567_plus n=3500,5000`, seeds `0,1`
- hard paper rows: `g3567 n=2500,3500,5000`, seeds `0,1`, `lambda=1.3`

Result:

- exact closure on all required rows: `10/10` exact,
- Step-3 decisive on all required rows (`diag_step3_decided=1` and
  `diag_step4_decided=0` for all rows),
- peak RSS remained bounded (about `4.8 GB` to `9.6 GB`).

### B) Implemented speedup pass (Phase B1 + B3)

Code change location:

- `solvers/cpp/stateful_dp_solver.cpp`

Implemented changes:

1. Pattern-generation bounded selection (`nth_element`) in:
   - DP per-work buckets,
   - DFS per-work buckets,
   - final flat trimming before retained-prefix sort.
2. Phase-1 feasible-beam partial selection (`nth_element`) before sorting kept
   prefix.

### C) Before/after outcome on required rows

Exactness gate:

- preserved (`10/10` exact after pass).

Runtime (mean) comparison:

- hard `g3567` required rows:
  - baseline: `1083.240s`
  - pass-1: `1252.507s`
  - delta: `+169.267s` (`+15.6%`)
- continuity required rows:
  - baseline: `415.663s`
  - pass-1: `426.928s`
  - delta: `+11.265s` (`+2.7%`)
- all required rows:
  - baseline: `816.209s`
  - pass-1: `922.275s`
  - delta: `+106.066s` (`+13.0%`)

Dominant shifted metric:

- `fwd_ec_time_pattern_generation` increased on most required rows,
- `fwd_ec_time_phase1` remains zero in this baseline package (`EC_TWO_PHASE=0`),
- `fwd_ec_time_exact_core` remained small and not dominant.

### D) PLAN_10 pass-1 verdict

This pass is exactness-safe but fails the runtime objective.

Decision:

- disqualify this B1/B3 implementation as a speedup package,
- keep the continuity-safe baseline package as the current K=4 best package,
- next PLAN10 iteration must reduce early-stage runtime (especially pattern
  generation) while preserving exactness and memory bounds.

## 2026-04-17 — PLAN_08 energy-core fortification campaign

Artifacts:

- `research/k_vs_arithmetic_axes_20260412/csv/plan08/PLAN08_energy_core_campaign.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan08/PLAN08_energy_core_campaign.json`

Run mode:

- binary: `solvers/cpp/build/stateful_compare`
- workflow: `ablation-stdin step1_exact_guided`
- env:
  - `PAST_RELAXED_BINPACK_SOLVER=energy_core`
  - `PAST_BLOCK_REPAIR_COMPLETION_MODE=direct`

Code note:

- added direct-completion size guard
  `PAST_BLOCK_REPAIR_COMPLETION_DIRECT_MAX_CELLS` (default `120000000`) with
  fallback to cheap completion when exceeded, to avoid large-row process kills.

### A) Required `g3567` rows (`lambda=1.3`, seeds 0/1)

| n | seed | runtime (s) | UB | LB | gap % | deciding step | exact? |
|---:|---:|---:|---:|---:|---:|---|---|
| 1000 | 0 | 428.1329 | 50,816,699 | 50,816,699 | 0.0000 | step3 | yes |
| 1000 | 1 | 52.8225 | 48,526,122 | 48,526,122 | 0.0000 | step3 | yes |
| 1500 | 0 | 1096.3171 | 74,225,131 | 74,225,131 | 0.0000 | step3 | yes |
| 1500 | 1 | 138.9668 | 73,679,253 | 73,679,253 | 0.0000 | step3 | yes |
| 2500 | 0 | 2266.1417 | 123,804,803 | 123,795,529 | 0.0075 | step4 | no |
| 2500 | 1 | 595.3949 | 122,885,430 | 122,882,084 | 0.0027 | step4 | no |
| 3500 | 0 | 3165.6152 | 173,210,721 | 173,198,908 | 0.0068 | step4 | no |
| 3500 | 1 | 727.1822 | 172,579,193 | 172,557,068 | 0.0128 | step4 | no |
| 5000 | 0 | 4876.1610 | 247,866,809 | 247,842,042 | 0.0100 | step4 | no |
| 5000 | 1 | 935.2845 | 248,084,131 | 248,052,411 | 0.0128 | step4 | no |

Aggregate:

- exact rows: `4/10` (all at `n<=1500`)
- Step-3 decisive: `4/10`; Step-4 decisive: `6/10`
- seed-runtime asymmetry persists (mean runtime seed0 `2366.47s` vs seed1
  `489.93s`).

### B) Fortified vs recovered baseline (apples-to-apples on `g3567`)

Baseline source for comparison:

- `research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- filtered to exact PLAN_07-recovered rows at same `(n, seed, lambda=1.3)`.

Observed deltas:

- runtime ratio (fortified / recovered baseline):
  - min `1.53x`, mean `1.90x`, max `2.10x`.
- quality:
  - recovered baseline: exact on all required rows.
  - fortified run: finite gaps on all rows with `n>=2500`.

Interpretation:

- current fortification package improves observability and robustness but does
  not yet preserve the recovered large-`n` K=4 closure quality/runtime frontier.

### C) Instrumentation highlights (`fwd_ec_*` diagnostics)

Across required `g3567` rows:

- pattern retention ratio stayed tight (`~1.0%` seed0, `~1.7%` seed1), so pool
  pruning is active.
- dominant time terms are:
  - pattern generation (`fwd_ec_time_pattern_generation`),
  - phase-1 feasible beam (`fwd_ec_time_phase1`).
- exact-core traversal (`fwd_ec_time_exact_core`) is a small fraction of runtime
  on most rows.
- hard rows frequently end with `fwd_ec_delta_used=-1` (no closing delta level
  reached before fallback), matching Step-4 entry.

### D) Historical continuity checks (`3567_plus`)

| n | seed | runtime (s) | UB | LB | gap % | deciding step | exact? |
|---:|---:|---:|---:|---:|---:|---|---|
| 3500 | 0 | 739.3686 | 69,468,305 | 69,450,708 | 0.0253 | step4 | no |
| 3500 | 1 | 389.2468 | 67,994,176 | 67,970,943 | 0.0342 | step4 | no |
| 5000 | 0 | 1251.5633 | 98,115,931 | 98,093,931 | 0.0224 | step4 | no |
| 5000 | 1 | 695.8961 | 98,116,929 | 98,082,929 | 0.0347 | step4 | no |

Continuity status:

- historical `3567_plus` closure is **not preserved** in current fortified
  configuration (all four rows finite-gap).

### E) Transfer checks

Required transfer rows (all exact):

- `g12357`: `n=1000,1500,2500` x seeds `0,1` => `6/6` exact.
- `g246810`: `n=1000,1500,2500` x seeds `0,1` => `6/6` exact.

Optional transfer rows (executed, all exact):

- `g12345678910`: `n=1000,1500` x seeds `0,1` => `4/4` exact.

Transfer interpretation:

- fortified energy-core path does not harm easier/lower-friction transfer
  families; required transfer quality remains strong.

### F) PLAN_08 acceptance snapshot (current)

Met:

- instrumentation delivered in structured CSV fields (`fwd_ec_*`).
- required paper-family and transfer runs completed with auditable artifacts.

Not met:

- K=4 stabilization goal (exact closure + reduced seed sensitivity at large `n`)
  is not yet achieved.
- historical `3567_plus` continuity is not preserved under current fortified
  settings.

## 2026-04-16 — Targeted K=4 energy-core revalidation

This section records the required K=4 recovery check requested before
continuing broad Plan-05 extension work.

Measurement artifact:

- `research/k_vs_arithmetic_axes_20260412/csv/plan05/K4_energy_core_recovery_comparison_20260416.csv`

Run mode:

- solver: `solvers/cpp/build/stateful_compare`
- workflow: `ablation-stdin step1_exact_guided`
- forced mode where requested:
  `PAST_RELAXED_BINPACK_SOLVER=energy_core`

### Phase 1 — Old `3567_plus` frontier under current code

Rows rerun on the exact historical instances:

- `paperext_profile_repair_smallk_nscale_plus_20260409/0009_profile_smallk_3567_plus_n3500_s1`
- `paperext_profile_repair_smallk_nscale_plus_20260409/0011_profile_smallk_3567_plus_n5000_s1`

Observed (forced energy-core):

| row | runtime (s) | UB | LB | gap % | fwd_pack_method | deciding step | exact-DP used | closes? |
|---|---:|---:|---:|---:|---|---|---:|---|
| 3567_plus n=3500 | 354.4763 | 172,475,616 | 172,415,824 | 0.0347 | block_repair_energy_core | step4 | 1 | no |
| 3567_plus n=5000 | 590.5615 | 248,943,407 | 248,815,508 | 0.0514 | block_repair_energy_core | step4 | 1 | no |

Comparison to earlier archive claim:

- Earlier archive text described full exact-guided closure on these frontier
  anchors.
- Current reruns do **not** reproduce that closure path under present code; both
  rows remain finite-gap and require Step-4 entry.

### Phase 2 — Paper-group `g3567={3,5,6,7}` with forced energy-core

| n | runtime (s) | UB | LB | gap % | fwd_pack_method | deciding step | exact-DP used |
|---:|---:|---:|---:|---:|---|---|---:|
| 1000 | 207.4820 | 50,815,862 | 50,815,862 | 0.0000 | block_repair_energy_core | step3 | 0 |
| 1500 | 547.7022 | 74,225,131 | 74,211,691 | 0.0181 | block_repair_energy_core | step4 | 1 |
| 2500 | 1053.6411 | 123,824,733 | 123,795,529 | 0.0236 | block_repair_energy_core | step4 | 1 |
| 3500 | 1953.3532 | 173,270,306 | 173,198,908 | 0.0412 | block_repair_energy_core | step3 | 0 |
| 5000 | 2396.4987 | 247,977,076 | 247,842,042 | 0.0545 | block_repair_energy_core | step3 | 0 |

### Phase 3 — Direct comparison on `g3567` (default vs forced energy-core)

| row | default result | energy-core result | better | why |
|---|---|---|---|---|
| g3567 n=1000 | gap 0.0016, 410.8945s, step4, method profile_repair_beam | gap 0.0000, 207.4820s, step3, method block_repair_energy_core | energy-core | exact closure + faster runtime |
| g3567 n=1500 | gap 0.0075, 1076.8157s, step4 | gap 0.0181, 547.7022s, step4 | default | materially tighter UB/LB gap |
| g3567 n=2500 | gap 0.0075, 2039.8695s, step3 | gap 0.0236, 1053.6411s, step4 | default | much better gap quality |
| g3567 n=3500 | gap 0.0068, 2910.5060s, step3 | gap 0.0412, 1953.3532s, step3 | default | much better gap quality |
| g3567 n=5000 | external timeout in run window | finite gap 0.0545, 2396.4987s | mixed/inconclusive | default missing due to timeout; energy-core gives a finite but weak-quality incumbent |

### Policy recommendation from measured reruns

1. The old K=4 closure mechanism (energy-core incumbent then exact-guided
   closure through `n=5000`) is **not reproduced** on current code for the
   historical `3567_plus` anchors.
2. Forced energy-core helps specific paper-group rows (`g3567 n=1000`) but is
   worse on gap quality for the larger tested paper-group rows.
3. Current K=4 default policy should **not** be changed to a blanket
   energy-core-first rule. Keep default mainline, with energy-core as a targeted
   diagnostic/override option.

## 2026-04-15 — Plan 03F validation (restore K=2 profile-repair exact mode)

This section records the mandatory rerun set from
`PLAN_03F_restore_k2_and_mmkp_selector.md` after updating Step-3 selector
policy in `stateful_dp_solver.cpp`.

Run method:

- binary: `solvers/cpp/build/stateful_compare`
- mode: `ablation-stdin step1_exact_guided`
- payload generator: `run_plan05_paper_groups_extension.py` (`build_payload`)
- machine: `twosby`

### 1) Mandatory K=2 family `{8,10}`

All rows used selector mode `exact` with reason `k2_exact_default`,
`fwd_pack_method=profile_realization_dp_exact`, and no Step-4 exact usage
(`diag_exact_dp_used=0`).

| family | n | Step-3 mode | selector reason | runtime (s) | UB | LB | gap % | deciding step | exact-DP used |
|---|---:|---|---|---:|---:|---:|---:|---|---:|
| `{8,10}` | 500 | exact | `k2_exact_default` | 3.0401 | 42,736,978 | 42,736,978 | 0.0000 | step3 | 0 |
| `{8,10}` | 600 | exact | `k2_exact_default` | 4.2729 | 52,288,682 | 52,288,682 | 0.0000 | step3 | 0 |
| `{8,10}` | 750 | exact | `k2_exact_default` | 6.6215 | 63,402,489 | 63,402,489 | 0.0000 | step3 | 0 |
| `{8,10}` | 1000 | exact | `k2_exact_default` | 11.7874 | 85,021,287 | 85,021,287 | 0.0000 | step3 | 0 |
| `{8,10}` | 1500 | exact | `k2_exact_default` | 27.0220 | 128,068,404 | 128,068,404 | 0.0000 | step3 | 0 |
| `{8,10}` | 2500 | exact | `k2_exact_default` | 82.6366 | 212,276,475 | 212,276,475 | 0.0000 | step3 | 0 |
| `{8,10}` | 3500 | exact | `k2_exact_default` | 185.7488 | 299,016,631 | 299,016,631 | 0.0000 | step3 | 0 |
| `{8,10}` | 5000 | exact | `k2_exact_default` | 359.9665 | 425,568,378 | 425,568,378 | 0.0000 | step3 | 0 |

Interpretation:

- The missing Step-3 K=2 behavior is restored in mainline policy.
- The two-type paper family again follows a low-memory profile-repair exact path
  rather than falling through unresolved or requiring global exact DP rescue.

### 2) Mandatory K=4 frontier representative

Probe row:

- `g3567`, `n=1000`, `lambda=1.3`, `seed=0`

Observed:

- Step-3 mode: beam
- selector decision/reason: `beam / merged_blocks`
- runtime: `410.8459s`
- `UB=50,816,699`, `LB=50,815,862`, gap `0.0016%`
- deciding step: `step4`
- exact-DP used: `1`

Interpretation:

- Selector correctly rejects exact profile mode when merged-block structure is
  outside tractable boundary and uses Step-3 beam incumbent path.
- Global exact DP remains the closure authority (Step 4), consistent with
  pipeline boundaries.

### 3) Mandatory K=6 representative (selector should prefer beam)

Probe row:

- `{4,6,8,10,12,14}`, `n=300`, `lambda=1.3`, `seed=0`

Observed:

- Step-3 mode: beam path active (`fwd_profile_beam_status=feasible`)
- selector decision/reason: `beam / state_space`
- runtime: `4.7162s`
- `UB=LB=24,739,369`, gap `0.0000%`
- deciding step: `step2`
- exact-DP used: `0`

Interpretation:

- Selector continues to avoid broad exact-mode entry on higher-K rows when
  frontier proxies indicate intractability.

### 4) One-paragraph policy statement (current)

Step 3 is profile-realization DP over recovered blocks: for `K=2` we use a
dedicated exact profile-realization mode by default (Mode A) with explicit
safety gates on state/composition size; for `K>=4` we choose exact mode only
when structural tractability tests pass (merged blocks, state-space estimate,
total/max composition estimates, branching estimates, and arithmetic hard-alarm)
and otherwise run the truncated beam mode (Modes B/C). Step 4 global exact DP
remains separate and is not the first rescue for rows that Step 3 is expected to
solve.

## Current imported observations

These are the key observations motivating this new archive. They come from the
current validated solver state but are reinterpreted here through the new
two-axis lens.

## 1. Easy-arithmetic high-`K` can already be exact at Step 1

On the `1..10` family:

- `K = 10`
- unit length is present,
- the semigroup is therefore as favorable as possible.

Current validated outcomes:

- exact at Step 1 through the tested `n=3500` rows.

This is the strongest immediate evidence that:

- large `K` alone is not the true frontier.

## 2. Harder arithmetic can dominate even at lower `K`

On the irregular six-type family `{2,3,4,5,7,11}`:

- Step 1 is no longer exact,
- incumbent generation matters,
- exact closure is no longer automatic.

This supports the new framing:

- the main open difficulty is not simply "more types,"
- but arithmetic-hard recovered-profile realization.

## 3. A second six-type family is easier but still not Step-1 exact

On the six-type family `{4,5,6,7,8,9}`:

- the corrected default solver behaves well,
- but still reports finite small gaps on the tested rows.

So the six-type story is already not uniform even before changing `K`.

## 4. Working interpretation

At archive creation, the current imported picture is:

- easy arithmetic:
  - high-`K` can be surprisingly easy,
  - often exact at Step 1
- hard arithmetic:
  - incumbent quality and repair matter much more,
  - even moderate `K` can stay open.

This is the core result that the new archive is designed to test properly.

## 5. First baseline grid under the new archive

Using the frozen baseline configuration
`PAST_BLOCK_REPAIR_LAGR_BEAM_POLISH_MAX_MERGED=0`, the first two-axis grid gave:

### Easy arithmetic

- `K3_contig_n300_s0`: exact at Step 1, `2.04s`
- `K4_contig_n300_s0`: exact at Step 1, `4.50s`
- `K5_contig_n300_s0`: exact at Step 1, `4.50s`
- `K6_contig_n200_s0`: exact at Step 1, `2.72s`
- `K7_contig_n100_s0`: exact at Step 1, `0.49s`
- `K8_contig_n200_s0`: exact at Step 1, `3.32s`
- `K10_1_10_n1000_s1`: exact at Step 1, `34.48s`
- `K10_1_10_n2500_s1`: exact at Step 1, `220.96s`

### Medium arithmetic

- `K6_456789_n1000_s1`: gap `0.0480%`, `25.37s`
- `K6_456789_n1500_s1`: gap `0.0442%`, `115.68s`

### Hard arithmetic

- `K6_2345711_n1000_s1`: gap `0.0356%`, `17.96s`
- `K6_2345711_n1500_s1`: gap `0.0294%`, `39.77s`
- `K6_2345711_n2500_s1`: gap `0.0238%`, `216.99s`

### Cross-cell

- `K7_irregular_n100_s0`: exact at Step 1, `0.33s`

Main takeaway:

- the first baseline grid already supports the new framing:
  - easy arithmetic scales far in `K`,
  - while the six-type medium/hard families remain open.

## 6. First structural code change: Level 3 separation

The first code change from the plan was then implemented:

- block assignments are now evaluated block-by-block on recovered windows,
- with exact dense per-block multiset DP when the local state space is small,
- and local ascending/descending `solve_fixed_sequence` fallback otherwise.

This does **not** yet redesign Level 2, but it is the first real separation of:

- Level 2: block assignment
- Level 3: within-block scheduling

## 7. Post-change representative results

### Easy arithmetic anchor

- `paperext_profile_repair_largek_nscale_20260409/0017_profile_largek_1_10_n1000_s1`
  - exact at Step 1, `36.57s`

So the easy/high-`K` exact branch is preserved.

### Medium arithmetic

- `.../0009_profile_largek_456789_n1000_s1`
  - `UB=61,118,062`
  - `LB=61,108,061`
  - gap `0.0164%`
  - runtime `49.35s`
- `.../0011_profile_largek_456789_n1500_s1`
  - `UB=91,632,095`
  - `LB=91,617,492`
  - gap `0.0159%`
  - runtime `111.08s`

### Hard arithmetic

- `.../0001_profile_largek_2345711_n1000_s1`
  - `UB=48,641,508`
  - `LB=48,637,514`
  - gap `0.0082%`
  - runtime `36.07s`
- `.../0003_profile_largek_2345711_n1500_s1`
  - `UB=74,102,952`
  - `LB=74,098,255`
  - gap `0.0063%`
  - runtime `76.40s`
- `.../0005_profile_largek_2345711_n2500_s1`
  - `UB=125,450,588`
  - `LB=125,442,130`
  - gap `0.0067%`
  - runtime `208.37s`

## 8. Immediate interpretation

This first Level 3 separation improves the open arithmetic-sensitive rows
substantially while preserving the easy-arithmetic exact story:

- `2345711 n=1000`: `0.0356% -> 0.0082%`
- `2345711 n=1500`: `0.0294% -> 0.0063%`
- `456789 n=1000`: `0.0480% -> 0.0164%`
- `456789 n=1500`: `0.0442% -> 0.0159%`

So the framework-driven first code change already paid off.

## 9. Remaining limitation

On these representative six-type rows, the winning Step-1 incumbent is still
reported as:

- `block_repair_feasible_beam`

This matters for interpretation:

- Level 3 is now modeled more honestly,
- but the main unsolved structural bottleneck still appears to be Level 2.

## 10. Phase-1 two-axis grid slice with arithmetic descriptors

Stored CSV:

- [csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase1.csv](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase1.csv)

Utility used:

- [run_two_axis_grid.py](/Users/mac/Documents/Study/PFE/PaST/scripts/run_two_axis_grid.py)

The utility records both:

- solver outputs,
- and arithmetic descriptors:
  - `has_one`
  - `gcd`
  - `multiplicity`
  - `contiguous`
  - `span`
  - `frobenius`
  - `apery_max`
  - `semigroup_density`

### Verified rows from the first slice

- `easy_k10_unit = {1,2,3,4,5,6,7,8,9,10}`, `n=300`
  - exact at Step 1
  - runtime `4.0521s`
  - method `ffd`
  - descriptors:
    - `has_one = 1`
    - `contiguous = 1`
    - `frobenius = -1`
    - `semigroup_density(100) = 1.000000`

- `hard_k4_irregular = {3,5,7,11}`, `n=300`
  - exact at Step 1
  - runtime `4.3910s`
  - method `ffd`
  - descriptors:
    - `has_one = 0`
    - `contiguous = 0`
    - `frobenius = 4`
    - `semigroup_density(100) = 0.970297`

- `hard_k8_irregular = {3,5,7,11,13,17,19,23}`, `n=300`
  - exact at Step 1
  - runtime `11.4293s`
  - method `ffd`
  - descriptors:
    - `has_one = 0`
    - `contiguous = 0`
    - `frobenius = 4`
    - `semigroup_density(100) = 0.970297`

- `hard_k10_irregular = {2,3,5,7,11,13,17,19,23,29}`, `n=300`
  - exact at Step 1
  - runtime `17.6653s`
  - method `ffd`
  - descriptors:
    - `has_one = 0`
    - `contiguous = 0`
    - `frobenius = 1`
    - `semigroup_density(100) = 0.990099`

- `medium_k6_dense = {4,5,6,7,8,9}`, `n=1000`
  - `UB = 62,412,903`
  - `LB = 62,404,265`
  - gap `0.0138%`
  - runtime `60.1463s`
  - incumbent method `block_repair_feasible_beam`
  - descriptors:
    - `has_one = 0`
    - `contiguous = 1`
    - `frobenius = 3`
    - `semigroup_density(100) = 0.970297`

- `hard_k6_2345711 = {2,3,4,5,7,11}`, `n=1000`
  - `UB = 52,575,221`
  - `LB = 52,568,409`
  - gap `0.0130%`
  - runtime `50.9597s`
  - incumbent method `block_repair_feasible_beam`
  - descriptors:
    - `has_one = 0`
    - `contiguous = 0`
    - `frobenius = 1`
    - `semigroup_density(100) = 0.990099`

## 11. What the first slice already says

Three conclusions are already supported:

1. large `K` is not automatically hard:
   - even irregular `K=8` and `K=10` families remain exact at Step 1 for
     `n=300`
2. the arithmetic axis is not captured by Frobenius number alone:
   - `hard_k6_2345711` has a small Frobenius number but still leaves a gap at
     `n=1000`
3. the current open six-type rows are still primarily a Level-2 story:
   - both medium and hard `K=6` rows at `n=1000` are owned by
     `block_repair_feasible_beam`

## 12. Still-open cross-cell

The main missing cell from this first slice remains:

- irregular `K=8` / `K=10` at larger `n` such as `1000`

Those rows were launched but not recorded as results because the first
controlled batch was interrupted before completion. They remain the next
high-value experimental target.

## 13. Phase-1 high-value cross-cells are now completed (`n=1000`)

Stored CSV:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase2.csv`

Required irregular high-`K` rows (seed `0`):

- `hard_k4_irregular`, `n=1000`
  - `UB = 63,958,209`
  - `LB = 63,952,923`
  - gap `0.0083%`
  - runtime `121.5103s`
  - winner `block_repair_energy_core`

- `hard_k8_irregular`, `n=1000`
  - `UB = 118,487,230`
  - `LB = 118,468,602`
  - gap `0.0157%`
  - runtime `268.1152s`
  - winner `block_repair_feasible_beam`

- `hard_k10_irregular`, `n=1000`
  - `UB = 120,229,235`
  - `LB = 120,202,617`
  - gap `0.0221%`
  - runtime `377.6920s`
  - winner `block_repair_feasible_beam`

Interpretation:

- all required high-value irregular high-`K` rows now finish cleanly,
- all remain finite-gap (not Step-1 exact),
- and Level-2 winners move from energy-core at `K=4` to feasible-beam at
  `K=8,10`.

So the open frontier is now recorded as a solved runtime/stability question but
still an assignment-quality question.

## 14. Optional phase-1 extensions (second seed on six-type families)

Also run and merged into phase-2 CSV:

- `medium_k6_dense`, `n=1000`, seed `1`
  - gap `0.0260%`
  - runtime `56.6861s`
  - winner `block_repair_feasible_beam`

- `hard_k6_2345711`, `n=1000`, seed `1`
  - gap `0.0064%`
  - runtime `36.9545s`
  - winner `block_repair_feasible_beam`

These rows keep the same qualitative ownership pattern as seed `0`:

- finite tiny gaps,
- beam-owned Level 2 in the six-type regime.

## 15. Lagrangian baseline recovery status (clean-policy validation)

Required anchor reruns under cleaned defaults (`seeded_beam=0`,
`beam_polish=0`, pricing off):

- `0001_profile_largek_2345711_n1000_s1`: gap `0.0082%`
- `0003_profile_largek_2345711_n1500_s1`: gap `0.0063%`
- `0005_profile_largek_2345711_n2500_s1`: gap `0.0067%`

All three are currently won by:

- `fwd_pack_method = block_repair_feasible_beam`

and not by `block_repair_lagrangian_assign`.

So for this branch, the current clean baseline is beam-owned; the previously
reported Lagrangian-owned quality regime was not recovered without hidden
hybrid behavior.

## 16. Robustness repair completed on the known crash row

During required high-`K` runs, `hard_k10_irregular n=1000` initially crashed
(`returncode=-11`).

ASAN diagnosis found a heap-buffer-overflow in exact DP seed initialization when
a type had zero multiplicity in the realized instance.

Fix implemented in `solvers/cpp/stateful_dp_solver.cpp`:

- skip seed transitions for `totals[i] <= 0` in:
  - `solve_exact_multiset_dp(...)`
  - `smart_reconstruct(...)`
  - `solve_sparse_exact_multiset_dp(...)`

After this fix, the previously crashing required row runs and is now archived in
phase-2 results.

## 17. Runner diagnostics now expose lightweight residual-core proxies

Updated `scripts/run_two_axis_grid.py` now records additional fields per row:

- `n_jobs_total`
- `active_method`
- `diag_merged_blocks`
- `diag_winner_is_ffd`
- `diag_winner_is_beam`
- `diag_winner_is_lagr`

This supports the plan’s requested minimal diagnostics without adding a large
instrumentation branch.

## 18. Plan 02B exact Level-2 diagnostic (supersedes Plan 02 Phase A claim)

Important correction:

- The earlier Plan-02 pool-membership diagnostic was tautological because beam
  and Lagrangian share the same generated pattern pool.
- So `beam_not_in_pool=0` is not evidence of "search gap only" by itself.

Plan 02B therefore implemented and ran a self-contained exact Level-2
branch-and-bound over the existing per-block pattern pools.

### New exact-L2 metrics now reported in CSV

- `fwd_beam_ub_for_exact_l2`
- `fwd_exact_l2_ub`
- `fwd_exact_l2_time`
- `fwd_exact_l2_nodes`
- `fwd_exact_l2_closed`
- `fwd_exact_l2_improved_over_beam`
- `fwd_exact_l2_beam_optimal_in_pool`
- `fwd_exact_l2_status`

Consolidated validation CSV:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase2b_exactl2_validation.csv`

### Required rows (Plan 02B) — results

1. `hard_k4_irregular n=1000 seed=0` (B=9)
   - LB `63,952,923`
   - beam UB `63,959,486` (beam gap `0.0103%`)
   - exact L2 UB `63,952,923` (exact gap `0.0000%`)
   - exact-L2: `closed`, `98.0751s`, `1,065,977` nodes
   - beam improved? `yes`

2. `hard_k6_2345711 n=1000 seed=0` (B=8)
   - LB `52,568,409`
   - beam UB `52,575,221` (beam gap `0.0130%`)
   - exact L2 UB `52,568,409` (exact gap `0.0000%`)
   - exact-L2: `closed`, `13.6711s`, `10,408,311` nodes
   - beam improved? `yes`

3. `hard_k6_2345711 n=1000 seed=1` (B=14)
   - LB `49,355,840`
   - beam UB `49,359,014` (beam gap `0.0064%`)
   - exact L2 UB `49,355,840` (exact gap `0.0000%`)
   - exact-L2: `closed` (with 600s limit), `485.8594s`, `1,913,515,174` nodes
   - beam improved? `yes`

4. `hard_k8_irregular n=1000 seed=0` (B=19)
   - LB `118,468,602`
   - beam UB `118,487,230` (beam gap `0.0157%`)
   - exact L2 UB `118,487,230` (same as beam before timeout)
   - exact-L2: `timeout`, `180.0001s`, `503,067,136` nodes
   - beam improved? `no`

5. `hard_k10_irregular n=1000 seed=0` (B=20)
   - LB `120,202,617`
   - beam UB `120,229,235` (beam gap `0.0221%`)
   - exact L2 UB `120,229,235` (same as beam before timeout)
   - exact-L2: `timeout`, `180.0001s`, `301,474,304` nodes
   - beam improved? `no`

6. `medium_k6_dense n=1000 seed=0` (B=9)
   - LB `62,404,265`
   - beam UB `62,412,903` (beam gap `0.0138%`)
   - exact L2 UB `62,404,265` (exact gap `0.0000%`)
   - exact-L2: `closed`, `12.3308s`, `19,186,729` nodes
   - beam improved? `yes`

### Gap diagnosis from exact-L2 evidence

- For B=8–9 (and B=14 with a larger exact-L2 budget), residual gap is clearly
  a **Level-2 in-pool search gap**: exact L2 closes to LB.
- For B=19–20, exact L2 timed out at 180s and matched beam's UB before timeout.
  This is evidence of a harder Level-2 search regime at larger B; it is NOT
  sufficient to claim the pool/profile is already the ceiling.

### Minimal redesign direction supported by evidence

- Use exact Level 2 on small/moderate merged-block counts (where it closes).
- Keep beam fallback on larger-B rows.
- If future exact-L2 runs at larger budgets still cannot improve/close B=19–20,
  then the next justified redesign is out-of-pool search (e.g., LNS/pricing).

## 19. Final method cleanup (Plan 03/04): mainline now matches 4-step story

The default mainline policy has been simplified to exactly four explainable steps:

1. semigroup profile recovery,
2. fast profile realization,
3. one unified hard-case repair family (`profile_repair_beam`),
4. exact DP fallback/certification.

### What changed in solver policy

- default recovered-profile solver is now `profile_repair_beam`.
- Step-3 implementation is unified as one method family:
  - feasibility-first beam over recovered blocks,
  - then bounded 2-block local destroy/repair intensification,
  - with exact/local per-block Level-3 evaluation.
- these are no longer active in default mainline:
  - `lagrangian_assign`,
  - `rg_beam`,
  - `feasible_counts`,
  - post-Lagrangian beam polish,
  - exact Level-2 B&B incumbent replacement.
- exact-L2 is demoted to diagnostic-only by default:
  - `PAST_BLOCK_REPAIR_EXACT_L2=0` default,
  - explicit `PAST_BLOCK_REPAIR_EXACT_L2_APPLY=1` is now required for it to
    alter incumbents.

### Exact fallback status after cleanup

- exact DP remains the only exact fallback in mainline:
  - sparse exact DP first,
  - dense exact DP fallback.
- exact stage still receives the best UB from earlier steps.

## 20. Focused validation of cleaned policy

Validation CSVs:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_easy.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_medium.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_medium_step3.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_hard_k6.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_hard.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_exactl2_demoted.csv`
- merged set:
  - `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03_validation.csv`

Representative rows and step ownership:

1. easy arithmetic (`easy_k10_unit n=300 seed=0`)
   - UB/LB/gap: `16,137,402 / 16,137,402 / 0.0000%`
   - runtime: `6.6759s`
   - deciding step: Step 2 (`fwd_pack_method=ffd`)
   - exact DP used: no
   - archival exact-L2 affecting result: no (`fwd_exact_l2_status=disabled`)

2. medium arithmetic, Step-3-active row (`medium_k6_dense n=1000 seed=0`, short budget)
   - UB/LB/gap: `62,412,903 / 62,404,265 / 0.0138%`
   - runtime: `39.1274s` (timed out)
   - deciding step before timeout: Step 3 (`fwd_pack_method=profile_repair_beam`)
   - exact DP used: no (budget exhausted)
   - archival exact-L2 affecting result: no

3. hard arithmetic with Step 3 and Step 4 fallback (`hard_k6_2345711 n=1000 seed=0`)
   - UB/LB/gap: `52,575,221 / 52,568,409 / 0.0130%`
   - runtime: `35.3151s`
   - Step 3 candidate: `fwd_pack_method=profile_repair_beam`
   - final deciding step: Step 4 (`winner_detail=exact`)
   - exact DP used: yes
   - archival exact-L2 affecting result: no

4. previously exact-L2-touched row (`hard_k4_irregular n=1000 seed=0`)
   - UB/LB/gap: `63,959,486 / 63,952,923 / 0.0103%`
   - runtime: `206.4341s` (timed out)
   - Step 3 method active: `fwd_pack_method=profile_repair_beam`
   - exact-L2 state: `fwd_exact_l2_status=disabled`
   - `diag_exact_l2_mainline_used=0`

This confirms exact-L2 is no longer part of default mainline behavior.

## 21. Plan 03B/04A continuation: strengthened Step 3 and clarified Step-4 diagnostics

Updated validation CSV:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_validation.csv`

with component files:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_easy.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_medium.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_hard_k6.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_hard_k8_n800.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260413_plan03b_hard_k4.csv`

### Representative rows (required coverage)

1. easy Step-2-decided row (`easy_k10_unit n=300`)
   - deciding step: Step 2 (`fwd_pack_method=ffd`)
   - `UB/LB/gap = 16,137,402 / 16,137,402 / 0.0000%`
   - runtime: `6.5965s`
   - exact used: `0`

2. medium Step-3-active row (`medium_k6_dense n=1000`)
   - Step-3 method: `fwd_pack_method=profile_repair_beam`
   - final deciding step: Step 4 (`winner_detail=exact`)
   - `UB/LB/gap = 62,411,449 / 62,404,265 / 0.0115%`
   - runtime: `58.8397s`
   - exact used: `1`
   - exact initial UB: `62,411,449`
   - exact diagnostics: `exact_diag_mode=sparse_skip_theoretical`

3. hard `K=6` row with Step 4 entered (`hard_k6_2345711 n=1000`)
   - Step-3 method: `fwd_pack_method=profile_repair_beam`
   - final deciding step: Step 4 (`winner_detail=exact`)
   - `UB/LB/gap = 52,574,872 / 52,568,409 / 0.0123%`
   - runtime: `44.6838s`
   - exact used: `1`
   - exact initial UB: `52,574,872`
   - exact diagnostics: `exact_diag_mode=sparse_skip_theoretical`

4. hard high-`K` row (`hard_k8_irregular n=800`)
   - deciding stage before timeout: Step 3
   - `UB/LB/gap = 94,145,122 / 94,128,308 / 0.0179%`
   - runtime: `186.8594s` (timed out)
   - exact used: `0`

5. stubborn tiny-gap row (`hard_k4_irregular n=1000`)
   - deciding stage before timeout: Step 3
   - `UB/LB/gap = 63,961,622 / 63,952,923 / 0.0136%`
   - runtime: `215.0008s` (timed out)
   - exact used: `0`

### Step-3 diagnostics now visible on active rows

Step-3 active rows now report non-trivial beam diagnostics, including:

- adaptive-width stats (`fwd_profile_beam_base_width`, `avg_width`, `max_width`)
- state-flow counts (`states_considered`, `states_kept`)
- pruning breakdown (`pruned_over`, `pruned_suffix`, `pruned_discrepancy`)
- discrepancy policy fields (`discrepancy_budget`, `discrepancy_depth`)

Example signals:

- `medium_k6_dense`: considered `70,401,289`, kept `1,436,761`,
  suffix-pruned `27,794,454`, discrepancy-pruned `872,593`
- `hard_k6_2345711`: considered `58,973,215`, kept `1,203,535`,
  suffix-pruned `22,455,811`, discrepancy-pruned `1,001,177`

### Exact-stage diagnostics interpretation (resolved)

Prior Plan-03B runs showed ambiguous exact diagnostics (`dense` with INF/zeros).
After diagnostics hardening, Step-4-used rows now clearly indicate the exact
path outcome using explicit skip modes:

- `sparse_skip_theoretical`: sparse exact skipped due to theoretical lattice
  guardrail,
- dense fallback may still be skipped for state-space limits (`dense_skip_*`),
  but no longer overwrites sparse diagnostics when sparse already reported a
  meaningful skip reason.

This resolves the diagnostics-interpretation blocker without changing the final
4-step method policy.

## 22. Plan 03C: Step-3 profile-realization DP unification (exact + truncated modes)

Step 3 is now implemented and described as one family:

- **profile-realization DP**, with
  - **exact mode**: fixed-block DP
  - **truncated mode**: profile-repair beam

This unifies the previous conceptual split without adding a new method family
and without merging Step 4 into Step 3.

### Code-structure unification completed

In `solvers/cpp/stateful_dp_solver.cpp`:

- added shared helpers used by both modes:
  - `build_profile_block_local_views(...)`
  - `evaluate_profile_block_counts(...)`
  - `profile_realization_block_order(...)`
- aligned exact and beam modes to use compatible:
  - recovered-block interpretation,
  - local exact block evaluator,
  - block-ordering policy interface.

In `solvers/cpp/stateful_compare.cpp`:

- Step-decider mapping now treats `profile_realization_dp_exact` as Step 3.

### Exact-safe enhancements transferred into exact fixed-block mode

Added to exact mode (fixed-block DP):

1. hardest-first block ordering (optional)
2. suffix min/max residual feasibility pruning (optional)
3. existing sparse frontier dedup retained

New Step-3 diagnostics recorded in CSV:

- `fwd_profile_realization_hardest_first`
- `fwd_profile_realization_exact_suffix_prune`

### Representative validation and measured effects

#### A. Mainline representative rows (default limits)

Files:

- `research/k_vs_arithmetic_axes_20260412/csv/plan03c/TMP_plan03c_mainline_easy.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan03c/TMP_plan03c_mainline_medium_postdefault.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan03c/TMP_plan03c_mainline_hardk6_postdefault.csv`

Outcomes:

- `easy_k10_unit n=300`: Step 2 (`ffd`), gap `0.0000%`
- `medium_k6_dense n=1000`: Step 3 beam active, Step 4 entered,
  gap `0.0115%`, `exact_diag_mode=sparse_skip_theoretical`
- `hard_k6_2345711 n=1000`: Step 3 beam active, Step 4 entered,
  gap `0.0123%`, `exact_diag_mode=sparse_skip_theoretical`

So mainline behavior remains consistent with the cleaned 4-step policy.

#### B. Where Step-3 exact mode is tractable

Default limits (forced exact mode) skip exact profile-realization DP on all
tested representative rows due to comp-est guardrail:

- `csv/plan03c/TMP_plan03c_exactmode_defaultlimits_n300.csv`
- `csv/plan03c/TMP_plan03c_exactmode_defaultlimits_k6_n1000.csv`

Raised limits (`MAX_COMP_EST/MAX_NC=1e9`) show tractable exact-mode islands:

- `easy_k4_unit n=300`: feasible (`state_space=31,224,600`, `comps=2,178`)
- `hard_k4_irregular n=300`: feasible (`state_space=32,433,024`, `comps=86`)
- `medium_k6_dense n=120`: feasible (`state_space=74,826,180`, `comps=372`)
- `hard_k6_2345711 n=120`: feasible (`state_space=77,565,600`, `comps=4,644`)

but K=6 at larger `n` remains skipped by practical guardrails.

#### C. Enhancement effects in exact mode (K=6, n=120)

Files:

- `csv/plan03c/TMP_plan03c_exactmode_hardk6_seedscan_post_cfgA.csv` (hardest-first=1, suffix=1)
- `csv/plan03c/TMP_plan03c_exactmode_hardk6_seedscan_post_cfgB_nohard.csv` (hardest-first=0, suffix=1)
- `csv/plan03c/TMP_plan03c_exactmode_hardk6_seedscan_post_cfgC_nosuffix.csv` (hardest-first=1, suffix=0)

Measured effects:

- hardest-first ordering: no speedup on this slice; no-hardest-first is slightly
  faster across seeds.
- suffix pruning: clear win; enabled pruning reduces
  - mean exact-mode runtime by ~`4.0%`, and
  - mean exact-mode `t_fwd_pack_block_dp` by ~`46.9%`
  versus no-suffix pruning on the same seed set.

### Interpretation

Plan-03C objective is met:

- Step 3 is now truthfully one profile-realization DP family with exact and
  truncated modes;
- fixed-block DP is retained and elevated as Step-3 exact mode;
- exact-safe pruning transfer (suffix residual checks) yields measurable benefit;
- Step 4 remains separate as global exact DP fallback.

## 23. Plan 03D: Step-3 exact-vs-beam selector validated

Plan 03D objective was to replace ad hoc Step-3 exact-vs-beam decisions with an
explicit, explainable selector based on profile structure and frontier
tractability estimates.

### Implemented selector policy

Mainline policy knob:

- `PAST_PROFILE_REALIZATION_SELECTOR_POLICY`
  - `auto_v1` (default)
  - `off` (legacy run-both behavior)
  - `force_exact`
  - `force_beam`

`auto_v1` decision rule:

- choose **Step-3 exact mode** iff all conditions hold:
  - merged blocks `<= 4`
  - count-state estimate `<= 1e8`
  - total composition estimate `<= 1e8`
  - max per-block composition estimate `<= 8e7`
  - no hard-arithmetic alarm
- otherwise choose **Step-3 beam mode**.

Hard-alarm condition (cheap structural flag):

- `has_one=0`, `contiguous=0`, `merged_blocks>=10`, and
  `semigroup_density<=0.975`.

### New diagnostics exposed for validation

Step-3 rows now include:

- selector policy/decision/reason and arithmetic snapshot,
- exact frontier estimates (`total_comp_estimate`, `max_comp_estimate`,
  `max_compositions_per_block`),
- mode-specific status/timing (`profile_beam_status`, `block_dp_status`,
  `t_pack_profile_beam`, `t_pack_block_dp_exact`),
- candidate comparison against Step 2 (`profile_step2_ub`,
  `profile_beam_candidate_ub`, `profile_exact_candidate_ub`, and
  improvement flags).

These are serialized in `stateful_compare` CSV output and used by the Plan-03D
validation table.

### Validation table

Consolidated selector table:

- `research/k_vs_arithmetic_axes_20260412/csv/plan03d/TMP_plan03d_selector_validation_table.csv`

Validation summary (13 representative rows):

1. **Auto chose exact correctly (8 rows)**
   - rows: `easy_k4 n=300`, `hard_k4 n=300`, `medium_k6 n=120` (seeds 0..2),
     `hard_k6 n=120` (seeds 0..2)
   - observed behavior: exact mode feasible in all cases, with
     `block_dp_status=feasible` and zero gaps.

2. **Auto chose beam correctly (5 rows)**
   - rows: `easy_k10 n=300`, `medium_k6 n=1000`, `hard_k4 n=1000`,
     `hard_k6 n=1000`, `hard_k8 n=800`
   - observed behavior: selector reasons are structural (`state_space` or
     `merged_blocks`), beam is feasible, and forced exact on these rows is
     skipped by comp-est guardrail and can leave no usable incumbent.

3. **Misclassification count on this set**
   - exact false-positive (auto exact but should beam): `0`
   - exact false-negative (auto beam but exact would be practical): `0`

### Why this boundary is currently reasonable

- merged-block count and frontier estimates separate the practical tractable
  exact-island rows from the large frontier blow-up rows;
- arithmetic signal is used as a secondary alarm (not as a sole driver), which
  keeps the selector consistent with the two-axis narrative;
- decision reasons are now explicit per row, making Step-3 method choice auditable.

### Current recommendation

- Keep `auto_v1` as the default selector in the unified Step-3 family now.
- Run one additional non-blocking calibration pass near thresholds (more seeds
  around merged=4..6 and comp-est near `1e8`) before freezing paper figures.

## 24. Plan 03D hardening pass: safety fallback and step-separated validation

This pass keeps `auto_v1` and addresses two known weaknesses:

1. exact-primary rows could exit Step 3 without a usable profile-realization
   incumbent,
2. selector confidence was overstated by counting Step-2-closed rows as
   selector evidence.

### A. Step-3 exact-primary fallback is now safe

Control-flow update in Step 3:

- for `auto_v1` rows where selector primary decision is exact,
  - run exact fixed-block DP first,
  - if exact yields no finite Step-3 candidate, run beam fallback immediately in
    the same Step-3 cycle.

Fallback trigger is candidate-based and includes the required failure statuses:

- `skipped_comp_est`
- `skipped_nc`
- `timeout`
- `reconstruct_failed`
- any non-finite exact candidate outcome.

This does not change method identity:

- Step 3 is still one profile-realization DP family,
- exact mode first, beam fallback second when exact is unusable.

### B. New diagnostics for fallback observability

Added row fields:

- `fwd_profile_exact_primary_fallback_to_beam`
- `fwd_profile_exact_primary_status_before_fallback`
- `fwd_profile_step3_incumbent_mode`

These are now emitted in `stateful_compare` CSV and allow direct auditing of
exact-primary fallback behavior.

### C. Fallback probes (explicit evidence)

Artifacts:

- `csv/plan03d/TMP_plan03d_exact_primary_fallback_probe.csv`
- `csv/plan03d/TMP_plan03d_exact_primary_fallback_probe_exactguided.csv`
- `csv/plan03d/TMP_plan03d_exact_primary_fallback_skipnc_probe.csv`
- `csv/plan03d/TMP_plan03d_exact_primary_fallback_skipcomp_probe.csv`

Observed behavior on probes:

- exact status before fallback recorded correctly (`timeout`, `skipped_nc`,
  `skipped_comp_est`),
- `fwd_profile_exact_primary_fallback_to_beam=1`,
- beam returns feasible candidate,
- final Step-3 incumbent mode is `beam`.

### D. Validation methodology fixed (Step-2 rows separated)

New boundary revalidation artifacts:

- raw auto/forced runs:
  - `csv/plan03d/TMP_plan03d_selector_boundary_reval_raw.csv`
- step-separated selector table:
  - `csv/plan03d/TMP_plan03d_selector_boundary_reval_table.csv`

The corrected table includes mandatory split flags:

- `step2_closed_row`
- `step3_selector_test_row`
- `step4_used_row`

and counts misclassification only on `step3_selector_test_row=1`.

From the rebuilt table:

- `step2_closed_rows = 10`
- `step3_selector_test_rows = 6`
- `step4_used_rows = 5`
- `misclassifications_on_step3_rows = 0`

So Step-2-closed rows are no longer treated as selector wins.

### E. Near-boundary calibration set used in this pass

Generated focused boundary datasets:

- `csv/plan03d/TMP_plan03d_boundary_scan_auto_v1.csv`
- `csv/plan03d/TMP_plan03d_boundary_scan_midn_auto_v1.csv`
- `csv/plan03d/TMP_plan03d_probe_hardk6_seed12.csv`
- `csv/plan03d/TMP_plan03d_probe_hardk4_boundary_seeds012.csv`
- `csv/plan03d/TMP_plan03d_calib_boundary_step3split.csv`

This set emphasizes merged-block and comp-est boundary behavior and explicitly
tracks whether Step 3 was actually needed.

### F. Updated recommendation

- `auto_v1` is now robust enough to keep as default because exact-primary rows
  have a safe in-cycle beam fallback and selector validation is now methodologically
  correct.
- Still run one additional iteration for confidence broadening (more Step-3-test
  rows near merged 4..6 and threshold-near comp-est), since current step3-test
  count is small.

## 25. Plan 04C v3: incumbent vs pruning with real sparse-exact expansions

Plan-04C v2 was dominated by rows where exact did not meaningfully run
(`none` / `sparse_skip_theoretical`). This v3 pass adds a targeted slice where
Step-4 sparse exact expands millions of states so incumbent/pruning effects are
observable.

### New v3 artifacts

- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_evidence_runs.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_phase1_incumbent_quality.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_phase1_best_incumbent_by_family.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_phase2_exactdp_variants.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_phase3_best_combos.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan04c/TMP_plan04c_v3_matrix_summary.csv`

### Anchor with usable exact counters

Main anchor:

- `hard_k8_irregular n=500 seed=0`

run with large sparse-theoretical guardrail:

- `PAST_SPARSE_EXACT_MAX_THEORETICAL=9000000000000000000`

so sparse exact runs rather than early-skipping.

### Phase 1 (incumbent source, exact variant fixed at `p0`)

On `hard_k8_irregular n=500`:

- `i1` and `i2` (same behavior):
  - final: `UB/LB/gap = 58,055,690 / 58,038,011 / 0.0305%`
  - runtime: ~`235s`
  - sparse exact: ~`13.39M` states expanded, exact elapsed ~`150.5s`
- `i3` and `i4` (same quality, much faster exact):
  - same final `UB/LB/gap`
  - runtime: ~`186.5s`
  - sparse exact: ~`3.23M` states expanded, exact elapsed ~`36.3s`
- `i0` is not viable here:
  - Step-3 incumbent failed (`fwd_pack_method=none`), exact timed out with
    unresolved final output (`ub/lb = -1/-1`).

Measured relative effect (`i3` vs `i2`):

- total runtime: about `-20.7%`
- exact elapsed: about `-75.9%`
- sparse-expanded states: about `-75.9%`

So on this hard anchor, incumbent handoff quality dominates practical exact cost.

### Phase 2 (exact variant, incumbent fixed)

#### A. Fixed `i2` on `hard_k8_irregular n=500`

- `p1` and `p3` vs `p0`/`p2`:
  - expanded states reduced (`13.39M -> 7.55M`),
  - pruned-bound counters reduced similarly,
  - but final UB/gap unchanged,
  - total runtime remains ~`235s`.

`p2` is effectively neutral on this slice (close to `p0`).

#### B. Fixed `i3` on `hard_k8_irregular n=500`

- `p1`/`p3` provide no additional expansion reduction over `p0`/`p2`
  (`3.23M` expanded in all),
- and add runtime overhead (~`+29s`).

### Additional stress check

On `medium_k6_dense n=600` with weak incumbent `i0`:

- sparse exact expands ~`3.82M` states in both `p0` and `p3`,
- `p3` shows nonzero type-aware pruning (`42`) as intended,
- but both runs remain unresolved within budget (`ub/lb = -1/-1`).

### Phase 3 best combinations (hard anchor)

Best practical pair for this anchor is incumbent-driven:

- `i3 + p0` (or `i4 + p0`) gives the fastest total behavior with same final gap.

`i2 + p3` reduces expansions relative to `i2 + p0` but does not improve final
quality or total time enough to beat `i3 + p0`.

### Plan-04C answer at this stage

Conditioned on rows where sparse exact actually expands:

- **incumbent quality is currently the stronger bottleneck lever**,
- pruning variant improvements are real in counters (especially `p1` under weak
  incumbents) but currently secondary in end-to-end impact.

## 26. PLAN16 fixed-n K-scaling (n=1000, seeds 0/1)

Artifact:

- `research/k_vs_arithmetic_axes_20260412/csv/plan16/PLAN16_k_scaling_n1000.csv`
- `research/k_vs_arithmetic_axes_20260412/csv/plan16/PLAN16_k_scaling_n1000_summary.csv`

Scope:

- families: `g24`, `g37`, `g810`, `g3567`, `g12357`, `g246810`,
  `g12345678910`, `g1234567891011121314151617181920`
- variants: `baseline`, `dense_step2_fastpath`
- fixed `n=1000`, `lambda=1.3`

Main outcomes:

- `K=2` split remains heterogeneous:
  - `g24` closes exactly at Step 2 (`2/2` seeds),
  - `g37` and `g810` close exactly via Step 3 `profile_realization_dp_exact`
    (`2/2` seeds each) when routed through the K=2 mainline profile path.
- `K=4` (`g3567`) closes exactly via Step 3 (`2/2` seeds).
- `K=5` (`g12357`, `g246810`) closes exactly via Step 2 (`2/2` seeds each).
- `K=10` (`g12345678910`) closes exactly via Step 2 (`2/2` seeds):
  - mean runtime `46.309s` (baseline) vs `32.069s` (dense fastpath), about `-30.7%`.
- `K=20` (`g1234567891011121314151617181920`) closes exactly via Step 2 (`2/2` seeds):
  - mean runtime `227.582s` (baseline) vs `199.066s` (dense fastpath), about `-12.5%`.

Interpretation:

- fixed-`n` scaling evidence supports continuing dense-unit Step-2 acceleration for
  large-`K` contiguous families (clear wins at `K=10` and `K=20`);
- the earlier unresolved `g37/g810` rows in this sweep were a runner-routing
  error: `energy_core` bypassed the Step-3 selector for `K=2`;
- after correction, the fixed-`n=1000` sweep no longer shows a `K=2` failure for
  `g37` or `g810`.
