# Blockers

## 2026-04-28 update — PLAN31 resolved: residual_aware plumbing fixed

PLAN31 Phase 0 identified and fixed the `residual_aware` bug. The `beam_diag.score_policy` and residual fields were computed inside `block_repair_feasible_beam_ub` but the copy from `beam_diag` to `RecoveredBlockPackingResult` was missing in `run_profile_beam_attempt`. Added copy lines for `profile_beam_score_policy`, `profile_beam_residual_weight`, `profile_beam_residual_mean_penalty`, `profile_beam_residual_max_penalty`, `profile_beam_late_frac`. Verified: setting `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY=residual_aware` now correctly shows `score_policy = residual_aware` with valid penalty values. **Note**: residual_aware still shows zero gap effect even when active.

## 2026-04-28 update — PLAN27 Step-3 adaptive survivor policy

---

## 2026-04-27 update — PLAN26 local corridor validation

### 1) Local corridor is invalid due to block-local schedulability mismatch

PLAN26 validated PLAN25's local corridor implementation and found it is mechanically unsound:

- The beam's chosen counts per block are generated from `generate_energy_core_patterns`, which filters by **work capacity** (total job lengths ≤ block capacity), not by **schedulability** (whether the specific multiset can be scheduled in the block given machine transition constraints).
- The beam validates its final solution by evaluating the **full global sequence** via `solve_fixed_sequence` on the entire horizon. It does NOT validate each block independently.
- `beam_corridor_local_dp` evaluates each block's counts independently via `evaluate_profile_block_counts` with local block views. This is stricter than the beam's global validation.
- Result: the base beam candidate fails block-local evaluation on at least 2 blocks per instance (`base_candidate_not_found_at_layer_0`).
- Coverage note: the planned `hardB_k10 seed=2 local_corridor_delta2_300s` row is missing from the PLAN26 raw artifact. The conclusion is still supported by the layer-0 failure on hardB delta1 and hardA delta1/delta2, but the Phase 1 grid is not fully complete.

### 2) Block count mismatch was a red herring

Initial suspicion was that `merged_seg` reconstructed from `fwd.block_profile` didn't match the blocks used by the beam. PLAN26 fixed this by propagating `merged_blocks` through `RecoveredBlockPackingResult` -> `RelaxedDPResult`. After the fix, `block_count_mismatch=0` on all rows, but the base path still fails.

### 3) No quick fix available

To make local corridor valid, one of the following would be required:
- Replace block-local evaluation with global sequence evaluation (major redesign of the layered DP).
- Generate only actually schedulable patterns per block (may severely limit candidate pool and change beam behavior).
- Abandon block-layered DP and use global state space with local offset encoding.

None of these are justified by current evidence. The beam is already near-optimal (gaps < 0.05%).

### 4) Existing Step-3 variant `ambig_scoreband_mult2` does not generalize

PLAN26 Phase 4 tested `ambig_scoreband_mult2` on 4 rows. It improved gap on hardB_k10 (2/2) but worsened on hardA_k10 (2/2). Promotion criterion (≥3/4 not worse) fails. This confirms PLAN22B's corrected decision.

The requested new Step-3 policies (`residual_aware`, `late_ambig`) were not implemented in PLAN26, so the Step-3 scoring branch is only partially tested.

### 5) Final decision

**C** — Local corridor invalid due to block/path mismatch. Do not use until fundamentally redesigned.

---

## 2026-04-26 update — PLAN24B forced-entry corridor exact DP diagnostic

### 1) Corridor exact DP cannot enter search due to int64 encoding overflow

PLAN24B tested whether corridor can prune states when exact DP is forced to enter. With `PAST_EXACT_CORRIDOR_FORCE_ENTRY=1`, the theoretical guardrail is bypassed, but:

- All forced rows hit `sparse_skip_overflow`: the int64 mixed-radix state encoding overflows for K=10 at n=1000.
- Product of (totals[i] + 1) ≈ 100^10 exceeds int64 range.
- Zero states generated → zero corridor pruning.

### 2) Encoding is the fundamental blocker, not the guardrail

PLAN24 showed `sparse_skip_theoretical` blocks search. PLAN24B shows that even after bypassing that guardrail, the encoding itself cannot represent the state space. The sparse exact DP is fundamentally limited to ~K=8 at n=1000 on hard irregular families due to int64 overflow.

### 3) Corridor still cannot be tested

Because no states are ever generated (due to encoding overflow, not guardrail policy), the corridor pruning machinery cannot be tested. No amount of guardrail relaxation or corridor delta tuning can overcome the encoding limit.

### 4) Next step for exact certification

The corridor approach cannot be tested under current sparse exact DP encoding. Any future exact certification work on K=10+ must either:
- Use a different encoding (e.g., multiple int64s, big integers)
- Use an entirely different exact method
- Accept that sparse exact DP is infeasible above K≈8

### 5) Promotion decision

**Decision: D** — Corridor still cannot enter meaningfully; abandon corridor under current exact DP.

## 2026-04-26 update — PLAN25 local corridor exact DP (offset encoding)

### 1) Local offset encoding successfully avoids int64 overflow

PLAN25 implemented `beam_corridor_local_dp()` with per-layer offset encoding `(2*delta+1)^K`. For K=10, delta=2, max states per layer ≈ 9.8M, well inside 32-bit range. No overflow observed.

### 2) Local corridor runs but returns infeasible-corridor

On hardA_k10 seed=0 and hardB_k10 seed=2:
- delta1: 25–24s, 40–60k states_seen, status `infeasible_corridor`
- delta2: 52–72s, 7.6–14.1M states_seen, status `infeasible_corridor`
- `best_ub = inf` on all rows; incumbent unchanged.

### 3) PLAN25 interpretation is not yet validated

The earlier claim that the beam prefix is too close to optimal is not supported yet. Because the beam base count vector is inserted as a candidate for each block, a valid implementation should record whether the base beam path survives the local DP. PLAN25 did not include that diagnostic.

Open correctness questions:
- Does `beam_chosen_counts` use the same block partition as the reconstructed `merged_seg`?
- Does every base beam count vector have finite local evaluation cost?
- Does the zero-offset beam path survive every layer?
- If a corridor-limited solution improves UB, does the code avoid incorrectly setting `lb=ub`?

### 4) Current blocker

The local offset representation is promising as a way around int64 overflow, but the PLAN25 method remains diagnostic/inconclusive until base-path and block-alignment checks are added.

### 5) Promotion decision

**Decision: diagnostic hold** — Keep local corridor code in solver but disable by default. Do not treat PLAN25 as evidence of corridor uselessness until PLAN25B/PLAN26 validates base-path survival and correct proof handling.

## 2026-04-26 update — PLAN24 beam-guided exact corridor evaluation

### 1) Corridor pruning never fires because sparse exact DP skips search

PLAN24 tested beam-guided Step-4 exact corridor on hard irregular K=10 rows. The corridor constrained exact DP states to be near the beam's prefix-count trajectory. However:

- Sparse exact DP uses mode `sparse_skip_theoretical` on all tested rows, meaning the theoretical bound check determines the gap is already tight enough and skips the exact search.
- Since no states are generated, no states are pruned. `exact_diag_corridor_pruned=0` for all 24 corridor rows.
- UB, LB, and gap are identical between standard and all corridor variants on every row.

### 2) No benefit from corridor in current regime

- No exact closure on any row (`is_optimal=0`).
- No gap improvement from any corridor variant vs `standard_step4`.
- No runtime improvement from corridor pruning.
- Corridor is neutral (no harm, no benefit) because no pruning opportunity arises.

### 3) Corridor approach unlikely to help without larger K or different instances

The beam produces incumbents that are strong enough to trigger the `sparse_skip_theoretical` guardrail, meaning the exact search never runs. On rows where exact DP would run (e.g., less-tight theoretical bounds), the corridor might still be relevant, but current evidence shows zero pruning. Larger K or different arithmetic families may be needed to test the corridor.

### 4) Next step for Step 4 certification

The remaining open question is not "can we prune states with beam guidance?" but "can we get exact DP to run at all on hard K=10+ irregular rows?" The theoretical bound is the main blocker, not the state space inside the corridor.

### 5) Promotion decision

**Decision: D** — No evidence that beam-guided exact corridor improves exact closure, gap, or runtime on hard irregular K=10 rows.

## 2026-04-25 update — PLAN23 role-based survivor policy evaluation

### 1) Gate 1 failed

PLAN23 tested role-based node evaluation (`role_mult3` and `role_mult3_feas`) on 5 Gate 1 rows:
- hardA_k10 seeds 0,1,2
- hardB_k10 seeds 0,2

Pass condition required:
- beat or tie standard on >= 4/5 rows
- improve gap on >= 2/5 rows
- runtime increase <= 20%

Results:
- `role_mult3`: wins=1, losses=1, ties=3; improved=1; rt_increase=+62.7%
- `role_mult3_feas`: wins=1, losses=1, ties=3; improved=1; rt_increase=+55.5%

Both variants failed all three criteria.

### 2) Role policy did not improve gap

On 3/5 rows, role variants produced exactly the same gap as standard_beam and uniform_mult2.
- hardA_k10 s0: 0.0172% (all variants identical)
- hardA_k10 s2: 0.0199% (all variants identical)
- hardB_k10 s0: 0.0391% (all variants identical)

The only win was hardB_k10 s2 (0.045% → 0.044%), a marginal 0.001% improvement.
The only loss was hardA_k10 s1 (0.0272% → 0.0283%).

### 3) Runtime increased substantially

Role policy generates more candidates per key (up to ROLE_MAX=3 representatives per FeasBeamKey). This increased per-layer processing time.
- hardA_k10 s0: standard ~315s → role ~547s (+74%)
- hardA_k10 s1: standard ~563s → role ~1257s (+123%)

### 4) Next step for beam quality

The role-based hypothesis (different representatives for score/local/arith/feas) is not validated. The remaining open question is whether beam-guided Step 4 certification (e.g., using the beam incumbent to guide exact DP pruning or ordering) can improve closure rate. No further survivor-policy tuning is justified.

### 5) Promotion decision

**Decision: E** — No survivor-policy change is validated; move next to beam-guided Step 4 certification.

## 2026-04-25 update — PLAN22B correction pass: ambig_scoreband_mult2 Gate 2 validation

### 1) Gate 2 validation completed

PLAN22B ran the missing `ambig_scoreband_mult2` rows on Gate 2:
- hardA_k10 seeds 2,3
- hardB_k10 seeds 0,1,2,3
- hardA_k12 seeds 0,1
- hardB_k12 seeds 0,1

Results vs standard_beam on Gate 2: wins=4, losses=5, ties=1.

### 2) PLAN22 decision corrected

The original PLAN22 Decision B (promote `ambig_scoreband_mult2` globally) is corrected to **Decision E**:
- Use `ambig_scoreband_mult2` only as a K=10 quality-improvement candidate, not as a global policy.

Rationale:
- It produces the best single gap (hardA_k10 s=0: 0.0094%) and helps on several K=10 seeds.
- But it does not generalize reliably beyond Gate 1 (4-5 on Gate 2).
- Mean gap across all 14 rows is 0.0357%, slightly worse than standard (0.0355%) and uniform (0.0353%).
- Mixed on hardB_k10 (2-2) and K=12 (maintains but does not improve incumbent production).

### 3) Next step for beam quality

The remaining open question is whether a refined scoreband (tighter band, alternative diversity features, or block-dependent eps) can improve robustness on K=8 and hardB_k10 seeds while keeping the K=10 anchor gains. No global promotion is justified yet.

## 2026-04-25 update — PLAN22 adaptive multiplicity validation (superseded by PLAN22B)

### 1) Adaptive node evaluation is partially validated

PLAN22 tested five adaptive multiplicity policies against standard beam on hard irregular K=8/10/12 rows.

- `ambig_scoreband_mult2` passed Gate 1 (3/4 not-worse, 3/4 improved) and produced the best single gap reduction: hardA_k10 seed 0 went from 0.0172% to 0.0094%.
- `hybrid_mult2` failed Gate 1 (2/4 not-worse) due to degradation on K=8 seeds.
- `early_mult2` passed Gate 1 but only improved runtime, never gap.
- Naive uniform multiplicity (`uniform_mult2`, `uniform_mult3_control`) failed Gate 1, confirming PLAN20B's seed-dependence warning.

### 2) Next step for beam quality

The remaining open question is not "should we use multiplicity at all?" but "can the `ambig_scoreband` filter be refined further (tighter score band, alternative diversity features, or block-dependent eps) to improve robustness on K=8 seeds while keeping the K=10 gap gains?"

### 3) Promotion decision (corrected by PLAN22B)

Original promotion candidate: `ambig_scoreband_mult2` (Decision B). This was an additive experimental toggle, not a default rewrite.

**Correction:** After Gate 2 validation, `ambig_scoreband_mult2` does not generalize reliably. Corrected decision is **E**: use only as a K=10 quality-improvement candidate.

## 2026-04-24 update — PLAN19 K=10/12 redesign completion

### 1) Exact closure at K=10/12 is confirmed infeasible under current fixed-block-DP budgets

PLAN19 tested three bounded additive redesigns:
- exact_after_beam C++ hook (post-beam exact mode activation)
- force_exact with guardrails raised to 1e12
- stronger K=12 beam

None recovered exact closure. `force_exact` immediately hits `skipped_comp_est`, confirming the exact fixed-block DP comp_est is astronomically large for K=10/12 irregular rows (B≈20, merged>16).

This is no longer a "maybe with better guardrails" blocker. It is a structural state-space boundary.

### 2) Routing override is justified and removes a runtime blocker

Baseline `energy_core` on K>=10 hard irregular rows consistently shows `selector_bypass` with no incumbent and wastes 500-1200s. Skipping it saves substantial runtime. This is an operational improvement, not a closure improvement.

### 3) Beam strengthening is not the path forward for K=12

`beam_plus` increased timeout rate (6/8 seeds) without improving incumbent quality. This redesign is disqualified.

### 4) Next blocker framing

The remaining open question is NOT "how to close K=10/12 exactly with fixed-block DP."

It is:
- can alternative exact methods (MIP, SAT) or better Step-2 heuristics close these gaps?
- or should the current ~0.02-0.06% gaps be accepted as the practical limit for this solver architecture?

No further fixed-block-DP redesign is justified from this evidence.

## 2026-04-24 update — PLAN18 K-boundary refinement completed

### 1) PLAN18 raw CSV is now complete (48/48 rows)

The two outstanding rows were rerun under patched memory-safe protocol:
- `hardB_k12 / irregular_reroute / seed=1` (replaced suspicious `rc=-15/no_csv_row` row)
- `hardB_k12 / irregular_reroute / seed=3` (previously missing)

Both reruns ended as clean external timeouts (`rc=-9`, `ub=-1`, `lb=-1`), confirming budget-limited behavior rather than anomalous failure.

### 2) Refined boundary near K=8/10/12

Completed PLAN18 evidence (seeds 0–3) sharpens the boundary:
- K=8: mixed exact vs finite-gap (2/4 exact on each ladder);
- K=10: no exact rows; finite-gap incumbents still usually produced via additive reroute;
- Historical PLAN18/19 K=12 status: mostly timeout/no-incumbent; only occasional
  finite-gap incumbents. Current PLAN33 status: tested hard K12 rows have valid
  certified finite gaps, so the remaining blocker is exact closure / HPC
  reproduction, not incumbent recovery.

So the exactness boundary is between K=8 and K=10. K=12 is beyond the current practical budget for irregular hard arithmetic at n=1000.

### 3) Dominant failure mode

At K=10: `finite_gap_after_step4` (via additive reroute) is the dominant best-of-route failure signature.
At K=12: `no_incumbent_timeout` dominates.

Baseline route continues to show `selector_bypass` (`non_mainline_solver`) on all high-K irregular rows, reinforcing that baseline energy_core is not the viable path here.

### 4) Memory-safety patch applied

`run_plan13_two_track_recovery.py` now avoids full-file `read_text()` on solver stdout/stderr. It reads only a trailing window (1 MB for stdout CSV, 8 KB for stderr tail). Default RSS cap was lowered from 16 GB to 12 GB.


## 2026-04-23 update — PLAN17 K-axis boundary at fixed `n=1000`

### 1) Prior ambiguity "is K itself hard?" is resolved for this budget

PLAN17 shows easy unit-contiguous families are exact through `K=20` at fixed `n=1000` (both seeds), so large K alone is not the active blocker in this regime.

### 2) Current active blocker shifts to irregular arithmetic from around `K=8`

For both hard irregular ladders, exact closure is robust at `K<=6`, then degrades starting around `K=8` (finite-gap / unresolved). At `K>=12`, the regime is mostly timeout-limited under the current per-row budget, with only a small residual finite-gap exception.

### 3) Budget-limited high-K irregular evidence is now explicit

Rows at irregular `K=12/16/20` are now measured under `900s`/row and `16 GB` cap. The active evidence is budget-limited high-`K` irregular behavior, not a missing-run gap.

### 4) K=2 routing blocker remains closed

PLAN17 K=2 rows use the intended route (`profile_repair_beam + auto_v1`) and do not show `non_mainline_solver` misrouting in final artifacts.

### 5) Immediate next blocker-resolving direction

If we need to sharpen the boundary beyond "around K=8", the next bounded step is a targeted rerun only on irregular `K=8/10/12` with slightly larger time budget per row, keeping route labels explicit and avoiding global solver redesign.


## 2026-04-22 update — PLAN_14 dense-unit large-K recovery (`g12345678910={1..10}`)

### 1) Prior `{1..10}` easy-family recovery blocker at `n=5000` is resolved under additive dense-unit Step-2 fast-path

With explicit additive toggle-gated behavior (`PAST_DENSE_UNIT_STEP2_FASTPATH=1`),
`{1..10}` now closes exactly at `n=5000` on seeds `0/1` via Step 2 (`ffd`), with
`UB=LB` and finite runtime in the PLAN14 artifact.

So the previously active blocker
("`{1..10}` runtime-window limited at `n=5000` in baseline")
is no longer a closure blocker for this family when the dense-unit fast-path
experiment is enabled.

### 2) Baseline path remains runtime-window limited; blocker reframed as pipeline-path issue, not intrinsic Step-2 hardness

The unchanged baseline still times out at `n=5000` in the tested window and emits
no incumbent on those timeout rows.

Given PLAN14 recovery under early Step-2 routing, the blocker is now best framed as:

- expensive generic pipeline behavior before/around Step 2 on dense unit-containing
  large-K rows,

rather than intrinsic hardness of `{1..10}` Step-2 closure.

### 3) Checkpoint/diagnostic quality blocker is reduced but not fully closed for baseline timeout rows

PLAN14 now records failure stage and peak RSS in dedicated checkpoint artifacts, and
includes control rows with finite `UB/LB` under fast-path.

However, baseline timeout rows can still end with `ub=-1/lb=-1` when no incumbent is
emitted before external termination. So "no useful incumbent on timeout rows" remains
a partial robustness blocker for baseline-only runs.

### 4) New near-term blocker for supervisor direction: `{1..20}` smoke execution wiring

`{1..20}` smoke rows were explicitly recorded as skipped in PLAN14 due to harness
family-map/payload wiring (family id not present in current paper-group mapping path).

So the immediate blocker for the next direction is now:

- add explicit `{1..20}` family wiring in the run harness/group map, then rerun
  smoke (`n=1000`, `n=2000`) under dense-unit fast-path.

### 5) Baseline integrity constraint remains active

Despite PLAN14 success, baseline policy is still unchanged and preserved.
Fast-path/count-based behavior remains additive and explicitly experimental until
formal promotion is approved.

## 2026-04-21 update — PLAN_13 two-track correction

### 1) `{1..10}` easy-family recovery blocker remains runtime-window limited

In the bounded PLAN13 pass, `g12345678910` at `n=5000` still timed out across:

- baseline energy-core route,
- additive mainline reroute probe,
- additive Step-2-incumbent-source probe (`i0`).

Under strict memory-safe execution, additive probes can also hit bounded
memory-limit kill before closure; this still does not provide a recovery at
`n=5000`.

So the easy-family target (`n>=5000`) is not yet recovered in this measured
window.

### 2) Prior `g37` closure blocker is resolved when rerouted to intended K=2 Step-3 exact path

Required reruns for `g37` (`n=750..5000`) under mainline K=2 selector/exact mode
show:

- `selector_decision=exact`, `selector_reason=k2_exact_default`,
- `step3_mode=exact`, `fwd_pack_method=profile_realization_dp_exact`,
- `fwd_block_dp_status=feasible`,
- exact closure (`UB=LB`) at Step 3 on tested rows/seeds.

Therefore the earlier unresolved `g37` rows were a routing/mode issue, not
evidence that K=2 Step-3 exact profile realization is intrinsically blocked on
that family through `n=5000`.

### 3) Paper-facing blocker framing after PLAN13

Current unresolved blocker for this two-track scope is now concentrated on:

- `{1..10}` runtime/termination behavior at `n>=5000`.

`g37` is no longer the active closure blocker at `n<=5000` once correctly
rerouted.

## 2026-04-20 update — PLAN_11 paper-group extension blockers

### 1) High-n robustness blocker (hard failure via `std::length_error`)

In the accepted baseline package, several paper groups now hit a hard runtime
failure mode at larger `n`:

- `g3567`: `n=8000` crashes (`returncode=-6`, `std::length_error`),
- `g246810`: crashes from `n=7000`,
- `g810`: crashes from `n=6000`.

This is a stronger blocker than finite-gap timeout because it prevents normal
diagnostic closure on those rows.

### 2) Regime break after newly extended exact points

Group-wise practical boundary now shows a clear regime break:

- `g3567`: exact at `n=6000`, then timeout/kill at `n=7000`,
- `g12357`: exact at `n=8000`, timeout at `n=10000`,
- `g12345678910`: timeout remains at and beyond `n=5000`.

So current blocker is no longer small-gap closure for these rows; it is runtime
and robustness at high `n` under fixed baseline policy.

### 3) `g37` remains closure-limited with Step-4 entry

Old `g37` ledger was exact only through `n=600`, but this is no longer the
current method status.

PLAN13 reroute shows tested `g37` rows through `n=5000` close through Step-3
`profile_realization_dp_exact`. Remaining `g37` blocker is only beyond the
corrected tested range and HPC reproduction.

Beyond that, the family does reach the exact fallback on some rows, but this is
not producing closure:

- `n=750,1000`: sparse exact times out,
- `n=1500,2500,3500,5000`: Step 4 is entered but still fails with no usable
  finite closure,
- `n=6000,7000`: unresolved despite Step-4 entry.

This remains a family-specific closure blocker distinct from the crash regime.

### 4) Additive experiment did not resolve stalled `g810`

A bounded additive experiment (`force_beam` selector on `g810`) was run in a
separate comparison artifact and produced the same failure mode as baseline.

No variant promotion is justified from current evidence.

### 5) Baseline integrity constraint remains active

Despite these blockers, accepted baseline policy is intentionally unchanged.

Any future mitigation must stay additive and explicitly experimental until it
proves better than baseline on direct comparison rows.

## 2026-04-19 update — PLAN_10 generator-policy pass (status after Phase A/C)

### 1) Prior K=4 pattern-generation bottleneck is resolved for the active gate

Phase-A generator decision testing (`PATTERN_DP_K=4`) on all active required K=4
rows preserved exactness (`10/10`) and produced a large runtime reduction,
including hard `g3567` seed-0 rows.

So the earlier blocker "K=4 pattern generation dominates runtime" is resolved
for this active scope under the selected K=4 policy.

### 2) Signature-dedup stage is low-value in current K=4 DP-generator regime

Measured comparison with/without signature dedup at K=4 showed negligible pattern
pool impact and a small runtime preference for dedup-off.

Resulting policy update: K=4 default disables signature-dedup.

### 3) Memory safety remains a standing operational constraint

All accepted heavy runs remain constrained by:

- one heavy row at a time,
- active RSS monitoring,
- hard kill threshold `16.5 GB`.

Observed accepted peaks in this pass remained well below the threshold (single
digit GBs), but memory guards remain mandatory for large-row campaigns.

### 4) What remains open after this pass

The specific K=4 generator bottleneck blocker is closed, but broader frontier
questions (outside this strict task) remain open, especially for non-K=4 or
different method regimes.

## 2026-04-19 update — PLAN_10 speedup blockers

### 1) Continuity-safe closure is restored, but required runtime speedup not yet achieved

Under the continuity-safe baseline package (`energy_core + direct`, fortified
features off, `state_keep=60000`), required K=4 rows are exact again (`10/10`
exact on the active plan10 scope).

So exactness is no longer the immediate blocker for this stage.

Current blocker is runtime: hard K=4 rows are still expensive.

### 2) First same-output optimization pass regressed runtime

Implemented pass (partial selection via `nth_element` in pattern buckets and
phase-1 beam trimming) preserved exactness, but increased runtime:

- hard required `g3567` rows: mean runtime `+15.6%`
- required continuity rows: mean runtime `+2.7%`
- overall required rows: mean runtime `+13.0%`

This pass is therefore disqualified as a speedup package.

### 3) Pattern generation remains the primary bottleneck in current package

On required hard rows, `fwd_ec_time_pattern_generation` still dominates the
early stage and grew after the pass-1 change.

`fwd_ec_time_exact_core` remains comparatively small and is not the first
optimization target.

### 4) Memory safety remains an active operational constraint

All accepted plan10 runs were executed with active RSS monitoring and a hard
kill threshold at `16.5 GB`.

Observed accepted peak RSS was below this threshold (roughly `4.8 GB` to
`9.7 GB` on required rows), but this memory guard must remain active for future
heavy runs.

## 2026-04-17 update — PLAN_08 fortification blockers

### 1) K=4 large-n closure regressed under fortified path

Required `g3567` PLAN_08 rows (`n=1000,1500,2500,3500,5000`, seeds `0,1`) now
show:

- exact closure only on `n<=1500` (`4/10` exact total),
- Step-4 finite-gap dependence on all rows with `n>=2500`.

This is weaker than the recovered baseline where the same rows were exact after
reruns.

### 2) Historical continuity not preserved (`3567_plus`)

Continuity checks (`n=3500,5000`, seeds `0,1`) all returned finite gaps
(`~0.022%–0.035%`) with Step-4 decision, instead of exact closure.

This is currently the most critical blocker for claiming a successful
fortification rollout.

### 3) Runtime concentration remains in Phase-1 beam + pool generation

New diagnostics show the dominant wall-time terms on hard `g3567` rows are:

- `fwd_ec_time_pattern_generation`, and
- `fwd_ec_time_phase1`.

`fwd_ec_time_exact_core` is comparatively small, so additional exact-core-only
tuning is unlikely to remove the main runtime bottleneck.

### 4) Seed sensitivity remains structurally high

On required `g3567` rows, seed-0 remains much slower than seed-1 (mean runtime
roughly `2366s` vs `490s`), so the targeted stabilization objective is not yet
met.

### 5) Memory/size risk still active on large direct-completion regimes

Without guardrails, large rows can still hit process-kill behavior. The new
completion-table cap avoids hard crashes, but this introduces a fallback branch
that can change quality behavior. This tradeoff is now explicit and must remain
tracked in evaluations.

### Practical next blocker-resolving direction

Given the diagnostics, the next bounded work should prioritize:

1. reducing phase-1 beam burden (width/ordering/trigger policy),
2. reducing pattern-generation cost for hard blocks,
3. preserving direct-completion strength only where safe,
4. re-validating continuity first before any broader K-growth claims.

## 2026-04-16 update — Targeted K=4 recovery check

### What was tested

Before continuing broad Plan-05 extension, reran K=4 with forced
`PAST_RELAXED_BINPACK_SOLVER=energy_core` under `step1_exact_guided` on:

- historical `3567_plus` frontier anchors (`n=3500,5000`),
- paper-group `g3567` rows (`n=1000,1500,2500,3500,5000`),
- and compared against default policy on the same paper-group rows.

### Current blocker status for K=4

The old archive closure path (energy-core incumbent then exact-guided closure
through `n=5000`) is not currently reproduced on the historical `3567_plus`
anchors.

Observed on current code (forced energy-core):

- `3567_plus n=3500`: gap `0.0347%` (Step 4 used)
- `3567_plus n=5000`: gap `0.0514%` (Step 4 used)

On paper-group `g3567`, forced energy-core helps only the `n=1000` row (exact
and faster), but gives weaker gap quality than default on the larger tested
rows.

### Practical implication

K=4 remains an active boundary blocker:

- do not promote a blanket energy-core-first K=4 policy,
- keep default mainline policy,
- treat energy-core as targeted override/diagnostic until a consistent
  cross-row win is demonstrated.

## 2026-04-15 update — Plan 03F blocker status

### Resolved blocker: Step-3 K=2 default bypass

Previously, mainline selector scope excluded `K=2` and beam kernels reject
`K<=2`, so the `{8,10}` family was not using a true Step-3 profile-repair path
in default policy.

This is now resolved:

- selector scope includes `K=2` as Mode A (exact-by-default with safety gates),
- `{8,10}` mandatory reruns (`n=500..5000`) all solved with
  `profile_realization_dp_exact` at Step 3,
- no Step-4 exact rescue needed on those rows (`diag_exact_dp_used=0`).

### Remaining active blocker

K=4 frontier quality/closure remains mixed:

- representative `g3567 n=1000` still takes Step-3 beam path and needs Step-4
  exact to close the final tiny gap.

So the blocker has shifted from “missing K=2 Step-3 path” to “improving K>=4
frontier quality while preserving the unified Step-3 exact-vs-beam policy.”

## 1. The current benchmark story is still partially confounded

### Problem

The current extension mixes:

- `K`,
- arithmetic structure,
- and incumbent-generation difficulty.

### Why that matters

Without separating the axes, we risk drawing the wrong conclusion, for example:

- "`K=6` is hard"

when the more accurate claim may be:

- "`K=6` with awkward arithmetic is hard."

## 2. The arithmetic descriptors are not yet formalized in the benchmark tables

### Problem

We have an intuition about:

- easy vs medium vs hard arithmetic,
- but not yet a standardized descriptor set for the paper.

### Immediate need

For each family we likely need a small arithmetic profile:

- presence of `1`,
- contiguity / density,
- rough Frobenius-type difficulty,
- maybe multiplicity and a simple semigroup descriptor.

We also need to distinguish clearly between:

- easy arithmetic,
- medium arithmetic,
- and hard arithmetic,

because the current archive structure already assumes three classes rather than
a binary split.

## 3. Current datasets may be uneven across the two axes

### Problem

The existing datasets already contain useful families, but they may not form a
perfect matrix over:

- `K`,
- arithmetic hardness,
- and `n`.

### Consequence

We may later need:

- either careful subsampling of existing families,
- or curated new family generation.

In particular, the current matrix may still be missing:

- hard-arithmetic high-`K` cross-cells,
- which are needed to test whether the two axes compound rather than merely
  coexist.

## 4. Hard-arithmetic incumbent refinement is still open

### Problem

Even after the current large-`K` repair progress, hard-arithmetic families still
leave small open gaps.

### Status at archive creation

This archive does not yet assume a new algorithm.

It first asks:

- is the residual difficulty really arithmetic-driven,
- and if yes, which branch should own it:
  - better incumbent generation,
  - better pattern generation,
  - or better exact certification?

## Immediate next directions

1. classify all currently relevant families into arithmetic classes
2. summarize validated results by arithmetic class, not only by `K`
3. first test hard-arithmetic incumbent refinement before changing the whole assignment layer
4. if quality still looks capped by the filtered pattern pool, escalate to dynamic pricing
5. keep arc-flow as the heavier follow-up only if pricing still leaves the same structural ceiling

## 5. Level 3 was under-modeled, but it was not the whole bottleneck

### What changed

The first structural code change from this archive was:

- evaluate recovered block assignments on block-local windows,
- with exact dense per-block multiset DP when the local state space is small,
- instead of scoring them only through a global ascending/descending surrogate.

### What happened

This change materially improved the open six-type rows:

- `2345711 n=1000`: gap dropped from `0.0356%` to `0.0082%`
- `2345711 n=1500`: gap dropped from `0.0294%` to `0.0063%`
- `456789 n=1000`: gap dropped from `0.0480%` to `0.0164%`
- `456789 n=1500`: gap dropped from `0.0442%` to `0.0159%`

So Level 3 approximation error was real.

### Why it is still not the full answer

Even after that change, the representative six-type incumbents still land via:

- `block_repair_feasible_beam`

rather than via a fully separated Level 2 winner.

Interpretation:

- Level 3 mattered enough to fix,
- but Level 2 remains the deeper structural frontier.

## 6. The current large gain still does not prove the final architecture

### Problem

The new Level 3 split improved results, but it did not yet yield the cleaner
end state we ultimately want:

- Level 1 profile recovery
- Level 2 assignment winner
- Level 3 exact within-block scheduling

The current solver is better, but still often behaves as:

- improved Level 3 evaluation
- feeding a strong but still heuristic Level 2 beam incumbent

### Consequence

The next major decision should not be another generic tuning cycle.

It should ask:

- does Level 2 now need one more clean seeded refinement,
- or are we already at the point where dynamic pricing is the right next
  structural step?

## 7. The hard-arithmetic high-`K` cross-cell is still too thin

### Status

The baseline grid now includes:

- easy arithmetic through `K=10`
- medium/hard arithmetic at `K=6`
- one small irregular cross-cell at `K=7`

### What is still missing

A stronger hard-arithmetic high-`K` anchor, for example:

- `K=8` or `K=10` with an irregular family at meaningful `n`

This is still needed to decide whether the two axes merely coexist or actually
compound.

## 8. The solver policy needed cleanup before the new grid could be trusted

### What was wrong

Before the current phase-1 grid work, the solver still had hidden mixed-policy
behavior:

- an internal seeded feasible-beam rescue inside the Lagrangian branch,
- and a post-Lagrangian beam-polish pass enabled by default.

That made it harder to interpret new results cleanly, because the default path
was quietly combining methods even when the archive wanted them compared as
separate Level-2 strategies.

### What was changed

The baseline policy was cleaned up:

- `PAST_BLOCK_REPAIR_LAGR_USE_SEEDED_BEAM` now defaults to `0`
- `PAST_BLOCK_REPAIR_LAGR_BEAM_POLISH_MAX_MERGED` now defaults to `0`

Interpretation:

- the default solver policy is now easier to explain,
- and future comparisons of Lagrangian vs beam are less confounded by hidden
  hybrid behavior.

## 9. The new two-axis runner needed one correctness fix

### Problem

The first draft of the new experiment utility interpreted `n` as jobs-per-type
instead of total jobs.

That silently inflated instance size by a factor of `K` and made the first
batch misleading.

### Resolution

The runner now samples `n` total jobs from the family length set with a fixed
deterministic seed, matching the benchmark convention used elsewhere.

### Consequence

The archived phase-1 grid slice is trustworthy, but earlier aborted runs from
the buggy runner should be ignored.

## 10. The hard-arithmetic high-`K`, large-`n` cell is still open

### What is now known

From the new runner:

- irregular `K=8`, `n=300` is exact at Step 1
- irregular `K=10`, `n=300` is exact at Step 1

So the cross-cell is no longer completely empty.

### What is still missing

We still do not yet have a clean recorded result for:

- irregular `K=8` or `K=10` at `n=1000`

### Why this matters

This is the first cell that can really answer whether:

- the arithmetic and `K` axes merely coexist,
- or whether they start to compound at larger total-job scales.

### Current status

The first controlled batch for those rows was intentionally interrupted and not
archived as a result. So this remains a real experimental blocker, not a
reported finding yet.

## 11. Status update: high-`K` irregular `n=1000` blocker is resolved

### What changed

The required cross-cells are now completed in:

- `research/k_vs_arithmetic_axes_20260412/csv/two_axis_grid/TWO_AXIS_GRID_20260412_phase2.csv`

Results:

- `hard_k4_irregular n=1000`: gap `0.0083%`, winner `block_repair_energy_core`
- `hard_k8_irregular n=1000`: gap `0.0157%`, winner `block_repair_feasible_beam`
- `hard_k10_irregular n=1000`: gap `0.0221%`, winner `block_repair_feasible_beam`

So this is no longer a missing-cell blocker.

### New interpretation

The unresolved issue shifts from:

- "can we run the row at all?"

to:

- "can Level 2 close the remaining tiny finite gaps on arithmetic-hard,
  larger-`K` rows?"

## 12. Robustness bug was real and fixed (exact-DP seed overflow)

### Problem

`hard_k10_irregular n=1000` initially crashed with `returncode=-11`.

### Diagnosis

ASAN showed a heap-buffer-overflow in `solve_exact_multiset_dp(...)` seed logic:

- code attempted seed transition `new_s = strides[i]` even when `totals[i] = 0`
- this can produce `new_s == NC` and out-of-bounds access in state arrays

### Fix

Added `totals[i] <= 0` guards in seed loops for:

- `solve_exact_multiset_dp(...)`
- `smart_reconstruct(...)`
- `solve_sparse_exact_multiset_dp(...)`

### Consequence

The previously crashing required row now runs to completion and is archived.

## 13. Current Level-2 baseline remains beam-owned under clean policy

### Validation summary

On required hard-six anchors (`n=1000,1500,2500`) with cleaned defaults
(seeded beam off, beam polish off), current winner remains:

- `block_repair_feasible_beam`

### Consequence

The earlier Lagrangian-owned baseline regime was not cleanly recovered on this
branch. This keeps the next structural question focused on Level 2 assignment
quality, not on Level 3 evaluation.

## 14. Decision at end of Plan 01

Recommendation selected:

- **Recommendation 2: Level 2 still needs one final algorithmic escalation**

Reason:

- required cross-cells are complete and stable,
- but beam still dominates important arithmetic-hard rows,
- and residual gaps remain assignment-driven.

Per plan, the next algorithmic step should be:

- dynamic pricing inside the Lagrangian loop,

not broad generic retuning.

## 15. Plan 02B update: exact Level-2 resolves small/moderate rows

### Correction to prior blocker statement

The prior Plan-02 pool-membership diagnostic (`beam_not_in_pool`) was
tautological and does not establish pool-vs-search by itself. Plan 02B replaced
that test with exact Level-2 branch-and-bound over the current pattern pool.

### What exact Level-2 showed

On required validation rows:

- **Closed with improvement over beam**
  - `hard_k4_irregular n=1000` (B=9)
  - `hard_k6_2345711 n=1000 seed=0` (B=8)
  - `hard_k6_2345711 n=1000 seed=1` (B=14, with longer exact-L2 budget)
  - `medium_k6_dense n=1000` (B=9)

- **Not closed within 180s**
  - `hard_k8_irregular n=1000` (B=19)
  - `hard_k10_irregular n=1000` (B=20)

### Current blocker framing (updated)

The key open blocker is now narrow and evidence-based:

- For larger merged-block counts (B≈19–20), exact in-pool Level-2 search is
  still too expensive at current time budgets, and beam remains the practical
  winner.

This does **not** yet prove the pool/profile ceiling; it proves a large-B
Level-2 search hardness regime.

### Recommended next step (minimal redesign)

Adopt a hybrid Level-2 policy:

- exact Level-2 as a diagnostic/improvement stage for small/moderate B,
- beam fallback for larger B,
- only escalate to out-of-pool redesign (LNS/pricing) once larger-B
  exact-in-pool runs no longer improve despite increased budget.

## 16. Plan 03/04 cleanup status: pipeline is now simplified and aligned

### What is no longer a blocker

The policy-clarity blocker is resolved. Default solver behavior now matches one
clean final story:

- Step 1: semigroup profile recovery
- Step 2: fast realization (FFD/BFD/random)
- Step 3: one unified hard-case method (`profile_repair_beam`)
- Step 4: exact DP fallback/certification

Demoted from default mainline (kept only archival/explicit):

- `lagrangian_assign`
- `rg_beam`
- `feasible_counts`
- post-Lagrangian beam polish
- exact-L2 incumbent replacement

Exact-L2 now defaults to disabled and does not affect default incumbents.

### Current blocker framing after cleanup

The remaining blocker is algorithmic quality on larger arithmetic-hard rows,
not method-story complexity:

- with default cleaned policy, Step 3 (`profile_repair_beam`) remains strong,
  but finite gaps persist on harder rows under practical budgets;
- Step 4 exact DP is the only exact fallback and can still time out on large
  instances.

### Next focused blocker to solve

Improve Step-3 quality within the same unified family (profile-guided beam +
local destroy/repair) so fewer hard rows need long exact fallback budgets.

## 17. Plan 03B/04A status: diagnostics interpretation blocker is resolved

### What changed

Exact-stage diagnostics were hardened so Step-4-attempted rows no longer emit
ambiguous `dense` + INF/zero counter combinations when exact is skipped by
guardrails.

Current explicit exact skip modes include:

- `sparse_skip_theoretical`
- `sparse_skip_overflow`
- `sparse_invalid_totals`
- `dense_skip_state_space`
- `dense_skip_memory`

And dense timeout now reports reached/expanded counters with
`exact_diag_exhaustive=0`.

### Consequence

The prior instrumentation-interpretation blocker is no longer active.

For Step-4-entered `K=6` rows in current validation, diagnostics now clearly
show exact attempted but skipped by sparse theoretical-lattice guardrail
(`sparse_skip_theoretical`) with initial/final UB handoff preserved.

### Remaining blocker (unchanged)

The open blocker remains algorithmic quality under practical budgets on
arithmetic-hard rows:

- Step 3 (`profile_repair_beam`) is significantly active and provides incumbents,
  but small finite gaps remain on hard rows;
- Step 4 exact DP is still bounded by sparse/dense state-space guardrails and
  runtime limits on larger instances.

So the next work should continue improving Step-3 quality and exact-DP
practical pruning/ordering inside the same 4-step architecture.

## 18. Plan 03C update: Step-3 method-family identity blocker is resolved

### What changed

Step 3 is now structurally and descriptively unified as one family:

- profile-realization DP
  - exact mode: fixed-block DP
  - truncated mode: profile-repair beam

Shared components between modes are now explicit in code:

- recovered blocks,
- count-state evolution,
- local exact block evaluator,
- compatible block-ordering policy,
- suffix residual feasibility checks.

### Consequence

The previous “beam vs fixed-block as unrelated methods” interpretation is no
longer the active blocker. Fixed-block DP is retained and positioned as Step-3
exact mode, not archival demotion.

### Remaining blocker after unification

The open blocker is practical tractability regime, not method identity:

- under default guardrails, Step-3 exact mode is often skipped on larger rows
  (`skipped_comp_est`),
- exact-mode tractability appears on smaller rows / raised guards,
- for larger arithmetic-hard rows, quality still depends on truncated Step-3 mode
  plus Step-4 fallback budgets.

### Measured exact-safe enhancement status

- suffix residual pruning in exact mode shows a meaningful runtime benefit on the
  K=6 seed-scan slice.
- hardest-first ordering did not show a win on the same slice; it remains an
  optional policy knob, not a demonstrated default gain yet.

## 19. Plan 03D update: selector-policy blocker is resolved

### What changed

Step 3 now has an explicit regime selector (no ad hoc exact-vs-beam decision):

- policy knob: `PAST_PROFILE_REALIZATION_SELECTOR_POLICY`
  (`auto_v1`, `off`, `force_exact`, `force_beam`),
- row-level diagnostics: selector decision + reason + arithmetic snapshot,
- exact/beam status and timing split recorded for every run.

`auto_v1` chooses exact only in a conservative tractability region:

- merged blocks `<= 4`
- state space `<= 1e8`
- total comp-est `<= 1e8`
- max block comp-est `<= 8e7`
- no hard-alarm trigger.

### Validation status

On the representative Plan-03D validation set
(`csv/plan03d/TMP_plan03d_selector_validation_table.csv`, 13 rows):

- exact chosen on 8 rows (all feasible),
- beam chosen on 5 rows (all appropriate for large/hard frontier),
- no misclassification observed in this slice.

Forced-exact controls on larger hard rows were frequently skipped by comp-est
guardrail and could leave no incumbent (`pack_method=none`,
`pack_outcome=exact_skipped_comp_est`), supporting beam preference there.

### Updated blocker framing

The blocker is no longer “we do not know when to run Step-3 exact mode.”

The remaining blocker is narrower:

- improve robustness/calibration of the selector boundary on near-threshold rows,
- and continue improving Step-3/Step-4 practicality on large frontier rows where
  beam remains the only practical Step-3 mode.

### Next focused work

1. run a targeted near-threshold calibration sweep (merged 4..6, comp-est near
   `1e8`) to stress false-positive/false-negative behavior,
2. add more hard-alarm-trigger families/seeds to confirm the alarm remains
   conservative and informative,
3. keep `auto_v1` as current default unless calibration reveals regressions.

## 20. Hardening update: exact-primary brittleness blocker is resolved

### What changed

For `auto_v1` exact-primary rows, Step 3 now has automatic in-cycle fallback:

- exact fixed-block DP first,
- if exact yields no finite Step-3 candidate, immediate beam fallback.

This covers the known unusable exact outcomes (`skipped_comp_est`,
`skipped_nc`, `timeout`, `reconstruct_failed`, and other non-finite exact
candidate cases).

### Evidence

Fallback diagnostics are now explicit in CSV:

- `fwd_profile_exact_primary_fallback_to_beam`
- `fwd_profile_exact_primary_status_before_fallback`
- `fwd_profile_step3_incumbent_mode`

Probe files confirming fallback activation:

- `csv/plan03d/TMP_plan03d_exact_primary_fallback_probe.csv`
- `csv/plan03d/TMP_plan03d_exact_primary_fallback_skipnc_probe.csv`
- `csv/plan03d/TMP_plan03d_exact_primary_fallback_skipcomp_probe.csv`

So this is no longer a hidden-policy or brittle-control-flow issue.

## 21. Validation-overclaim blocker is resolved (method split enforced)

### What changed

Selector revalidation now explicitly separates row roles:

- Step-2-closed rows,
- Step-3 selector-test rows,
- Step-4-used rows.

Primary table:

- `csv/plan03d/TMP_plan03d_selector_boundary_reval_table.csv`

### Consequence

Misclassification is now counted only on Step-3 selector-test rows, not on
Step-2-closed controls.

Current counts from rebuilt table:

- `step2_closed_rows = 10`
- `step3_selector_test_rows = 6`
- `step4_used_rows = 5`
- `misclassifications_on_step3_rows = 0`

The earlier overclaim risk from mixed-row counting is therefore removed.

## 22. Remaining blocker after this pass

The remaining blocker is confidence width, not correctness of methodology or
fallback control flow:

- boundary sample where Step 3 is genuinely exercised is still small,
- more near-threshold seeds are needed before claiming a stable selector
  boundary at publication scale.

Practical policy after this pass:

- keep `auto_v1` as default,
- continue boundary-focused accumulation of Step-3-test rows.

## 23. Plan 04C update: skip-dominated matrix blocker is partially resolved

### What changed

The prior Plan-04C matrix was largely uninformative for pruning because many
rows ended with:

- `exact_diag_mode=none`, or
- `sparse_skip_theoretical` with zero expansion counters.

A targeted v3 pass was run with raised sparse-theoretical guardrail and
hard-anchor slices where sparse exact expands millions of states.

Primary evidence files:

- `csv/plan04c/TMP_plan04c_v3_evidence_runs.csv`
- `csv/plan04c/TMP_plan04c_v3_phase1_incumbent_quality.csv`
- `csv/plan04c/TMP_plan04c_v3_phase2_exactdp_variants.csv`
- `csv/plan04c/TMP_plan04c_v3_phase3_best_combos.csv`

### Consequence

For `hard_k8_irregular n=500`, exact counters are now usable, so incumbent vs
pruning interaction is measurable.

Observed blocker shift:

- not "no exact visibility" anymore on this anchor,
- now "limited breadth of anchors with usable exact expansions."

### Current remaining blocker (Plan 04C)

Evidence breadth is still narrow:

- one strong hard anchor (`hard_k8 n=500`) now has good exact counters,
- additional K6/K8 rows still often either close early, fail Step-3 incumbent for
  `i0`, or time out before producing broad comparative conclusions.

So we now have a clear directional result (incumbent lever stronger on the main
hard anchor), but not yet a wide multi-row confirmation set.

### Next blocker-solving direction

1. Add 2-3 nearby hard rows where exact expands with finite incumbent and
   practical budget (same measurement protocol).
2. Keep `i2` and `i3` as primary incumbent contrasts; keep `p0/p1/p3` as pruning
   contrasts.
3. Record whether `p1` counter gains ever translate into UB/gap gains under any
   nontrivial finite-incumbent row.

## 24. PLAN16 fixed-n K-scaling blocker update

Completed fixed-`n=1000` K-scaling sweep (`plan16`) shows:

- large-`K` contiguous dense families are not currently blocked at this `n`:
  - `K=10` and `K=20` close exactly at Step 2 on both seeds;
  - dense-unit fastpath reduces runtime at both `K=10` and `K=20`.
- corrected K=2 reroute results at the same `n` show:
  - `g37` and `g810` close exactly via Step 3 `profile_realization_dp_exact`
    on both seeds once routed through `profile_repair_beam + auto_v1`.

Current blocker framing after PLAN16:

- there is no remaining fixed-`n=1000` blocker on `g37/g810`;
- the incorrect unresolved rows were caused by a runner-routing mistake, not by
  a real method wall at small `K`.
