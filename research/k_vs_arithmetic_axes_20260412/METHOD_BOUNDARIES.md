# Method Boundaries and Pipeline Roles

Last updated: 2026-04-30 (PLAN33 verified — Decision A)

2026-04-28 PLAN31 family-aware survivor policy note (`n=1000`, `lambda=1.3`, K=10):

- Family-aware survivor selection passes PLAN31 Gate A: 6/8 improved, 7/8 not worse vs standard.
- Best: hardA=uniform_mult2, hardB=ambig_scoreband_mult2 (or late_ambig). Mean gap 0.0309–0.0318% vs baseline 0.0345%.
- `residual_aware` plumbing fixed (missing `beam_diag` copy); policy now activates correctly but shows zero gap effect.
- `fine_plus_coarse_lookahead` smoke test shows worse gap; not promoted.
- Family asymmetry is structural: hardA benefits from uniform multiplicity (has job length 2), hardB benefits from ambiguous scoreband (no job length 2, coarser arithmetic).
- **Decision: A** — Promote family-aware survivor selection for hard K=10 beam rows.
- Global fallback: `uniform_mult2` (PLAN27 A).

2026-04-28 PLAN30 easy-vs-hard fixed-n K-scaling boundary note (`n=1000`, `lambda=1.3`, implements PLAN_16):

- Easy contiguous-unit families (`{1..K}`) remain exact through `K=40` at fixed `n=1000`.
- `K=24`: mean runtime 364s, 4/4 exact, Step 2 (`ffd`).
- `K=30`: mean runtime 683s, 4/4 exact, Step 2 (`ffd`).
- `K=40`: mean runtime 1552s, 4/4 exact, Step 2 (`ffd`).
- All memory-safe (peak RSS 1.7–4.8 GB).
- Hard irregular exact closure degrades around `K=8–10` (mixed exact/finite-gap
  at K=8, finite-gap at K=10). PLAN33 later gives certified finite-gap K12
  incumbents, so K12 is no longer a no-incumbent regime in the current method.
- **Boundary sharpened**: The practical K boundary is not a universal K threshold; it depends on arithmetic structure. Easy families scale to much larger K than hard families.
- **Decision: A** — K-scaling story is sufficiently documented.
- **Note**: PLAN28 remains reserved for block-realizability diagnostics and repair (see below).

2026-04-28 PLAN27 Step-3 survivor policy note (`n=1000`, `lambda=1.3`, K=10):

- Tested `uniform_mult2`, `ambig_scoreband_mult2`, `late_ambig`, `residual_aware`, `late_residual_ambig` on hardA_k10/hardB_k10 seeds 0-3.
- `uniform_mult2` passes promotion: 6/8 not worse, mean gap improves (0.0343% vs 0.0345%), runtime decreases 14.3%.
- `late_ambig` and `late_residual_ambig` show real signal (5W/3L) but fail the 6/8 not-worse threshold.
- `residual_aware` has zero gap effect. The `score_policy` diagnostic stays `default` despite runner setting `PAST_PROFILE_REPAIR_BEAM_SCORE_POLICY=residual_aware`. Root cause unknown; possibly build/env propagation issue.
- Multiplicity policies are family-dependent: `uniform_mult2` helps hardA, `ambig_scoreband_mult2`/`late_ambig` help hardB.
- **Boundary**: `uniform_mult2` is the best validated global Step-3 survivor policy. `late_ambig` is a per-family candidate. `residual_aware` is blocked pending root-cause diagnosis.
- **Decision: A with caveat**.

2026-04-27 PLAN26 local corridor validation note (`n=1000`, `lambda=1.3`, K=10):

- Implemented correctness repairs: fixed `lb=ub` bug, added alignment diagnostics, fixed `merged_blocks` propagation, added base-path survival simulation.
- Tested on hardA_k10 s0 and hardB_k10 s2 with delta=1,2.
- All rows memory-safe (RSS 4–8 GB).
- **Base path does NOT survive**: `base_path_survives=0` on all local corridor rows.
- **Reject reason**: `base_candidate_not_found_at_layer_0`.
- **Root cause**: `generate_energy_core_patterns` generates patterns by work capacity, not schedulability. The beam validates the full global sequence, not individual blocks. `evaluate_profile_block_counts` (block-local scheduler) is stricter than the beam's global validation.
- **Block count mismatch fixed**: `merged_blocks` propagation ensures `block_count_mismatch=0`. The mismatch was a red herring.
- **Boundary sharpened**: Local corridor DP is fundamentally invalid as designed. Block-local evaluation cannot be used with beam counts that are only guaranteed to be feasible globally.
- **Decision: C** — Local corridor invalid due to block/path mismatch. Do not use until fundamentally redesigned (e.g., global sequence evaluation instead of block-local evaluation).

2026-04-26 PLAN25 local corridor exact DP note (`n=1000`, `lambda=1.3`, K=10):

- Implemented `beam_corridor_local_dp()` with local offset encoding `(2*delta+1)^K` per layer, avoiding global mixed-radix int64 overflow.
- Candidate generation perturbs beam prefix counts (single + pair moves), keeps top 50 per block by cost.
- Hard state cap 5M, time cap 300s.
- Tested on hardA_k10 seed=0 and hardB_k10 seed=2 with delta=1,2.
- All rows memory-safe (RSS 4–8 GB). Local offset state exploration runs:
  - delta1: 25–24s, 40–60k states_seen
  - delta2: 52–72s, 7.6–14.1M states_seen
- Status consistently `infeasible_corridor`.
- `best_ub = inf` on all local corridor rows; incumbent unchanged.
- **Boundary correction**: PLAN25 validates the local offset representation, not the full corridor method. Since the beam base count vector should normally define at least one corridor path, `infeasible_corridor` requires follow-up diagnostics for base-path survival, block alignment, and candidate evaluation.
- **Decision: diagnostic hold** — Keep code disabled by default. Do not use PLAN25 as proof that the beam prefix is too close to optimal or that the local corridor has no useful completion.

Last updated: 2026-04-26 (PLAN24B forced-entry corridor diagnostic)

2026-04-26 PLAN24B forced-entry corridor note (`n=1000`, `lambda=1.3`, K=10):

- Added `PAST_EXACT_CORRIDOR_FORCE_ENTRY=1` env var to bypass `sparse_skip_theoretical` and force sparse exact DP to enter search.
- Clamps internal time budget to `PAST_EXACT_CORRIDOR_TIME_LIMIT` (default 300s) and states to `PAST_EXACT_CORRIDOR_MAX_STATES` (default 50M).
- Tested on hardA_k10 seed=0 and hardB_k10 seed=2 with corridor delta=1,2.
- **All forced rows hit `sparse_skip_overflow`**: int64 mixed-radix encoding overflows for K=10 at n=1000. Product of (totals[i] + 1) exceeds int64.
- Zero states generated, zero corridor pruning. Identical gaps to standard.
- **Boundary sharpened**: The sparse exact DP encoding is fundamentally limited to ~K=8 at n=1000. The blocking issue is int64 overflow, not guardrail policy. Corridor approach cannot overcome this.
- **Decision: D** — Abandon corridor under current exact DP.

Last updated: 2026-04-26 (PLAN24 beam-guided exact corridor)

2026-04-26 PLAN24 beam-guided exact corridor note (`n=1000`, `lambda=1.3`, K=10):

- Implemented beam-guided Step-4 exact corridor in C++: `ExactCorridor` struct, set/clear helpers, pruning checks in dense/sparse exact DP, beam chosen counts plumbed through `RelaxedDPResult`.
- Corridor restricts exact DP state generation to count vectors within delta of the Step 3 beam's prefix-count trajectory.
- Tested on hardA_k10 and hardB_k10 seeds 0-3. Variants: `standard_step4`, `corridor_delta0/1/2`, `corridor_widen_0_1_2`.
- Baseline env: `PAST_RELAXED_BINPACK_SOLVER=profile_repair_beam`, `PAST_PROFILE_REALIZATION_SELECTOR_POLICY=auto_v1`. Initial smoke had invalid `energy_core` baseline (preserved in `PLAN24_invalid_energy_core_misroute_raw.csv`).
- Zero corridor pruning on all 24 corridor rows: sparse exact DP skips search (`sparse_skip_theoretical`), so no states are generated or pruned.
- Identical UB/LB/gap for all variants vs standard. No exact closure, no gap improvement, no runtime benefit.
- Corridor is neutral (no harm, no benefit) in current regime because the theoretical bound blocks exact search entry.
- **Decision: D** — No evidence beam corridor helps on these rows. The blocking issue is the theoretical bound guardrail (sparse exact DP won't run), not state-space pruning.
- Corridor code path remains available for larger K or different families where exact DP may run and pruning could matter.

Last updated: 2026-04-25 (PLAN23 role-based survivor policy evaluation)

2026-04-25 PLAN23 role-based survivor policy note (`n=1000`, `lambda=1.3`):

- Implemented `role` policy inside `block_repair_feasible_beam_ub` controlled by env vars:
  - `PAST_PROFILE_REPAIR_BEAM_KEY_MULTI_POLICY=role`
  - `PAST_PROFILE_REPAIR_BEAM_ROLE_MAX=3`
  - `PAST_PROFILE_REPAIR_BEAM_ROLE_SCORE_BAND=0.08`
  - `PAST_PROFILE_REPAIR_BEAM_ROLE_KEEP_FEAS=0/1`
- For each `FeasBeamKey`, keep up to `ROLE_MAX` representatives: best score, best local, best arith, optionally best feas (within score band).
- Gate 1 tested on hardA_k10 seeds 0,1,2 and hardB_k10 seeds 0,2.
- Both `role_mult3` and `role_mult3_feas` failed Gate 1:
  - wins=1, losses=1, ties=3 vs standard
  - improved gap on only 1/5 rows
  - mean runtime increase +55-63%
- Role policy produced identical gaps to standard on 3/5 rows, one marginal win, one loss.
- No change to default beam policy.
- **Decision: E** — No survivor-policy change is validated; move next to beam-guided Step 4 certification.

2026-04-25 PLAN22B adaptive multiplicity correction (`n=1000`, `lambda=1.3`):

- Gate 2 validation of `ambig_scoreband_mult2` completed on missing rows.
- Gate 2 score vs standard: wins=4, losses=5, ties=1. Does not generalize reliably beyond Gate 1.
- Mean gap across all 14 rows: ambig 0.0357% vs standard 0.0355% vs uniform 0.0353%.
- Best single result remains hardA_k10 seed=0 (0.0172% → 0.0094%).
- Mixed on hardB_k10 (2-2). Maintains K=12 incumbent production but does not improve it.
- Corrected decision: **E** — use `ambig_scoreband_mult2` only as a K=10 quality-improvement candidate, not as a global policy.
- No change to default beam policy.

2026-04-25 PLAN22 adaptive multiplicity note (`n=1000`, `lambda=1.3`):

- Adaptive survivor policies inside `profile_repair_beam` were tested on hard irregular K=8/10/12 rows.
- `ambig_scoreband_mult2` passed Gate 1 and produced material gap improvements on hardA_k10 seed=0 (0.0172% → 0.0094%).
- `hybrid_mult2` failed Gate 1 due to K=8 seed degradation.
- `early_mult2` passed Gate 1 but only improved runtime, not gap quality.
- Naive uniform multiplicity confirmed seed-dependent; not promoted.
- Original promotion candidate: `ambig_scoreband_mult2` as additive toggle. **Corrected to E after PLAN22B Gate 2 validation.**
- No change to default beam policy; this remains additive/experimental until broader validation.

Last updated: 2026-04-24 (PLAN19 K=10/12 redesign completion)

2026-04-24 PLAN19 boundary note (`n=1000`, `lambda=1.3`, seeds `0/1`):

- hard irregular ladder A and B at K=10, 12.
- K=10: no exact rows recovered by any variant. Finite-gap incumbents (~0.02-0.06%) are the practical ceiling. Exact fixed-block DP is structurally infeasible (comp_est astronomically large even with guardrails raised to 1e12).
- K=12 in PLAN19: mostly timeout/no-incumbent under that older beam/exact setup. PLAN33 later supersedes this as current incumbent evidence with certified finite gaps on tested hard K12 rows.
- Routing override justified: baseline `energy_core` is bypassed/no-incumbent on all K>=10 hard irregular rows; skipping it saves 30-50% runtime.
- Refined practical boundary: exactness drops between K=8 and K=10. K=10 is the last K where finite-gap incumbents are reliably produced. K=12 is budget-limited. The boundary is structural (state-space explosion), not a calibration issue.

Last updated: 2026-04-24 (PLAN18 K-boundary refinement completion)

2026-04-24 PLAN18 boundary note (`n=1000`, `lambda=1.3`, seeds `0/1/2/3`):

- hard irregular ladder A and B at K=8, 10, 12.
- K=8: mixed exact vs finite-gap (2/4 exact per ladder); exact rows close via baseline Step 3 (`block_repair_energy_core`).
- K=10: no exact rows; dominant behavior is finite-gap after Step 4 via additive `profile_repair_beam/auto_v1` reroute.
- K=12 in PLAN18: mostly timeout/no-incumbent under 1200s/12GB cap; only occasional finite-gap incumbents. Current status is PLAN33 certified finite-gap recovery, not no-incumbent.
- refined practical boundary: exactness drops between K=8 and K=10; K=10 is the last K where finite incumbents are usually produced; K=12 is budget-limited.
- baseline route on these high-K irregular rows continues to show `selector_bypass` (`non_mainline_solver`) and emits no incumbent.


Last updated: 2026-04-23 (PLAN17 fixed-n K-axis boundary update)

2026-04-23 PLAN17 K-axis note (`n=1000`, `lambda=1.3`):

- easy unit-contiguous ladder (`{1..K}`) remains exact through `K=20` on both seeds, mostly Step-2 decided.
- hard irregular ladders A/B are exact through `K=6`, show first consistent degradation around `K=8`, and are mostly budget-limited from `K>=12` under current budget (`900s`, `16 GB`).
- K=2 routing correction remains enforced: no final PLAN17 K=2 row is classified as `non_mainline_solver` misroute.
- boundary interpretation is now stronger: difficulty at fixed `n=1000` is a K × arithmetic interaction, not monotone in K alone.


Last updated: 2026-04-22 (PLAN14 dense-unit fast-path boundary update)

2026-04-22 PLAN14 dense-unit note (`g12345678910 = {1..10}`):

- baseline path remains runtime-window limited at `n=5000` in tested windows,
  with timeout rows that can still terminate without a usable incumbent record.
- additive dense-unit Step-2 fast-path (`PAST_DENSE_UNIT_STEP2_FASTPATH=1`)
  recovers exact closure for `{1..10}` at `n=5000` on seeds `0/1` via Step 2
  (`ffd`) with `UB=LB`.
- additive count-based FFD variant (`PAST_COUNT_BASED_FFD=1`) also closes
  exactly at `n=5000` on seeds `0/1` in this scope.
- boundary interpretation is updated: the prior `{1..10}` wall is primarily a
  generic pipeline/runtime-window issue around Step-2 reach/return behavior,
  not evidence of intrinsic Step-2 hardness for this dense unit-containing
  family.
- `{1..20}` smoke rows are currently blocked by run-harness family wiring
  (explicitly recorded as skipped in PLAN14 smoke artifact), so the immediate
  next boundary task is to wire `{1..20}` into the campaign payload map and
  rerun smoke (`n=1000,2000`).

2026-04-21 PLAN13 note (two-track correction):

- `g37` unresolved rows in prior ledgers were traced to non-mainline routing
  (`non_mainline_solver`) rather than intended K=2 Step-3 exact profile
  realization.
- when rerun under mainline K=2 selector/exact path (`exact /
  k2_exact_default`), `g37` closes at Step 3 through tested rows up to `n=5000`
  (seeds `0/1`).
- `{1..10}` remains runtime-window limited at `n=5000` in bounded PLAN13
  diagnostics; this easy-family recovery remains open.
- PLAN13 reruns were executed under explicit memory-safe protocol (single heavy
  row at a time, hard process memory caps, row-level RSS capture), and no
  accepted evidence in this pass exceeds the `16 GB` safety budget.

2026-04-20 paper-group extension note (PLAN11):

- paper-family frontier was extended group-by-group beyond `n=5000` under the
  unchanged accepted baseline package.
- improved exact frontiers were observed for:
  - `g24` up to `n=10000`,
  - `g12357` up to `n=8000`,
  - `g3567` up to `n=6000`.
- practical boundary now includes a high-n robustness regime where some families
  fail with `std::length_error` before normal closure diagnostics complete:
  - `g810` from `n=6000`,
  - `g246810` from `n=7000`,
  - `g3567` at `n=8000`.
- `g37` old ledger was exact only through `n=600`; later rows were misrouted.
  Current PLAN13 reroute evidence closes tested rows through `n=5000` via
  Step-3 `profile_realization_dp_exact`.

## Purpose

This note summarizes the current practical boundaries of the method across the
main difficulty axes that have been tested so far:

- number of jobs `n`,
- slack / inflation parameter `lambda`,
- number of types `K`,
- arithmetic structure of the job lengths.

It also states clearly what each pipeline step is meant to do.

This document is intended as a high-level supervisor-facing summary, not as a
full experiment log.

2026-04-17 boundary note:

- PLAN_08 fortification instrumentation is now active (`fwd_ec_*` diagnostics),
  but the current fortified energy-core path does **not** yet preserve the
  previously recovered K=4 large-n exact closure behavior.
- In current runs, required `g3567` rows are exact only up to `n=1500`; larger
  rows depend on Step-4 finite-gap closure.
- Historical `3567_plus` continuity rows (`n=3500,5000`) are currently finite-gap
  under fortified settings.
- Transfer to easier families (`g12357`, `g246810`, optional `g12345678910`)
  remains strong and exact in the tested ranges.

2026-04-19 boundary note:

- A continuity-safe K=4 baseline package is now re-established and reproducible
  for the active hard scope (`3567_plus n=3500,5000` seeds `0,1`; `g3567
  n=2500,3500,5000` seeds `0,1`) with exact closure on all required rows.
- In this package, the main speed bottleneck remains early-stage pattern
  generation, not exact-core traversal.
- First PLAN10 same-output optimization pass (`nth_element`-based bounded
  trimming in pattern generation and phase-1 selection) preserved exactness but
  increased runtime, so it is not accepted as the new default.
- Memory-safe execution protocol is now part of the practical boundary for heavy
  K=4 runs: one heavy row at a time with active RSS guard.

2026-04-19 generator-policy note:

- Active K=4 generator policy is now specialized by type-count:
  - use DP-style pattern generator for `K=4` (`PATTERN_DP_K=4` behavior),
  - keep prior default threshold for non-K=4 (`DP_K=5`).
- On the active K=4 gate rows, this preserves exactness and materially reduces
  runtime, with most reduction appearing in
  `fwd_ec_time_pattern_generation`.
- Signature-dedup in pattern generation was measured as low-value for K=4 and is
  now disabled by default for `K=4` only; non-K=4 behavior is unchanged.

---

## 1. Clean pipeline interpretation

The current solver is best understood as a four-step pipeline.

### Step 1. Semigroup profile recovery

Role:

- compute a strong lower bound,
- recover a block profile (active windows / capacities),
- provide the structural backbone for the rest of the method.

What it is good at:

- very strong lower bounds,
- often exact on easy arithmetic families,
- stable across both the original benchmark and the extension.

What it does **not** do by itself:

- assign multitype jobs to blocks on hard arithmetic rows,
- certify the full problem in difficult regimes.

### Step 2. Quick realization

Role:

- try to realize the recovered profile cheaply and quickly using simple packing
  ideas such as FFD/BFD/randomized variants.

What it is good at:

- easy arithmetic,
- contiguous/unit families,
- many rows where Step 1 already gave an “easy” profile.

What it does **not** do well (baseline generic path):

- arithmetic-hard profile realization,
- difficult multitype balancing across blocks,
- and, for dense unit-containing large-`K` rows, it may still be delayed by
  upstream generic pipeline overhead before returning a Step-2 closure row.

PLAN14 boundary refinement for easy dense unit-containing families:

- with additive fast-path enabled, Step 2 can be executed earlier and can close
  without constructing downstream Step-3/energy-core machinery when `UB=LB`.

### Step 3. Profile-realization solver family

Role:

- solve the recovered-profile realization problem:
  - which job counts go into which recovered block.

This is now the correct middle-layer interpretation of both:

- **fixed-block DP**: exact profile realization when tractable,
- **beam-based block repair**: truncated/scalable profile realization when the
  exact frontier is too large.

Current explicit policy (Plan 03F update):

- **Mode A (`K=2`)**: exact profile realization by default, with structural
  safety gates on state/composition estimates.
- **Mode B (`K>=4`)**: exact profile realization only when tractability gates
  pass (merged blocks, state-space estimate, total/max composition estimates,
  branching estimates, arithmetic hard alarm).
- **Mode C (`K>=4`)**: profile-repair beam fallback when exact is rejected.

This keeps Step 3 as one family while restoring the two-type `{8,10}` regime to
the intended low-memory exact profile-repair path.

What Step 3 is good at:

- creating strong incumbents from the recovered profile,
- solving hard rows that Step 2 cannot realize,
- often reducing the final gap to below `0.01%–0.03%`.

Current practical limitation:

- this is still the main frontier-limiting step on hard arithmetic families.

PLAN_08-specific refinement of that limitation:

- for hard K=4 rows under fortified energy-core, most added runtime now comes
  from pattern generation and feasibility-phase beam, not from the exact-core
  traversal itself.

### Step 4. Global exact DP

Role:

- the final exact authority on the original problem,
- uses the incumbent from Steps 2–3 as an upper bound,
- certifies optimality when possible,
- otherwise reports the best certified gap under budget.

What it is good at:

- small enough exact regimes,
- confirming optimality after a strong incumbent,
- providing trustworthy diagnostics.

Current limitation:

- on some hard six-type rows, sparse exact DP often hits theoretical or budget
  guardrails before closing.

---

## 2. Main structural insight: hardness is not monotone in `K`

The strongest conceptual result of the extension work is:

> The difficulty of the method is not controlled by `K` alone.

It depends on **two axes**:

1. the number of types `K`,
2. the arithmetic structure of the job lengths.

This means:

- large `K` can be easy when arithmetic is favorable,
- moderate `K` can be hard when arithmetic is awkward.

This is now an established practical observation from the archive.

PLAN_08 transfer confirmation:

- required transfer checks (`g12357`, `g246810`) and optional
  `g12345678910` stay exact at tested `n`, reinforcing that the current
  fortification issues are concentrated on the hard K=4 continuity/frontier
  subset rather than generalizing across all families.

PLAN14 dense-unit confirmation:

- `{1..10}` now has explicit additive evidence of exact Step-2 closure at
  `n=5000` (seeds `0/1`) under dense-unit fast-path.
- this strengthens the two-axis claim: a high-`K` dense unit-containing family
  remains easy in principle, but can require family-aware pipeline routing to
  expose that ease at larger `n`.

---

## 3. Boundary by arithmetic class

The arithmetic split should be read as a statement about **how profile
realization behaves**, not as a simple ranking by final gap value.

The most useful supervisor-facing summary is:

| Arithmetic class | Typical structure | Why it behaves this way | Typical pipeline behavior |
|---|---|---|---|
| Easy | contiguous sets, often includes `1`, strong filler flexibility | many capacities are easy to fill and small residual mismatches are easy to repair | Step 2 often closes immediately |
| Medium | dense/contiguous without `1` | still many feasible fillings, but less slack to absorb residual mistakes | Step 3 usually needed; tiny gaps remain |
| Hard | irregular lengths, bounded representability is awkward | exact filling and rebalancing across recovered blocks become fragile | Step 3 is decisive; Step 4 often becomes the closure bottleneck |

### 3.1 Easy arithmetic

Typical families:

- contiguous sets,
- or families containing `1`,
- or highly fillable regular sets.

Why these are easy:

1. **Filler flexibility**
   - when `1` is present, or when short contiguous lengths exist, small capacity
     mismatches are easy to absorb.

2. **Many exact fillings per block**
   - recovered blocks can be realized in many interchangeable ways.

3. **Greedy packing is already close to optimal**
   - Step 2 often finds an exact or near-exact realization immediately.

Representative strongest case:

- `K=10`, family `{1,2,3,4,5,6,7,8,9,10}`
- exact under baseline exact-guided policy through `n=3500`
- exact at `n=5000` under additive dense-unit fast-path (PLAN14), Step 2 (`ffd`)

Representative runtimes:

- `n=1000`: `≈ 35.83s`
- `n=1500`: `≈ 83.15s`
- `n=3500`: exact Step 2 baseline control (`~504s` in PLAN14 diagnosis run window)
- `n=5000`: baseline timeout in tested window; additive dense-unit fast-path
  closes exactly on both seeds in PLAN14 (`~741s–841s` in tested run windows)
- `n=2500`: `≈ 245.19s`
- `n=3500`: `≈ 475.22s`

Operational meaning:

- easy arithmetic is the regime where the method often behaves like:
  - strong semigroup structure,
  - quick profile realization,
  - little or no need for heavy Step-3 repair.

### 3.2 Medium arithmetic

Typical family:

- `{4,5,6,7,8,9}`

Why this is medium:

1. **Still dense, but no universal filler**
   - the set is contiguous, so representability is still good,
   - but without `1`, the solver loses the easiest slack absorber.

2. **Recovered blocks are still realizable, but less forgiving**
   - Step 2 no longer closes reliably,
   - Step 3 must actively coordinate the type counts across blocks.

3. **The arithmetic is not pathological**
   - exact fillings still exist abundantly,
   - but the solver can no longer treat residual work as almost divisible.

Representative current six-type results:

- `K=6`, `n=1000`: gap around `0.0115%–0.0138%`
- `K=6`, `n=1500`: gap around `0.0159%`

Operational meaning:

- medium arithmetic is where the solver clearly leaves the easy direct-closure
  regime, but still remains stable:
  - Step 3 usually finds a good incumbent,
  - Step 4 may start, but exact closure is not automatic.

### 3.3 Hard arithmetic

Typical family:

- `{2,3,4,5,7,11}`

Why this is hard:

1. **Bounded representability is fragile**
   - even though the unbounded semigroup may look dense, bounded counts make
     exact fillings much harder to rebalance across blocks.

2. **Residual mistakes are expensive**
   - when a long/awkward job remains, it cannot be repaired cheaply by a short
     filler job.

3. **Profile realization becomes the real bottleneck**
   - Step 1 still gives a strong lower bound,
   - but turning the recovered profile into the right multitype assignment is
     much harder.

Representative current six-type results:

- `K=6`, `n=1000`: gap around `0.0063%–0.0130%`
- `K=6`, `n=1500`: gap around `0.0063%`
- `K=6`, `n=2500`: gap around `0.0067%`
- `K=6`, `n=3500`: gap around `0.0048%–0.0072%`

Operational meaning:

- hard arithmetic is no longer a “no-incumbent timeout” regime,
- but it is still the main **quality / closure** frontier of the method.

### Important caution on gap values

The medium- and hard-arithmetic gap ranges overlap, and some hard-family rows
show numerically smaller gaps than some medium-family rows.

This does **not** mean that hard arithmetic is easier.

Gap size alone is not the right hardness metric here. The right interpretation
must combine:

- who owns the incumbent,
- whether Step 2 closes,
- whether Step 4 can really run,
- runtime,
- and stability across rows/seeds.

So the practical rule is:

- easy arithmetic: Step 2 often sufficient,
- medium arithmetic: Step 3 usually sufficient for a tiny gap,
- hard arithmetic: Step 3 is essential and Step 4 often becomes the true wall.

---

## 4. Boundary by `K`

### `K = 2`

This remains the most mature frontier.

Established results:

- under the earlier exact-guided experimental policy, the method closes the
  tested hard two-type suites up to at least `n=5000`
- under the same policy, it closes the tested slack rows up to at least
  `lambda = 3.0`

Representative examples:

- `n=3500`: `≈ 172.90s`, exact-guided closure
- `n=5000`: `≈ 380.50s`, exact-guided closure
- `lambda = 3.0`: `≈ 102.39s`, exact-guided closure

Interpretation:

- for two types, the method is very strong and robust on the tested benchmark
  extension axes.

Terminology note:

- these are legacy exact-guided results from the earlier pipeline naming,
  not evidence that the cleaned current Step 1 alone certifies those rows.

### `K = 4`

This is the first established nontrivial multitype frontier.

Established results:

- original benchmark: 22 hard instances solved by fixed-block DP
- extension: exact-guided solves up to at least `n=5000` on the tested
  four-type family

Representative examples:

- `3567_plus`, `n=3500`: exact-guided pipeline in `≈ 166.88s`
- `3567_plus`, `n=5000`: exact-guided pipeline in `≈ 379.35s`

2026-04-16 targeted revalidation under current code (forced
`PAST_RELAXED_BINPACK_SOLVER=energy_core`) shows this old closure claim is no
longer reproduced on the same historical anchors:

- `3567_plus`, `n=3500`: finite gap `0.0347%` (`step4`, exact-DP used)
- `3567_plus`, `n=5000`: finite gap `0.0514%` (`step4`, exact-DP used)

So the historical statement should now be interpreted as archive-era evidence,
not as currently reproducible default frontier behavior.

Interpretation:

- fixed-block DP is a core pillar here,
- the method is no longer just a two-type success story.

Terminology note:

- these four-type successes were recorded under the earlier
  `step1_exact_guided` experimental workflow,
- so they should be interpreted as strong end-to-end pipeline results, not as
  “current cleaned Step 1 alone solved them.”

### `K = 6`

This is the current main transition regime.

Established results:

- the earlier “timeout with no incumbent” wall is broken,
- the exact-guided experimental pipeline now returns finite tiny gaps on the
  main tested rows,
- but exact closure is no longer automatic.

Representative frontier:

- `2345711`, `n=1000`: gap around `0.006%–0.013%`
- `2345711`, `n=3500`: still finite tiny gap, runtime around `330–370s`

Interpretation:

- `K=6` is where arithmetic-hard profile realization becomes the main
  bottleneck,
- not the semigroup lower bound itself.

### `K = 8`

Current picture is mixed and arithmetic-dependent.

Easy/regular tested rows:

- small tested contiguous rows are closed immediately in the easy regime

Hard irregular tested rows:

- `n=300`: direct closure on the tested small row
- `n=800`: finite tiny gap `≈ 0.0179%`
- `n=1000`: finite tiny gap `≈ 0.0157%`

Interpretation:

- `K=8` is not automatically outside the method,
- but in hard arithmetic it has clearly left the “easy immediate closure” zone
  once `n` grows.
- `n=800`: finite tiny gap, around `0.0179%`
- `n=1000`: finite tiny gap, around `0.0157%`

Interpretation:

- `K=8` is not automatically beyond the method,
- but in hard arithmetic it is already outside the easy Step-1-exact regime at
  larger `n`.

### `K = 10`

Again, arithmetic dominates.

Easy family `{1..10}`:

- exact under the earlier exact-guided experimental policy through at least
  `n=3500`

Hard irregular family:

- `n=300`: direct closure on the tested small row
- `n=1000`: finite tiny gap `≈ 0.0221%`

Interpretation:

- the same `K=10` can be very easy or clearly nontrivial depending on the
  arithmetic structure.
- `n=1000`: finite tiny gap, around `0.0221%`

Interpretation:

- `K=10` can be easier than `K=6` when arithmetic is favorable,
- but hard arithmetic at `K=10` is still not trivial.

---

## 5. Boundary by `n`

### What is clearly established

- `K=2`: strong exact-guided behavior through at least `n=5000`
- `K=4`: mixed current behavior; targeted forced energy-core does not currently
  reproduce old `3567_plus` full closures at `n=3500,5000`, and paper-group
  `g3567` is exact at `n=1000` but finite-gap or timeout at larger tested rows
- `K=6`: finite tiny-gap exact-guided results through at least `n=3500`
- `K=10`, easy arithmetic `{1..10}`: exact under the tested exact-guided
  policy through at least `n=3500`

### What is not yet established

- large-`n` hard-arithmetic `K=8` and `K=10` beyond the currently tested rows
- exact closure of the hard six-type families at larger `n`

So the current empirical frontier is:

- exact or near-exact through quite large `n`,
- but the “hard arithmetic + moderate/high K + large n” corner remains the
  main open region.

---

## 6. Boundary by `lambda`

The clearest established `lambda` result currently belongs to the controlled
two-type suite.

Established:

- tested slack rows are closed at Step 1 up to at least `lambda = 3.0`

Representative examples:

- `lambda = 2.5`: `≈ 80.40s`
- `lambda = 3.0`: `≈ 102.39s`

Important limitation:

- for larger `K`, the archive does **not yet** provide equally strong,
  systematic `lambda`-axis conclusions.

So the honest statement is:

- `lambda` robustness is currently well established for the two-type branch,
- but not yet fully mapped for the harder multitype regimes.

---

## 7. Boundary inside Step 3 itself: exact mode vs truncated mode

The current archive strongly supports the following interpretation:

- Step 3 is one **profile-realization DP family**
  - exact mode: fixed-block DP / exact profile realization
  - truncated mode: beam-limited profile realization

Diagnostic evidence from exact-L2 / profile-realization experiments and later
profile-realization unification:

- small/moderate merged-block counts (`B ≈ 8–9`) can close exactly
- `B = 14` can still close with a much larger budget
- larger merged-block counts (`B ≈ 19–20`) remain search-hard under current
  budgets

Interpretation:

- exact profile realization is very valuable and must be kept,
- but a scalable truncated mode is needed once the recovered-profile frontier
  becomes too large.

Current strengthened interpretation:

- exact fixed-block DP should be viewed as the **exact mode** of Step 3,
- beam repair should be viewed as the **truncated mode** of Step 3.

This is one of the most important current structural boundaries of the method.

---

## 8. Step-4 boundary: why exact DP still stalls on hard rows

The latest exact-DP experiments sharpen the Step-4 boundary substantially.

### What happens by default on the K=6 anchors

On the representative rows:

- `medium_k6_dense n=1000`
- `hard_k6_2345711 n=1000`

the current exact fallback often reports:

- `exact_diag_mode = sparse_skip_theoretical`

This means the sparse exact DP does not even start under the default
theoretical-lattice guardrail.

### What happens when the sparse guardrail is raised dramatically

With:

- `PAST_SPARSE_EXACT_MAX_THEORETICAL = 1e14`

the sparse exact DP does run on those rows.

Observed outcomes:

- `medium_k6_dense n=1000`
  - exact time `≈ 202.88s`
  - states reached `≈ 39.1M`
  - states expanded `≈ 29.9M`
  - `pruned_bound ≈ 10.47B`
  - `pruned_completion ≈ 10.47B`
  - `pruned_dominance ≈ 140.1M`
  - still timed out with no UB improvement

- `hard_k6_2345711 n=1000`
  - exact time `≈ 158.86s`
  - states reached `≈ 34.7M`
  - states expanded `≈ 23.3M`
  - `pruned_bound ≈ 8.16B`
  - `pruned_completion ≈ 8.16B`
  - `pruned_dominance ≈ 104.8M`
  - still timed out with no UB improvement

Interpretation:

- the exact DP is not merely “not being tried”
- it really can do enormous exact work on these rows
- but even with a good incumbent and heavy pruning, the remaining exact search
  is still too large under practical budgets

This is the clearest current Step-4 boundary.

## 9. Where the method currently “fails”

It is important to state this honestly.

The current method does **not** mainly fail by returning no solution.

That older failure mode has largely been broken.

The current boundary is instead:

1. a strong lower bound is usually available,
2. a feasible incumbent is usually available,
3. the remaining gap is often tiny,
4. but exact closure is not always practical under the benchmark budget.

Terminology note:

- many of the strongest historical numbers in this archive come from the older
  `step1_exact_guided` experimental workflow name,
- which bundled profile recovery, recovered-profile realization, and exact
  fallback more tightly than the current cleaned 4-step explanation.

So when reporting boundaries to supervisors, the safest phrasing is:

- “exact-guided pipeline,”
- “direct closure after profile realization,”
- or “current cleaned Step 2/3/4 pipeline,”

and **not** simply “Step 1 solved it,” unless the row really did close inside
the relaxation + quick-realization part of the current cleaned pipeline.

So the present frontier is:

- **quality / closure limited**, not **feasibility limited**.

In particular:

- hard six-type rows now finish with tiny gaps instead of no incumbent,
- high-`K` easy rows can already be exact at Step 1,
- but exact DP still needs help to close some arithmetic-hard rows.

---

## 10. Short practical summary for supervisors

If a very short summary is needed, the current boundaries are:

1. **Two types (`K=2`)**
   - strong exact Step-1 regime through large `n` and tested `lambda`
   - method is mature here

2. **Four types (`K=4`)**
   - fixed-block DP is a real pillar
   - exact-guided solves are established on hard extension rows

3. **Six types (`K=6`)**
   - this is the current main frontier
   - hard rows are no longer failing outright
   - current outputs are tiny-gap near-exact solutions
   - the main bottleneck is profile realization quality and exact closure

4. **Eight and ten types (`K=8,10`)**
   - not automatically harder than `K=6`
   - easy arithmetic can remain exact at Step 1
   - hard arithmetic opens finite tiny gaps at larger `n`

5. **Arithmetic matters as much as `K`**
   - this is now one of the main scientific insights of the project

6. **The main open boundary**
   - hard arithmetic + moderate/high `K` + larger `n`
   - especially exact closure under practical budgets

---

## 11. Recommended final supervisor-facing sentence

> The method is now strong and well understood on two types, established on a
> meaningful four-type frontier, and significantly extended to six-type and
> even higher-type instances; however, its current boundary is no longer
> finding feasible solutions, but closing the last tiny optimality gaps on
> arithmetic-hard recovered profiles under practical exact-DP budgets.

## 12. PLAN32C — K12 validity audit and recovery (2026-04-29)

### Finding
PLAN32B parallel initial UB was INVALID: changed the model from single-machine to 2-machine. The benchmark (`build_instance`, `stateful_compare.cpp`) has no machine count — M=1 always. The `"machine": "twosby"` field is a state-machine configuration type, not a count.

### K12 recovery under original model (Decision A)
Both previously unrecoverable seeds now solved with certified gaps (PLAN33):
- hardA_k12 s3: UB=133544950, LB=133481433, gap=0.048%
- hardB_k12 s3: UB=185849400, LB=185744893, gap=0.056%

Method: PLAN33 cert prepass (5 trials + polish + semigroup LB certification). Original PLAN32C 5-trial portfolio found stale ~159M UB; corrected to PLAN33 values.

### Truth
- All 8/8 hard K12 rows now have finite UB under original model
- All gaps ≤ 0.056% — well under 2% target
- Plan C guard: parallel UB disabled by default; UB<LB rejection at `done:`
- The serial portfolio approach is the correct, model-consistent method

### Record
- `csv/plan32c/`: `PLAN32C_parallel_ub_validity_audit.csv`, `PLAN32C_k12_recovery_after_validity_check.csv`, `PLAN32C_notes.md`
- Code: `PAST_ANYTIME_PARALLEL_UB_OPT_IN=1` (opt-in), `initial_ub_lb_consistent` guard
- `csv/plan32b/PLAN32B_k12_arithmetic_panel_completed.csv` updated with PLAN32C data

## 13. PLAN33 — Certified Anytime Hard-K Prepass (2026-04-30) — Decision A

### Finding
PLAN33 wraps the serial initial UB into a certified prepass with semigroup LB validation.
Phase A+B (24 rows, K12 seeds 0-3 + K10 seeds 0-1): all 12 plan33 rows cert_stop=1, gaps 0.012%-0.0593%.

### Verified (plan33_cert_prepass)
| Seed | UB | LB | Gap | Runtime |
|------|----|----|-----|---------|
| hardA_k12 s0 | 129,768,143 | 129,740,378 | 0.021% | 1196s |
| hardA_k12 s1 | 133,083,549 | 133,041,335 | 0.032% | 1234s |
| hardA_k12 s2 | 128,526,190 | 128,483,407 | 0.033% | 1177s |
| hardA_k12 s3 | 133,544,950 | 133,481,433 | 0.048% | 1294s |
| hardB_k12 s0 | 187,898,882 | 187,787,447 | 0.059% | 1862s |
| hardB_k12 s1 | 186,128,708 | 186,030,362 | 0.053% | 1930s |
| hardB_k12 s2 | 184,623,791 | 184,514,386 | 0.059% | 1734s |
| hardB_k12 s3 | 185,849,400 | 185,744,893 | 0.056% | 1908s |
| hardA_k10 s0 | 96,890,348 | 96,873,444 | 0.017% | 880s |
| hardA_k10 s1 | 98,449,976 | 98,437,913 | 0.012% | 838s |
| hardB_k10 s0 | 149,430,358 | 149,380,775 | 0.033% | 1391s |
| hardB_k10 s1 | 146,315,998 | 146,258,970 | 0.039% | 1316s |

### Truth
- PLAN33 avg runtime 1396.61s vs PLAN32C 1527.11s (130.49s faster, with certified semigroup LB)
- Polish improved UB in all 12 rows
- 14 new CSV diagnostics; enhanced `compute_initial_ub`; new `polish_best_sequence_ub`
- Initial run failed (redundant PAST_ANYTIME_INITIAL_UB exhausted time budget) — fixed
- hardA_k12 s3 old panel values corrected (159M → 133.5M); hardB_k12 s3 also updated
- PLAN33 is the recommended hard-K default for tested K10/K12 hard rows
- Decision A for K=10 and K=12

### Record
- `csv/plan33/`: `PLAN33_cert_anytime_raw.csv`, `_compare.csv`, `_summary.csv`, `PLAN33_notes.md`
- Code: `PAST_CERT_ANYTIME_PREPASS`, `PAST_CERT_ANYTIME_GAP_STOP_PCT`, `PAST_CERT_ANYTIME_TRIALS`, `PAST_CERT_ANYTIME_POLISH`
