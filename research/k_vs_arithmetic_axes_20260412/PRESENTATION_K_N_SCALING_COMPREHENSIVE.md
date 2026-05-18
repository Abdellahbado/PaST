# Comprehensive K- and n-Scaling Experiment Summary

Status date: 2026-05-03

This document is a presentation-facing summary of the scaling experiments in `research/k_vs_arithmetic_axes_20260412`. It consolidates the fixed-`n` K-axis experiments, the earlier paper-group `n`-extension experiments, and the method variants tested to improve the hard cases.

## Executive Summary

The current evidence supports a clear separation:

| Regime | Main observation | Best current method | Current boundary |
|---|---|---|---|
| Easy contiguous unit families `{1,...,K}` | Even large `K` remains easy because the greedy realization closes the lower bound | Step 2 FFD / dense-unit fastpath | Completed exact through `K=40` at `n=1000` |
| Small `K=2` non-unit families such as `{3,7}` and `{8,10}` | Easy when routed through the intended K=2 exact Step-3 path | Step 3 exact profile realization | exact through `n=5000` for `g37` and `g810` evidence |
| Moderate hard irregular families | Exact closure starts failing around `K=8` but some seeds remain exact | Step 3 profile repair beam + exact fallback | `K=8` mixed exact / finite gap |
| Hard irregular large `K` | Exact proof does not close, but PLAN33 gives certified finite gaps | Step 3 profile repair beam / PLAN33 certified anytime prepass | `K=10/12` hard rows have certified gaps ≤ `0.0593%` in PLAN33 |
| K=4 paper family `{3,5,6,7}` | Pattern generation was the bottleneck and was fixed | Step 3 energy core with K=4 DP-style generator | exact to `n=6000`; runtime greatly improved |

The important message for supervisors is:

> Scaling difficulty is not controlled by `K` alone. It is controlled by the interaction between `K` and arithmetic structure. Contiguous unit-containing families scale to much larger `K`; irregular families become hard around `K=8--10`.

## Current Data Quality Notes

| Item | Status |
|---|---|
| `PLAN16`, `PLAN17`, `PLAN18`, `PLAN19`, `PLAN27`, `PLAN28` | Completed enough to support presentation claims |
| `PLAN30_easy_k_scaling_raw.csv` | Headered raw artifact for the PLAN30 easy K-scaling runs |
| `K=40` easy-family run | Complete in PLAN30: all 4/4 rows exact |
| `PLAN29` multi-view block reconstruction | Complete: no coarsening view generalized; do not promote |
| HPC reproducibility map | `PAPER_HPC_REPRODUCIBILITY_MAP.md` gives responsible scripts, solver code, and env toggles |

## Method Glossary

### Step 1: Relaxed Profile / Lower-Bound Recovery

Step 1 builds a simplified schedule profile and a lower bound. It groups the time horizon into recovered blocks and estimates what amount of work should be done in each block.

Simple interpretation:

> Step 1 gives the solver a cheap optimistic picture of the schedule. It is useful for bounds and structure, but its blocks are not guaranteed to be directly packable with the real job sizes.

### Step 2: FFD / Direct Realization

Step 2 tries to turn the relaxed profile into a real schedule using a fast greedy packing rule such as FFD.

Simple interpretation:

> Step 2 is the fast path. If it finds a feasible schedule whose cost equals the lower bound, the instance is solved exactly.

This is why easy families `{1,...,K}` close very quickly relative to hard irregular families: they have many small job sizes and include size `1`, so the relaxed profile is easier to realize.

### Dense-Unit Fastpath

This is an additive shortcut for contiguous unit-containing families such as `{1,2,...,10}` or `{1,2,...,20}`.

Simple interpretation:

> If a family is dense and contains job size `1`, try the greedy Step-2 closure early and avoid expensive generic preparation.

This recovered `{1,...,10}` at `n=5000` and improves runtime for fixed-`n` large-`K` easy families.

### Step 3 Exact Profile Realization

This is an exact dynamic program over the recovered profile, mainly useful for small `K`, especially `K=2`.

Simple interpretation:

> The solver checks exactly whether the recovered block/profile structure can be realized with the available job sizes.

This is the method that closes `g37={3,7}` and `g810={8,10}` when routing is correct.

### Step 3 Energy Core

Energy core is a Step-3 repair method used mainly for the hard K=4 family `g3567={3,5,6,7}`.

Simple interpretation:

> Instead of considering every possible block pattern, the solver focuses on the patterns most relevant to the relaxed energy profile. It keeps a compact "core" of promising patterns, then repairs/completes around that core.

The K=4 acceleration came from generating those patterns with a DP-style generator instead of expensive generic generation.

### Step 3 Profile Repair Beam

The beam method is the main scalable method for hard irregular larger-`K` rows.

Simple interpretation:

> The solver scans the recovered blocks and keeps only a limited set of promising partial schedules. At each step, many candidates are generated, scored, and pruned; only the best survivors continue.

This gives strong incumbents for hard `K=10` rows, but not exact closure.

### Step-3 Survivor Policy

The survivor policy decides which beam states survive after pruning.

Simple interpretation:

> If the beam keeps the wrong partial states, the optimal or near-optimal completion may be lost. The survivor policy controls that risk.

The best validated global policy so far is `uniform_mult2`, which keeps two representatives per key and improves runtime/gap modestly on `K=10` hard rows.

### Step 4 Exact Fallback

Step 4 tries to prove optimality after Step 3 gives an incumbent.

Simple interpretation:

> Step 4 is the proof attempt. It can certify that no better schedule exists, but for hard large-`K` rows its state space becomes too large.

For hard irregular `K=10/12`, Step 4 often leaves a small finite gap or cannot enter meaningfully because the exact state representation is too large.

## K-Scaling Experiments

### PLAN16: Fixed `n=1000` on Current Paper Groups

Purpose: pivot from scaling `n` to scaling `K`, using the current paper groups at fixed `n=1000`.

Source: `csv/plan16/PLAN16_k_scaling_n1000_summary.csv`

| Family | K | Best status at `n=1000` | Deciding method | Mean runtime baseline |
|---|---:|---|---|---:|
| `g24={2,4}` | 2 | exact `2/2` | Step 2 FFD | 6.560s |
| `g37={3,7}` | 2 | exact `2/2` | Step 3 exact profile realization | 7.081s |
| `g810={8,10}` | 2 | exact `2/2` | Step 3 exact profile realization | 11.591s |
| `g3567={3,5,6,7}` | 4 | exact `2/2` | Step 3 energy/core profile path | 16.938s |
| `g12357={1,2,3,5,7}` | 5 | exact `2/2` | Step 2 FFD | 14.350s |
| `g246810={2,4,6,8,10}` | 5 | exact `2/2` | Step 2 FFD | 21.982s |
| `g12345678910={1,...,10}` | 10 | exact `2/2` | Step 2 FFD | 46.309s |
| `g1234567891011121314151617181920={1,...,20}` | 20 | exact `2/2` | Step 2 FFD | 227.582s |

Dense fastpath impact:

| Family | K | Baseline mean runtime | Dense fastpath mean runtime | Interpretation |
|---|---:|---:|---:|---|
| `{1,...,10}` | 10 | 46.309s | 32.069s | same exact result, faster |
| `{1,...,20}` | 20 | 227.582s | 199.066s | same exact result, faster |

Conclusion:

> At fixed `n=1000`, the original paper groups do not show a simple monotone `K` difficulty curve. `K=10` and `K=20` contiguous unit families are exact at Step 2, while smaller irregular families may require Step 3.

### PLAN17: Controlled Easy-vs-Hard K-Axis Study

Purpose: isolate arithmetic structure at fixed `n=1000`.

Source: `csv/plan17/PLAN17_k_axis_n1000_summary_by_k.csv`

Families:

| Class | Example | Meaning |
|---|---|---|
| easy unit | `{1,2,...,K}` | dense contiguous family containing `1` |
| hard irregular A | irregular ladder family | no simple unit-contiguous packing structure |
| hard irregular B | second irregular ladder family | harder irregular arithmetic pattern |

Headline table:

| K | Easy unit result | Hard irregular A result | Hard irregular B result |
|---:|---|---|---|
| 2 | exact `2/2` | not primary hard class | not primary hard class |
| 4 | exact `2/2` | exact `2/2` | exact `2/2` |
| 6 | exact `2/2` | exact `2/2` | exact `2/2` |
| 8 | exact `2/2` | mixed exact / unresolved or finite gap | mixed exact / finite gap |
| 10 | exact `2/2` | finite gap under profile reroute | finite gap or timeout/no incumbent |
| 12 | exact `2/2` | timeout/no incumbent in PLAN17 budget | timeout/no incumbent |
| 16 | exact `2/2` | timeout/no incumbent | timeout/no incumbent |
| 20 | exact `2/2` | timeout/no incumbent | timeout/no incumbent |

Conclusion:

> Easy arithmetic scales to at least `K=20` at `n=1000`. Hard irregular arithmetic starts degrading around `K=8`, and becomes non-exact by `K=10`.

### PLAN18: K-Boundary Refinement on Hard Irregular Families

Purpose: repeat the hard boundary with more seeds (`0,1,2,3`) and a stricter memory-safe budget.

Source: `csv/plan18/PLAN18_k_boundary_refine_summary_by_k.csv`

| Family class | K | Exact rows | Worst status | Mean runtime |
|---|---:|---|---|---:|
| hard irregular A | 8 | `2/4` | finite gap | 157.878s |
| hard irregular B | 8 | `2/4` | finite gap | 490.679s |
| hard irregular A | 10 | `0/4` | finite gap | 487.999s |
| hard irregular B | 10 | `0/4` | finite gap | 950.154s |
| hard irregular A | 12 | `0/4` | finite gap | 933.130s |
| hard irregular B | 12 | `0/4` | timeout/no incumbent | 1188.499s |

Conclusion:

> The exactness boundary for hard irregular arithmetic is around `K=8`. `K=10` is still useful because it usually gives finite-gap incumbents. `K=12` is often beyond the current practical budget.

### PLAN19: K=10/12 Redesign Attempts

Purpose: test whether hard irregular `K=10/12` could be closed by stronger exact or beam variants.

Source: `csv/plan19/PLAN19_k10_k12_method_notes.md`

Tested ideas:

| Idea | What it tried | Result |
|---|---|---|
| exact after beam | run bounded exact fixed-block DP after beam incumbent | did not close; exact mode hit complexity guardrails |
| force exact | force exact profile mode with high guardrails | hit `skipped_comp_est`; exact state space too large |
| routing override | skip unhelpful baseline energy-core for hard `K>=10` | useful; saves runtime without quality loss |
| stronger beam (`beam_plus`) | increase beam strength for K=12 | more timeouts; no gap improvement |

Conclusion:

> Fixed-block exact DP is not the right closure mechanism for hard irregular `K=10/12` under current budgets. The useful path is direct profile-repair beam, not energy-core or forced exact mode.

### PLAN20: Beam Diagnostics

Purpose: inspect why profile repair beam works but does not close all hard rows.

Source: `csv/plan20/PLAN20_phaseA_beam_diagnostics.md`

Key observations:

| Observation | Interpretation |
|---|---|
| beam time dominates many hard rows | Step 3 itself is expensive near the boundary |
| many states are pruned | survivor choice matters |
| discrepancy budget is small | beam diversity may be too limited |
| gaps are small | beam gives good incumbents, but proof remains hard |

Conclusion:

> The critical Step-3 question is not only beam width. It is which states survive pruning.

### PLAN22 and PLAN22B: Adaptive Multiplicity / Ambiguity Scoreband

Purpose: improve beam survivor selection.

Sources:

- `csv/plan22/PLAN22_adaptive_node_eval_notes.md`
- `csv/plan22b/PLAN22B_ambig_scoreband_validation_notes.md`

Tested idea:

> Keep extra representatives for ambiguous states, especially when their scores are close but their arithmetic/local structure differs.

Result:

| Variant | Result |
|---|---|
| `ambig_scoreband_mult2` | strong on one anchor row, but does not generalize reliably |
| `early_mult2` | safer runtime effect, little gap improvement |
| naive uniform multiplicity in early PLAN22 | mixed, required later validation |

Corrected conclusion after PLAN22B:

> `ambig_scoreband_mult2` should not be promoted globally. It is a targeted K=10 option, but seed-dependent.

### PLAN23: Role-Based Beam Survivors

Purpose: keep different "roles" in the beam, such as best score, best local quality, best arithmetic quality.

Source: `csv/plan23/PLAN23_role_based_beam_notes.md`

Result:

| Variant | Outcome |
|---|---|
| `role_mult3` | failed Gate 1; runtime increased |
| `role_mult3_feas` | failed Gate 1; no validated improvement |

Conclusion:

> Role-based survivor selection was more expensive and less reliable than simpler multiplicity policies.

### PLAN24 and PLAN24B: Beam-Guided Exact Corridor

Purpose: use the Step-3 beam trajectory to restrict Step-4 exact DP to a narrow promising corridor.

Sources:

- `csv/plan24/PLAN24_beam_corridor_exact_notes.md`
- `csv/plan24b/PLAN24B_forced_corridor_notes.md`

Result:

| Attempt | What happened |
|---|---|
| normal corridor | exact DP skipped search, so corridor pruned nothing |
| forced-entry corridor | sparse exact representation overflowed for K=10 |

Conclusion:

> Beam-guided Step-4 corridor is blocked under the current sparse exact encoding. The issue is representation scale, not only pruning.

### PLAN25 and PLAN26: Local Corridor DP

Purpose: avoid global mixed-radix overflow by encoding only local offsets around the beam path.

Sources:

- `csv/plan25/PLAN25_local_corridor_dp_notes.md`
- `csv/plan26/PLAN26_multi_idea_notes.md`

Result:

| Finding | Interpretation |
|---|---|
| local offset encoding avoids global overflow | implementation direction was mechanically plausible |
| base beam path does not survive local block checks | recovered blocks do not align with independently schedulable block boundaries |
| corridor gives no valid improvement | local block feasibility is mismatched with the beam's global validation |

Conclusion:

> Local corridor DP is not valid as currently designed. It assumes each recovered block is locally schedulable, but the beam only guarantees global sequence feasibility.

### PLAN27: Step-3 Adaptive Survivor Policy

Purpose: return to Step 3 and compare survivor policies directly on hard `K=10`.

Source: `csv/plan27/PLAN27_step3_adaptive_survivor_notes.md`

Rows: hardA_k10 and hardB_k10, seeds `0,1,2,3`, `n=1000`.

| Variant | Rows | Mean gap | Mean runtime | Wins/losses/ties vs standard | Decision |
|---|---:|---:|---:|---|---|
| standard beam | 8 | 0.0345% | 624.9s | baseline | reference |
| `uniform_mult2` | 8 | 0.0343% | 535.4s | 4/2/2 | passes promotion |
| `ambig_scoreband_mult2` | 8 | 0.0326% | 569.8s | 5/3/0 | fails not-worse gate |
| `late_ambig` | 8 | 0.0327% | 683.1s | 5/3/0 | fails gate |
| `residual_aware` | 8 | 0.0345% | 695.6s | 0/0/8 | no gap effect |
| `late_residual_ambig` | 8 | 0.0326% | 665.4s | 5/3/0 | fails gate |

Conclusion:

> `uniform_mult2` is the best validated global Step-3 survivor policy so far. It gives a modest gap improvement and a material runtime reduction. Family-aware survivor selection may be better, but is not yet fully validated.

### PLAN30 Easy K Scaling: K=24, K=30, and K=40 (implements PLAN_16)

Purpose: test supervisor-requested larger `K` on easy contiguous unit families.

Source: `csv/plan30/PLAN30_easy_k_scaling_raw.csv`

Completed exact rows:

| Family | K | Seed | Variant | Status | Runtime | Peak RSS |
|---|---:|---:|---|---|---:|---:|
| `{1,...,24}` | 24 | 0 | baseline | exact, FFD | 396.3156s | 1.893GB |
| `{1,...,24}` | 24 | 0 | dense fastpath | exact, FFD | 403.2663s | 4.318GB |
| `{1,...,24}` | 24 | 1 | baseline | exact, FFD | 336.4027s | 4.690GB |
| `{1,...,24}` | 24 | 1 | dense fastpath | exact, FFD | 321.0931s | 3.716GB |
| `{1,...,30}` | 30 | 0 | baseline | exact, FFD | 738.0361s | 4.361GB |
| `{1,...,30}` | 30 | 0 | dense fastpath | exact, FFD | 730.0118s | 3.829GB |
| `{1,...,30}` | 30 | 1 | baseline | exact, FFD | 639.2387s | 3.664GB |
| `{1,...,30}` | 30 | 1 | dense fastpath | exact, FFD | 625.6885s | 4.833GB |
| `{1,...,40}` | 40 | 0 | baseline | exact, FFD | 1719.8555s | 2.625GB |
| `{1,...,40}` | 40 | 0 | dense fastpath | exact, FFD | 1678.7807s | 3.095GB |
| `{1,...,40}` | 40 | 1 | baseline | exact, FFD | 1410.8058s | 3.072GB |
| `{1,...,40}` | 40 | 1 | dense fastpath | exact, FFD | 1398.3403s | 1.716GB |

Status of `K=40`:

> `K=40` is complete in PLAN30: all 4/4 rows are exact and close through Step 2 (`ffd`).

Conclusion:

> Easy contiguous unit families scale far beyond the hard irregular boundary. The current completed evidence supports exact closure through `K=40`.

### PLAN28 Block-Realizability Diagnostics

Purpose: test whether local block realizability explains easy-vs-hard behavior.

Source: `csv/plan28/PLAN28_block_realizability_notes.md`

Result:

| Diagnostic | Finding |
|---|---|
| base path survives local block checks | `0` for all diagnosed rows |
| bad-block rate | overlaps between easy and hard |
| finite pattern count | mostly depends on K, not family hardness |

Conclusion:

> Local block realizability diagnostics did not separate easy from hard cases. They are too strict because recovered blocks are relaxed blocks, not necessarily independently schedulable blocks.

### PLAN29 Multi-View Adjacent Block Reconstruction

Purpose: test whether Step 3 improves when given alternative adjacent coarsenings of recovered blocks.

Source: `csv/plan29/PLAN29_multiview_block_reconstruction_raw.csv`

Status:

> In progress. Partial rows exist, but no final summary or decision has been produced.

Partial pattern visible so far:

| Observation | Interpretation |
|---|---|
| coarse views often reduce runtime | fewer blocks make the beam/exact fallback cheaper |
| coarse views often worsen gap | merging blocks can destroy useful relaxed structure |
| target/price-preserving views sometimes match baseline | selective reconstruction may be safer than simple coarsening |

Do not present PLAN29 as a validated method yet.

## n-Scaling Experiments

### Paper-Group n-Frontiers

Sources:

- `PAPER_RESULTS_READY.md`
- `PAPER_GROUPS_EXTENSION_SUMMARY.md`
- `csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv`
- `csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv`

| Family | K | Last exact n | First observed issue | Main method |
|---|---:|---:|---|---|
| `g24={2,4}` | 2 | 10000 | none through 10000 | Step 2 FFD |
| `g12357={1,2,3,5,7}` | 5 | 8000 | timeout at 10000 | Step 2 FFD |
| `g3567={3,5,6,7}` | 4 | 6000 | timeout at 7000; length error at 8000 | Step 3 energy core |
| `g246810={2,4,6,8,10}` | 5 | 6000 | crash from 7000 | Step 2 FFD |
| `g810={8,10}` | 2 | 5000 | crash from 6000 | Step 3 exact profile realization |
| `g37={3,7}` | 2 | 5000 under PLAN13 reroute | old unresolved rows were misrouted | Step 3 exact profile realization |
| `g12345678910={1,...,10}` | 10 | 5000 with PLAN14 fastpath | baseline timeout at 5000 | Step 2 dense-unit fastpath |

Main correction:

> The old `g37` ledger understated the boundary because high-`n` rows were routed through the wrong non-mainline path. PLAN13 rerouted `g37` through the intended K=2 Step-3 exact profile-realization path and closed tested rows through `n=5000`.

### PLAN10 K=4 Generator Acceleration

Purpose: accelerate the `g3567={3,5,6,7}` K=4 hard family.

Artifacts:

- `csv/plan10/PLAN10_k4_generator_dp4.csv`
- `csv/plan10/PLAN10_k4_generator_compare.csv`
- `csv/plan10/PLAN10_k4_speedup_ablation.csv`

Result:

| Metric | Before | After K=4 DP generator | Improvement |
|---|---:|---:|---:|
| hard `g3567` mean runtime | 1083.240s | 250.839s | -76.8% |
| continuity mean runtime | 415.663s | 294.919s | -29.0% |
| all required rows mean runtime | 816.209s | 268.471s | -67.1% |
| pattern generation time | very large | near negligible | about -99% |

Method explanation:

> The old pattern generator spent most of the runtime constructing candidate block patterns. The new K=4 DP-style generator creates the useful candidate patterns directly for `K=4`, avoiding expensive generic enumeration.

Conclusion:

> This is one of the strongest validated method improvements. It preserved exactness and made the K=4 hard family much faster.

### PLAN13 g37 Reroute

Purpose: verify whether `g37={3,7}` was truly hard or simply misrouted.

Artifact:

- `csv/plan13/PLAN13_g37_k2_reroute.csv`

Result:

| n values tested | Seeds | Status | Method |
|---|---|---|---|
| 750, 1000, 1500, 2500, 3500, 5000 | 0, 1 | exact | K=2 Step-3 exact profile realization |

Conclusion:

> `g37` was not fundamentally open up to `n=5000`. The intended K=2 exact Step-3 route solves it. The old failures were routing failures, not method failures.

### PLAN14 Dense-Unit Recovery for `{1,...,10}`

Purpose: recover `{1,...,10}` at `n=5000`, where generic baseline timed out.

Artifact:

- `csv/plan14/PLAN14_g12345678910_fastpath_compare.csv`

Result:

| Family | n | Variant | Status | Runtime |
|---|---:|---|---|---:|
| `{1,...,10}` | 3500 | baseline | exact Step 2 | 565.3569s |
| `{1,...,10}` | 3500 | dense fastpath | exact Step 2 | 384.4468s |
| `{1,...,10}` | 5000 | baseline | external timeout | 1200s window |
| `{1,...,10}` | 5000 | dense fastpath, seed 0 | exact Step 2 | 840.5427s |
| `{1,...,10}` | 5000 | dense fastpath, seed 1 | exact Step 2 | 741.2518s |

Conclusion:

> `{1,...,10}` was not intrinsically hard. The generic pipeline spent time in unnecessary preparation. The dense-unit fastpath lets Step 2 close the row directly.

## What We Learned About Arithmetic Hardness

### Easy Arithmetic

Easy arithmetic means the job sizes give many ways to fill small residual gaps. The strongest example is `{1,2,...,K}`.

Why it is easy:

| Feature | Effect |
|---|---|
| contains job size `1` | almost any leftover capacity can be filled |
| contiguous sizes | many interchangeable combinations |
| dense set | the relaxed profile is easier to realize greedily |

Observed behavior:

> Easy unit-contiguous families close at Step 2 even for large K. Completed PLAN30 evidence reaches `K=40`.

### Hard Irregular Arithmetic

Hard arithmetic means the job sizes leave fewer exact combinations for filling recovered blocks.

Why it is hard:

| Feature | Effect |
|---|---|
| no unit job | residual gaps are harder to fill |
| irregular size spacing | fewer combinations match the relaxed block structure |
| large K with irregular sizes | beam search must choose among many competing partial assignments |

Observed behavior:

> Hard irregular families are exact through `K=6`, mixed at `K=8`, finite-gap at `K=10`, and no longer lack incumbents at `K=12` after PLAN33. The remaining K12 limitation is exact proof, not feasible-solution recovery.

## Recommended Presentation Story

Use three layers:

| Slide theme | Claim | Evidence |
|---|---|---|
| n-scaling on paper groups | existing pipeline scales several paper groups far beyond original sizes | `g24` to 10000, `g12357` to 8000, `g3567` to 6000, `g37/g810` to 5000 |
| K-scaling at fixed n | K alone is not the difficulty axis | easy `{1,...,K}` exact through K40 completed, while hard irregular exact closure fails around K8-K10 |
| method evolution | failures were used to improve routing and Step 3 | K=4 DP generator, K=2 reroute, dense-unit fastpath, uniform beam survivor |

Short supervisor-facing statement:

> We found that scaling by number of job sizes is meaningful only when arithmetic structure is controlled. For dense unit-containing families, the method scales to much larger K because Step 2 realizes the relaxation directly. For irregular families, Step 3 or the certified anytime prepass becomes essential. Exact closure becomes difficult around K=8--10 at n=1000, but PLAN33 still gives certified gaps below 0.06% on tested hard K10/K12 rows.

## Current Method Status

| Component | Status | Keep? |
|---|---|---|
| Step 2 FFD/direct realization | strong for easy families | yes |
| dense-unit fastpath | validated for `{1,...,10}` and useful for large easy K | yes |
| K=2 exact profile realization | validated for `g37`, `g810` | yes |
| K=4 energy-core DP generator | validated and highly beneficial | yes |
| profile repair beam | main scalable hard-K incumbent method | yes |
| `uniform_mult2` beam survivor | best validated global Step-3 survivor improvement | yes, candidate default for hard K |
| `ambig_scoreband_mult2` | seed/family-dependent | keep as targeted option, not global |
| role-based survivor | failed | no promotion |
| beam-guided exact corridor | blocked by exact-DP representation | no promotion |
| local corridor DP | invalid under current block/path assumptions | no promotion |
| block-realizability local diagnostic | not discriminative | no promotion |
| multi-view block reconstruction | failed to generalize | no promotion |

## Current Open Questions

| Question | Current answer |
|---|---|
| Have we scaled easy K beyond 20? | Yes. Completed exact rows through K=40 in PLAN30 |
| Is hard K=10 solved exactly? | No. It gives small finite gaps, usually with good incumbents |
| Is hard K=12 solved exactly? | No. But PLAN33 gives valid finite incumbents and certified gaps ≤ 0.0593% on tested hard K10/K12 rows |
| Should we present fixed `n=1000` as realistic scheduling horizon? | No. Present it as a controlled synthetic stress test for K/arithmetic hardness |
| What is the realistic study still needed? | A fixed-horizon study where K varies but the calendar horizon stays realistic |
| What is the most promising remaining work? | HPC rerun/cleanup and final table generation; additional method redesign is secondary |

## Evidence Files

Primary summaries:

| File | Role |
|---|---|
| `PAPER_RESULTS_READY.md` | current paper-facing result snapshot |
| `PAPER_GROUPS_EXTENSION_SUMMARY.md` | n-scaling paper-group frontier summary |
| `METHOD_PROVENANCE.md` | method-to-code mapping |
| `CURRENT_RESULTS_INDEX.md` | current artifact index |
| `PAPER_HPC_REPRODUCIBILITY_MAP.md` | code/provenance map for HPC reruns |

K-scaling artifacts:

| Plan | Main files |
|---|---|
| PLAN16 | `csv/plan16/PLAN16_k_scaling_n1000_summary.csv` |
| PLAN17 | `csv/plan17/PLAN17_k_axis_n1000_summary_by_k.csv` |
| PLAN18 | `csv/plan18/PLAN18_k_boundary_refine_summary_by_k.csv` |
| PLAN19 | `csv/plan19/PLAN19_k10_k12_method_notes.md` |
| PLAN20 | `csv/plan20/PLAN20_phaseA_beam_diagnostics.md` |
| PLAN22/22B | `csv/plan22*/PLAN22*_notes.md` |
| PLAN23 | `csv/plan23/PLAN23_role_based_beam_notes.md` |
| PLAN24/24B | `csv/plan24*/PLAN24*_notes.md` |
| PLAN25/26 | `csv/plan25/PLAN25_local_corridor_dp_notes.md`, `csv/plan26/PLAN26_multi_idea_notes.md` |
| PLAN27 | `csv/plan27/PLAN27_step3_adaptive_survivor_notes.md` |
| PLAN28 | `csv/plan28/PLAN28_block_realizability_notes.md` |
| PLAN30 | `csv/plan30/PLAN30_easy_k_scaling_raw.csv`, `csv/plan30/PLAN30_easy_vs_hard_notes.md` |
| PLAN29 | `csv/plan29/PLAN29_multiview_block_reconstruction_raw.csv` |

n-scaling artifacts:

| Plan | Main files |
|---|---|
| PLAN05 | `csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv` |
| PLAN10 | `csv/plan10/PLAN10_k4_generator_compare.csv` |
| PLAN11 | `csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv` |
| PLAN13 | `csv/plan13/PLAN13_g37_k2_reroute.csv` |
| PLAN14 | `csv/plan14/PLAN14_g12345678910_fastpath_compare.csv` |
