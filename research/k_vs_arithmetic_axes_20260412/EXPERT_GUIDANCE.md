# Expert Guidance

This file collects expert-reviewed assessments, critiques, and concrete next-step
direction for the coder. It is maintained by the expert reviewer and the research
lead. The coder should consult this file before starting new work.

Last updated: 2026-04-15

Supersession note (Plan 03/04):

- The final method policy is now a 4-step pipeline with unified Step 3
  (`profile_repair_beam`) and exact DP as the only exact fallback.
- Earlier guidance in this file that treated Lagrangian recovery or exact-L2 as
  default-mainline directions should be read as historical context, not current
  default policy.

---

## Overall assessment of current state

### 2026-04-15 policy checkpoint (Plan 03F)

The Step-3 policy now matches the intended unified formulation with explicit
mode selection:

- Mode A (`K=2`): exact profile realization by default (structural safety
  gates retained)
- Mode B (`K>=4`): exact profile realization only when tractable
- Mode C (`K>=4`): beam fallback

Validated outcome of this checkpoint:

- `{8,10}` `n=500..5000` restored to Step-3 exact profile realization with
  `UB=LB` and no Step-4 exact rescue,
- K=4/K=6 probes still use beam when structural gates reject exact.

Guidance implication:

- keep this policy stable as the default narrative,
- focus next improvement effort on K>=4 Step-3 incumbent quality rather than
  reopening branch sprawl.

### What the coder got right

1. **The two-axis separation is the key insight.** Difficulty is not
   monotone in K. The arithmetic structure of job lengths is an independent
   and often dominant axis. This reframes the paper from "scaling K" to
   "understanding and handling arithmetic hardness." The coder correctly
   identified this and built the archive around it.

2. **The three-level decomposition is clean.**
   - Level 1: block profile (semigroup DP / SPACES)
   - Level 2: block assignment (which jobs go in which block)
   - Level 3: within-block scheduling (optimal ordering per block)

   This is the right architecture. It matches the problem structure and
   connects to established OR literature (MMKP, cutting stock, bin packing).

3. **Phase sequencing is disciplined.** Experiments before code, framework
   before implementation, evidence-based decision points. The plan's
   fallback ladder is well-ordered.

4. **The Level 3 separation was a genuine win.** Gap improvements of 3–5×
   on all hard-arithmetic rows (e.g., 0.0356% → 0.0082% on 2345711 n=1000).
   This confirms that within-block evaluation quality was a real source of
   UB inflation.

5. **The literature review is well-targeted.** Rosales & García-Sánchez for
   semigroup theory, Chvátal/Pisinger for instance-dependent hardness,
   Gilmore-Gomory/Vanderbeck for column generation, Valério de
   Carvalho/Brandão for arc-flow. All relevant and well-positioned.

### What still needs attention

1. **Raise Step-3 quality inside the unified family.** Keep default policy as
   one hard-case method (`profile_repair_beam`) and improve neighborhoods,
   ordering, and incumbent retention without reintroducing branch sprawl.

2. **Keep exact DP as sole exact fallback and improve practicality.** Focus on
   admissible bounds, safe dominance, and expansion order in sparse/dense exact
   DP rather than adding a second exact method.

3. **Use exact-L2 only as archival diagnosis.** Keep it gated and non-default;
   do not let it alter default incumbent policy unless explicitly requested.

---

## Concrete next steps (in priority order)

### Step 1. Improve Step 3 inside one family (`profile_repair_beam`)

**Priority: IMMEDIATE.**

Work only within the current Step-3 family. Improve incumbent quality via
bounded, explainable operators:

- better neighborhood ordering,
- stronger but bounded 2-block destroy/repair,
- optional bounded 3-block neighborhoods only if runtime-capped and justified,
- better incumbent reuse and pruning.

Do not add co-equal default branches.

### Step 2. Strengthen exact DP as the only exact fallback

**Priority: HIGH.**

Keep exact fallback as DP-only (sparse + dense fallback). Improve through:

- stronger admissible lower bounds,
- safe dominance,
- better expansion order,
- natural state/node merging where already compatible.

Do not add a second exact family in the mainline story.

### Step 3. Run focused policy validation after each change

**Priority: HIGH.**

Always validate at least one row in each role:

- easy row that should close in Step 2,
- medium/hard row where Step 3 is active,
- hard row where Step 4 exact fallback is used,
- one previously exact-L2-touched row confirming exact-L2 stays non-mainline.

Report per-row: deciding step, UB/LB/gap, runtime, exact-DP-used flag,
and archival-branch-disabled flags.

### Step 4. Keep archival branches explicit and non-default

**Priority: MEDIUM.**

Legacy methods may remain for diagnosis/comparison, but must be opt-in only:

- `lagrangian_assign`, `rg_beam`, `feasible_counts`, exact-L2 apply mode,
- fixed-profile `block_dp_exact` diagnostics.

No hidden fallback should alter default policy.

### Step 5. Keep archive handoff documents synchronized with policy

**Priority: MEDIUM.**

When default policy changes, update `LOG.md`, `RESULTS.md`, `BLOCKERS.md`, and
this file in the same cycle so future coders see one consistent action list.

---

## Assessments of specific ideas tried

### Lagrangian + beam combination

**Verdict: regression, do not keep as default.**

The combined branch was slower on every row and quality-equivalent to
beam-only on most rows. The beam polish did not overwrite better
incumbents (note_pack_candidate protects against this), but the
Lagrangian state itself had regressed during the integration work. The
combination is a valid concept but requires the Lagrangian to be at its
best state first, and the beam to be seeded with the Lagrangian's
assignment — neither condition was met.

### Adjacent-pair local improver

**Verdict: too weak, not worth keeping.**

The pattern-swap neighborhood (swap patterns between adjacent blocks
while preserving type totals) found zero improvements on all tested rows.
This is because the filtered pattern set has too few valid swap pairs. A
job-level (not pattern-level) local search would be more fine-grained
but hasn't been tried.

### Pattern widening (per_work_keep=12, global_keep=96)

**Verdict: helps small rows, hurts medium rows. NOT a default change.**

Widening helped 2345711 n=1000 (0.0129% → 0.0072%) but destabilized
the Lagrangian on n=2500 (fell back to beam, gap worsened). Blanket
pattern widening is not the answer; dynamic pricing is the principled
alternative.

### R_feas / later LB stages

**Verdict: no help on the hard-arithmetic rows.**

On 2345711 n=1000: lb_after_fwd = lb_after_feas = lb_after_fl =
48,637,514. The relaxation hierarchy is already tight. The remaining gap
is entirely on the UB side.

### Level 3 separation (exact per-block multiset DP)

**Verdict: strong win, keep permanently.**

Gap improvements of 3–5× across all hard-arithmetic rows. The per-block
exact DP with a 50K cell threshold and 50ms time limit is well-
calibrated. The fallback to ascending/descending solve_fixed_sequence
on large blocks is appropriate.

---

## Potential gaps in the framework (for future investigation)

These are secondary concerns, not immediate priorities. They should be
explored AFTER the main plan's Phases 1–5 are complete.

### 1. Block profile quality (Level 1)

The three-level decomposition assumes Level 1 produces the right block
profile. But the semigroup relaxation's optimal profile is for the
RELAXED problem. The exact optimal might use slightly different block
boundaries.

**Diagnostic test:** perturb the block profile slightly (merge two
adjacent small blocks, or split one large block) and see if the UB
improves. If it does, the gap is partly a Level 1 issue.

### 2. Price signal interaction with Level 2

The two axes (K, arithmetic) characterize combinatorial difficulty. But
the TOU price signal could also affect Level 2: it might be better to
assign more jobs to blocks in cheap price periods. The current Level 2
treats all blocks equally (just capacity constraints). A price-aware
Level 2 would weight blocks by average price.

However, the currently observed 0.007% gaps suggest this is a minor
effect. Test only if the main plan doesn't close the gaps.

### 3. Sequential vs. joint optimization

Levels 1 → 2 → 3 are solved sequentially. A fully joint model (e.g.,
arc-flow) would solve Levels 2+3 simultaneously. This is already on the
fallback ladder but worth noting as a theoretical ideal.

---

## Novel ideas from high-multiplicity / n-fold literature review

A second literature search identified papers from the high-multiplicity
scheduling and n-fold IP communities. Some are relevant. Many require
careful scoping because our problem is NOT standard bin packing — it is
assignment of bounded-count job types to FIXED blocks with per-block
scheduling costs.

### Important distinction for the coder

Our Level 2 problem is:

> Given B FIXED blocks with known capacities, assign K types of jobs
> (with bounded counts) to blocks such that each block's capacity is
> exactly met, global type totals are satisfied, and total scheduling
> cost (evaluated per-block by Level 3) is minimized.

This is NOT bin packing (which minimizes the NUMBER of bins). The blocks
already exist and have fixed capacities from the relaxation. Our problem
is closer to the **Multiple Subset Sum Problem** or a cost-weighted
**multitype assignment** problem. Do not overclaim connections to bin
packing polynomial-time results.

### Idea A: Configuration LP + residual core (from Knop-Koutecký 2020)

**Source:** Knop, Koutecký. "Scheduling Kernels via Configuration LP."
arXiv:2003.02187.

**The idea:** After LP preassignment of bulk job counts to blocks, only
a small "residual core" of O(K²) jobs remains to be reassigned. Solve
only the residual exactly; keep the rest of the LP solution.

**Relevance to us:** Our pipeline IS a version of this. The semigroup
relaxation + FFD packing is the LP preassignment. The Lagrangian/beam
repair is the residual solver. Making this connection explicit has two
benefits:

1. It gives the paper a principled theoretical justification —
   our pipeline implements a known algorithmic paradigm, not an ad-hoc
   heuristic.
2. It suggests a concrete diagnostic: **compute the residual core size**
   (how many blocks have nonzero deviation from the FFD assignment) and
   report it as an explanatory variable. If the core is small, the
   problem is easy. If large, it's hard.

**Action for the coder:** After FFD packing, count the number of blocks
where type counts deviate from the target proportions. Report this as
`residual_core_blocks` in the baseline grid. This takes ~5 lines of
code.

### Idea B: Proximity bounds (from Brinkop, Fischer, Jansen 2022)

**Source:** Brinkop, Fischer, Jansen. "Structural Results for
High-Multiplicity Scheduling on Uniform Machines." arXiv:2203.01741.

**The idea:** The optimal integer solution is within ℓ₁ distance d(K)
of the LP solution, where d depends only on K (not n). This means
instead of searching the full assignment space, you only need to search
within a bounded neighborhood of the relaxation's assignment.

**Relevance to us:** If applicable, this would justify replacing the
Lagrangian/beam with a provably complete bounded search:

1. Compute the LP assignment (semigroup relaxation + FFD/BFD)
2. Enumerate all integer assignments within ℓ₁ distance d(K)
3. Evaluate each by Level 3
4. Take the best

**Caveat:** The proximity results are for scheduling on identical
machines, not for our TOU-cost single-machine setting. The bound d(K)
may not directly transfer. But the CONCEPT (search near the LP) is
sound. The question is whether d(K) is small enough for K=6 to make
enumeration tractable.

**Action for the coder:** This is a theoretical investigation, not an
immediate code task. Compute the ℓ₁ distance between the FFD assignment
and the beam's winning assignment on the hard-arithmetic rows. If this
distance is consistently small (say, <20 job transfers), proximity-based
search is promising.

### Idea C: Double-exponential lower bound (Jansen et al. 2025)

**Source:** Jansen, Ohnesorge, Pirotton. "Tight Double-Exponential Lower
Bound for High-Multiplicity Bin Packing." arXiv:2512.02691.

**The idea:** For general high-multiplicity bin packing, the encoding
dependence on K is doubly exponential. This is a complexity barrier.

**Relevance to us:** Supports two paper claims:

1. Arithmetic-aware methods are NECESSARY — there is no universal
   polynomial dependence on K for general instances.
2. The two-axis framework is the right decomposition — easy arithmetic
   can dodge the lower bound (our K=10 {1..10} exact results), while
   hard arithmetic hits it.

**Action for the coder:** Cite this result in the paper's complexity
discussion. Do not try to improve the general-K dependence — the lower
bound says it's intractable. Instead, characterize which instances
avoid the worst case (our easy-arithmetic branch).

### Idea D: Switching-cost bridge (Gabay et al. 2015)

**Source:** Gabay et al. "High Multiplicity Scheduling with Switching
Costs for Few Products." arXiv:1504.00201.

**The idea:** High-multiplicity scheduling with machine state transitions
modeled as switching costs.

**Relevance to us:** Our machine-state model (on/off/standby transitions)
is structurally similar to switching costs. This is the closest existing
high-multiplicity reference to our specific problem variant.

**Action for the coder:** Add this as a citation bridge in the paper.
It strengthens the claim that our problem fits within the high-multiplicity
scheduling framework, with the additional structure of TOU pricing.

### Idea E: Augmentation-based Level 2 (from n-fold IP theory)

**Source:** Knop, Koutecký. "Scheduling meets n-fold Integer
Programming." arXiv:1603.02611.

**The idea:** In n-fold IP, feasible solutions can be improved by
"augmenting steps" — small changes that improve the objective while
maintaining feasibility. The augmenting steps have bounded support
(depend on K, not n).

**Relevance to us:** The current local improver (adjacent-pair pattern
swap) failed because it was too coarse. Augmenting-step theory suggests
a more principled local search:

1. Start from the current assignment (from Lagrangian or beam)
2. At each step, find the best "augmentation" that changes at most
   d(K) blocks and improves the cost
3. Repeat until no improving augmentation exists

The key difference from the current local improver: augmentation theory
tells you the RIGHT neighborhood size and guarantees convergence.

**Caveat:** n-fold IP augmentation requires the problem to have genuine
n-fold structure. Our problem has B blocks (analogous to n agents) with
different capacities, which is only approximately n-fold. The theory
may not directly apply.

**Action for the coder:** This is a medium-term research direction, not
an immediate task. If dynamic pricing doesn't close the Level 2 gaps,
augmentation-based search is the next theoretical option to investigate.

### What NOT to pursue from this literature

- Do NOT implement a full n-fold IP solver — it's theoretically elegant
  but practically slower than our current pipeline for these problem
  sizes.
- Do NOT claim our problem is polynomial for fixed K based on
  Goemans-Rothvoss — their result is for bin packing (minimize bins),
  not for our fixed-block cost-minimization assignment.
- Do NOT cite arXiv:2203.03600 — it is withdrawn with a major proof
  error.

---

## Analytical refinements from external review (vetted)

An external review proposed several analytical improvements. After
filtering out overambitious parts (full branch-and-price solver,
certifying optimality via column exhaustion), these are the ideas
worth adopting:

### 1. Three-gap decomposition

The remaining optimality gap should be decomposed into three distinct
sources:

- **Relaxation gap**: difference between the semigroup LB and the true
  optimal. Currently ≈ 0 on all tested rows (R_feas = R_semi).
- **Configuration-pool gap**: missed patterns — the optimal assignment
  uses a block-fill pattern that is NOT in the filtered pool.
- **Search gap**: the optimal pattern IS in the pool, but the
  Lagrangian/beam fails to find the right combination.

The observed 0.006–0.016% is pool gap + search gap. Distinguishing
these two is the key diagnostic for deciding between dynamic pricing
(fixes pool gap) and better search (fixes search gap).

**Action for the coder:** After a beam run, check whether the beam's
winning patterns are a SUBSET of the patterns available to the
Lagrangian. If yes, the gap is search. If the beam uses patterns the
Lagrangian couldn't see, the gap is pool.

### 2. Hardness should be a triplet, not just lengths

Arithmetic hardness should be defined on the triplet:

> (length set L, block capacities C_1..C_B, remaining multiplicities n_1..n_K)

Not just the length set L alone. Two instances with the same lengths
but different block profiles (from different relaxations) can have
very different Level 2 difficulty. The block capacities come from
Level 1 and interact with bounded representability in instance-specific
ways.

**Action for the coder:** When reporting arithmetic descriptors, also
record the distribution of block capacities (min, max, mean, std) and
whether specific capacities fall near semigroup gaps.

### 3. Verify block-boundary state separability

If machine-state transitions carry over between blocks (e.g., the
machine is left ON at the end of block b and must be ON at the start
of block b+1), then Level 3 costs are NOT fully separable by block.
The current Level 3 evaluator uses block-local SPACES views with
c_start and c_end costs. Verify that these correctly account for
inter-block transition costs. If not, the per-block exact DP could
underestimate the true cost, and Level 2 would need to include
boundary state labels.

**Action for the coder:** On one representative hard row, compare the
sum of per-block Level 3 costs against the cost of the global
solve_fixed_sequence on the same assignment. If they match, the
separability is correct. If they differ, the boundary handling needs
fixing.

### 4. Non-monotonicity formalization for the paper

The non-monotonicity of difficulty in K can be stated formally:

> Adding a new type with length L_{K+1} to the generator set adds a
> new generator to the numerical semigroup. If L_{K+1} fills gaps in
> the existing semigroup, the density increases and capacity matching
> becomes easier. Therefore difficulty can DECREASE as K increases.

This explains why K=10 {1..10} is easier than K=6 {2,3,4,5,7,11}:
lengths 1,6,8,9,10 fill all remaining gaps in the 6-type semigroup.

**Action for the coder:** Include a small example in the paper showing
that adding length 1 to {2,3,4,5,7,11} makes all block capacities
representable, reducing the problem to easy arithmetic.

### 5. Additional diagnostics to report

Beyond the current metrics (gap, runtime, method), report:

- **Residual-core size**: number of blocks where the FFD assignment
  deviates from the target type proportions (Idea A above)
- **Infeasibility distance**: ℓ₁ norm of the Lagrangian's best
  near-feasible assignment (the "best_l1" value)
- **Fill uniqueness**: for each block capacity, how many distinct
  feasible fill patterns exist in the bounded pattern pool? Blocks
  with only 1 pattern are "rigid"; blocks with many are "flexible"

These help explain WHY specific rows are hard and support the
three-gap decomposition.

---

## Latest implementation findings (2026-04-12 evening)

### 1. The Lagrangian's precise failure mode is repair handoff

Tracing on the hard n=1000 row showed: the Lagrangian converges to
best_l1 = 23 (very close to feasible, out of ~1000 total jobs). The
dual trajectory is fine. The failure is in the REPAIR step: converting
a 23-unit L1 residual into a fully feasible assignment.

**Implication:** The next Level 2 improvement should target the
repair/rounding step specifically, not the dual search. Possible
approaches:
- Widen the repair radius (currently repair_l1 is adaptive)
- Try multiple repair starting points (from different near-feasible
  iterates, not just the best one)
- Use the three-gap decomposition to identify whether repair needs
  different patterns (pool gap) or different combinations (search gap)

### 2. Level 3 broke the Lagrangian's search guidance

The Level 3 change (exact per-block DP) inadvertently changed what the
Lagrangian was optimizing. The dual loop started optimizing the exact
evaluator instead of the old proxy. The exact evaluator is better for
scoring final candidates but WORSE for guiding dual steps (noisier,
causes the search to overshoot).

**Resolution:** The coder correctly split these: proxy for dual search
guidance, exact evaluator for final candidate scoring. This is the
right architecture — search and evaluation should use different cost
functions. Verify this split is active in the current codebase.

### 3. Seeded beam handoff does NOT recover Lagrangian quality

The coder tried seeding the beam with the Lagrangian's best iterate.
The beam still won over the Lagrangian. This means the beam's advantage
is in its EXPLORATION (searching different count-vector combinations),
not in its starting point. Warm-starting doesn't help.

### 4. Dynamic pricing is deferred

The first implementation (one priced bounded-knapsack pattern per block
per iteration) caused instability on larger rows. Currently disabled
behind `PAST_BLOCK_REPAIR_LAGR_DYNAMIC_PRICING`. Additionally, the
pool-vs-search question was not properly answered (Phase A diagnostic
was tautological — see "Latest implementation findings" below).

Dynamic pricing remains a valid future direction but is not the
immediate priority.

### 5. Cross-cell results (RESOLVED — Plan 01)

All cross-cell rows now run cleanly after fixing a heap-buffer-overflow
in exact DP seed initialization (guard for `totals[i] <= 0`).

Results at n=1000:
- **K=4 irregular**: gap 0.0083%, winner `block_repair_energy_core`
- **K=8 irregular**: gap 0.0157%, winner `block_repair_feasible_beam`
- **K=10 irregular**: gap 0.0221%, winner `block_repair_feasible_beam`

The axes compound mildly at n=1000 (gaps grow with K on irregular
arithmetic), but all gaps remain < 0.025%.

---

## Paper-level guidance

### Recommended paper structure

**Section 5: Easy arithmetic — K-scaling with favorable length sets**
- Show: R_semi + FFD closes the gap up to K=10
- Characterize when Step 1 is exact (semigroup completeness)

**Section 6: Hard arithmetic — the block assignment bottleneck**
- Show: the gap comes from Level 2, not Level 1
- Show: Lagrangian + exact Level 3 handles moderate cases
- Remaining gaps are 0.006–0.016%

**Section 7: Adaptive pipeline with arithmetic-aware method selection**
- Easy arithmetic → R_semi + FFD (no heavy machinery needed)
- Hard arithmetic → R_semi + Lagrangian MMKP + per-block exact DP

### Key claims to support

1. "Difficulty is not monotone in K" — supported by K=10 exact vs K=6 gap
2. "Arithmetic structure of job lengths is an independent hardness axis"
   — supported by the two-axis grid
3. "The semigroup relaxation is tight regardless of arithmetic class"
   — supported by the LB invariance (R_feas = R_semi on hard rows)
4. "The bottleneck for hard arithmetic is feasible block assignment
   (Level 2), not relaxation quality (Level 1) or within-block scheduling
   (Level 3)" — supported by Level 3 separation results + LB invariance

### What NOT to overclaim

- Do NOT claim Frobenius number alone predicts difficulty (it doesn't;
  bounded representability matters more)
- Do NOT claim K is irrelevant (K affects exact DP state space; it's just
  not the dominant effect with the current heuristic pipeline)
- Do NOT claim the method is optimal on hard arithmetic (0.006% gaps
  remain open)

---

## Latest implementation findings (2026-04-13 Plan 02B exact-L2)

1. **The Plan-02A diagnostic issue is now resolved correctly.**

   The earlier pool-membership check (`beam_not_in_pool`) was correctly
   identified as tautological because beam and Lagrangian share the same
   generated pool. Plan 02B replaced this with a real test: solve Level 2
   exactly over that shared pool.

2. **Self-contained exact Level-2 branch-and-bound is implemented and wired.**

   The solver now includes an exact Level-2 B&B over block pattern choices,
   with residual-count branching, suffix feasibility pruning, suffix min-cost
   pruning, and beam incumbent seeding. It is gated and timed by env flags,
   and reports diagnostics in CSV (`exact_l2_ub`, `time`, `nodes`, `closed`,
   improved-over-beam, beam-optimal-in-pool, status).

3. **Exact-L2 evidence cleanly separates regimes by merged-block count.**

   On required validation rows:

   - B=8/9: exact L2 closes to LB and improves beam.
   - B=14: exact L2 also closes and improves beam when given a longer budget.
   - B=19/20: exact L2 times out at 180s and matches beam UB before timeout.

4. **Updated diagnosis: Level-2 in-pool search gap is real on small/moderate B.**

   For B up to moderate size, remaining gap is primarily Level-2 search inside
   the existing pool. For B=19-20, current evidence shows large-B search
   hardness under present budgets; it does NOT prove pool/profile ceiling.

5. **Recommended operating policy is now hybrid.**

   - Run exact L2 on small/moderate B where it closes quickly or reliably.
   - Keep feasible beam fallback for larger-B rows.
   - Escalate to out-of-pool redesign only if larger-budget exact-in-pool runs
     still fail to improve/close the large-B rows.

---

## Unexplored ideas ranked by expert assessment (post-Plan 02B)

These are the main unimplemented upgrades after exact-L2 completion.
Ranked by expected impact x implementation risk:

### Rank 1: Large Neighborhood Search (out-of-pool, assignment-focused)

Unfix 2-3 blocks, re-fill by enumerating feasible count vectors (not restricted
to current pool), and re-evaluate with Level-3 exact DP. This is the most direct
way to test whether large-B plateaus are caused by pool limitations.

**Status: not implemented. Highest-value next redesign if large-B remains open.**

### Rank 2: Block coordinate descent with occasional exact subproblem closure

Fix all but one (or two) blocks, re-optimize local assignment exactly, iterate.
Cheap and practical as a post-beam/post-exact improver for large-B rows.

**Status: not implemented. Low complexity, medium expected gain.**

### Rank 3: Pattern perturbation + selective pool augmentation

Generate capacity-neutral pattern perturbations from incumbent patterns and
inject only high-value candidates. Keeps architecture simple while enabling
controlled out-of-pool exploration.

**Status: not implemented.**

### Rank 4: Alternative block profiles from Level-1 DP

Store near-optimal profile alternatives and rerun Level 2 on a small profile set.
This addresses the remaining Level-1-quality concern without full redesign.

**Status: not implemented. Requires DP path instrumentation.**

### Rank 5: Swap-chain local search (job-level, capacity-neutral chains)

Perform fine-grained multi-block exchange chains beyond adjacent-pattern swaps.
Potentially useful, but engineering complexity is higher than expected gain.

**Status: not implemented.**

### Completed from prior ranking

- Self-contained exact Level-2 B&B: implemented and validated (Plan 02B).

### NOT recommended

- Full n-fold IP solver: too complex, marginal practical benefit
- Commercial/external solvers: against project constraints
- Arc-flow reformulation: equivalent to pattern enumeration

---

## Latest implementation findings (2026-04-13 Plan 03/04 cleanup)

1. **Default pipeline is now aligned to the final four-step story.**

   Mainline behavior now follows:

   - Step 1: semigroup profile recovery
   - Step 2: fast profile realization (FFD/BFD/random)
   - Step 3: unified `profile_repair_beam`
   - Step 4: exact DP fallback (sparse then dense)

2. **Step 3 has been collapsed to one family.**

   The default hard-case repair now runs one beam-centered method with
   bounded local destroy/repair intensification, under the single paper-facing
   name:

   - `profile_repair_beam`

3. **Exact-L2 is demoted from mainline to diagnostic.**

   Exact Level-2 B&B remains available for archival diagnosis but no longer
   influences default results:

   - default `PAST_BLOCK_REPAIR_EXACT_L2=0`
   - optional `PAST_BLOCK_REPAIR_EXACT_L2_APPLY=1` is required to apply its UB

4. **Legacy Level-2 branches are no longer default-active.**

   `lagrangian_assign`, `feasible_counts`, `rg_beam`, and post-Lagrangian beam
   polish are now explicit archival/diagnostic paths only.

5. **Fixed-profile exact block-DP is no longer in default mainline.**

   It remains callable for diagnostics, but the final exact story is now only:

   - semigroup-guided exact DP fallback (sparse + dense)

6. **Validation confirms intended policy behavior.**

   On focused cleanup rows:

   - easy row closes in Step 2 (`ffd`)
   - medium/hard rows use Step-3 method (`profile_repair_beam`)
   - exact DP is used as fallback when budget allows and gap remains
   - previously exact-L2-touched row shows exact-L2 disabled and non-influential

---

## Updated next-step priority (post-cleanup)

1. **Improve Step-3 quality within the unified family (highest priority).**

   Work only inside `profile_repair_beam`:

   - stronger but bounded local destroy/repair operators,
   - better neighborhood ordering,
   - better incumbent reuse,
   - no reintroduction of co-equal Lagrangian/exact-L2 defaults.

2. **Keep exact DP as the only exact fallback and make it cheaper.**

   Continue admissible-bound and dominance improvements in sparse/dense exact DP,
   without adding a second exact family.

3. **Keep exact-L2 strictly archival unless explicitly requested.**

    Use it only for diagnosis/comparison tables, not default method claims.

## Latest implementation findings (2026-04-15 Plan 04C targeted v3)

1. **Exact-stage observability is now real on a hard anchor.**

   A targeted Plan-04C v3 pass (raised sparse-theoretical guardrail on selected
   rows) produced runs where sparse exact expands millions of states, so pruning
   counters are now meaningful on at least one hard anchor:

   - `hard_k8_irregular n=500 seed=0`.

2. **Incumbent quality currently dominates practical exact behavior on that anchor.**

   With exact variant fixed (`p0`), switching from `i2` to `i3` (or `i4`) kept
   final gap unchanged but substantially reduced exact work and runtime:

   - sparse-expanded states: `~13.39M -> ~3.23M`
   - exact elapsed: `~150.6s -> ~36.3s`
   - total runtime: `~235s -> ~186.5s`

   This is a strong sign that better Step-3 incumbent handoff remains the main
   practical lever in this regime.

3. **Pruning variants show counter-level gains but weaker end-to-end gains.**

   On `hard_k8 n=500` with incumbent fixed to `i2`:

   - `p1/p3` reduce expansions vs `p0/p2` (`13.39M -> 7.55M`),
   - but final UB/gap remains unchanged,
   - runtime stays effectively flat at ~`235s`.

   With stronger incumbent `i3`, `p1/p3` did not reduce expansions further and
   increased runtime.

4. **Type-aware LB implementation is active but effect is row-dependent.**

   Nonzero `pruned_type_aware` appears on weaker-incumbent K6 probe rows, but on
   the hard K8 anchor the value remains zero even when expansions differ. This
   suggests interaction with incumbent quality and search trajectory, not a simple
   monotone effect.

5. **Guidance implication for next cycle.**

   Keep Plan-04C focus, but prioritize:

   - strengthening incumbent handoff quality inside Step-3 unified family,
   - then retaining only pruning variants that improve end-to-end time/gap (not
     counters alone),
   - while expanding evidence breadth to 2-3 additional hard rows with real exact
     expansions before finalizing defaults.

---

## Latest implementation findings (2026-04-14 Plan 03B/04A continuation)

1. **Step-3 strengthening remains inside one unified family.**

   The active hard-case method remains `profile_repair_beam` (no branch sprawl),
   with arithmetic-aware ranking, bounded discrepancy handling, and adaptive beam
   width reflected in populated beam diagnostics on active rows.

2. **Exact-stage diagnostics interpretation issue is resolved.**

   The exact fallback now reports explicit skip/tightness modes instead of
   ambiguous `dense`+INF/zero outputs. New observed modes include:

   - `sparse_skip_theoretical`
   - `dense_skip_state_space`
   - `dense_skip_memory`

   This makes Step-4 behavior interpretable without changing solver policy.

3. **K=6 Step-4-entered rows now show clear exact handoff behavior.**

   For representative medium/hard K=6 rows, diagnostics show:

   - `exact_diag_initial_ub = Step-3 UB`
   - explicit sparse skip mode
   - `exact_diag_exhaustive=0`

   So exact DP is attempted under the final policy, but bounded by current
   sparse lattice guardrails on these rows.

4. **Current bottleneck remains practical exactness, not policy clarity.**

    The pipeline story is now stable and diagnosable. Remaining work should focus
    on reducing finite gaps/timeouts by improving Step-3 quality and exact-DP
    pruning/order practicality, while keeping exact-L2 archival-only.

## Latest implementation findings (2026-04-14 Plan 03C DP-family unification)

1. **Step 3 is now technically aligned as one profile-realization DP family.**

   The code now treats Step 3 as:

   - exact mode: fixed-block DP
   - truncated mode: `profile_repair_beam`

   with shared recovered-block semantics, shared local exact block evaluator,
   and compatible ordering/pruning interfaces.

2. **Fixed-block DP is retained and elevated, not archival-demoted.**

   Exact fixed-block realization remains present as Step-3 exact mode
   (`profile_realization_dp_exact`) and is now integrated into default/profile
   solver flow under existing tractability guardrails.

3. **Exact-safe transfer status: suffix pruning helps; hardest-first is mixed.**

   On a focused K=6 exact-mode seed scan (`n=120`, raised tractability limits):

   - enabling suffix min/max residual pruning reduced mean exact-mode runtime
     (~4.0%) and exact-mode block-DP time (~46.9%) vs no-suffix.
   - hardest-first ordering did not improve this slice; no-hardest-first was
     slightly faster across tested seeds.

4. **Tractability frontier remains the practical blocker for Step-3 exact mode.**

   Under default guardrails, exact Step-3 mode still skips larger rows
   (`skipped_comp_est`), while truncated mode + Step-4 fallback remains the
   practical path on hard `n=1000` rows.

5. **Guidance implication for next cycle.**

   Keep the unified Step-3 family framing fixed. Continue exact-safe pruning and
   tractability improvements inside exact mode, but treat hardest-first ordering
   as an empirical knob (not a default win) until broader evidence supports it.

## Latest implementation findings (2026-04-14 Plan 03D selector cycle)

1. **Step-3 exact-vs-beam decision is now explicit and auditable.**

   A selector policy is implemented in mainline Step 3 via:

   - `PAST_PROFILE_REALIZATION_SELECTOR_POLICY`
     (`auto_v1`, `off`, `force_exact`, `force_beam`)

   and row-level diagnostics now expose policy/decision/reason plus arithmetic
   context and exact/beam status split.

2. **Auto selector uses a conservative tractability gate.**

   `auto_v1` chooses exact only when merged-block and frontier-estimate limits
   are all below practical thresholds and no arithmetic hard alarm is active.
   This matches the project requirement to avoid K-only decisions.

3. **Validation indicates correct regime split on representative rows.**

   In `csv/plan03d/TMP_plan03d_selector_validation_table.csv`:

   - exact chosen on small exact-island rows (K4 n=300, K6 n=120 seeds), all
     feasible;
   - beam chosen on larger/harder rows (K6/K4 n=1000, K8 n=800, easy K10 with
     huge state-space), with selector reasons recorded as structural limits.

   No misclassification appears on this representative slice.

4. **Forced-exact controls confirm why beam remains primary on large rows.**

   For representative larger rows, forced exact is frequently blocked by
   comp-est guardrails and can fail to produce any usable incumbent
   (`pack_method=none`, `exact_skipped_comp_est`). This is direct evidence that
   beam-first is the practical policy there.

5. **Guidance implication after Plan 03D.**

   Keep `auto_v1` as default selector now. Treat the next cycle as calibration,
   not redesign:

   - add near-threshold seeds,
   - stress hard-alarm rows,
   - adjust only if false-positive/false-negative evidence appears.

## Latest implementation findings (2026-04-14 Plan 03D hardening pass)

1. **Exact-primary Step-3 policy is now fail-safe.**

   For `auto_v1` exact-primary rows, Step 3 now executes exact first and
   automatically falls back to beam in the same cycle when exact has no finite
   candidate. This removes the prior brittle outcome where Step 3 could exit
   without a usable profile-realization incumbent.

2. **Fallback is explicit in diagnostics (no hidden rescue).**

   New CSV diagnostics now expose:

   - `fwd_profile_exact_primary_fallback_to_beam`
   - `fwd_profile_exact_primary_status_before_fallback`
   - `fwd_profile_step3_incumbent_mode`

   Probe rows show expected activation on `timeout`, `skipped_nc`, and
   `skipped_comp_est` exact statuses.

3. **Selector validation now enforces step-role separation.**

   The rebuilt selector table (`csv/plan03d/TMP_plan03d_selector_boundary_reval_table.csv`)
   explicitly separates:

   - Step-2-closed controls,
   - Step-3 selector-test rows,
   - Step-4-used rows.

   Misclassification is computed only on Step-3 selector-test rows.

4. **Current hardening verdict.**

   `auto_v1` is robust enough to stay default after this pass because:

   - exact-primary brittleness is fixed,
   - validation methodology no longer overstates confidence.

   However, confidence width is still limited by small Step-3-test sample size,
   so another boundary-focused accumulation cycle is still recommended.

5. **Priority for next cycle (unchanged in spirit).**

   Keep selector rule stable; gather more near-threshold Step-3-test rows before
   any threshold changes. Do not redesign selector family unless new
   misclassification evidence appears.
