# PLAN 05 — Extend the Paper's 7 Processing-Time Groups in `n` and `lambda`

Last updated: 2026-04-15

## Purpose

This plan defines a clean paper-facing benchmark campaign built on the **seven
processing-time groups from Section 5.2 / Table 2** of
`Papers/arXiv 2506.10405.txt`.

The goal is **not** to revisit the already-established Table-2 reproduction.
The goal is to **extend those same seven groups** to:

1. larger job counts `n`,
2. varied horizon inflation / slack (`lambda`),
3. and report how the cleaned current four-step pipeline behaves.

This extension should be completed **before** investing substantial additional
effort into harder custom families, because it gives the cleanest supervisor-
 and paper-facing statement:

> On the paper's own processing-time families, our method remains strong and
> scalable beyond the original benchmark range.

---

## Why this plan exists

What is already true:

- The archive already contains strong claims and evidence that the original
  Table-2 benchmark families were solved.
- In particular, the paper's hard pair `{8,10}` has already been studied far
  beyond the original `n <= 200` regime in dedicated large-`n` notes.

What is **not** yet cleanly established:

- a single systematic extension of **all seven Section-5.2 groups**
- under the current cleaned four-step method
- with larger `n`
- and controlled variation in `lambda`.

So the missing piece is not "can we solve the original benchmark?" but:

> "How far do the original paper families continue to scale under our method?"

---

## The 7 target families

These are the exact Section-5.2 / Table-2 processing-time groups from the
paper:

1. `{1,2,3,4,5,6,7,8,9,10}`
2. `{1,2,3,5,7}`
3. `{2,4,6,8,10}`
4. `{2,4}`
5. `{3,5,6,7}`
6. `{3,7}`
7. `{8,10}`

These seven groups are the **only required families** for this plan.

Do **not** mix in:

- the Section-5.3 harder groups,
- the six-type archive families `{2,3,4,5,7,11}` and `{4,5,6,7,8,9}`,
- or custom irregular high-`K` families.

Those belong to later plans.

---

## Method policy to use

Use the cleaned current mainline solver policy:

1. Step 1: semigroup profile recovery
2. Step 2: quick realization
3. Step 3: profile-realization DP family
   - exact mode when tractable
   - beam mode otherwise
4. Step 4: global exact DP fallback

Do **not** use:

- exact-L2 as a mainline method,
- Lagrangian as a co-equal default branch,
- archived non-mainline solver modes unless explicitly requested for a side
  diagnostic.

This benchmark extension must reflect the **actual final method story**.

---

## Main questions this plan must answer

### Q1. Scaling in `n`

For each of the seven paper families:

- how far can we scale `n` under the current method?
- when does the dominant deciding step switch?
- does the family remain exact, near-exact, or open-gap?

### Q2. Scaling in `lambda`

For each of the seven paper families:

- how sensitive is the method to horizon slack / inflated horizon length?
- does larger `lambda` help closure, hurt runtime, or both?
- which families are robust and which become difficult as `lambda` grows?

### Q3. Boundary classification

For each of the seven paper families:

- is it `easy-scalable`,
- `step3-dominated but practical`,
- or `step4-limited` under extension?

---

## Required experiment design

## Phase 1 — Reconfirm paper-family baseline under current cleaned policy

Before extending anything, run a compact baseline slice on the 7 families using
the paper-like protocol:

- `n ∈ {50,100,150,200}`
- `lambda ∈ {1.3,1.6,1.9,2.2}`
- 1-3 seeds per point is acceptable for the first pass if full 20-seed
  reproduction is too expensive immediately, but the exact seed policy must be
  recorded.

Purpose:

- ensure the current cleaned pipeline still behaves consistently on the paper's
  family space,
- establish a fresh baseline table under the **current** method,
- avoid comparing large-`n` extension rows against stale pre-cleanup numbers.

Deliverable:

- one baseline CSV
- one short summary table:
  - family
  - `n`
  - `lambda`
  - runtime
  - UB/LB/gap
  - deciding step

---

## Phase 2 — Extend `n` first, keeping `lambda` paper-like

This is the highest-priority extension phase.

Fix:

- machine/state setting consistent with the paper protocol,
- `lambda = 1.3` first.

Run for each of the 7 families:

- `n ∈ {300,400,500,600,750,1000}`

If a family remains clearly easy/practical:

- extend further to `n ∈ {1500,2500,3500,5000}`

Do this progressively:

1. first all seven families through `n=1000`
2. then only the strongest/easiest families through `1500+`

Reason:

- we want complete coverage through moderate extension first,
- then deeper scaling only where justified.

### Priority order inside Phase 2

Run in this order:

1. `{8,10}`
2. `{1,2,3,4,5,6,7,8,9,10}`
3. `{3,5,6,7}`
4. `{3,7}`
5. `{2,4,6,8,10}`
6. `{1,2,3,5,7}`
7. `{2,4}`

Rationale:

- `{8,10}` is the paper's hardest iconic family
- `{1..10}` is the clean easiest high-variability family
- the remaining families then fill the interior of the paper's spectrum

Deliverable:

- one `n`-extension CSV
- one family-by-family summary table

Required summary columns:

- family
- `K`
- `n`
- `lambda`
- runtime
- UB
- LB
- gap
- deciding step
- Step-3 mode if used (`exact` / `beam`)
- exact-DP entered or not

---

## Phase 3 — Extend `lambda` second, after `n` behavior is known

Only after Phase 2 is stable.

For each family, choose representative `n` values:

- one small `n` that is safely easy,
- one medium `n`,
- one larger `n` that is near the practical boundary if available.

Then vary:

- `lambda ∈ {1.3,1.6,1.9,2.2,2.5,3.0}`

Important:

- treat this as the correct way to vary the horizon / "end"
- do **not** create arbitrary custom `h` values divorced from the paper recipe
  unless there is a very explicit separate reason.

Reason:

- the paper defines horizon through `lambda`,
- so the clean extension story is "vary `lambda`", not "choose arbitrary end
  points."

Deliverable:

- one `lambda`-extension CSV
- per-family runtime/gap trends vs `lambda`

---

## Phase 4 — Family classification

Using Phases 1-3, classify each of the seven families into one of:

### A. Easy-scalable

Typical behavior:

- Step 1 or Step 2 closes regularly
- Step 3 seldom needed
- Step 4 rarely needed
- larger `n` and `lambda` remain practical

### B. Step-3-dominated but practical

Typical behavior:

- Step 2 stops closing reliably
- Step 3 becomes the main incumbent producer
- exact fallback may be entered but is not the main performance wall

### C. Step-4-limited / closure-limited

Typical behavior:

- strong incumbents exist
- final gaps remain tiny
- exact certification becomes the main bottleneck

This classification is a required output.

---

## Explicit questions to answer for each family

For each of the seven groups, answer all of the following:

1. Largest tested `n` at `lambda=1.3`
2. Largest tested `n` with finite tiny gap
3. Largest tested `n` with exact closure
4. Which step usually decides the result?
5. Does increasing `lambda` help or hurt?
6. Is the family easy, Step-3-dominated, or Step-4-limited?

These answers should be written in prose in the archive, not only left in CSV.

---

## What is already partially covered

The coder should assume these are **partially** covered already and should be
verified/reused rather than blindly rerun from scratch:

- `{8,10}`: large-`n` extension notes already exist and reach at least
  `n=1000`
- `{1..10}`: large-`n` exact rows already exist in the archive
- `{3,5,6,7}`: partial larger-`n` evidence exists via `3567_plus`

But for this plan, these must still be brought into one coherent extension
table together with the other four families.

---

## What is probably missing and should be added

Most likely missing as a clean current extension sweep:

- `{1,2,3,5,7}`
- `{2,4,6,8,10}`
- `{2,4}`
- `{3,7}`

especially at:

- `n >= 300`
- and across multiple `lambda` values.

These are high priority for filling the paper-family matrix.

---

## Success criteria

This plan succeeds if all of the following become true:

1. The seven paper Section-5.2 families have one coherent extension table under
   the current four-step method.
2. For each family, we know the practical `n` boundary at `lambda=1.3`.
3. For each family, we know how `lambda` changes difficulty.
4. We can state which pipeline step dominates each family.
5. We can honestly say whether the paper's own family space is now fully under
   control before moving to harder custom families.

---

## Failure modes and fallback rules

### If full all-family all-grid runs are too expensive

Fallback:

1. complete Phase 2 through `n=1000` for all 7 families first
2. complete Phase 3 only for 3 representative families:
   - easy: `{1..10}`
   - medium/interior: `{3,5,6,7}` or `{1,2,3,5,7}`
   - hard iconic: `{8,10}`

### If some large-`n` runs time out broadly

Fallback:

- record the last practical exact / tiny-gap point cleanly
- do not skip the family
- the boundary itself is a valid result

### If a family behaves unexpectedly compared with the paper

Required action:

- record whether the difference is due to:
  - current pipeline differences,
  - instance-generation mismatch,
  - or stronger method behavior.

Do not smooth over these differences.

---

## Required archive updates

Update at least:

- `LOG.md`
- `RESULTS.md`
- `METHOD_BOUNDARIES.md`
- `EXPERT_GUIDANCE.md`

Add a dedicated paper-family summary note if needed, for example:

- `PAPER_GROUPS_EXTENSION_SUMMARY.md`

---

## Recommended final paper-facing message

If this plan succeeds, the clean claim should be:

> On the seven processing-time groups used in the paper's main real-price
> benchmark, our four-step method not only reproduces the benchmark but remains
> effective under substantial extensions in job count and horizon slack.

Only after that should the project pivot to beyond-paper harder custom cases.
