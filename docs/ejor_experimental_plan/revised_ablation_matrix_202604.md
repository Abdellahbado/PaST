# Revised Ablation Matrix For The Current PaST Method

Date: 2026-04-01

## Why The Old Ablation Package Should Be Updated

The previous ablation package was useful, but it no longer matches the current
method tightly enough for a journal submission.

Three things changed:

1. the forward semigroup path became much faster after the sharpened automatic
   max-gap rule;
2. the practical benchmark story is now split across two roles:
   - large-`n` `{8,10}` for scalability,
   - realistic bounded-count 3-type families for backup necessity;
3. the old component-ablation story was built around a pipeline in which later
   fallback stages mattered more often than they do now.

As a result, the existing studies are in three different states:

| Study | Status | Keep? | Reason |
|---|---|---|---|
| Relaxation quality | still valid | Yes | still measures unit vs gcd vs semigroup cleanly |
| G sweep | partially stale | Update | must be reframed around the new `auto_max_gap` logic |
| SPACES banded vs full | partially stale | Update | should be rerun with the new sharpened gap rule |
| Component ablation | inadequate | Replace | current report is weak and does not reflect the present winning path |

---

## What A Journal-Ready OR Ablation Should Look Like

For an OR / exact-algorithm paper, the computational study should focus on:

1. **component contribution**
   which major algorithmic ideas matter;
2. **parameter robustness**
   whether the key tuning rule is stable and safe;
3. **structure sensitivity**
   which instance characteristics make the method hard;
4. **reproducibility**
   per-instance results, explicit benchmark definitions, and runnable scripts.

It should not look like a broad ML-style variant sweep. Controlled comparisons
are more appropriate here.

This is consistent with:

- Hooker's argument for controlled algorithmic experimentation rather than
  loose competitive testing,
- Dolan and Moré's benchmarking philosophy,
- and EJOR's explicit reproducibility requirements.

Sources:

- [EJOR Guide for Authors](https://www.sciencedirect.com/journal/european-journal-of-operational-research/publish/guide-for-authors)
- [Hooker 1982](https://kilthub.cmu.edu/articles/journal_contribution/Testing_Heuristics_We_Have_It_All_Wrong/6708278)
- Dolan and Moré (2002), performance profiles

---

## Recommended Revised Study Package

The revised package should have four core studies.

### Study A. Relaxation Quality

Purpose:

- quantify the lower-bound hierarchy
- `unit -> gcd -> semigroup`

Primary dataset:

- original `benedikt2025b_groups`

Question answered:

- how much of the strength comes from divisibility only, and how much from full
  semigroup reachability?

### Study B. Max-Gap / Banding Robustness

Purpose:

- validate the new sharpened automatic max-gap rule
- compare it against full SPACES and scaled variants

Primary datasets:

- original `benedikt2025b_groups`
- formal `scalability_large_n` suite

Question answered:

- is the new automatic band limit both safe and materially faster?

### Study C. Certification Contribution

Purpose:

- isolate the role of fixed-profile certification in the current method
- especially on large-`n` hard families

Primary datasets:

- formal `scalability_large_n`
- optionally the original `{8,10}` group inside `benedikt2025b_groups`

Question answered:

- when do the relaxed profile and simple packing heuristics suffice,
- and when does exact fixed-profile block certification make the difference?

### Study D. Backup Necessity

Purpose:

- justify `R_feas` in the paper
- show that semigroup is usually enough but not always sufficient

Primary dataset:

- formal `backup_realistic` suite

Question answered:

- when does semigroup return a weaker or unpackable relaxed profile,
- and does `R_feas` repair the problem?

---

## Optional Secondary Study

### Study E. Increasing-`K` Structural Boundary

Purpose:

- probe whether many distinct job sizes create a practical limit

Primary dataset:

- formal `k_boundary` suite

Current status:

- useful as a boundary probe
- not yet a strong main-paper claim, because the first realistic pass up to
  `K=8` did not expose a clear failure boundary

So this should be:

- appendix or secondary section,
- not a headline ablation.

---

## Concrete Matrix

| Study | Main claim | Dataset(s) | Main outputs |
|---|---|---|---|
| A. Relaxation quality | semigroup is the right default lower bound | `benedikt2025b_groups` | mean/max gap, strict-improvement counts |
| B. Max-gap robustness | sharpened auto-gap is safe and faster | `benedikt2025b_groups`, `paperext_scalability_large_n_202604` | runtime, mismatch count vs full, average chosen gap |
| C. Certification contribution | exact fixed-profile certification matters on hard scalable cases | `paperext_scalability_large_n_202604` | phase reached, winning submethod, runtime |
| D. Backup necessity | `R_feas` repairs realistic bounded-count failures | `paperext_backup_realistic_202604` | LB change, packability rescue, exact confirmation |
| E. `K` boundary (secondary) | larger `K` alone may or may not be the boundary | `paperext_k_boundary_202604` | phase reached, runtime by `K`, whether backups help |

---

## What We Should Report In The Paper

### Main paper tables

1. baseline comparison against the paper solver
2. relaxation hierarchy table
3. max-gap / SPACES robustness table
4. certification contribution table
5. backup-realistic rescue table

### Appendix material

1. increasing-`K` boundary study
2. full per-instance CSVs
3. optional statistical comparisons or performance profiles

---

## Mapping To Current Solver Knobs

The current code already supports the main controls needed:

| Concept | Current mechanism |
|---|---|
| unit / gcd / semi LB comparison | `relaxation-stdin` |
| banded vs full SPACES | `ablation-stdin full` vs `full_spaces` |
| profile all lower bounds | `ablation-stdin bounds_profile` |
| omit smart reconstruction | `ablation-stdin no_smart_recon` |
| relaxed/front-only phase | `ablation-stdin step1_only` |
| vary auto gap | `PAST_MAX_GAP_OVERRIDE`, `PAST_MAX_GAP_SCALE` |
| semigroup vs feasible packability | `relax-pack-stdin` + `relax-hierarchy-stdin` |

So the update is mainly:

- a new experimental structure,
- not a major new instrumentation effort.

---

## Practical Recommendation

If we want a compact, strong journal package, the priority order should be:

1. refresh relaxation quality,
2. replace the old `G` study by a max-gap robustness study,
3. replace the old component study by a certification study,
4. add the realistic backup-necessity study,
5. keep the increasing-`K` study as a secondary structural probe.

That is the cleanest study package for the current version of the method.
