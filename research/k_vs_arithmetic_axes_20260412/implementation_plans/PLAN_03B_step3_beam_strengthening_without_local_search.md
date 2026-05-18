# Plan 03B: Step 3 Beam Strengthening Without Local Search

## Goal

Strengthen Step 3 while keeping the method clean, predictable, and close to the
original theory:

- no local search as a separate method family,
- no second exact method,
- no return to a Level-2 method zoo,
- no hidden fallback branches.

Recommended Step-3 method:

**Arithmetic-aware profile repair beam with discrepancy and pricing-lite
extensions.**

Short paper name:

- `profile_repair_beam`

This remains one method family. The improvements below are changes to the beam,
not new co-equal methods.

---

## Why this direction

The current evidence says:

1. the hard cases live in Level 2,
2. the current beam is the strongest existing Step-3 method,
3. local-search-style add-ons make the story less clean,
4. exact-L2 was useful diagnostically but should not survive in the final
   method,
5. the right question is not “what new heuristic should we add?” but “how do we
   make the beam less brittle on arithmetic-hard rows?”

So the right redesign is:

- keep the beam,
- make its branching/scoring/generation stronger,
- and make its output more useful to Step 4.

---

## Literature-backed beam directions

### A. Variable / recovering beam width

Beam width should not be fixed blindly. Literature on scheduling beam search
supports adaptive beam/filter widths and recovering variants:

- Sabuncuoglu and Bayiz (2008), beam search variants for scheduling:
  https://doi.org/10.1016/j.cor.2006.11.004

Interpretation for this solver:

- easy layers: narrow beam,
- arithmetic-hard or high-uncertainty layers: wider beam,
- optionally retain a small recovery set when pruning is aggressive.

This is still beam search, not local search.

### B. Limited-discrepancy beam

If the ranking heuristic is usually good but occasionally wrong near the top of
the search tree, discrepancy-based beam search is a clean intensification:

- Furcy and Koenig, “Limited Discrepancy Beam Search” (IJCAI 2005):
  https://www.ijcai.org/Proceedings/05/Papers/0818.pdf

Interpretation for this solver:

- let the beam follow the preferred pattern ordering by default,
- but allow a bounded number of early “non-top-ranked” choices,
- especially in the first few recovered blocks where mistakes are most costly.

This is much cleaner than local neighborhood search because it remains a tree
search over the same Step-3 object.

### C. Pricing-lite / dynamic pattern augmentation

If the filtered pattern pool is the real ceiling on hard-arithmetic rows, then
the clean extension is to generate missing patterns on demand during the beam:

- Gilmore and Gomory (1961), column generation for pattern problems:
  https://doi.org/10.1287/opre.9.6.849
- Brandão and Pedroso (2016), arc-flow / compressed pattern representation:
  https://doi.org/10.1016/j.cor.2015.11.009

Interpretation for this solver:

- keep the beam as the outer search,
- but for blocks/states where the current pool is weak, generate a few new
  candidate patterns using a bounded knapsack pricing subproblem,
- do not jump straight to full branch-price or arc-flow in the mainline.

This is the cleanest way to strengthen Step 3 without switching method family.

---

## Recommended Step-3 redesign

### Stage 1. Keep the current beam skeleton

Preserve:

- count-vector state,
- suffix feasibility pruning,
- bounded width,
- exact Level-3 per-block evaluation,
- incumbent tracking.

### Stage 2. Improve the ranking function

The beam score should become explicitly arithmetic-aware.

Recommended ranking components:

1. **primary feasibility pressure**
   - residual type deficit/excess,
   - suffix reachability pressure,
   - impossible or near-impossible residual signatures get pushed down hard.

2. **exact local cost**
   - keep exact Level-3 block evaluation inside the score.

3. **arithmetic-risk term**
   - penalize residual states whose remaining work/capacity signature is harder
     to realize under the current lengths.
   - use cheap descriptors only:
     - residue mismatch modulo small generators,
     - low bounded-density residuals,
     - small “filler slack” when no short lengths remain.

This is still one beam score, not a new method.

### Stage 3. Add bounded discrepancy search

Add a discrepancy budget inside the beam:

- each time the search takes a non-top-ranked pattern for a block, charge one
  discrepancy,
- allow only a small discrepancy budget,
- bias discrepancies toward early layers.

Suggested first version:

- only for the first `d` merged blocks,
- discrepancy budget `0, 1, 2`,
- fixed small multiplier on beam width.

Goal:

- recover from early heuristic mistakes,
- without leaving the beam-search framework.

### Stage 4. Add variable-width control

Beam width should depend on difficulty of the current layer, not only on a
global constant.

Suggested signals:

- number of reachable states before truncation,
- pattern entropy / diversity in this block,
- residual arithmetic risk,
- whether incumbent is already finite and tight.

Policy idea:

- narrow width when the layer is easy or the incumbent is already strong,
- wider width when the layer is combinatorially diverse or arithmetic-hard.

### Stage 5. Add pricing-lite only if needed

Only if beam improvements still stall and evidence points to a pool ceiling:

- for a small number of high-impact blocks, solve a bounded knapsack pricing
  subproblem to create a few extra patterns,
- inject them into the current beam layer,
- keep the rest of the method unchanged.

This should be the first out-of-pool escalation, ahead of any full arc-flow
redesign.

---

## What not to do

To keep Step 3 clean:

- do not add generic local search,
- do not add a separate LNS phase,
- do not reintroduce Lagrangian as a co-equal default,
- do not add exact-L2 back into the mainline,
- do not add full arc-flow or full column generation before the simpler beam
  strengthening stages are tested.

---

## Development order

### Phase 1. Beam-only cleanup and scoring

Implement first:

1. arithmetic-aware ranking cleanup,
2. discrepancy budget,
3. variable beam width.

This is the highest-priority Step-3 plan.

### Phase 2. Validate against current hard rows

Evaluate on the existing benchmark rows where Step 3 matters:

- hard `K=6` rows,
- medium `K=6` rows,
- representative hard `K=4`,
- at least one hard high-`K` cross-cell.

Primary metrics:

- UB improvement,
- Step-3 runtime,
- exact-DP handoff quality,
- whether Step 4 closes more often or faster.

### Phase 3. Decide if pool ceiling remains

If the strengthened beam still returns the same incumbents as before, and the
diagnostics suggest the search is no longer the bottleneck, then the next
extension is:

- pricing-lite / dynamic pattern augmentation.

If the strengthened beam clearly improves incumbents, continue in this beam
family and do not escalate yet.

---

## Success criterion

Step 3 should be explainable as:

> “For hard arithmetic, we solve profile realization with an arithmetic-aware
> beam search, strengthened by discrepancy control and adaptive width, while
> evaluating each candidate by exact per-block scheduling.”

That is clean, theoretically defensible, and still very close to the original
method.
