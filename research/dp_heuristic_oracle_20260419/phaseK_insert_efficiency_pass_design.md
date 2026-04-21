# Phase K Insert Efficiency Pass Design

Date: 2026-04-20

## 1) What Phase J proved

At `61/347`:

- old screened `vnd_exact_dp`: `6944`
- no-screen diagnostic best: `6920`
- Phase J `vnd_exact_dp_insert_rank_v1`: `6908`
- Phase J `vnd_exact_dp_insert_rank_diverse`: `6884`

So analytical insert screening can recover and exceed no-screen signal.

## 2) Remaining bottleneck

- quality signal exists, but screening structure is heavy.
- Phase J diverse still screened `29160` insert candidates and incurred high runtime/RSS overhead.

## 3) Efficiency-focused ideas considered

1. **Idea A (pool structure tightening)**
   - lower source top-k
   - lower per-source keep
   - add per-target quota in per-source selection
   - lower shortlist/exact caps

2. **Idea B (staged exact-eval budget)**
   - evaluate in stages (`small -> medium -> full`)
   - stop stage expansion when score quality drops enough

3. **Idea C (source-machine focusing refinement)**
   - preserve gap-aware source priority
   - in trimmed mode, keep only top-priority sources above threshold

4. **Idea D (tiny secondary swap_inter budget)**
   - considered but deferred in this pass to avoid confounding insert-screening efficiency evidence.

## 4) Selected now for implementation

1. `vnd_exact_dp_insert_rank_diverse_trimmed`
   - Idea A + Idea C
2. `vnd_exact_dp_insert_rank_diverse_budgeted`
   - Idea A + Idea B + Idea C

## 5) Why these are the best last non-ML pass

- they keep the same successful Phase J method family,
- modify only candidate-pool and budget structure,
- remain bounded/single-point,
- produce direct evidence for “continue handcrafted vs pivot” without broadening scope.

## 6) Continue vs pivot criterion after Phase K

Continue handcrafted only if one variant does at least one:

- beats `6884`, or
- keeps ~`6884` quality with **material** runtime/RSS improvement.

If not, pivot to learning-based screening/ranking with this insert-focused exact-DP acceptance engine as base.
