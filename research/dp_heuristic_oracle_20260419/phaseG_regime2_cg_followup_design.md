# Phase G Regime-2 CG Follow-up Design (Duplicate-Stop Correction)

Date: 2026-04-20

## 1) Duplicate-stop bug

In the first Phase G run, the loop stopped as soon as the single best negative reduced-cost pricing candidate was already in the pool.

This was too aggressive: another class (or another near-best candidate) could still provide a new negative reduced-cost column.

## 2) Correction implemented

- Keep per-iteration best-single pricing across all classes.
- If negative candidates exist but all are duplicates, run an additional bounded threshold pricing pass (`paper-instance-threshold`) for the affected classes.
- Build a small candidate list and select the best **non-duplicate** negative column.
- Stop only when no negative column exists, or when negative columns exist but none are new (`no_new_negative_rc_column`).

## 3) Small enhancement beyond minimum fix

Yes (bounded): class-local candidate diversification via threshold enumeration for duplicate-hit classes only.

Bounded by:

- only duplicate-hit classes are expanded,
- strict per-class time limit,
- capped candidate list size.

No method-family change; still restricted master + bounded pricing loop.

## 4) Exact bounded rerun

- one rerun only: instance `61`, epsilon `347`
- same initial pool policy and same master setup
- same iteration cap (12)
- regenerated artifacts in `temp/phaseG_regime2_cg/`

## 5) “Enough signal to continue” criterion

Continue only if corrected run shows at least one:

1. material TEC improvement over `7040`,
2. materially stronger LP bound,
3. clear evidence previous outcome was mainly duplicate-stop limited (new columns added with meaningful effect).

Otherwise treat regime-2 restricted-CG branch as likely non-competitive in this bounded form.
