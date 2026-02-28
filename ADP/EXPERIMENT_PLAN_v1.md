# Rigorous Experimental Evaluation Plan: Learning-Accelerated ADP for Single-Machine TOU Scheduling

**Date:** February 27, 2026  
**Status:** Planning phase — not yet implemented

---

## Overview

Systematically evaluate the learning-accelerated beam-pruned DP method across three orthogonal axes:

1. **Pricing profile complexity** — from simple `daily_tou` to jagged, multi-peak, and randomly-generated profiles
2. **Forecast noise intensity** — full-factorial sweep of σ × ρ × spike configurations to find the *breaking point*
3. **Model capacity** — polynomial degree comparison (deg 2 → 3 → 4) and MLP width/depth variants

**Training sizes:** small and medium only (large training skipped as per scope).  
**Evaluation sizes:** small, medium, and large for all trained models.  
**Key reuse:** The pre-built medium pooled dataset at `ADP/Data/Pooled Medium Daily Optimal Path Training Data.npz` is reused for all `daily_tou` medium-trained experiments — no regeneration needed.

---

## Code Changes Required

### Change 1 — New Pricing Profiles + Custom CLI

**File:** `sandbox/eval_pooled_vhat.py`

#### 1a. New named profiles

Add the following profile functions alongside `daily_tou_20()` in the codebase (and in `sandbox/train_eval_vhat_beam_dp.py`):

| Profile name | Description | 20-element price vector |
|---|---|---|
| `complex_5tier` | 5 tiers, irregular durations, valley → ramp → sharp peak → descent | `[1,1,1.5,1.5,2,2,2,3,3,3,4,5,5,4,3,2.5,2.5,2,1.5,1]` |
| `jagged` | Rapid alternation between low and high, tests regime-switching | `[1,3,1,3,2,4,2,4,1,5,1,5,2,4,2,4,1,3,1,3]` |
| `double_peak` | Two separate peaks per day (non-unimodal structure) | `[1,1,2,4,4,2,1,1,2,3,5,5,3,2,1,1,2,4,2,1]` |

Add all three as new choices in `--daily-price-profile` alongside the existing `daily_tou` and `generate_data`.

#### 1b. Generic custom price vector

- Add CLI argument `--custom-prices` accepting a comma-separated 20-element float string  
  (e.g. `--custom-prices "1,1,2,3,4,4,3,2,1.5,1.5,2,3,5,5,4,3,2,2,1,1"`)
- Add `"custom"` as a profile choice in `--daily-price-profile`
- Parse and validate in `main()` at the profile dispatch block (~line 1583)
- The plumbing already supports arbitrary 20-element float vectors via `build_instance(daily_prices_20=...)`; only CLI parsing needs extending

---

### Change 2 — Configurable Polynomial Degree

**Files:** `solvers/vhat_models.py`, `sandbox/eval_pooled_vhat.py`

Currently the polynomial model is hardcoded to degree 2 and raises a `ValueError` for any other degree.

#### Changes needed

1. **Generalize `_poly_powers_degree2(d_in)`** → `_poly_powers(d_in, degree)` using `itertools.combinations_with_replacement`
2. **Update `fit_poly_ridge()`** to accept and use any integer degree ≥ 2
3. **Add `--poly-degree` CLI argument** (default=2, choices=[2, 3, 4]) to the argparser
4. **Include degree in model checkpoint** — save/load must persist degree metadata alongside the weight vector

#### Feature dimension estimates (45 base features for small instances)

| Degree | Formula | Approx. features | Relative size |
|--------|---------|-----------------|---------------|
| 2 | $1 + d + \frac{d(d+1)}{2}$ | **1,081** | 1× (baseline) |
| 3 | $\binom{d+3}{3}$ | **~17,296** | ~16× |
| 4 | $\binom{d+4}{4}$ | **~163,185** | ~151× |

- **Degree 3:** Tractable. A 17K×17K ridge normal-equations solve takes ~1–3s; inference per state is ~1ms.
- **Degree 4:** Borderline. Dense ridge on 163K features risks memory issues. Add a time/memory guard; if it exceeds 30s for fitting, report as infeasible and skip.

---

### Change 3 — MLP Hidden Size CLI (lower priority)

**File:** `sandbox/eval_pooled_vhat.py`

- Add `--mlp-hidden` argument accepting comma-separated dims (e.g. `"128,64"` or `"256,128,64"`)
- Pass to `MLPValueModel` constructor instead of hardcoded `[64, 32]`
- Default: `"64,32"` (preserves existing behavior)
- This allows testing wider/deeper MLPs without code changes

---

## Experiment Scripts

### Experiment A — Noise Stress Test

**Script:** `scripts/exp_noise_stress_test.sh`  
**Goal:** Find the noise level at which the method breaks (gap exceeds 1%, 2%, 5%)

#### Noise parameter grid (full factorial)

| Parameter | Values |
|-----------|--------|
| σ (sigma, AR(1) stddev) | 0.0, 0.25, 0.5, 1.0, 2.0, 4.0 |
| ρ (rho, AR(1) correlation) | 0.5, 0.9 |
| Spike configuration | off (prob=0, mag=0), moderate (prob=0.02, mag=2.0, dur=2), extreme (prob=0.05, mag=4.0, dur=3) |

**Total noise configs:** 6 × 2 × 3 = **36 combinations**

#### Setup

- **No retraining** — noise is applied at evaluation time only; all trained checkpoints are reused
- Eval seeds: 400–429 (medium, 30 instances), 500–519 (large, 20 instances)
- Price mode: `forecast_realized` (model guided by clean forecast, costs evaluated on noisy realized prices)
- Beams: [2, 5, 10]
- Models: poly (deg 2), poly (deg 3), mlp, lgbm
- Profiles: `daily_tou`, `complex_5tier`
- Two pretrained checkpoints per combo: trained-on-small and trained-on-medium

**Output:** `ADP/logs/noise_stress_test/`

**Key metrics to report:**
- Mean gap vs σ curve per (profile, model) — the "breaking point" chart
- Gap distribution (box plots) at each σ level
- Speed ratio (guided beam / exact DP) — should remain constant under noise since noise only affects costs

---

### Experiment B — Profile Complexity Sweep

**Script:** `scripts/exp_profile_sweep.sh`  
**Goal:** Quantify how profile complexity affects approximation quality and cross-profile generalization

#### Profile grid

| Profile | Style |
|---------|-------|
| `daily_tou` | Baseline: 3-tier, smooth (existing) |
| `complex_5tier` | 5 tiers, irregular durations |
| `jagged` | Rapid alternation, hard regime detection |
| `double_peak` | Non-unimodal structure |
| `generate_data` (seed=20260109) | Randomly generated, integer prices ∈ [1,8] |

#### Training setup

| Train size | Seeds | N range | D range | Target util |
|------------|-------|---------|---------|-------------|
| Small | 0–999 | 20–60 | 2–4 | 0.80 |
| Medium | 0–99 | 100–200 | 5–15 | 0.85 |

**Reuse for daily_tou + medium:** Load `ADP/Data/Pooled Medium Daily Optimal Path Training Data.npz` via `--load-pooled-data`. Skip data collection entirely.

#### Evaluation setup

| Eval category | Seeds | N range | D range | DP time limit |
|---------------|-------|---------|---------|---------------|
| Small | 100–129 | 20–60 | 2–4 | 5s |
| Medium | 400–429 | 100–200 | 5–15 | 30s |
| Large | 500–519 | 250–500 | 10–30 | 60s |

- Price mode: **deterministic** only (noise is handled by Experiment A)
- Models: poly (deg 2), poly (deg 3), mlp, lgbm
- Beams: [2, 5, 10]

**Output:** `ADP/logs/profile_sweep/`

**Key metrics:**
- Cross-size generalization table (train ↔ eval size pairs)
- Per-profile R² and MAE of the value function model
- Optimality gap per (profile, model, train-size, eval-size) combination

---

### Experiment C — Polynomial Degree Comparison

**Script:** `scripts/exp_poly_degree.sh`  
**Goal:** Determine whether higher-degree polynomials improve approximation quality meaningfully, and at what cost

#### Degree configurations

| Config | Degree | L2 reg | X-noise | Dropout |
|--------|--------|--------|---------|---------|
| `poly2_std` | 2 | 1e-3 | 0.0 | 0.0 |
| `poly2_gen` | 2 | 1e-1 | 0.02 | 0.10 |
| `poly3_std` | 3 | 1e-3 | 0.0 | 0.0 |
| `poly3_gen` | 3 | 1e-1 | 0.02 | 0.10 |
| `poly4_std` | 4 | 1e-3 | 0.0 | 0.0 (if feasible) |

- Profiles: `daily_tou`, `complex_5tier`
- Train on small → eval small, medium, large
- Train on medium → eval medium, large
- Beams: [2, 5, 10]
- Also run `forecast_realized` eval with σ=0.5, ρ=0.9, moderate spikes for all degree configs

**Output:** `ADP/logs/poly_degree/`

**Key metrics to report:**
- R² and MAE on training/test split (model quality)
- Optimality gap per eval size
- Model fitting time
- Inference time per beam step (critical: must remain faster than exact DP)
- Memory usage for degree 3 and 4

**Decision threshold:** Degree 3 is preferred over 2 only if it reduces mean gap by >10% (relative) on medium and large instances without exceeding a 3× inference time penalty.

---

### Experiment D — Combined Noise + Profile Stress

**Script:** `scripts/exp_noise_profile_combined.sh`  
**Goal:** Find the practical operating envelope — where does the method fail under both hard profiles and hard noise?

#### Configuration

- Profiles: `complex_5tier`, `jagged`, `double_peak`
- σ: 1.0, 2.0, 4.0
- ρ: 0.9
- Spikes: extreme (prob=0.05, mag=4.0, dur=3)
- Models: poly (deg 2), poly (deg 3), lgbm
- Train on small → eval on medium; train on medium → eval on large
- Beams: [2, 5, 10]

**Output:** `ADP/logs/noise_profile_combined/`

**Key metric:** Gap heatmap (profile × σ) — visualize the "zone of competence" of the method.

---

## Master Orchestrator

**Script:** `scripts/run_all_rigorous_experiments.sh`

```
Experiment B (Profile sweep) → Experiment C (Poly degree) → Experiment A (Noise stress) → Experiment D (Combined)
```

Features:
- `RESUME=1` — checks for existing output CSVs and skips completed runs
- Per-experiment skip flags: `SKIP_PROFILE=1`, `SKIP_DEGREE=1`, `SKIP_NOISE=1`, `SKIP_COMBINED=1`
- Auto-detects pre-built medium pooled NPZ and passes `--load-pooled-data` accordingly
- Aggregates all output CSVs into a unified summary at the end

---

## Analysis Script

**Script:** `scripts/analyze_experiments.py`

Reads all output CSVs from `ADP/logs/{noise_stress_test,profile_sweep,poly_degree,noise_profile_combined}/` and produces:

| Output | Description |
|--------|-------------|
| **Gap vs σ curve** | Per (profile, model) — the "breaking point" visualization |
| **Profile complexity bar chart** | Mean gap grouped by profile, per train/eval size pair |
| **Poly degree tradeoff table** | R², gap, fitting time, inference time per degree |
| **Cross-size generalization matrix** | Train-size × eval-size → mean gap heatmap |
| **Noise × profile heatmap** | 2D heatmap showing zone of competence |
| **Speed ratio table** | Speedup vs exact DP per beam width and instance size |

Plots saved as PDF and PNG to `ADP/logs/analysis/`.

---

## Verification Steps

1. **Smoke test** (after code changes): Train poly deg 2 and deg 3 on 5 seeds → eval 3 seeds. Verify deterministic gap = 0% (matches existing behavior).
2. **Regression check**: Re-run train-small / eval-medium with `daily_tou` + poly deg 2 and verify identical results to the attached logs.
3. **Noise monotonicity check**: Gap should weakly increase with σ for all (profile, model) combos — flag anomalies.
4. **Degree 4 feasibility gate**: If fitting time > 60s or matrix memory > 32GB, auto-skip and log a warning.

---

## Compute Estimates

| Experiment | Runs | Est. time/run | Total estimate |
|------------|------|--------------|----------------|
| Profile sweep (B) — small training | 5 profiles × 4 models | ~15 min | ~5 h |
| Profile sweep (B) — medium training | 4 new profiles × 4 models | ~60 min | ~16 h |
| Poly degree (C) | 5 configs × 2 profiles × 2 train sizes | ~30–90 min | ~10 h |
| Noise stress (A) | 36 noise configs × 2 profiles × 2 train sizes × 4 models | ~2–5 min (eval only) | ~10–24 h |
| Combined (D) | 3 profiles × 3 σ × 2 train sizes × 3 models | ~10 min | ~5 h |
| **Total** | | | **~46–60 h** |

**Recommended:** Run on the 754GB RAM machine with `WORKERS=8`, using `NOHUP=1` mode.  
Experiments A and D can be parallelized (nested GNU parallel over noise configs) to reduce wall time.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Full factorial noise sweep (σ × ρ × spike) | Required for paper rigor — σ-only sweep would hide interaction effects |
| Poly degree up to 4 (with feasibility gate) | Degree 3 likely sweet spot; degree 4 establishes upper bound on the degree→quality curve |
| Skip linear model | Consistently weakest performer; compute budget better spent on degree sweep |
| Skip large-size training | Data generation and DP solving is prohibitively expensive; small and medium cover the generalization story |
| Reuse pre-built medium pooled data | No benefit to regenerating identical data; saves ~10h of compute for `daily_tou` medium experiments |
| Deterministic eval for profile sweep | Cleanly separates profile complexity axis from noise axis (Experiment A handles noise) |
| `--custom-prices` CLI flag | Enables future ad-hoc experimentation without any Python code changes |

---

## Implementation Order

1. [ ] **Code changes** — new profiles, `--custom-prices`, `--poly-degree`, `--mlp-hidden`
2. [ ] **Smoke test** — verify no regressions after code changes
3. [ ] **Experiment B scripts** — profile sweep (longest to train; start first)
4. [ ] **Experiment C scripts** — poly degree (depends on code changes)
5. [ ] **Experiment A scripts** — noise stress (uses pre-trained checkpoints from B/C; runs last)
6. [ ] **Experiment D scripts** — combined stress (subset of A configs with new profiles)
7. [ ] **Master orchestrator** — wires everything together with resume/skip logic
8. [ ] **Analysis script** — reads all CSVs, produces paper-ready tables and figures
