# Rigorous Experimental Evaluation Plan: Learning-Accelerated ADP for Single-Machine TOU Scheduling

**Date:** February 27, 2026  
**Status:** Planning phase — not yet implemented  
**Revision:** v2 — Major pivot: ditched degree-3/4 polynomials; added LASSO/ElasticNet, richer degree-2 features, batch-vectorized beam DP for fair timing comparison.

---

## Table of Contents

1. [Overview](#overview)
2. [Code Changes Required](#code-changes-required)
   - [Change 1 — New Pricing Profiles + Custom CLI](#change-1--new-pricing-profiles--custom-cli)
   - [Change 2 — New Features for phi_for_state()](#change-2--new-features-for-phi_for_state)
   - [Change 3 — ElasticNet / LASSO Model](#change-3--elasticnet--lasso-model)
   - [Change 4 — Batch-Vectorized Beam DP (Fair Comparison)](#change-4--batch-vectorized-beam-dp-fair-comparison)
3. [Experiments](#experiments)
   - [Experiment A — Noise Stress Test](#experiment-a--noise-stress-test)
   - [Experiment B — Profile Complexity Sweep](#experiment-b--profile-complexity-sweep)
   - [Experiment C — Regularization and Feature Enrichment Comparison](#experiment-c--regularization-and-feature-enrichment-comparison)
   - [Experiment D — Combined Noise + Profile Stress](#experiment-d--combined-noise--profile-stress)
4. [Scripts](#scripts)
5. [Analysis](#analysis)
6. [Verification Steps](#verification-steps)
7. [Compute Estimates](#compute-estimates)
8. [Design Decisions](#design-decisions)
9. [Implementation Order](#implementation-order)

---

## Overview

Systematically evaluate the learning-accelerated beam-pruned DP method across four axes:

1. **Pricing profile complexity** — from simple `daily_tou` to jagged, multi-peak, and randomly-generated profiles
2. **Forecast noise intensity** — full-factorial sweep of sigma x rho x spike to find the *breaking point*
3. **Feature engineering + regularization** — enable existing hidden features, add new ones, compare Ridge vs LASSO vs ElasticNet
4. **Fair speed comparison** — batch-vectorize the beam DP inference path so timing comparison with exact DP is apples-to-apples

**Training sizes:** small and medium only (large training skipped as per scope).  
**Evaluation sizes:** small, medium, and large for all trained models.  
**Key reuse:** The pre-built medium pooled dataset at `ADP/Data/Pooled Medium Daily Optimal Path Training Data.npz` is reused for all `daily_tou` medium-trained experiments.

### What changed from v1

| Item | v1 | v2 | Reason |
|------|----|----|--------|
| Polynomial degree | Sweep deg 2, 3, 4 | **Degree 2 only** | deg-3 = 17K features, 2.4 GB accumulator, marginal gain. Better to enrich base features. |
| Regularization | Ridge only | **Ridge + LASSO + ElasticNet** | LASSO zeros out useless polynomial interactions; ElasticNet balances both. |
| Features | Defaults only (19+2K dims) | **All optional flags ON + new features** | 4 existing feature flags are coded but off by default; additional features are either computed-but-dropped or trivially derivable. |
| Beam DP speed | Python dicts + serial vhat | **Batch vectorized decode, phi, predict + np.argpartition** | Current comparison is structurally unfair (exact DP uses Cython C hash maps; beam DP uses pure Python). |
| MLP hidden CLI | Planned | **Deferred** | Low priority, not needed for core story. |

---

## Code Changes Required

### Change 1 — New Pricing Profiles + Custom CLI

**File:** `sandbox/eval_pooled_vhat.py`

#### 1a. New named profiles

Add profile functions. The 20-element price vectors are:

| Profile name | Description | 20-element price vector |
|---|---|---|
| `complex_5tier` | 5 tiers, irregular durations, valley, ramp, sharp peak, descent | `[1,1,1.5,1.5,2,2,2,3,3,3,4,5,5,4,3,2.5,2.5,2,1.5,1]` |
| `jagged` | Rapid alternation between low and high, tests regime-switching | `[1,3,1,3,2,4,2,4,1,5,1,5,2,4,2,4,1,3,1,3]` |
| `double_peak` | Two separate peaks per day (non-unimodal) | `[1,1,2,4,4,2,1,1,2,3,5,5,3,2,1,1,2,4,2,1]` |

**Where to add (exact locations):**

1. **Profile constants** — Add around line 30-60 of `eval_pooled_vhat.py` (where imports and constants live). Define:
   ```python
   _NAMED_PROFILES = {
       "daily_tou": [3,3,3,3,3,1,1,1,1,2,2,2,5,5,5,5,3,3,3,3],  # existing
       "complex_5tier": [1,1,1.5,1.5,2,2,2,3,3,3,4,5,5,4,3,2.5,2.5,2,1.5,1],
       "jagged": [1,3,1,3,2,4,2,4,1,5,1,5,2,4,2,4,1,3,1,3],
       "double_peak": [1,1,2,4,4,2,1,1,2,3,5,5,3,2,1,1,2,4,2,1],
   }
   ```

2. **CLI `--daily-price-profile` choices** — Currently at line ~1420, update to:
   ```python
   ap.add_argument(
       "--daily-price-profile",
       type=str,
       default="daily_tou",
       choices=["daily_tou", "generate_data", "complex_5tier", "jagged", "double_peak", "custom"],
   )
   ```

3. **CLI `--custom-prices`** — Add new argument right after `--daily-price-profile`:
   ```python
   ap.add_argument(
       "--custom-prices",
       type=str,
       default="",
       help="Comma-separated 20-element float vector for custom profile.",
   )
   ```

4. **Profile dispatch** — Around line 1585, where `daily_prices_20` is resolved. Currently:
   ```python
   daily_prices_20: List[float] | None = None
   if str(args.daily_price_profile).strip().lower() == "generate_data":
       daily_prices_20 = _make_generate_data_daily_prices(...)
   ```
   Replace with:
   ```python
   daily_prices_20: List[float] | None = None
   prof = str(args.daily_price_profile).strip().lower()
   if prof == "generate_data":
       daily_prices_20 = _make_generate_data_daily_prices(
           seed=int(args.gd_seed), T=20,
           Tk_choices=(2, 3, 5),
           ck_low=int(args.gd_ck_low), ck_high=int(args.gd_ck_high),
       )
   elif prof == "custom":
       daily_prices_20 = [float(x) for x in str(args.custom_prices).split(",")]
       assert len(daily_prices_20) == 20, f"--custom-prices must have 20 elements, got {len(daily_prices_20)}"
   elif prof in _NAMED_PROFILES:
       daily_prices_20 = list(_NAMED_PROFILES[prof])
   # else: daily_prices_20 stays None -> default daily_tou behavior via build_instance()
   ```

**Testing:** `python sandbox/eval_pooled_vhat.py --daily-price-profile complex_5tier --train-seeds 0-4 --eval-seeds 100-102 --N 30 --D 3 --model-type poly` — verify no crash + sane gaps.

---

### Change 2 — New Features for phi_for_state()

**File:** `solvers/vhat_linear.py`  
**Function:** `phi_for_state()` (lines 37-215)  
**Dataclass:** `FeatureSpec` (lines 11-31)

#### Current state of features

The function currently builds features in this order with these dimensions:

| Block | Features | Dim | Notes |
|-------|----------|-----|-------|
| Bias | 1.0 | 1 | Always on |
| Regime one-hot | reg_oh[0..2] | 3 | Dimensionless |
| Distance to next off/cheap | d_off, d_cheap | 2 | Bounded by H=20 |
| Workload summary | N, W, R, S_pos (div norm) | 4 | Core features |
| Meta (optional, OFF) | log1p(T), log1p(N), log1p(W), util, slack_ratio | 5 | `spec.include_meta` |
| Slack-regime interaction | S*off, S*peak | 2 | |
| Cheap capacity | c_off, c_peak, pressure_off, pressure_cheap | 4 | **c_sh computed but DROPPED** |
| Bins (optional, ON) | short, long | 2 | |
| Len histogram (optional, OFF) | hist[1..pmax] | pmax | `spec.include_len_hist` |
| Price shape (optional, OFF) | mean,std,min,max + 3*(cos,sin) | 10 | `spec.include_price_shape` |
| Per-class counts (ON) | remaining[0..K-1] | K | |
| Per-class now-cost (ON) | cost_now*nk for each class + agg | K+1 | |

**Default ON:** bias(1) + regime(3) + distance(2) + workload(4) + slack_interact(2) + capacity(4) + bins(2) + per_class_counts(K) + per_class_now_cost(K+1) = **19 + 2K**  
**All flags ON:** + meta(5) + len_hist(pmax) + price_shape(10) = **34 + 2K + pmax**

#### 2a. Fix dropped feature: c_sh

**Bug location:** `solvers/vhat_linear.py` lines ~84-90 and ~128.

The variable `c_sh` (shoulder regime count from `ctx.count_regime[1, t]`) is computed on line ~87:
```python
c_sh = float(int(ctx.count_regime[1, t]))
```
But it is NEVER appended to `feats`. Only `c_off` and `c_peak` are appended at line ~128:
```python
feats.extend([c_off / norm, c_peak / norm, pressure_off, pressure_cheap])
```

**Fix:** Change line ~128 to:
```python
feats.extend([c_off / norm, c_sh / norm, c_peak / norm, pressure_off, pressure_cheap])
```

This adds 1 dimension to the feature vector.

#### 2b. Add new features

All these values are **already available** in `phi_for_state()` scope — no new data sources needed.

| # | Feature name | Expression | Rationale |
|---|---|---|---|
| 1 | `c_sh / norm` | Already computed, just dropped (see 2a above) | Shoulder capacity affects scheduling flexibility |
| 2 | `h_frac` | `h / H` where `h = t % ctx.H` | Continuous hour-of-day position (complements regime one-hot) |
| 3 | `days_remaining` | `(T - t) / H` | Number of full days remaining — useful for multi-day horizons |
| 4 | `budget_ratio` | `W / (R + 1.0)` | Utilization of remaining horizon (different from pressure_* which uses regime counts) |
| 5 | `current_price / mean_price` | `ctx.day[h] / (np.mean(ctx.day) + 1e-8)` | Price signal: is current slot expensive? |
| 6 | `min_class_slack` | `max(0, S - max(lengths) * max(remaining)) / norm` | Tightest margin — proxy for urgency |

#### 2c. Implementation plan

1. **Add `include_extra` flag to `FeatureSpec`** — `solvers/vhat_linear.py` line ~11-31:
   ```python
   @dataclass
   class FeatureSpec:
       include_per_class_counts: bool = True
       include_per_class_now_cost: bool = True
       include_bins: bool = True
       normalize: bool = False
       include_len_hist: bool = False
       pmax_for_hist: int = 12
       include_price_shape: bool = False
       include_meta: bool = False
       include_extra: bool = False  # <<< NEW FIELD
   ```

2. **Add new feature block** in `phi_for_state()` after the capacity block (~line 130), before the bins block:
   ```python
   # Extra features (continuous time, budget, price signal, urgency)
   if spec.include_extra:
       h_frac = float(h) / float(max(ctx.H, 1))
       days_rem = float(T - t) / float(max(ctx.H, 1))
       budget_ratio = float(W) / (R + 1.0)
       day_arr = np.asarray(ctx.day, dtype=np.float64)
       cur_price_ratio = float(day_arr[h]) / (float(np.mean(day_arr)) + 1e-8)
       max_len = int(np.max(lengths_arr)) if int(lengths_arr.size) > 0 else 1
       max_rem = int(np.max(remaining)) if int(remaining.size) > 0 else 0
       min_class_slack = max(0.0, float(S) - float(max_len * max_rem)) / norm
       feats.extend([h_frac, days_rem, budget_ratio, cur_price_ratio, min_class_slack])
   ```

3. **Add CLI flag** in `sandbox/eval_pooled_vhat.py` (after line ~1510, near other `--feat-*` args):
   ```python
   ap.add_argument("--feat-extra", type=int, default=0,
                    help="1 = include extra features (h_frac, days_rem, budget_ratio, etc.)")
   ```

4. **Wire the flag** in the FeatureSpec construction block (~line 1622). Extend the existing `if bool(args.feat_len_hist) or ...` block:
   ```python
   if bool(args.feat_len_hist) or bool(args.feat_price_shape) or bool(args.feat_meta) or bool(getattr(args, 'feat_extra', 0)):
       spec = FeatureSpec(
           include_per_class_counts=spec.include_per_class_counts,
           include_per_class_now_cost=spec.include_per_class_now_cost,
           include_bins=spec.include_bins,
           normalize=spec.normalize,
           include_len_hist=(spec.include_len_hist or bool(args.feat_len_hist)),
           pmax_for_hist=pmax_for_hist,
           include_price_shape=(spec.include_price_shape or bool(args.feat_price_shape)),
           include_meta=(spec.include_meta or bool(args.feat_meta)),
           include_extra=(spec.include_extra or bool(getattr(args, 'feat_extra', 0))),
       )
   ```

**Resulting dimensions with ALL flags ON including include_extra:**

| Config | Raw features (K=10, pmax=12) | Poly deg-2 features |
|--------|------------------------------|---------------------|
| Default (current) | 39 | ~780 |
| All flags ON (no extra) | 54 | ~1,485 |
| All flags ON + extra | 59 | ~1,770 |
| Transferable + all flags | 39 | ~780 |

All comfortably within Ridge/LASSO capacity at degree 2.

---

### Change 3 — ElasticNet / LASSO Model

**Files:** `solvers/vhat_models.py`, `sandbox/eval_pooled_vhat.py`

#### Current state

The only polynomial fitting is Ridge regression:
- `_stream_fit_poly_ridge()` in `sandbox/eval_pooled_vhat.py` (line 650) — chunked normal equations: `A += X_poly.T @ X_poly`, `b += X_poly.T @ y`, then `A += l2 * I; np.linalg.solve(A, b)`
- `fit_poly_ridge()` in `solvers/vhat_models.py` (line 213) — in-memory version, also Ridge only

Neither supports LASSO or ElasticNet. There is no `sklearn` import anywhere in the codebase.

#### 3a. Add ElasticNetPolyValueModel class to vhat_models.py

**Insert after** `PolyRidgeValueModel` class (around line 240):

```python
@dataclass
class ElasticNetPolyValueModel:
    """Polynomial value function with ElasticNet (L1+L2) regularization.

    Uses sklearn.linear_model.ElasticNet (or Lasso when l1_ratio=1.0).
    Produces sparse weights — zeroes out irrelevant polynomial interactions.
    """
    weights: np.ndarray      # shape (d_poly,)
    intercept: float
    powers: np.ndarray       # shape (n_terms, d_in)
    spec: FeatureSpec
    H: int = 20
    l1_ratio: float = 0.5    # 1.0 = pure LASSO, 0.0 = pure Ridge

    def predict_from_used(self, *, t, used, totals, lengths, ctx) -> float:
        phi = phi_for_state(t=t, used=used, totals=totals, lengths=lengths,
                            ctx=ctx, spec=self.spec)
        x_poly = _poly_expand_single(phi, self.powers)
        return float(np.dot(self.weights, x_poly) + self.intercept)

    def predict_batch(self, X_raw: np.ndarray) -> np.ndarray:
        """Vectorized prediction on (N, d_raw) feature matrix."""
        X_poly = _poly_expand_batch(X_raw, self.powers)
        return X_poly @ self.weights + self.intercept
```

#### 3b. Add fit_elasticnet() function to vhat_models.py

**Insert after** `fit_poly_ridge()` (after line ~240):

```python
def fit_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1e-3,
    l1_ratio: float = 0.5,
    spec: FeatureSpec,
    H: int = 20,
    train_frac: float = 0.85,
    split_seed: int = 42,
    max_iter: int = 5000,
) -> Tuple["ElasticNetPolyValueModel", Dict[str, float]]:
    """Fit degree-2 polynomial with ElasticNet regularization.

    Args:
        alpha: Overall regularization strength (sklearn convention).
        l1_ratio: 1.0 = pure LASSO, 0.5 = elastic net, 0.0 = pure Ridge.
    """
    from sklearn.linear_model import ElasticNet as _ElasticNet

    d_in = int(X.shape[1])
    powers = _poly_powers_degree2(d_in)

    # Train/test split
    n = int(X.shape[0])
    rng = np.random.default_rng(int(split_seed))
    mask = rng.random(n) < float(train_frac)

    X_poly = _poly_expand_batch(X, powers)
    X_train, y_train = X_poly[mask], y[mask]
    X_test, y_test = X_poly[~mask], y[~mask]

    model = _ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio),
                        max_iter=int(max_iter), fit_intercept=True)
    model.fit(X_train, y_train)

    w = np.asarray(model.coef_, dtype=np.float64)
    intercept = float(model.intercept_)

    # Metrics
    yhat_train = X_train @ w + intercept
    yhat_test = X_test @ w + intercept
    sse_train = float(np.sum((y_train - yhat_train) ** 2))
    sse_test = float(np.sum((y_test - yhat_test) ** 2))
    var_train = float(np.sum((y_train - np.mean(y_train)) ** 2))
    var_test = float(np.sum((y_test - np.mean(y_test)) ** 2))

    n_nonzero = int(np.count_nonzero(w))
    n_total = int(w.shape[0])

    metrics = {
        "r2_train": 1.0 - sse_train / max(var_train, 1e-12),
        "r2_test": 1.0 - sse_test / max(var_test, 1e-12),
        "mae_train": float(np.mean(np.abs(y_train - yhat_train))),
        "mae_test": float(np.mean(np.abs(y_test - yhat_test))),
        "n_nonzero": float(n_nonzero),
        "n_total_features": float(n_total),
        "sparsity": 1.0 - float(n_nonzero) / float(n_total),
    }

    vm = ElasticNetPolyValueModel(
        weights=w, intercept=intercept, powers=powers,
        spec=spec, H=H, l1_ratio=float(l1_ratio),
    )
    return vm, metrics
```

**Note:** `sklearn` must be added to `requirements.txt`. It is NOT currently a dependency.

#### 3c. Wire ElasticNet/LASSO in eval_pooled_vhat.py CLI and training dispatch

1. **`--model-type` choices** (line ~1370): Add `"elasticnet"` and `"lasso"`:
   ```python
   choices=["linear", "poly", "mlp", "poly_mlp", "factored_mlp", "lgbm",
            "mlp_hist", "mlp_price", "mlp_meta", "mlp_all",
            "elasticnet", "lasso"],
   ```

2. **New CLI args** (after line ~1400):
   ```python
   ap.add_argument("--elasticnet-alpha", type=float, default=1e-3,
                    help="ElasticNet/LASSO regularization strength (sklearn alpha).")
   ap.add_argument("--elasticnet-l1-ratio", type=float, default=0.5,
                    help="L1 ratio: 1.0=LASSO, 0.5=ElasticNet, 0.0=approx Ridge.")
   ```

3. **Model dispatch** — In the training block where `model_type == "poly"` is handled (around line 1750-1800), add an `elif` branch:
   ```python
   elif model_type in ("elasticnet", "lasso"):
       from solvers.vhat_models import fit_elasticnet, ElasticNetPolyValueModel
       l1_ratio = 1.0 if model_type == "lasso" else float(args.elasticnet_l1_ratio)
       vm, fit_metrics = fit_elasticnet(
           X_all, y_all,
           alpha=float(args.elasticnet_alpha),
           l1_ratio=l1_ratio,
           spec=spec, H=int(args.H),
       )
       log(f"ElasticNet fit: R2={fit_metrics['r2_test']:.4f}, "
           f"sparsity={fit_metrics['sparsity']:.1%} "
           f"({int(fit_metrics['n_nonzero'])}/{int(fit_metrics['n_total_features'])} nonzero)")
   ```

4. **vhat closure** (line ~2580) — The `ElasticNetPolyValueModel` has the same `predict_from_used()` interface as `PolyRidgeValueModel`, so **no change needed** to the existing vhat closure. It calls `model.predict_from_used(...)` which works for all model types.

5. **Add `scikit-learn` to `requirements.txt`:**
   ```
   scikit-learn>=1.3
   ```

#### 3d. Streaming ElasticNet variant (optional, lower priority)

The current `_stream_fit_poly_ridge()` uses chunked normal equations — this only works for Ridge (closed-form). ElasticNet requires iterative optimization.

**Options:**
- **Option A (simple, recommended):** Load all data into memory, call `sklearn.ElasticNet.fit()`. Works for small/medium instances (at most about 1M samples x 2K features = ~16 GB).
- **Option B (scalable):** Use `sklearn.linear_model.SGDRegressor(penalty='elasticnet')` with `partial_fit()` in chunks. Requires tuning learning rate.

**Start with Option A.** Only implement Option B if memory is an issue.

---

### Change 4 — Batch-Vectorized Beam DP (Fair Comparison)

**Problem statement:** The current timing comparison is structurally unfair.

| Solver | Language | Data structure | Inference |
|--------|----------|---------------|-----------|
| **Exact DP** | Cython C (`solvers/_sparse_dp_cython.pyx`) | C open-addressing hash map (`CStateMap`, 46 B/state) | N/A — no vhat calls |
| **Beam DP** | Pure Python (`solvers/optimal_benchmark_dp_numba.py` line 637) | Python `Dict[int, Tuple[float, int]]` (~260 B/entry) | Serial: per-state `vhat(t, state)` -> `phi_for_state()` -> `model.predict_from_used()` |

The beam DP is 5-6x slower than it needs to be, purely due to Python overhead, not the algorithm itself.

**Key fact:** `solvers/_sparse_dp_cython.pyx` contains ONLY exact DP functions. There is NO beam/guided DP in Cython. Confirmed by grep — only `solve_sparse_dp_cython()` exists, no beam variant.

**Files to modify:**
- `solvers/vhat_linear.py` — add batch feature extraction
- `solvers/vhat_models.py` — add batch predict to all model classes
- `solvers/optimal_benchmark_dp_numba.py` — refactor beam pruning step
- `solvers/optimal_benchmark_dp.py` — pass vhat_batch through dispatcher
- `sandbox/eval_pooled_vhat.py` — construct vhat_batch closure

#### 4a. Batch feature extraction: phi_for_states_batch()

**File:** `solvers/vhat_linear.py`  
**Insert after** `phi_for_state()` function (after line 215).

This function computes features for B states at the same time step t. The key insight is that all scalar context (t, h, regime, prices) is shared — only remaining job counts vary across the batch.

```python
def phi_for_states_batch(
    *,
    t: int,
    used_batch: np.ndarray,     # shape (B, K), dtype int32
    totals: np.ndarray,          # shape (K,)
    lengths: np.ndarray,         # shape (K,)
    ctx: "TOUFeatureContext",
    spec: FeatureSpec,
) -> np.ndarray:
    """Vectorized feature extraction for B states at the same time step t.

    Returns float64 array of shape (B, d_feat).

    Invariant: phi_for_states_batch(...)[i] == phi_for_state(t, used_batch[i], ...)
    """
    B = int(used_batch.shape[0])
    K = int(totals.shape[0])
    T = int(ctx.T)
    t = max(0, min(int(t), T))
    H = int(ctx.H)
    norm = float(T) if spec.normalize else 1.0

    lengths_arr = np.asarray(lengths, dtype=np.int32)
    totals_np = np.asarray(totals, dtype=np.int32)

    remaining = totals_np[np.newaxis, :] - used_batch  # (B, K)

    N = np.sum(remaining, axis=1).astype(np.float64)                           # (B,)
    W = np.sum(remaining * lengths_arr[np.newaxis, :], axis=1).astype(np.float64)  # (B,)
    R = float(T - t)
    S = R - W                                                                   # (B,)
    S_pos = np.maximum(S, 0.0)                                                  # (B,)

    h = int(t % H)
    reg = int(ctx.day_regime[h])

    # Scalar features — same for all states at this time step
    reg_oh = np.zeros(3, dtype=np.float64)
    if 0 <= reg < 3:
        reg_oh[reg] = 1.0

    c_off = float(int(ctx.count_regime[0, t]))
    c_sh  = float(int(ctx.count_regime[1, t]))
    c_peak = float(int(ctx.count_regime[2, t]))
    d_off = float(int(ctx.dist_to_next_off[h]))
    d_cheap = float(int(ctx.dist_to_next_cheap[h]))
    pressure_off   = W / (c_off + 1.0)              # (B,)
    pressure_cheap = W / (c_off + c_sh + 1.0)       # (B,)

    # Build column blocks as (B, d_i) arrays, then hstack at the end
    cols = []

    # Bias
    cols.append(np.ones((B, 1), dtype=np.float64))

    # Regime one-hot (broadcast scalar)
    cols.append(np.tile(reg_oh, (B, 1)))  # (B, 3)

    # Distances (broadcast scalar)
    cols.append(np.full((B, 2), [d_off, d_cheap], dtype=np.float64))

    # Workload summary
    cols.append(np.column_stack([N / norm, W / norm,
                                  np.full(B, R / norm), S_pos / norm]))  # (B, 4)

    # Meta (optional)
    if spec.include_meta:
        util = W / float(T) if T > 0 else np.zeros(B)
        slack_ratio = S_pos / (W + 1.0)
        cols.append(np.column_stack([
            np.full(B, float(np.log1p(float(T)))),
            np.log1p(N), np.log1p(W), util, slack_ratio,
        ]))  # (B, 5)

    # Slack-regime interactions
    cols.append(np.column_stack([
        (S_pos / norm) * reg_oh[0],
        (S_pos / norm) * reg_oh[2],
    ]))  # (B, 2)

    # Cheap capacity (with c_sh fix)
    cols.append(np.column_stack([
        np.full(B, c_off / norm),
        np.full(B, c_sh / norm),      # <<< was dropped in phi_for_state
        np.full(B, c_peak / norm),
        pressure_off, pressure_cheap,
    ]))  # (B, 5)

    # Extra features (optional)
    if spec.include_extra:
        h_frac = float(h) / float(max(H, 1))
        days_rem = float(T - t) / float(max(H, 1))
        budget_ratio = W / (R + 1.0)
        day_arr = np.asarray(ctx.day, dtype=np.float64)
        cur_price_ratio = float(day_arr[h]) / (float(np.mean(day_arr)) + 1e-8)
        max_len = int(np.max(lengths_arr)) if int(lengths_arr.size) > 0 else 1
        max_rem_per_row = np.max(remaining, axis=1).astype(np.float64)
        min_class_slack = np.maximum(0.0, S - float(max_len) * max_rem_per_row) / norm
        cols.append(np.column_stack([
            np.full(B, h_frac), np.full(B, days_rem),
            budget_ratio, np.full(B, cur_price_ratio),
            min_class_slack,
        ]))  # (B, 5)

    # Bins (optional)
    if spec.include_bins:
        short_mask = (lengths_arr <= 2)
        median_len = int(np.median(lengths_arr)) if int(lengths_arr.size) > 0 else 0
        long_thr = max(3, median_len)
        long_mask = (lengths_arr >= long_thr)
        short = np.sum(remaining[:, short_mask], axis=1).astype(np.float64) / norm
        long  = np.sum(remaining[:, long_mask],  axis=1).astype(np.float64) / norm
        cols.append(np.column_stack([short, long]))  # (B, 2)

    # Len histogram (optional)
    if spec.include_len_hist:
        pmax_h = int(max(1, int(spec.pmax_for_hist)))
        hist = np.zeros((B, pmax_h), dtype=np.float64)
        for k in range(K):
            L = int(lengths_arr[k])
            idx = min(max(L, 1), pmax_h) - 1
            hist[:, idx] += remaining[:, k].astype(np.float64)
        cols.append(hist / norm)  # (B, pmax_h)

    # Price shape (optional)
    if spec.include_price_shape:
        day = np.asarray(ctx.day, dtype=np.float64)
        shape_feats = [float(np.mean(day)), float(np.std(day)),
                       float(np.min(day)), float(np.max(day))]
        x_ang = np.arange(H, dtype=np.float64)
        for k in (1, 2, 3):
            ang = 2.0 * np.pi * float(k) * x_ang / float(H)
            shape_feats.append(float(np.mean(day * np.cos(ang))))
            shape_feats.append(float(np.mean(day * np.sin(ang))))
        cols.append(np.tile(np.array(shape_feats), (B, 1)))  # (B, 10)

    # Per-class counts (optional)
    if spec.include_per_class_counts:
        cols.append(remaining.astype(np.float64) / norm)  # (B, K)

    # Per-class now-cost (optional)
    if spec.include_per_class_now_cost:
        cost_now_per_class = np.zeros(K, dtype=np.float64)
        for k in range(K):
            L = int(lengths_arr[k])
            if L <= H:
                cost_now_per_class[k] = float(ctx.day_window_cost[L][h])
            else:
                cost_now_per_class[k] = float(get_day_window_cost(ctx, L)[h])
        # (B, K): remaining * cost_now  (broadcast)
        per_class = remaining.astype(np.float64) * cost_now_per_class[np.newaxis, :] / norm
        agg = np.sum(per_class, axis=1, keepdims=True)
        cols.append(np.hstack([per_class, agg]))  # (B, K+1)

    return np.hstack(cols)  # (B, d_feat)
```

**Critical invariant:** `phi_for_states_batch(t=t, used_batch=U)[i]` MUST equal `phi_for_state(t=t, used=U[i], ...)` for all i. Write a unit test that checks this with `np.allclose(atol=1e-10)`.

#### 4b. Batch predict for all model classes

**File:** `solvers/vhat_models.py`

Add `predict_batch(self, X_raw: np.ndarray) -> np.ndarray` to each model class:

1. **`PolyRidgeValueModel`** (line ~240):
   ```python
   def predict_batch(self, X_raw: np.ndarray) -> np.ndarray:
       """Vectorized prediction. X_raw: (B, d_raw) -> (B,) costs."""
       X_poly = _poly_expand_batch(X_raw, self.powers)
       return X_poly @ self.weights
   ```
   Note: `_poly_expand_batch` already exists (line 127) and handles batches efficiently.

2. **`MLPValueModel`** (line ~300):
   ```python
   def predict_batch(self, X_raw: np.ndarray) -> np.ndarray:
       """Vectorized MLP prediction."""
       import torch
       x = torch.tensor(X_raw, dtype=torch.float32)
       with torch.no_grad():
           return self.net(x).squeeze(-1).numpy()
   ```

3. **`ElasticNetPolyValueModel`** — already has `predict_batch()` in the class definition above (Change 3a).

4. **`LGBMValueModel`** (line ~1059):
   ```python
   def predict_batch(self, X_raw: np.ndarray) -> np.ndarray:
       X_poly = _poly_expand_batch(X_raw, self.powers)
       return self.booster.predict(X_poly)
   ```

#### 4c. Refactor beam pruning step

**File:** `solvers/optimal_benchmark_dp_numba.py`  
**Function:** `solve_sparse_dp_python_beam()` (line 637)

**Current pruning code** (lines ~745-755):
```python
if len(layer) > prune_threshold:
    items = list(layer.items())
    best_items = heapq.nsmallest(
        int(beam_width),
        items,
        key=lambda it: score_state(t, it[0], float(it[1][0]), int(it[1][1])),
    )
    layer = {s: (float(cp[0]), int(cp[1])) for s, cp in best_items}
    dp_layers[t] = layer
```

Where `score_state` calls `vhat(int(t), int(state))` — ONE Python function call per state, which internally calls `decode_used()` -> `phi_for_state()` -> `model.predict_from_used()`.

**New function signature** — add `vhat_batch` parameter:
```python
def solve_sparse_dp_python_beam(
    lengths: np.ndarray,
    totals: np.ndarray,
    prefix: np.ndarray,
    T: int,
    radices: np.ndarray,
    mult: np.ndarray,
    K: int,
    final_state: int,
    *,
    vhat: Optional[callable] = None,       # legacy: (t, state) -> float
    vhat_batch: Optional[callable] = None,  # NEW: (t, states_array) -> costs_array
    beam_width: int,
    prune_factor: float = 2.0,
    time_limit: float = -1.0,
    tie_break: str = "early",
) -> Tuple[float, int, Dict, bool, Optional[Tuple[int, int, float]]]:
```

**Refactored pruning block:**
```python
if len(layer) > prune_threshold:
    states = np.array(list(layer.keys()), dtype=np.int64)
    costs  = np.array([layer[s][0] for s in states], dtype=np.float64)
    pens   = np.array([layer[s][1] for s in states], dtype=np.int64)

    # Batch heuristic evaluation
    if vhat_batch is not None:
        h_vals = vhat_batch(t, states)  # (M,) float64 — vectorized
    elif vhat is not None:
        h_vals = np.array([vhat(int(t), int(s)) for s in states])  # fallback serial
    else:
        h_vals = np.zeros(len(states), dtype=np.float64)

    h_vals = np.where(np.isfinite(h_vals), h_vals, 0.0)
    scores = costs + h_vals

    # np.argpartition is O(M) vs O(M log B) for heapq.nsmallest
    if len(states) > beam_width:
        idx = np.argpartition(scores, beam_width)[:beam_width]
    else:
        idx = np.arange(len(states))

    layer = {int(states[i]): (float(costs[i]), int(pens[i])) for i in idx}
    dp_layers[t] = layer
```

**Note on tie-breaking:** The current code uses `(cost + h, pen)` as a tie-breaking key. The batch version above ignores pen in the sort (uses scores only). For most practical purposes this is fine — pen is a secondary criterion. If strict tie-break parity is needed, use `np.lexsort((pens[idx], scores[idx]))` after the argpartition.

#### 4d. Wire batch path through the call chain

**File:** `solvers/optimal_benchmark_dp.py`  
**Function:** `solve_optimal_benchmark_dp()` (line 518)

The guided beam dispatch is at lines 604-627. Currently:
```python
result = solve_sparse_dp_python_beam(
    ..., vhat=vhat, beam_width=beam_width, ...
)
```

Change to:
```python
result = solve_sparse_dp_python_beam(
    ..., vhat=vhat, vhat_batch=vhat_batch,
    beam_width=beam_width, ...
)
```

And add `vhat_batch=None` to `solve_optimal_benchmark_dp()` signature.

**File:** `sandbox/eval_pooled_vhat.py`  
**Location:** Around line 2580-2600, where the beam DP is called.

**Add vhat_batch closure** after the existing vhat closure:
```python
def vhat_batch_fn(t_: int, states_: np.ndarray) -> np.ndarray:
    """Batch cost-to-go estimate for multiple states at time t."""
    B = len(states_)
    # Vectorized state decoding
    used_batch = np.zeros((B, K), dtype=np.int32)
    x = states_.copy()
    for i in range(K):
        used_batch[:, i] = x % int(radices[i])
        x //= int(radices[i])
    # Batch feature extraction
    from solvers.vhat_linear import phi_for_states_batch
    X_raw = phi_for_states_batch(
        t=int(t_), used_batch=used_batch,
        totals=totals_arr, lengths=lengths_arr,
        ctx=ctx, spec=spec,
    )
    # Batch model prediction
    return model.predict_batch(X_raw)
```

Then pass it to the solve call:
```python
result = solve_optimal_benchmark_dp(
    ..., guided=True, vhat=vhat, vhat_batch=vhat_batch_fn,
    beam_width=beam, ...
)
```

#### Expected speedup from batch vectorization

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| State decoding | Python `decode_used()` per state | Vectorized numpy mod/div | ~50x |
| Feature extraction | Python list -> np.asarray per state | Batch numpy `phi_for_states_batch()` | ~20-50x |
| Model prediction | `predict_from_used()` per state | `predict_batch()` on full matrix | ~100x (poly), ~10x (MLP) |
| Pruning selection | `heapq.nsmallest` O(M log B) | `np.argpartition` O(M) | ~3-5x |
| **Total prune step** | ~600K Python calls / instance | ~T batch numpy ops | **~20-50x overall** |

This makes the guided beam DP timing competitive with or faster than exact Cython DP.

---

## Experiments

### Experiment A — Noise Stress Test

**Script:** `scripts/exp_noise_stress_test.sh`  
**Goal:** Find the noise level at which the method breaks (gap exceeds 1%, 2%, 5%)

#### Noise parameter grid (full factorial)

| Parameter | Values |
|-----------|--------|
| sigma (AR(1) stddev) | 0.0, 0.25, 0.5, 1.0, 2.0, 4.0 |
| rho (AR(1) correlation) | 0.5, 0.9 |
| Spike config | off (prob=0, mag=0), moderate (prob=0.02, mag=2.0, dur=2), extreme (prob=0.05, mag=4.0, dur=3) |

**Total noise configs:** 6 x 2 x 3 = **36 combinations**

#### Setup

- **No retraining** — noise is applied at evaluation time only; all trained checkpoints are reused
- Eval seeds: 400-429 (medium, 30 instances), 500-519 (large, 20 instances)
- Price mode: `forecast_realized`
- Beams: [2, 5, 10]
- Models: poly (Ridge), elasticnet, mlp, lgbm
- Profiles: `daily_tou`, `complex_5tier`
- Two pretrained checkpoints per combo: trained-on-small and trained-on-medium

**Output:** `ADP/logs/noise_stress_test/`

**Key metrics:**
- Mean gap vs sigma curve per (profile, model) — the "breaking point" chart
- Gap distribution (box plots) at each sigma level
- Speed ratio (guided beam / exact DP) — should remain constant under noise

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
| `generate_data` (seed=20260109) | Randomly generated, integer prices in [1,8] |

#### Training setup

| Train size | Seeds | N range | D range | Target util |
|------------|-------|---------|---------|-------------|
| Small | 0-999 | 20-60 | 2-4 | 0.80 |
| Medium | 0-99 | 100-200 | 5-15 | 0.85 |

**Reuse for daily_tou + medium:** Load `ADP/Data/Pooled Medium Daily Optimal Path Training Data.npz` via `--load-pooled-data`.

#### Evaluation setup

| Eval category | Seeds | N range | D range | DP time limit |
|---------------|-------|---------|---------|---------------|
| Small | 100-129 | 20-60 | 2-4 | 5s |
| Medium | 400-429 | 100-200 | 5-15 | 30s |
| Large | 500-519 | 250-500 | 10-30 | 60s |

- Price mode: **deterministic** only (noise handled by Experiment A)
- Models: poly (Ridge), elasticnet, lasso, mlp, lgbm
- Beams: [2, 5, 10]

**Output:** `ADP/logs/profile_sweep/`

---

### Experiment C — Regularization and Feature Enrichment Comparison

**Script:** `scripts/exp_regularization_features.sh`  
**Goal:** Compare Ridge vs LASSO vs ElasticNet on degree-2 polynomials with varying feature richness

#### Feature configurations

| Config name | CLI flags | Raw dim (K=10, pmax=12) | Approx Poly dim |
|-------------|-----------|-------------------------|-----------------|
| `default` | (none) | 39 | ~780 |
| `all_flags` | `--feat-len-hist 1 --feat-price-shape 1 --feat-meta 1` | 54 | ~1,485 |
| `all_extra` | `--feat-len-hist 1 --feat-price-shape 1 --feat-meta 1 --feat-extra 1` | 59 | ~1,770 |
| `transferable_all` | `--transferable-features --feat-len-hist 1 --feat-price-shape 1 --feat-meta 1 --feat-extra 1` | 39 | ~780 |

#### Regularization grid

| Model type | L2/alpha | L1 ratio | CLI args |
|------------|----------|----------|----------|
| `poly` (Ridge) | 1e-3 | 0.0 | `--model-type poly --l2 1e-3` |
| `poly` (Ridge strong) | 1e-1 | 0.0 | `--model-type poly --l2 1e-1` |
| `elasticnet` | 1e-3 | 0.5 | `--model-type elasticnet --elasticnet-alpha 1e-3 --elasticnet-l1-ratio 0.5` |
| `elasticnet` (strong) | 1e-2 | 0.5 | `--model-type elasticnet --elasticnet-alpha 1e-2 --elasticnet-l1-ratio 0.5` |
| `lasso` | 1e-3 | 1.0 | `--model-type lasso --elasticnet-alpha 1e-3` |
| `lasso` (strong) | 1e-2 | 1.0 | `--model-type lasso --elasticnet-alpha 1e-2` |

**Total configs:** 4 feature setups x 6 regularizations = **24 model variants**

#### Setup

- Train on small (seeds 0-999) -> eval on small (100-129), medium (400-429), large (500-519)
- Train on medium (seeds 0-99) -> eval on medium (400-429), large (500-519)
- Profile: `daily_tou` only (isolates regularization effect from profile complexity)
- Beams: [2, 5, 10]

**Output:** `ADP/logs/regularization_features/`

**Key metrics:**
- R-squared and MAE (model quality) by config
- Sparsity (% of zero weights) for LASSO/ElasticNet
- Optimality gap by eval size
- Generalization gap (train-R2 minus test-R2)
- Feature importance ranking from LASSO weights (which polynomial interactions survive?)

---

### Experiment D — Combined Noise + Profile Stress

**Script:** `scripts/exp_noise_profile_combined.sh`  
**Goal:** Find the practical operating envelope

#### Configuration

- Profiles: `complex_5tier`, `jagged`, `double_peak`
- sigma: 1.0, 2.0, 4.0
- rho: 0.9
- Spikes: extreme (prob=0.05, mag=4.0, dur=3)
- Models: poly (Ridge), elasticnet, lgbm
- Train on small -> eval on medium; train on medium -> eval on large
- Beams: [2, 5, 10]

**Output:** `ADP/logs/noise_profile_combined/`

**Key metric:** Gap heatmap (profile x sigma) — visualize the "zone of competence."

---

## Scripts

### Master Orchestrator

**Script:** `scripts/run_all_rigorous_experiments.sh`

Execution order:
```
Experiment C (Regularization) -> Experiment B (Profile) -> Experiment A (Noise) -> Experiment D (Combined)
```

Features:
- `RESUME=1` — checks for existing output CSVs and skips completed runs
- Per-experiment skip flags: `SKIP_REG=1`, `SKIP_PROFILE=1`, `SKIP_NOISE=1`, `SKIP_COMBINED=1`
- Auto-detects pre-built medium pooled NPZ and passes `--load-pooled-data`
- Aggregates all output CSVs into a unified summary at the end

### Per-experiment scripts

Each experiment script follows the same template:
```bash
#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="ADP/logs/<experiment_name>"
mkdir -p "$LOG_DIR"

RESUME="${RESUME:-0}"

for PROFILE in ...; do
  for MODEL in ...; do
    for TRAIN_SIZE in small medium; do
      OUT_CSV="$LOG_DIR/${PROFILE}_${MODEL}_train${TRAIN_SIZE}.csv"
      [ -f "$OUT_CSV" ] && [ "$RESUME" = "1" ] && echo "SKIP $OUT_CSV" && continue

      python sandbox/eval_pooled_vhat.py \
        --daily-price-profile "$PROFILE" \
        --model-type "$MODEL" \
        --train-seeds "$TRAIN_SEEDS" \
        --eval-seeds "$EVAL_SEEDS" \
        --beams "2,5,10" \
        --feat-len-hist 1 --feat-price-shape 1 --feat-meta 1 --feat-extra 1 \
        --out-csv "$OUT_CSV" \
        ... (size-specific args)
    done
  done
done
```

---

## Analysis

**Script:** `scripts/analyze_experiments.py`

Reads all output CSVs from `ADP/logs/` subdirectories and produces:

| Output | Description |
|--------|-------------|
| **Gap vs sigma curve** | Per (profile, model) — the "breaking point" visualization |
| **Profile complexity bar chart** | Mean gap grouped by profile, per train/eval size pair |
| **Regularization comparison table** | R-squared, gap, sparsity per (feature-config, regularization) |
| **Feature importance from LASSO** | Top-30 polynomial interactions by absolute weight |
| **Cross-size generalization matrix** | Train-size x eval-size -> mean gap heatmap |
| **Noise x profile heatmap** | 2D heatmap showing zone of competence |
| **Speed ratio table** | Speedup vs exact DP per beam width and instance size |
| **Before/after batch speedup** | Timing comparison: serial vs batch beam DP |

Plots saved as PDF and PNG to `ADP/logs/analysis/`.

---

## Verification Steps

1. **Unit test for batch features:** `phi_for_states_batch(t, used_batch)[i] == phi_for_state(t, used_batch[i])` for 100 random states. Use `np.allclose(atol=1e-10)`.
2. **Unit test for batch predict:** `model.predict_batch(X)[i] == model.predict_from_used(...)` for all model types.
3. **Smoke test:** Train poly (Ridge) + elasticnet on 5 seeds, eval 3 seeds. Verify deterministic gap is approximately 0% (matches existing behavior).
4. **Regression check:** Re-run train-small / eval-medium with `daily_tou` + poly (Ridge) and verify identical results to the attached logs.
5. **Noise monotonicity check:** Gap should weakly increase with sigma — flag anomalies.
6. **ElasticNet sanity:** With `l1_ratio=0.0` and `alpha=l2`, ElasticNet should approximately match Ridge results.
7. **Batch speedup measurement:** Time the beam DP step before and after vectorization on the same instance — expect at least 10x speedup.

---

## Compute Estimates

| Experiment | Runs | Est. time/run | Total estimate |
|------------|------|--------------|----------------|
| Regularization + features (C) | 24 configs x 2 train sizes | ~15-60 min | ~12-24 h |
| Profile sweep (B) — small training | 5 profiles x 5 models | ~15 min | ~6 h |
| Profile sweep (B) — medium training | 5 profiles x 5 models | ~60 min | ~25 h |
| Noise stress (A) | 36 noise x 2 profiles x 2 train x 4 models | ~2-5 min (eval only) | ~10-24 h |
| Combined (D) | 3 profiles x 3 sigma x 2 train x 3 models | ~10 min | ~5 h |
| **Total** | | | **~58-84 h** |

**Post batch-vectorization:** Eval times should drop significantly (beam DP is the bottleneck for large instances), potentially reducing total by 30-50%.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Ditch degree 3/4 polynomial** | Degree 3 = 17K features, 2.4 GB accumulator, 7.2 GB solve peak. Marginal quality gain. Better to enrich base features at degree 2. |
| **Add LASSO/ElasticNet** | LASSO zeroes out useless polynomial interactions — effective feature selection. With ~2K poly features, many cross-terms are noise; sparsity is desirable. |
| **Enable ALL optional feature flags** | 4 existing flags (meta, len_hist, price_shape, bins) are coded and tested but OFF in bash scripts. Free improvement. |
| **Add extra features** | c_sh is computed but dropped (bug). h_frac, days_rem, budget_ratio, cur_price/mean_price, min_class_slack are all derivable from existing context with zero new data. |
| **Batch vectorize beam DP** | Makes timing fair: exact DP uses Cython C hash maps (46 B/state); beam uses pure Python dicts (260 B/state) + 600K serial calls. After vectorization, both use optimized C/numpy. |
| **np.argpartition instead of heapq** | O(M) vs O(M log B) for pruning. With M=100K states and B=1000, 5x faster. |
| **sklearn for ElasticNet** | Standard, well-tested, supports warm starts. Not worth implementing coordinate descent from scratch. |
| **Keep legacy vhat(t, state) interface** | Backward compatibility. Batch path is additive. |
| **Full factorial noise sweep** | Required for rigor — sigma-only sweep hides interaction effects between correlation and spikes. |
| **Skip linear model** | Consistently weakest; compute budget better spent on regularization sweep. |

---

## File-by-File Change Summary

| File | Changes | Key lines | Priority |
|------|---------|-----------|----------|
| `solvers/vhat_linear.py` | Fix c_sh drop (line ~128), add include_extra to FeatureSpec (line ~11-31), add new feature block (after line ~130), add `phi_for_states_batch()` (after line 215) | 11-31, 84-90, 128, 215+ | **HIGH** |
| `solvers/vhat_models.py` | Add `ElasticNetPolyValueModel` class + `fit_elasticnet()` (after line ~240), add `predict_batch()` to PolyRidge (line ~240), MLP (line ~300), LGBM (line ~1059) | 240, 300, 1059 | **HIGH** |
| `sandbox/eval_pooled_vhat.py` | Add _NAMED_PROFILES dict (line ~30-60), extend --daily-price-profile choices (line ~1420), add --custom-prices/--feat-extra/--elasticnet-* CLI args (line ~1400-1510), add elasticnet training branch (line ~1750-1800), add vhat_batch closure (line ~2580) | 30, 1370, 1400, 1510, 1750, 2580 | **HIGH** |
| `solvers/optimal_benchmark_dp_numba.py` | Add vhat_batch param to function signature (line 637), refactor prune block (lines 745-755) | 637, 745-755 | **HIGH** |
| `solvers/optimal_benchmark_dp.py` | Add vhat_batch param to `solve_optimal_benchmark_dp()` (line 518), pass through to beam call (line ~606-627) | 518, 606-627 | MEDIUM |
| `requirements.txt` | Add `scikit-learn>=1.3` | End of file | LOW |
| `tests/test_batch_features.py` | Unit tests: phi_for_states_batch vs phi_for_state numerical equivalence | New file | MEDIUM |
| `tests/test_batch_predict.py` | Unit tests: predict_batch vs predict_from_used for all model types | New file | MEDIUM |

---

## Implementation Order

1. [ ] **Change 2a** — Fix c_sh bug in `vhat_linear.py` (1 line change)
2. [ ] **Change 2b+2c** — Add `include_extra` to FeatureSpec + new feature block in `phi_for_state()`
3. [ ] **Change 4a** — Add `phi_for_states_batch()` to `vhat_linear.py`
4. [ ] **Change 3a+3b** — Add `ElasticNetPolyValueModel` + `fit_elasticnet()` to `vhat_models.py`
5. [ ] **Change 4b** — Add `predict_batch()` to all model classes in `vhat_models.py`
6. [ ] **Change 1** — New profiles + custom CLI in `eval_pooled_vhat.py`
7. [ ] **Change 3c** — Wire ElasticNet/LASSO model type in `eval_pooled_vhat.py`
8. [ ] **Change 4c** — Refactor beam pruning in `optimal_benchmark_dp_numba.py`
9. [ ] **Change 4d** — Wire batch vhat in `eval_pooled_vhat.py` + `optimal_benchmark_dp.py`
10. [ ] **requirements.txt** — Add scikit-learn
11. [ ] **Unit tests** — batch features, batch predict
12. [ ] **Smoke test** — verify no regressions
13. [ ] **Experiment C scripts** — regularization + features
14. [ ] **Experiment B scripts** — profile sweep
15. [ ] **Experiment A scripts** — noise stress (eval only)
16. [ ] **Experiment D scripts** — combined stress
17. [ ] **Master orchestrator** — wires everything + resume logic
18. [ ] **Analysis script** — reads CSVs, produces figures
