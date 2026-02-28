#!/usr/bin/env python3
"""Smoke test for all code changes from EXPERIMENT_PLAN v2."""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

# ── 1. Feature extraction tests ──────────────────────────────────────
from PaST.solvers.vhat_linear import FeatureSpec, phi_for_state, phi_for_states_batch
from PaST.solvers.vhat_tou_features import build_tou_feature_context

prices = np.array([1.0, 2.0, 5.0, 3.0] * 5, dtype=np.float64)
ctx = build_tou_feature_context(prices, H=20, validate_repeating=True)
totals = np.array([2, 1], dtype=np.int32)
lengths = [2, 3]

spec_old = FeatureSpec(normalize=True, include_bins=True)
spec_new = FeatureSpec(normalize=True, include_bins=True, include_extra=True)

phi_old = phi_for_state(
    t=0, used=(0, 0), totals=totals, lengths=lengths, ctx=ctx, spec=spec_old
)
phi_new = phi_for_state(
    t=0, used=(0, 0), totals=totals, lengths=lengths, ctx=ctx, spec=spec_new
)

print(f"Old feature dim: {phi_old.shape[0]}")
print(f"New feature dim: {phi_new.shape[0]}")
extra = phi_new.shape[0] - phi_old.shape[0]
assert extra == 5, f"Expected 5 extra features, got {extra}"
print("[PASS] Extra features block adds exactly 5 features")

# ── 2. Batch extraction ──────────────────────────────────────────────
radices = totals + 1
used_cache = {}
states = [0, 1, 2]
X = phi_for_states_batch(
    t=0,
    states=states,
    totals=totals,
    lengths=lengths,
    ctx=ctx,
    spec=spec_new,
    radices=radices,
    used_cache=used_cache,
)
assert X.shape == (3, phi_new.shape[0]), f"Batch shape mismatch: {X.shape}"
print(f"[PASS] phi_for_states_batch returns shape {X.shape}")

# ── 3. Model classes ─────────────────────────────────────────────────
from PaST.solvers.vhat_linear import LinearRidgeValueModel
from PaST.solvers.vhat_models import (
    PolyRidgeValueModel,
    ElasticNetPolyValueModel,
    MLPValueModel,
    PolyMLPValueModel,
    FactoredMLPValueModel,
    LGBMValueModel,
    fit_elasticnet,
)

# Test predict_batch exists on all models
D = phi_new.shape[0]
dummy_X = np.random.randn(5, D)

lr = LinearRidgeValueModel(np.zeros(D), 0.0, spec_new)
out = lr.predict_batch(dummy_X)
assert out.shape == (5,), f"LinearRidge predict_batch shape: {out.shape}"
print("[PASS] LinearRidgeValueModel.predict_batch")

# ── 4. ElasticNet fit + predict ───────────────────────────────────────
X_train = np.random.randn(50, D)
y_train = np.random.randn(50)
w, powers, intercept = fit_elasticnet(X_train, y_train, alpha=1.0, l1_ratio=0.5)
print(
    f"[PASS] fit_elasticnet: w.shape={w.shape}, powers.shape={powers.shape}, intercept={intercept:.4f}"
)

en = ElasticNetPolyValueModel(
    weights=w, intercept=intercept, spec=spec_new, powers_=powers
)
pred = en.predict_batch(dummy_X)
assert pred.shape == (5,), f"ElasticNet predict_batch shape: {pred.shape}"
print("[PASS] ElasticNetPolyValueModel.predict_batch")

# ── 5. Named pricing profiles ────────────────────────────────────────
from PaST.sandbox.eval_pooled_vhat import _NAMED_PROFILES

expected_profiles = {"flat", "two_block", "ramp", "double_peak", "weekend_weekday"}
assert expected_profiles.issubset(
    set(_NAMED_PROFILES.keys())
), f"Missing profiles: {expected_profiles - set(_NAMED_PROFILES.keys())}"
for name, vals in _NAMED_PROFILES.items():
    assert len(vals) == 20, f"Profile '{name}' has {len(vals)} slots, expected 20"
print(f"[PASS] All {len(_NAMED_PROFILES)} named profiles present with 20 slots each")

# ── 6. Save/load round-trip for ElasticNet ────────────────────────────
import tempfile, json, os

with tempfile.TemporaryDirectory() as tmp:
    fpath = os.path.join(tmp, "en_model.npz")
    en.save(fpath)
    en2 = ElasticNetPolyValueModel.load(fpath)
    assert en2.spec.include_extra == True
    pred2 = en2.predict_batch(dummy_X)
    assert np.allclose(pred, pred2), "Round-trip mismatch"
print("[PASS] ElasticNet save/load round-trip")

# ── 7. Batch beam DP signature check ─────────────────────────────────
import inspect
from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp
from PaST.solvers.optimal_benchmark_dp_numba import solve_sparse_dp_python_beam

sig1 = inspect.signature(solve_optimal_benchmark_dp)
sig2 = inspect.signature(solve_sparse_dp_python_beam)
assert (
    "vhat_batch" in sig1.parameters
), "vhat_batch missing from solve_optimal_benchmark_dp"
assert (
    "vhat_batch" in sig2.parameters
), "vhat_batch missing from solve_sparse_dp_python_beam"
print("[PASS] vhat_batch parameter wired through DP functions")

print("\n=== ALL SMOKE TESTS PASSED ===")
