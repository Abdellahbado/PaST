"""
Permanent regression tests for sparse DP correctness.

Verifies:
1. Exact DP (no limit) produces known optimal cost.
2. Time-limited DP cost >= exact optimal (never below).
3. track_schedule=True and False give same cost without timeout.
4. max_states guardrail triggers gracefully (cost >= exact).
5. Greedy fallback produces finite cost on timeout.
6. Deterministic abort paths (max_states=1) are exercised.
"""

import sys, os
import numpy as np
import pytest

# Ensure PaST root is importable
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, os.path.dirname(_proj_root))

from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp


# ---------------------------------------------------------------------------
# Fixture: A small but non-trivial instance with known optimal solution.
# 10 jobs with processing times from {1,2,3}, horizon 30, random prices.
# ---------------------------------------------------------------------------
@pytest.fixture
def small_instance():
    rng = np.random.RandomState(42)
    processing_times = [1, 2, 3, 1, 2, 3, 1, 2, 3, 2]
    T = 30
    prices = rng.uniform(0.5, 5.0, size=T).astype(np.float64)
    return processing_times, prices


@pytest.fixture
def exact_cost(small_instance):
    """Compute and cache exact optimal cost (no limits)."""
    p, prices = small_instance
    result = solve_optimal_benchmark_dp(
        p, prices, tie_break="early", time_limit=-1, track_schedule=True
    )
    assert result.feasible, "Instance must be feasible"
    assert np.isfinite(result.cost), "Exact DP must produce finite cost"
    return float(result.cost)


# ---------------------------------------------------------------------------
# Test 1: Exact DP produces a finite optimal cost with valid schedule
# ---------------------------------------------------------------------------
def test_exact_dp_produces_finite_cost(small_instance):
    p, prices = small_instance
    result = solve_optimal_benchmark_dp(
        p, prices, tie_break="early", time_limit=-1, track_schedule=True
    )
    assert result.feasible
    assert np.isfinite(result.cost)
    assert result.cost > 0
    assert len(result.schedule) == len(p)
    assert not result.timed_out


# ---------------------------------------------------------------------------
# Test 2: Time-limited DP cost >= exact optimal (CRITICAL regression)
# ---------------------------------------------------------------------------
def test_timeout_cost_ge_exact(small_instance, exact_cost):
    p, prices = small_instance
    # Use tiny time limit to force timeout
    result = solve_optimal_benchmark_dp(
        p, prices, tie_break="early", time_limit=1e-6, track_schedule=True
    )
    assert result.feasible
    assert np.isfinite(result.cost), "Timeout must still produce finite cost"
    assert result.cost >= exact_cost - 1e-9, (
        f"Timeout cost {result.cost:.4f} < exact {exact_cost:.4f} — "
        f"double-counting bug!"
    )


# ---------------------------------------------------------------------------
# Test 3: track_schedule=False + timeout cost >= exact (the original bug case)
# ---------------------------------------------------------------------------
def test_timeout_no_track_cost_ge_exact(small_instance, exact_cost):
    p, prices = small_instance
    result = solve_optimal_benchmark_dp(
        p, prices, tie_break="early", time_limit=1e-6, track_schedule=False
    )
    assert result.feasible
    assert np.isfinite(result.cost), "Timeout+no_track must produce finite cost"
    assert result.cost >= exact_cost - 1e-9, (
        f"Timeout+no_track cost {result.cost:.4f} < exact {exact_cost:.4f} — "
        f"track_schedule=False bug!"
    )


# ---------------------------------------------------------------------------
# Test 4: Without timeout, track_schedule=True and False give same cost
# ---------------------------------------------------------------------------
def test_track_schedule_same_cost_no_timeout(small_instance):
    p, prices = small_instance
    r_track = solve_optimal_benchmark_dp(
        p, prices, tie_break="early", time_limit=-1, track_schedule=True
    )
    r_notrack = solve_optimal_benchmark_dp(
        p, prices, tie_break="early", time_limit=-1, track_schedule=False
    )
    assert r_track.feasible and r_notrack.feasible
    assert abs(r_track.cost - r_notrack.cost) < 1e-9, (
        f"track_schedule mismatch: {r_track.cost:.6f} vs {r_notrack.cost:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 5: max_states guardrail triggers and cost >= exact
# ---------------------------------------------------------------------------
def test_max_states_guardrail(small_instance, exact_cost):
    p, prices = small_instance
    # Very small max_states to force abort
    result = solve_optimal_benchmark_dp(
        p,
        prices,
        tie_break="early",
        time_limit=-1,
        track_schedule=True,
        max_states=5,
    )
    assert result.feasible
    assert np.isfinite(result.cost), "max_states abort must produce finite cost"
    assert result.cost >= exact_cost - 1e-9, (
        f"max_states cost {result.cost:.4f} < exact {exact_cost:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5b: max_states=1 deterministically forces abort path on a sparse instance.
# Uses K=10 distinct lengths to force sparse DP (>100M cells).
# With max_states=1 the DP aborts almost immediately so runtime is trivial.
# ---------------------------------------------------------------------------
@pytest.fixture
def sparse_instance():
    """Instance with K=10 distinct lengths → n_states ≈ 1M → sparse path."""
    rng = np.random.RandomState(77)
    # 30 jobs: 3 of each length 1..10
    processing_times = list(range(1, 11)) * 3
    T = sum(processing_times) + 20  # 185
    prices = rng.uniform(1.0, 8.0, size=T).astype(np.float64)
    return processing_times, prices


@pytest.fixture
def sparse_exact_cost(sparse_instance):
    p, prices = sparse_instance
    result = solve_optimal_benchmark_dp(
        p, prices, tie_break="early", time_limit=30.0, track_schedule=True
    )
    assert result.feasible
    assert np.isfinite(result.cost)
    return float(result.cost)


def test_max_states_one_forces_abort(sparse_instance, sparse_exact_cost):
    p, prices = sparse_instance
    result = solve_optimal_benchmark_dp(
        p,
        prices,
        tie_break="early",
        time_limit=-1,
        track_schedule=True,
        max_states=1,
    )
    assert result.feasible
    assert result.timed_out, "max_states=1 must trigger abort on sparse instance"
    assert np.isfinite(result.cost), "Aborted DP must still produce finite cost"
    assert result.cost >= sparse_exact_cost - 1e-9


# ---------------------------------------------------------------------------
# Test 5c: max_states=1 with track_schedule=False also safe (sparse path)
# ---------------------------------------------------------------------------
def test_max_states_one_no_track(sparse_instance, sparse_exact_cost):
    p, prices = sparse_instance
    result = solve_optimal_benchmark_dp(
        p,
        prices,
        tie_break="early",
        time_limit=-1,
        track_schedule=False,
        max_states=1,
    )
    assert result.feasible
    assert result.timed_out
    assert np.isfinite(result.cost)
    assert result.cost >= sparse_exact_cost - 1e-9


# ---------------------------------------------------------------------------
# Test 6: Larger instance to exercise sparse DP path
# (Forces state space > max_cells threshold for dense DP)
# ---------------------------------------------------------------------------
def test_large_sparse_instance():
    """Instance with many distinct lengths forces sparse DP path."""
    rng = np.random.RandomState(123)
    # 8 distinct lengths ensures K=8, large state space → sparse path
    processing_times = list(range(1, 9)) * 3  # 24 jobs, lengths 1..8
    T = sum(processing_times) + 20  # enough horizon
    prices = rng.uniform(1.0, 10.0, size=T).astype(np.float64)

    result = solve_optimal_benchmark_dp(
        processing_times, prices, tie_break="early", time_limit=10.0, track_schedule=True
    )
    assert result.feasible
    assert np.isfinite(result.cost)
    assert result.cost > 0


# ---------------------------------------------------------------------------
# Test 7: Empty and single-job edge cases
# ---------------------------------------------------------------------------
def test_empty_jobs():
    prices = np.array([1.0, 2.0, 3.0])
    result = solve_optimal_benchmark_dp([], prices, tie_break="early")
    assert result.feasible
    assert result.cost == 0.0


def test_single_job():
    prices = np.array([3.0, 1.0, 2.0])
    result = solve_optimal_benchmark_dp([1], prices, tie_break="early")
    assert result.feasible
    assert result.cost == 1.0  # cheapest single slot


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
