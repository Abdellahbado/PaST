import os
import sys

import numpy as np

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _proj_root not in sys.path:
    sys.path.insert(0, os.path.dirname(_proj_root))

from PaST.data.stateful_paper_benchmark import (
    compute_paper_horizon,
    generate_stateful_paper_dataset,
)
from PaST.solvers.machine_states import MachineStateConfig
from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp


def test_paper_nosby_alias_matches_figure2_model():
    cfg = MachineStateConfig.paper_nosby()
    assert cfg.states == ("off", "proc", "idle")
    assert cfg.get_transition_time("off", "proc") == 2
    assert cfg.get_transition_time("proc", "off") == 1
    assert cfg.get_transition_power("proc", "proc") == 4.0


def test_compute_paper_horizon_matches_formula():
    cfg = MachineStateConfig.paper_nosby()
    processing_times = [1, 2, 5]
    horizon = compute_paper_horizon(processing_times, 1.6, cfg)
    assert horizon == 18


def test_generate_stateful_paper_dataset_ranges_and_solver_smoke():
    dataset = generate_stateful_paper_dataset(
        seeds=[7],
        n_values=[8],
        lambda_values=[1.3],
        machine_model="paper_nosby",
    )
    assert len(dataset.instances) == 1

    instance = dataset.instances[0]
    assert instance.machine_model == "paper_nosby"
    assert all(1 <= p <= 5 for p in instance.processing_times)
    assert all(1 <= c <= 10 for c in instance.prices)
    assert instance.horizon == compute_paper_horizon(
        instance.processing_times,
        instance.lambda_scale,
        MachineStateConfig.paper_nosby(),
    )

    res = solve_optimal_benchmark_dp(
        processing_times=instance.processing_times,
        prices=np.asarray(instance.prices, dtype=np.float64),
        machine_config=MachineStateConfig.paper_nosby(),
        track_schedule=False,
    )
    assert res.feasible
    assert np.isfinite(res.cost)
