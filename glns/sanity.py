"""Sanity-check layer for LLM-generated operators.

Validates syntax, signature, correctness, and runtime on tiny toy instances
before admitting an operator into the population.
"""

from __future__ import annotations

import copy
import inspect
import logging
import random
import time
from typing import List, Optional, Tuple

from glns.config import SandboxConfig
from glns.sandbox import run_operator_sandboxed
from glns.schemas import OperatorSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Toy instance helpers
# ---------------------------------------------------------------------------


def make_toy_instances() -> List[dict]:
    """Return 2-3 tiny instances for fast sanity checks."""
    return [
        {
            "m": 3,
            "n": 6,
            "T": 50,
            "p": [2, 1, 3, 1, 4, 3],
            "e": [1, 2, 2],
            "ct": [
                4,
                4,
                1,
                1,
                4,
                3,
                2,
                2,
                3,
                3,
                1,
                1,
                4,
                2,
                3,
                1,
                2,
                4,
                3,
                1,
                2,
                3,
                1,
                4,
                2,
                1,
                3,
                4,
                2,
                1,
                3,
                2,
                1,
                4,
                3,
                2,
                1,
                3,
                4,
                2,
                1,
                2,
                3,
                4,
                1,
                2,
                3,
                1,
                2,
                4,
            ],
        },
        {
            "m": 2,
            "n": 4,
            "T": 20,
            "p": [3, 2, 4, 1],
            "e": [1, 3],
            "ct": [1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 1, 1, 3, 3, 2, 2, 4, 4, 1, 1],
        },
    ]


def _make_trivial_solution(inst: dict) -> List[List[int]]:
    """Round-robin assignment for sanity checking."""
    m = inst["m"]
    seqs: List[List[int]] = [[] for _ in range(m)]
    for j in range(inst["n"]):
        seqs[j % m].append(j)
    return seqs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sanity_check(
    spec: OperatorSpec,
    sandbox_cfg: SandboxConfig,
    toy_instances: Optional[List[dict]] = None,
) -> Tuple[bool, Optional[str]]:
    """Run a full sanity check on an operator.

    Returns (passed, error_message).  error_message is None on success.
    """
    if toy_instances is None:
        toy_instances = make_toy_instances()

    fn_name = spec.type  # "destroy" or "repair"

    # ── 1. Syntax was already validated by Pydantic field_validator,
    #       but double-check:
    try:
        compile(spec.code, "<sanity>", "exec")
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # ── 2. Functional test on each toy instance ──────────────────────
    rng = random.Random(12345)

    for idx, inst in enumerate(toy_instances):
        sol = _make_trivial_solution(inst)
        n = inst["n"]
        m = inst["m"]

        if spec.type == "destroy":
            destroy_cnt = max(1, n // 3)
            ok, result = run_operator_sandboxed(
                fn_code=spec.code,
                fn_name="destroy",
                args=(copy.deepcopy(sol), destroy_cnt, inst, rng),
                cfg=sandbox_cfg,
            )
            if not ok:
                return False, f"Destroy crashed on toy instance {idx}: {result}"

            # Validate output shape.
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                return False, (
                    f"Destroy must return (removed_jobs, partial_solution), "
                    f"got {type(result).__name__} of length {len(result) if isinstance(result, (tuple,list)) else '?'}"
                )
            removed_jobs, partial_sol = result
            if not isinstance(removed_jobs, list):
                return (
                    False,
                    f"removed_jobs must be a list, got {type(removed_jobs).__name__}",
                )
            if not isinstance(partial_sol, list):
                return (
                    False,
                    f"partial_solution must be a list of lists, got {type(partial_sol).__name__}",
                )

            if len(partial_sol) != m:
                return (
                    False,
                    f"partial_solution must have length m={m}, got {len(partial_sol)}",
                )

            # Check removed_jobs are valid indices.
            all_in_partial = set()
            for seq in partial_sol:
                if not isinstance(seq, list):
                    return False, "partial_solution must be a list of lists"
                all_in_partial.update(seq)
            removed_set = set(removed_jobs)
            if len(removed_set) != len(removed_jobs):
                return False, "removed_jobs contains duplicates"
            if not removed_set.issubset(set(range(n))):
                return False, f"removed_jobs contains invalid indices (not in 0..{n-1})"
            # Union should be the full job set.
            if removed_set | all_in_partial != set(range(n)):
                return False, (
                    f"removed_jobs ∪ partial_solution jobs ≠ {{0..{n-1}}}. "
                    f"Missing: {set(range(n)) - (removed_set | all_in_partial)}, "
                    f"extra: {(removed_set | all_in_partial) - set(range(n))}"
                )
            # Should have no overlap.
            if removed_set & all_in_partial:
                return False, "removed_jobs and partial_solution overlap"

        else:  # repair
            # Create a partial solution by removing some jobs.
            destroy_cnt = max(1, n // 3)
            partial = copy.deepcopy(sol)
            removed: List[int] = []
            for _ in range(destroy_cnt):
                # Remove random jobs.
                non_empty = [i for i, s in enumerate(partial) if s]
                if not non_empty:
                    break
                mi = rng.choice(non_empty)
                j = partial[mi].pop(rng.randrange(len(partial[mi])))
                removed.append(j)

            ok, result = run_operator_sandboxed(
                fn_code=spec.code,
                fn_name="repair",
                args=(copy.deepcopy(partial), list(removed), inst, rng),
                cfg=sandbox_cfg,
            )
            if not ok:
                return False, f"Repair crashed on toy instance {idx}: {result}"

            if not isinstance(result, list):
                return (
                    False,
                    f"Repair must return list of lists, got {type(result).__name__}",
                )

            if len(result) != m:
                return False, f"Repair output must have length m={m}, got {len(result)}"

            all_jobs = set()
            for seq in result:
                if not isinstance(seq, list):
                    return False, "Repair must return a list of lists"
                all_jobs.update(seq)

            if all_jobs != set(range(n)):
                missing = set(range(n)) - all_jobs
                extra = all_jobs - set(range(n))
                return False, (
                    f"Repair output doesn't contain all jobs exactly once. "
                    f"Missing: {missing}, extra: {extra}"
                )
            total_len = sum(len(seq) for seq in result)
            if total_len != n:
                return (
                    False,
                    f"Repair output has {total_len} jobs, expected {n} (duplicates?)",
                )

    return True, None
