"""Population management for destroy and repair operator pools.

Uses **stable operator IDs** (not positional indices) for fitness and synergy
tracking, preventing index corruption across pruning/insertion cycles.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from glns.config import PopulationConfig
from glns.schemas import OperatorRecord

logger = logging.getLogger(__name__)


class OperatorPool:
    """A bounded pool of operators (either all destroy or all repair)."""

    def __init__(self, capacity: int, pool_type: str) -> None:
        self.capacity = capacity
        self.pool_type = pool_type  # "destroy" or "repair"
        self.operators: List[OperatorRecord] = []

    def add(self, op: OperatorRecord) -> bool:
        """Add an operator if there is room.  Returns True on success."""
        if len(self.operators) >= self.capacity:
            logger.warning(
                "Pool %s is full (%d); cannot add.", self.pool_type, self.capacity
            )
            return False
        self.operators.append(op)
        return True

    def ids(self) -> List[str]:
        return [op.id for op in self.operators]

    def get_by_id(self, op_id: str) -> Optional[OperatorRecord]:
        for op in self.operators:
            if op.id == op_id:
                return op
        return None

    def remove_by_ids(self, ids_to_remove: set[str]) -> None:
        self.operators = [op for op in self.operators if op.id not in ids_to_remove]

    def empty_slots(self) -> int:
        return max(0, self.capacity - len(self.operators))

    def __len__(self) -> int:
        return len(self.operators)

    def __iter__(self):
        return iter(self.operators)


class PopulationManager:
    """Manages both pools plus global fitness and synergy metrics.

    Synergy is stored as ``Dict[(destroy_id, repair_id), float]`` — keyed by
    **stable** operator IDs so that pruning/insertion never corrupts indices.
    """

    def __init__(self, cfg: PopulationConfig) -> None:
        self.cfg = cfg
        self.destroy_pool = OperatorPool(cfg.N, "destroy")
        self.repair_pool = OperatorPool(cfg.N, "repair")

        # Metrics — reset each generation after pruning + evolution.
        self.F_d: Dict[str, float] = {}
        self.F_r: Dict[str, float] = {}
        self.synergy: Dict[Tuple[str, str], float] = {}

    # ----- metric helpers -------------------------------------------------

    def update_metrics(
        self,
        F_d: Dict[str, float],
        F_r: Dict[str, float],
        synergy: Dict[Tuple[str, str], float],
    ) -> None:
        """Absorb evaluation-phase metrics."""
        self.F_d = F_d
        self.F_r = F_r
        self.synergy = synergy

    def reset_metrics(self) -> None:
        """Zero out F and S after a generation (G-LNS paper §3.4)."""
        self.F_d = {op.id: 0.0 for op in self.destroy_pool}
        self.F_r = {op.id: 0.0 for op in self.repair_pool}
        self.synergy = {}

    # ----- ranking & pruning ----------------------------------------------

    def rank_and_prune(self) -> int:
        """Rank operators by global fitness and prune the bottom M from each pool.

        Returns the total number of pruned slots (across both pools).
        """
        M = self.cfg.M
        pruned_total = 0

        for pool, F in [(self.destroy_pool, self.F_d), (self.repair_pool, self.F_r)]:
            if len(pool) <= M:
                continue  # don't prune below M
            # Rank: lowest fitness first.
            ranked = sorted(pool.operators, key=lambda op: F.get(op.id, 0.0))
            to_remove = set(op.id for op in ranked[:M])
            pool.remove_by_ids(to_remove)
            pruned_total += len(to_remove)
            pruned_info = [
                (op.id, round(float(F.get(op.id, 0.0)), 4)) for op in ranked[:M]
            ]
            logger.info(
                "Pruned %d operator(s) from %s pool: %s",
                len(to_remove),
                pool.pool_type,
                pruned_info,
            )

        # Clean stale synergy entries.
        valid_d = set(self.destroy_pool.ids())
        valid_r = set(self.repair_pool.ids())
        self.synergy = {
            (d, r): v
            for (d, r), v in self.synergy.items()
            if d in valid_d and r in valid_r
        }
        return pruned_total

    # ----- selection for evolution ----------------------------------------

    def top_reference_ops(self, k: int = 3) -> List[OperatorRecord]:
        """Return up to k top-fitness operators across both pools (for in-context examples)."""
        all_ops = list(self.destroy_pool) + list(self.repair_pool)
        F_all = {**self.F_d, **self.F_r}
        all_ops.sort(key=lambda op: F_all.get(op.id, 0.0), reverse=True)
        return all_ops[:k]

    def select_for_mutation(
        self, pool: OperatorPool, strategy: str, rng
    ) -> OperatorRecord:
        """Select a parent for mutation.

        strategy:
            'logic'  → prefer lower-ranked elites (more room for innovation).
            'param'  → prefer top-ranked elites (fine-tune stability).
        """
        F = self.F_d if pool.pool_type == "destroy" else self.F_r
        ranked = sorted(pool.operators, key=lambda op: F.get(op.id, 0.0), reverse=True)
        if strategy == "param":
            # Top third.
            idx = rng.randint(0, max(0, len(ranked) // 3))
        else:
            # Bottom two-thirds.
            start = max(1, len(ranked) // 3)
            idx = rng.randint(start, len(ranked) - 1) if start < len(ranked) else 0
        return ranked[idx]

    def select_for_homo_crossover(
        self, pool: OperatorPool, rng
    ) -> Tuple[OperatorRecord, OperatorRecord]:
        """Fitness-proportionate selection of two parents from the same pool."""
        F = self.F_d if pool.pool_type == "destroy" else self.F_r
        ops = list(pool.operators)
        if len(ops) < 2:
            return ops[0], ops[0]
        weights = [max(F.get(op.id, 0.0), 1e-6) for op in ops]
        total = sum(weights)
        probs = [w / total for w in weights]

        def _sample() -> int:
            r = rng.random()
            cum = 0.0
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    return i
            return len(probs) - 1

        a = _sample()
        b = _sample()
        while b == a and len(ops) > 1:
            b = _sample()
        return ops[a], ops[b]

    def select_for_joint_crossover(
        self,
    ) -> Tuple[OperatorRecord, OperatorRecord, float]:
        """Select the (destroy, repair) pair with the highest synergy score.

        Returns (destroy_op, repair_op, synergy_score).
        """
        if not self.synergy:
            # Fallback: pick first from each pool.
            d = self.destroy_pool.operators[0]
            r = self.repair_pool.operators[0]
            return d, r, 0.0

        best_key = max(self.synergy, key=self.synergy.get)  # type: ignore[arg-type]
        d_id, r_id = best_key
        d_op = self.destroy_pool.get_by_id(d_id) or self.destroy_pool.operators[0]
        r_op = self.repair_pool.get_by_id(r_id) or self.repair_pool.operators[0]
        return d_op, r_op, self.synergy[best_key]
