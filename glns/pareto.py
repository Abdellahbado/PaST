"""Bi-objective Pareto archive for (Cmax, TEC).

Maintains a set of non-dominated solutions with crowding-distance pruning when
the archive exceeds its capacity.  Hypervolume is computed *only* for logging
(never in the inner loop).
"""

from __future__ import annotations

import copy
import math
import random as stdlib_random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lightweight solution record
# ---------------------------------------------------------------------------


@dataclass
class ArchiveEntry:
    """One non-dominated solution stored in the archive."""

    makespan: int
    energy: float
    # Per-machine job sequences (the representation operators work on).
    sequences: List[List[int]]
    # Benchmark instance identity. Entries from different instances are NOT comparable.
    instance_id: int = 0
    # Per-machine start times from DP (for analysis / debugging).
    start_times: Optional[List[List[int]]] = None
    # Number of consecutive generations without improvement from this entry.
    stagnation: int = 0


# ---------------------------------------------------------------------------
# Pareto archive
# ---------------------------------------------------------------------------


class ParetoArchive:
    """Bounded archive of non-dominated (Cmax, TEC) solutions."""

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self.entries: List[ArchiveEntry] = []

    def _default_instance_id(self) -> int:
        if not self.entries:
            return 0
        return min(e.instance_id for e in self.entries)

    def entries_for(self, instance_id: Optional[int]) -> List[ArchiveEntry]:
        """Return the list of entries for a given instance id.

        If instance_id is None, uses the archive's default instance id.
        """
        iid = self._default_instance_id() if instance_id is None else int(instance_id)
        return [e for e in self.entries if e.instance_id == iid]

    # ----- dominance logic -------------------------------------------------

    @staticmethod
    def dominates(a: ArchiveEntry, b: ArchiveEntry) -> bool:
        """True if *a* weakly dominates *b* (≤ on both, < on at least one)."""
        return (
            a.makespan <= b.makespan
            and a.energy <= b.energy
            and (a.makespan < b.makespan or a.energy < b.energy)
        )

    def is_dominated(self, entry: ArchiveEntry) -> bool:
        """True if any existing entry dominates *entry*."""
        pool = self.entries_for(entry.instance_id)
        return any(self.dominates(e, entry) for e in pool)

    # ----- insertion -------------------------------------------------------

    def add(self, entry: ArchiveEntry) -> Tuple[bool, int]:
        """Try to add *entry*.

        Returns:
            (entered, n_dominated_removed)
            entered:  True if the entry was inserted into the archive.
            n_dominated_removed:  number of previously-stored entries that the
                new entry dominated (and which were therefore removed).  This is
                > 0 only when the new entry strictly dominates at least one
                existing archive member (i.e., it improves the Pareto set, not
                just adds a trade-off point).

        Steps:
        1. If dominated by an existing entry → reject.
        2. Remove all existing entries dominated by the new one.
        3. Insert.
        4. If archive exceeds capacity, prune via crowding distance.
        """
        if self.is_dominated(entry):
            return False, 0

        # Remove dominated members from the SAME instance only.
        old_len = len(self.entries)
        self.entries = [
            e
            for e in self.entries
            if (e.instance_id != entry.instance_id) or (not self.dominates(entry, e))
        ]
        n_removed = old_len - len(self.entries)
        self.entries.append(entry)

        # Capacity is enforced per instance (entries from different instances are incomparable).
        if len(self.entries_for(entry.instance_id)) > self.max_size:
            # Prefer pruning existing entries over the just-added one so that
            # "entered" means the entry is part of the maintained archive.
            self._prune_to_capacity(instance_id=entry.instance_id, protect=entry)

        survived = any(e is entry for e in self.entries)
        if not survived:
            # The candidate was pruned away due to archive capacity.
            return False, 0

        return True, n_removed

    # ----- crowding distance -----------------------------------------------

    def crowding_distances(self, instance_id: Optional[int] = None) -> List[float]:
        """NSGA-II crowding distance for each entry (within one instance)."""
        entries = self.entries_for(instance_id)
        n = len(entries)
        if n <= 2:
            return [float("inf")] * n

        dist = [0.0] * n

        for key in ("makespan", "energy"):
            vals = [getattr(e, key) for e in entries]
            indices = sorted(range(n), key=lambda i: vals[i])
            lo, hi = vals[indices[0]], vals[indices[-1]]
            span = float(hi - lo) if hi != lo else 1.0
            dist[indices[0]] = float("inf")
            dist[indices[-1]] = float("inf")
            for rank in range(1, n - 1):
                dist[indices[rank]] += (
                    vals[indices[rank + 1]] - vals[indices[rank - 1]]
                ) / span

        return dist

    def _prune_to_capacity(
        self,
        instance_id: Optional[int],
        protect: Optional[ArchiveEntry] = None,
    ) -> None:
        """Remove entries with smallest crowding distance until within capacity.

        If *protect* is provided, we avoid pruning that specific entry unless
        it is unavoidable.
        """
        iid = self._default_instance_id() if instance_id is None else int(instance_id)
        while len(self.entries_for(iid)) > self.max_size:
            # Work on an index list of entries of this instance.
            idxs = [i for i, e in enumerate(self.entries) if e.instance_id == iid]
            local_entries = [self.entries[i] for i in idxs]
            # Compute distances in local index space.
            # (Reuse crowding_distances by passing the instance_id.)
            dists = self.crowding_distances(iid)
            candidates_local = [
                li
                for li in range(len(dists))
                if math.isfinite(dists[li])
                and (protect is None or local_entries[li] is not protect)
            ]
            if not candidates_local:
                candidates_local = [
                    li for li in range(len(dists)) if math.isfinite(dists[li])
                ]
            min_local = min(candidates_local, key=lambda li: dists[li], default=0)
            # Map local index back to self.entries index.
            self.entries.pop(idxs[min_local])

    # ----- selection strategies -------------------------------------------

    def select(
        self,
        strategy: str,
        rng: stdlib_random.Random,
        instance_id: Optional[int] = None,
    ) -> ArchiveEntry:
        """Pick an entry to perturb.

        Strategies:
            crowded   – prefer crowded regions (small distance) → diversity.
            extreme_tec – best TEC entry.
            extreme_cmax – best Cmax entry.
            stagnated – longest-stagnated entry.
            random    – uniform.
        """
        entries = self.entries_for(instance_id)
        if not entries:
            raise RuntimeError("Archive is empty; cannot select.")

        if strategy == "random" or len(entries) == 1:
            return rng.choice(entries)

        if strategy == "extreme_tec":
            return min(entries, key=lambda e: e.energy)

        if strategy == "extreme_cmax":
            return min(entries, key=lambda e: e.makespan)

        if strategy == "stagnated":
            return max(entries, key=lambda e: e.stagnation)

        if strategy == "crowded":
            dists = self.crowding_distances(instance_id)
            # Prefer *small* crowding distance → tournament among 3 random.
            candidates = rng.sample(range(len(entries)), k=min(3, len(entries)))
            pick = min(candidates, key=lambda i: dists[i])
            return entries[pick]

        # Fallback.
        return rng.choice(entries)

    # ----- stagnation bookkeeping -----------------------------------------

    def increment_stagnation(self, instance_id: Optional[int] = None) -> None:
        for e in self.entries_for(instance_id):
            e.stagnation += 1

    def reset_stagnation(self, entry: ArchiveEntry) -> None:
        entry.stagnation = 0

    # ----- metrics (logging only, never hot-path) -------------------------

    def hypervolume(
        self, ref_makespan: int, ref_energy: float, instance_id: Optional[int] = None
    ) -> float:
        """2-D hypervolume w.r.t. a reference point (larger is better).

        Uses the simple sweep-line algorithm for 2 objectives.
        """
        entries = self.entries_for(instance_id)
        if not entries:
            return 0.0

        pts = sorted(
            [(e.makespan, e.energy) for e in entries],
            key=lambda p: p[0],
        )
        hv = 0.0
        prev_energy = float(ref_energy)
        for cmax, tec in pts:
            if cmax >= ref_makespan:
                break
            if tec < prev_energy:
                hv += float(ref_makespan - cmax) * (prev_energy - tec)
                prev_energy = tec
        return hv

    def size(self, instance_id: Optional[int] = None) -> int:
        if instance_id is None:
            return len(self.entries)
        return len(self.entries_for(instance_id))

    def front(self, instance_id: Optional[int] = None) -> List[Tuple[int, float]]:
        """Return the current Pareto front as (Cmax, TEC) pairs.

        Important: Pareto fronts are only meaningful *within the same instance*.
        If instance_id is None, returns the front for the archive's default
        instance_id.
        """
        entries = self.entries_for(instance_id)
        return sorted(
            [(e.makespan, e.energy) for e in entries],
            key=lambda p: p[0],
        )
