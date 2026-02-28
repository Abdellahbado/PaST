"""Load instances from the original Wang2018 benchmark (Benchmark/Data/*.txt).

Each instance j (j = 1..90) is stored as three files:
    Data_c{j}.txt  – time-slot energy prices  (one per line → T values → ct)
    Data_e{j}.txt  – machine energy prices     (one per line → m values → e)
    Data_p{j}.txt  – job processing times      (one per line → n values → p)

Scales follow the paper convention:
    1-30  → small
    31-60 → mls   (medium / large-scale)
    61-90 → large
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

SCALE_RANGES = {
    "small": range(1, 31),
    "mls": range(31, 61),
    "large": range(61, 91),
}


def _scale_for_id(instance_id: int) -> str:
    for scale, rng in SCALE_RANGES.items():
        if instance_id in rng:
            return scale
    return "unknown"


def _read_values(path: Path) -> List[int]:
    """Read a single-column numeric file and return integer values."""
    vals: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals.append(int(round(float(line))))
    return vals


def load_benchmark_instance(instance_id: int, data_dir: Path) -> dict:
    """Load a single benchmark instance from Data_c/e/p files.

    Returns a dict in the same format as ``load_instances_from_json``:
        {m, n, T, p, e, ct, instance_id, scale}

    instance_id is offset by +1000 to avoid collision with instances_90.json IDs.
    """
    ct = _read_values(data_dir / f"Data_c{instance_id}.txt")
    e = _read_values(data_dir / f"Data_e{instance_id}.txt")
    p = _read_values(data_dir / f"Data_p{instance_id}.txt")

    T = len(ct)
    m = len(e)
    n = len(p)

    return {
        "instance_id": 1000 + instance_id,  # offset to avoid ID collision
        "benchmark_id": instance_id,  # original paper ID
        "paper": "Wang2018_benchmark",
        "scale": _scale_for_id(instance_id),
        "m": m,
        "n": n,
        "T": T,
        "p": p,
        "e": e,
        "ct": ct,
    }


def load_benchmark_instances(
    data_dir: str | Path,
    instance_ids: List[int] | None = None,
) -> List[dict]:
    """Load multiple benchmark instances.

    Parameters
    ----------
    data_dir : path to the folder containing Data_c*.txt, Data_e*.txt, Data_p*.txt
    instance_ids : which instances to load (1..90). None → all 90.

    Returns
    -------
    list of instance dicts, ready for operator evaluation.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Benchmark data directory not found: {data_dir}")

    if instance_ids is None:
        instance_ids = list(range(1, 91))

    instances: List[dict] = []
    for iid in instance_ids:
        try:
            inst = load_benchmark_instance(iid, data_dir)
            instances.append(inst)
        except FileNotFoundError:
            logger.warning(
                "Benchmark instance %d: missing data file(s) in %s", iid, data_dir
            )
        except Exception as exc:
            logger.warning("Benchmark instance %d: load error: %s", iid, exc)

    logger.info(
        "Loaded %d benchmark instances from %s (scales: %s)",
        len(instances),
        data_dir,
        {s: sum(1 for i in instances if i["scale"] == s) for s in SCALE_RANGES},
    )
    return instances


def split_benchmark_instances(
    instances: List[dict],
    evo_fraction: float = 2 / 3,
    seed: int = 42,
) -> tuple[List[dict], List[dict]]:
    """Stratified split of benchmark instances into evolution and evaluation sets.

    Splits **within each scale group** so that both sets have proportional
    coverage of small, mls, and large instances.

    Parameters
    ----------
    instances : list of instance dicts (as returned by ``load_benchmark_instances``)
    evo_fraction : fraction of each scale group assigned to evolution (default 2/3)
    seed : random seed for reproducible shuffling

    Returns
    -------
    (evo_instances, eval_instances)
        evo_instances  – used for LLM operator fitness evaluation (67%)
        eval_instances – held out for benchmark comparison (33%)
    """
    import random as _random

    rng = _random.Random(seed)

    evo: List[dict] = []
    eval_: List[dict] = []

    for scale in ("small", "mls", "large"):
        group = sorted(
            [i for i in instances if i["scale"] == scale],
            key=lambda x: x["instance_id"],
        )
        rng.shuffle(group)
        n_evo = max(1, round(len(group) * evo_fraction))
        evo.extend(group[:n_evo])
        eval_.extend(group[n_evo:])

    # Restore sorted order for determinism in logs.
    evo.sort(key=lambda x: x["instance_id"])
    eval_.sort(key=lambda x: x["instance_id"])

    scale_counts_evo = {
        s: sum(1 for i in evo if i["scale"] == s) for s in ("small", "mls", "large")
    }
    scale_counts_eval = {
        s: sum(1 for i in eval_ if i["scale"] == s) for s in ("small", "mls", "large")
    }
    logger.info(
        "Benchmark adaptation split (frac=%.2f seed=%d): " "evo=%d %s | eval=%d %s",
        evo_fraction,
        seed,
        len(evo),
        scale_counts_evo,
        len(eval_),
        scale_counts_eval,
    )
    return evo, eval_
