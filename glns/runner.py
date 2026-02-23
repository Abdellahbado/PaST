"""G-LNS main runner — orchestrates the full evolutionary loop.

Phases per generation:
1. Evaluate  (multi-episode ALNS — no LLM)
2. Prune     (rank & remove bottom M — no LLM)
3. Evolve    (ONE batched LLM call to fill all pruned slots)
4. Validate  (sanity check + retries — may use LLM for fix-up)
5. Reset     (zero out F, S for next generation)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from glns.config import GLNSConfig
from glns.evaluation import (
    evaluate_sequences,
    load_instances_from_json,
    make_initial_solution,
    run_evaluation_phase,
)
from glns.evolution import evolve_generation
from glns.llm_client import GroqOperatorClient
from glns.pareto import ArchiveEntry, ParetoArchive
from glns.population import PopulationManager
from glns.sanity import sanity_check
from glns.schemas import OperatorRecord
from glns.seed_operators import (
    build_seed_destroy_operators,
    build_seed_repair_operators,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instance split helpers
# ---------------------------------------------------------------------------


def _default_evolution_ids(instances: List[dict]) -> List[int]:
    """Auto-select small + first 10 medium instances for evolution."""
    small = [inst["instance_id"] for inst in instances if inst.get("scale") == "small"]
    mls = [inst["instance_id"] for inst in instances if inst.get("scale") == "mls"]
    return small + mls[:10]


def _split_instances(
    instances: List[dict],
    evolution_ids: Optional[List[int]],
    test_ids: Optional[List[int]],
) -> Tuple[List[dict], List[dict]]:
    if evolution_ids is None:
        evolution_ids = _default_evolution_ids(instances)
    evo_set = set(evolution_ids)

    if test_ids is None:
        test_ids_set = set(inst["instance_id"] for inst in instances) - evo_set
    else:
        test_ids_set = set(test_ids)

    evo = [inst for inst in instances if inst["instance_id"] in evo_set]
    test = [inst for inst in instances if inst["instance_id"] in test_ids_set]
    return evo, test


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------


def _seed_archive(
    instances: List[dict], archive: ParetoArchive, rng: random.Random
) -> None:
    """Build initial solutions via LPT round-robin for a few instances and seed the archive."""
    for inst in instances[:5]:
        inst_id = int(inst.get("instance_id", 0))
        seqs = make_initial_solution(inst, rng)
        energy, cmax, starts = evaluate_sequences(seqs, inst, inst["T"])
        if energy < float("inf"):
            archive.add(
                ArchiveEntry(
                    instance_id=inst_id,
                    makespan=cmax,
                    energy=energy,
                    sequences=seqs,
                    start_times=starts,
                )
            )


def _init_populations(
    pop: PopulationManager,
    llm: Optional[GroqOperatorClient],
    cfg: GLNSConfig,
) -> None:
    """Inject seed operators and optionally top-up with LLM-generated ones."""
    # Seed operators.
    for op in build_seed_destroy_operators():
        pop.destroy_pool.add(op)
    for op in build_seed_repair_operators():
        pop.repair_pool.add(op)

    # Top-up with LLM if pools not full.
    d_need = pop.destroy_pool.empty_slots()
    r_need = pop.repair_pool.empty_slots()
    if (d_need > 0 or r_need > 0) and llm is not None:
        logger.info(
            "Requesting %d destroy + %d repair operators from LLM for initialisation",
            d_need,
            r_need,
        )
        try:
            batch = llm.generate_init_batch(n_destroy=d_need, n_repair=r_need)
            for spec in batch.operators:
                passed, err = sanity_check(spec, cfg.sandbox)
                if passed:
                    rec = OperatorRecord(spec=spec, generation_born=0)
                    pool = (
                        pop.destroy_pool if spec.type == "destroy" else pop.repair_pool
                    )
                    pool.add(rec)
                else:
                    logger.warning("LLM init operator failed sanity: %s", err)
        except Exception as exc:
            logger.error("LLM initialisation call failed: %s", exc)

    # If still not full (LLM failed), duplicate seeds.
    while pop.destroy_pool.empty_slots() > 0:
        seeds = build_seed_destroy_operators()
        fb = seeds[len(pop.destroy_pool) % len(seeds)]
        fb.id = f"dup_d{len(pop.destroy_pool)}"  # type: ignore[assignment]
        pop.destroy_pool.add(fb)
    while pop.repair_pool.empty_slots() > 0:
        seeds = build_seed_repair_operators()
        fb = seeds[len(pop.repair_pool) % len(seeds)]
        fb.id = f"dup_r{len(pop.repair_pool)}"  # type: ignore[assignment]
        pop.repair_pool.add(fb)


# ---------------------------------------------------------------------------
# Logging / persistence
# ---------------------------------------------------------------------------


def _log_generation(
    gen: int,
    archive: ParetoArchive,
    pop: PopulationManager,
    elapsed_sec: float,
    ref_cmax: int,
    ref_tec: float,
    tracking_instance_id: int,
) -> Dict:
    hv = archive.hypervolume(ref_cmax, ref_tec, instance_id=tracking_instance_id)
    front = archive.front(instance_id=tracking_instance_id)
    size_total = archive.size()
    size_inst = archive.size(tracking_instance_id)
    entry = {
        "generation": gen,
        "tracking_instance_id": tracking_instance_id,
        "archive_size": size_inst,
        "archive_size_total": size_total,
        "hypervolume": hv,
        "front": front,
        "n_destroy": len(pop.destroy_pool),
        "n_repair": len(pop.repair_pool),
        "elapsed_sec": round(elapsed_sec, 2),
    }
    logger.info(
        "Gen %3d | inst %d | archive %d (total %d) | HV %.2f | front[0]=%s | front[-1]=%s | %.1fs",
        gen,
        tracking_instance_id,
        size_inst,
        size_total,
        hv,
        front[0] if front else "?",
        front[-1] if front else "?",
        elapsed_sec,
    )
    return entry


def _save_results(
    output_dir: Path,
    archive: ParetoArchive,
    pop: PopulationManager,
    gen_log: List[Dict],
    cfg: GLNSConfig,
    tracking_instance_id: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Archive front.
    front = archive.front(instance_id=tracking_instance_id)
    with open(output_dir / "pareto_front.json", "w") as f:
        json.dump({"instance_id": tracking_instance_id, "front": front}, f, indent=2)

    # Full archive with sequences.
    archive_data = []
    for e in archive.entries:
        archive_data.append(
            {
                "instance_id": e.instance_id,
                "makespan": e.makespan,
                "energy": e.energy,
                "sequences": e.sequences,
                "start_times": e.start_times,
            }
        )
    with open(output_dir / "archive_full.json", "w") as f:
        json.dump(archive_data, f, indent=2)

    # Operators (code + metadata).
    ops = []
    for pool in (pop.destroy_pool, pop.repair_pool):
        for op in pool:
            ops.append(
                {
                    "id": op.id,
                    "type": op.op_type,
                    "idea": op.idea,
                    "code": op.code,
                    "generation_born": op.generation_born,
                }
            )
    with open(output_dir / "operators.json", "w") as f:
        json.dump(ops, f, indent=2)

    # Generation log.
    with open(output_dir / "generation_log.json", "w") as f:
        json.dump(gen_log, f, indent=2)

    # Config snapshot.
    with open(output_dir / "config.json", "w") as f:
        f.write(cfg.model_dump_json(indent=2))

    logger.info("Results saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Test phase
# ---------------------------------------------------------------------------


def run_test_phase(
    pop: PopulationManager,
    test_instances: List[dict],
    archive: ParetoArchive,
    cfg: GLNSConfig,
    rng: random.Random,
) -> ParetoArchive:
    """Run the final test evaluation with T_test iterations on held-out instances."""
    logger.info(
        "=== TEST PHASE: %d instances, T=%d iterations ===",
        len(test_instances),
        cfg.eval.T_test,
    )

    test_eval_cfg = cfg.eval.model_copy()
    test_eval_cfg.T_iters = cfg.eval.T_test
    test_eval_cfg.K_episodes = max(3, cfg.eval.K_episodes // 2)

    test_archive = ParetoArchive(max_size=cfg.archive.max_size * 2)
    # Seed test archive from evolution archive.
    for e in archive.entries:
        test_archive.add(copy.deepcopy(e))

    run_evaluation_phase(
        destroy_pool=list(pop.destroy_pool),
        repair_pool=list(pop.repair_pool),
        instances=test_instances,
        archive=test_archive,
        eval_cfg=test_eval_cfg,
        sandbox_cfg=cfg.sandbox,
        rng=rng,
    )
    return test_archive


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_glns(cfg: GLNSConfig) -> ParetoArchive:
    """Execute the full G-LNS evolutionary loop.

    Returns the final Pareto archive.
    """
    # Allow runtime override without code changes.
    # Example: GLNS_LOG_LEVEL=DEBUG python run_glns.py
    level_name = os.environ.get("GLNS_LOG_LEVEL", "INFO").upper().strip()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Reduce third-party noise; keep our own loggers informative.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    rng = random.Random(cfg.seed)
    t_start = time.perf_counter()

    # ----- Load instances -------------------------------------------------
    instances_path = str(cfg.instances.instances_json)
    if not os.path.isabs(instances_path):
        # Resolve relative to the PaST project root.
        project_root = Path(__file__).resolve().parent.parent
        instances_path = str(project_root / instances_path)

    logger.info("Loading instances from %s", instances_path)
    all_instances = load_instances_from_json(instances_path)
    evo_instances, test_instances = _split_instances(
        all_instances, cfg.instances.evolution_ids, cfg.instances.test_ids
    )
    logger.info(
        "Evolution: %d instances | Test: %d instances",
        len(evo_instances),
        len(test_instances),
    )

    tracking_instance_id = int(evo_instances[0]["instance_id"]) if evo_instances else 0

    # ----- Initialise archive, populations, LLM --------------------------
    archive = ParetoArchive(max_size=cfg.archive.max_size)
    _seed_archive(evo_instances, archive, rng)
    logger.info(
        "Seeded archive with %d initial solutions (tracking instance %d has %d)",
        archive.size(),
        tracking_instance_id,
        archive.size(tracking_instance_id),
    )

    pop = PopulationManager(cfg.population)

    api_key = os.environ.get("GROQ_API_KEY", "")
    llm: Optional[GroqOperatorClient] = None
    if api_key:
        llm = GroqOperatorClient(cfg.llm, api_key=api_key)
    else:
        logger.warning(
            "GROQ_API_KEY not set — running with seed operators only (no LLM evolution)"
        )

    _init_populations(pop, llm, cfg)
    logger.info(
        "Populations initialised: %d destroy, %d repair",
        len(pop.destroy_pool),
        len(pop.repair_pool),
    )

    # Reference point for hypervolume logging.
    ref_cmax = max(inst["T"] for inst in evo_instances)
    ref_tec = float(
        ref_cmax
        * max(max(inst["e"]) for inst in evo_instances)
        * max(max(inst["ct"]) for inst in evo_instances)
    )

    gen_log: List[Dict] = []

    # ----- Main evolutionary loop ----------------------------------------
    for gen in range(cfg.evolution.G_max):
        t_gen = time.perf_counter()

        logger.info(
            "=== Generation %d/%d | inst=%d | archive=%d (total %d) | pools: D=%d R=%d ===",
            gen + 1,
            cfg.evolution.G_max,
            tracking_instance_id,
            archive.size(tracking_instance_id),
            archive.size(),
            len(pop.destroy_pool),
            len(pop.repair_pool),
        )

        # Phase 1: Evaluate.
        F_d, F_r, synergy = run_evaluation_phase(
            destroy_pool=list(pop.destroy_pool),
            repair_pool=list(pop.repair_pool),
            instances=evo_instances,
            archive=archive,
            eval_cfg=cfg.eval,
            sandbox_cfg=cfg.sandbox,
            rng=rng,
        )
        pop.update_metrics(F_d, F_r, synergy)

        # Phase 2: Prune.
        pruned = pop.rank_and_prune()

        # Phase 3+4: Evolve (one batched LLM call) + validate.
        if llm is not None and pruned > 0:
            outcome = evolve_generation(pop, llm, cfg.evolution, cfg.sandbox, gen, rng)
            if outcome.llm_ok:
                logger.info(
                    "Gen %d: added %d new operators (LLM ok; fallback_used=%d)",
                    gen + 1,
                    len(outcome.inserted),
                    outcome.used_fallback,
                )
            else:
                logger.info(
                    "Gen %d: added %d operators via fallback (rate_limited_for=%.1fs)",
                    gen + 1,
                    len(outcome.inserted),
                    float(outcome.rate_limited_for_sec or 0.0),
                )
        else:
            if pruned > 0:
                # No LLM — fill with fallback.
                from glns.evolution import _fallback_fill

                _fallback_fill(
                    pop,
                    pop.destroy_pool.empty_slots(),
                    pop.repair_pool.empty_slots(),
                    gen,
                )

        # Phase 5: Reset metrics.
        pop.reset_metrics()

        # Logging.
        elapsed = time.perf_counter() - t_gen
        entry = _log_generation(
            gen,
            archive,
            pop,
            elapsed,
            ref_cmax,
            ref_tec,
            tracking_instance_id,
        )
        gen_log.append(entry)

    # ----- Test phase -----------------------------------------------------
    if test_instances:
        test_archive = run_test_phase(pop, test_instances, archive, cfg, rng)
        # Merge test results into main archive.
        for e in test_archive.entries:
            archive.add(copy.deepcopy(e))

    # ----- Save -----------------------------------------------------------
    total_time = time.perf_counter() - t_start
    logger.info("G-LNS completed in %.1f seconds", total_time)
    _save_results(cfg.output_dir, archive, pop, gen_log, cfg, tracking_instance_id)

    return archive
