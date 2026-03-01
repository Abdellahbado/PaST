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
import signal
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
from glns.evaluation_v2 import run_evaluation_phase_v2
from glns.evolution import evolve_generation
from glns.llm_client import GroqOperatorClient
from glns.pareto import ArchiveEntry, ParetoArchive
from glns.population import PopulationManager
from glns.sanity import sanity_check, sanity_check_assignment
from glns.schemas import OperatorRecord
from glns.seed_operators import (
    build_seed_destroy_operators,
    build_seed_repair_operators,
)
from glns.seed_operators_v2 import (
    build_seed_destroy_operators_v2,
    build_seed_repair_operators_v2,
)
from glns.sequencing import (
    evaluate_assignment,
    make_initial_assignment,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instance split helpers
# ---------------------------------------------------------------------------


def _default_evolution_ids(instances: List[dict]) -> List[int]:
    """Auto-select small + ALL medium instances for evolution."""
    small = [inst["instance_id"] for inst in instances if inst.get("scale") == "small"]
    mls = [inst["instance_id"] for inst in instances if inst.get("scale") == "mls"]
    return small + mls


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
# Initialisation helpers (assignment-only v2)
# ---------------------------------------------------------------------------


def _seed_archive_v2(
    instances: List[dict],
    archive: ParetoArchive,
    rng: random.Random,
    sequencing_mode: str = "auto",
) -> None:
    """Build initial solutions via LPT assignment + optimal DP sequencing."""
    for inst in instances[:8]:
        inst_id = int(inst.get("instance_id", 0))
        assign = make_initial_assignment(inst)
        energy, cmax, seqs, starts = evaluate_assignment(
            assign,
            inst,
            sequencing_mode=sequencing_mode,
        )
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


def _init_populations_v2(
    pop: PopulationManager,
    llm: Optional[GroqOperatorClient],
    cfg: GLNSConfig,
) -> None:
    """Inject assignment-only seed operators and optionally LLM top-up."""
    for op in build_seed_destroy_operators_v2():
        pop.destroy_pool.add(op)
    for op in build_seed_repair_operators_v2():
        pop.repair_pool.add(op)

    # Top-up with LLM if pools not full.
    d_need = pop.destroy_pool.empty_slots()
    r_need = pop.repair_pool.empty_slots()
    if (d_need > 0 or r_need > 0) and llm is not None:
        logger.info(
            "Requesting %d destroy + %d repair operators from LLM for initialisation (v2 assignment mode)",
            d_need,
            r_need,
        )
        try:
            batch = llm.generate_init_batch(n_destroy=d_need, n_repair=r_need)
            for spec in batch.operators:
                passed, err = sanity_check_assignment(spec, cfg.sandbox)
                if passed:
                    rec = OperatorRecord(spec=spec, generation_born=0)
                    pool = (
                        pop.destroy_pool if spec.type == "destroy" else pop.repair_pool
                    )
                    pool.add(rec)
                else:
                    logger.warning("LLM init operator (v2) failed sanity: %s", err)
        except Exception as exc:
            logger.error("LLM initialisation call (v2) failed: %s", exc)

    # Fallback: duplicate seed operators.
    while pop.destroy_pool.empty_slots() > 0:
        seeds = build_seed_destroy_operators_v2()
        fb = seeds[len(pop.destroy_pool) % len(seeds)]
        fb.id = f"dup_d{len(pop.destroy_pool)}"  # type: ignore[assignment]
        pop.destroy_pool.add(fb)
    while pop.repair_pool.empty_slots() > 0:
        seeds = build_seed_repair_operators_v2()
        fb = seeds[len(pop.repair_pool) % len(seeds)]
        fb.id = f"dup_r{len(pop.repair_pool)}"  # type: ignore[assignment]
        pop.repair_pool.add(fb)


# ---------------------------------------------------------------------------
# Logging / persistence
# ---------------------------------------------------------------------------


def _compute_aggregate_hv(
    archive: ParetoArchive,
    inst_ref: Dict[int, Tuple[int, float]],
) -> Tuple[float, int, Dict[int, float]]:
    """Compute per-instance HV and return (mean_hv, n_instances_with_entries, per_inst_hv).

    Each instance uses its own reference point so HV values are scale-appropriate.
    Mean is taken over instances that have at least 1 archive entry.
    """
    per_inst_hv: Dict[int, float] = {}
    for iid, (rc, rt) in inst_ref.items():
        if archive.size(iid) > 0:
            per_inst_hv[iid] = archive.hypervolume(rc, rt, instance_id=iid)
    n = len(per_inst_hv)
    mean_hv = sum(per_inst_hv.values()) / n if n > 0 else 0.0
    return mean_hv, n, per_inst_hv


def _log_generation(
    gen: int,
    archive: ParetoArchive,
    pop: PopulationManager,
    elapsed_sec: float,
    ref_cmax: int,
    ref_tec: float,
    tracking_instance_id: int,
    prev_hv: Optional[float],
    inst_ref: Optional[Dict[int, Tuple[int, float]]] = None,
) -> Dict:
    hv = archive.hypervolume(ref_cmax, ref_tec, instance_id=tracking_instance_id)
    front = archive.front(instance_id=tracking_instance_id)
    size_total = archive.size()
    size_inst = archive.size(tracking_instance_id)
    hv_delta = (hv - prev_hv) if prev_hv is not None else None

    # Aggregate HV across all evolution instances.
    agg_hv = 0.0
    agg_n = 0
    if inst_ref:
        agg_hv, agg_n, _ = _compute_aggregate_hv(archive, inst_ref)

    entry = {
        "generation": gen + 1,
        "tracking_instance_id": tracking_instance_id,
        "archive_size": size_inst,
        "archive_size_total": size_total,
        "hypervolume": hv,
        "hypervolume_delta": hv_delta,
        "agg_hv_mean": round(agg_hv, 2),
        "agg_hv_n_instances": agg_n,
        "front": front,
        "n_destroy": len(pop.destroy_pool),
        "n_repair": len(pop.repair_pool),
        "elapsed_sec": round(elapsed_sec, 2),
    }
    logger.info(
        "Gen %3d | inst %d | archive %d (total %d) | HV %.2f%s | aggHV %.2f (%d inst) | front[0]=%s | front[-1]=%s | %.1fs",
        gen + 1,
        tracking_instance_id,
        size_inst,
        size_total,
        hv,
        f" (Δ{hv_delta:+.2f})" if hv_delta is not None else "",
        agg_hv,
        agg_n,
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
    benchmark_eval_archive: Optional["ParetoArchive"] = None,
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

    # Benchmark adaptation eval archive (held-out eval set — separate from main archive).
    if benchmark_eval_archive is not None:
        eval_data = []
        for e in benchmark_eval_archive.entries:
            eval_data.append(
                {
                    "instance_id": e.instance_id,
                    "makespan": e.makespan,
                    "energy": e.energy,
                    "sequences": e.sequences,
                    "start_times": e.start_times,
                }
            )
        with open(output_dir / "archive_benchmark_eval.json", "w") as f:
            json.dump(eval_data, f, indent=2)
        logger.info(
            "Benchmark eval archive saved to %s (%d entries)",
            output_dir / "archive_benchmark_eval.json",
            len(eval_data),
        )

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

    if cfg.assignment_mode:
        run_evaluation_phase_v2(
            destroy_pool=list(pop.destroy_pool),
            repair_pool=list(pop.repair_pool),
            instances=test_instances,
            archive=test_archive,
            eval_cfg=test_eval_cfg,
            sandbox_cfg=cfg.sandbox,
            rng=rng,
            sequencing_mode=cfg.sequencing_mode,
            n_workers=cfg.max_workers,
        )
    else:
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


def run_benchmark_probe(
    pop: PopulationManager,
    benchmark_instances: List[dict],
    cfg: GLNSConfig,
    rng: random.Random,
    label: str = "benchmark",
) -> Tuple[ParetoArchive, float, int]:
    """Run a lightweight ALNS evaluation on external benchmark instances.

    Returns (probe_archive, mean_hv, n_instances_with_entries).
    """
    probe_eval = cfg.eval.model_copy()
    probe_eval.K_episodes = cfg.eval.benchmark_eval_episodes
    probe_eval.T_iters = cfg.eval.benchmark_eval_iters

    probe_archive = ParetoArchive(max_size=cfg.archive.max_size * 2)

    if cfg.assignment_mode:
        run_evaluation_phase_v2(
            destroy_pool=list(pop.destroy_pool),
            repair_pool=list(pop.repair_pool),
            instances=benchmark_instances,
            archive=probe_archive,
            eval_cfg=probe_eval,
            sandbox_cfg=cfg.sandbox,
            rng=rng,
            sequencing_mode=cfg.sequencing_mode,
            n_workers=cfg.max_workers,
        )
    else:
        run_evaluation_phase(
            destroy_pool=list(pop.destroy_pool),
            repair_pool=list(pop.repair_pool),
            instances=benchmark_instances,
            archive=probe_archive,
            eval_cfg=probe_eval,
            sandbox_cfg=cfg.sandbox,
            rng=rng,
        )

    # Compute per-instance HV with instance-specific ref points.
    bm_ref: Dict[int, Tuple[int, float]] = {}
    for inst in benchmark_instances:
        iid = int(inst["instance_id"])
        bm_ref[iid] = (inst["T"], float(inst["T"] * max(inst["e"]) * max(inst["ct"])))
    mean_hv, n_inst, _ = _compute_aggregate_hv(probe_archive, bm_ref)

    logger.info(
        "Benchmark probe [%s]: %d entries across %d instances | mean HV = %.2f",
        label,
        probe_archive.size(),
        n_inst,
        mean_hv,
    )
    return probe_archive, mean_hv, n_inst


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
    benchmark_instances: List[dict] = []

    if (
        cfg.instances.benchmark_adaptation
        and cfg.instances.benchmark_data_dir is not None
    ):
        # ── Benchmark-adaptation mode: evolve directly on Wang2018 ──────────
        # Load ALL Wang2018 instances, then stratified-split into evo / eval.
        bm_path = str(cfg.instances.benchmark_data_dir)
        if not os.path.isabs(bm_path):
            project_root = Path(__file__).resolve().parent.parent
            bm_path = str(project_root / bm_path)

        if os.path.isdir(bm_path):
            from glns.benchmark_loader import (
                load_benchmark_instances as _load_bm,
                split_benchmark_instances as _split_bm,
            )

            all_bm_instances = _load_bm(bm_path)
            evo_instances, benchmark_instances = _split_bm(
                all_bm_instances,
                evo_fraction=cfg.instances.benchmark_adaptation_evo_frac,
                seed=cfg.instances.benchmark_adaptation_split_seed,
            )
            test_instances = []  # no separate internal test set in this mode
            logger.info(
                "BENCHMARK ADAPTATION MODE: evolving on %d Wang2018 instances | "
                "held-out evaluation: %d instances",
                len(evo_instances),
                len(benchmark_instances),
            )
        else:
            logger.error(
                "benchmark_adaptation=True but benchmark_data_dir not found: %s — "
                "falling back to standard mode",
                bm_path,
            )
            cfg.instances.benchmark_adaptation = False  # type: ignore[assignment]
            evo_instances = []
            test_instances = []

    if not cfg.instances.benchmark_adaptation or not evo_instances:
        # ── Standard mode: evolve on instances_90.json ──────────────────────
        instances_path = str(cfg.instances.instances_json)
        if not os.path.isabs(instances_path):
            project_root = Path(__file__).resolve().parent.parent
            instances_path = str(project_root / instances_path)

        logger.info("Loading instances from %s", instances_path)
        all_instances = load_instances_from_json(instances_path)
        evo_instances, test_instances = _split_instances(
            all_instances, cfg.instances.evolution_ids, cfg.instances.test_ids
        )
        logger.info(
            "Evolution: %d instances | Test (internal): %d instances",
            len(evo_instances),
            len(test_instances),
        )

        # Load external Wang2018 benchmark (for probing only — no evolution).
        if cfg.instances.benchmark_data_dir is not None:
            bm_path = str(cfg.instances.benchmark_data_dir)
            if not os.path.isabs(bm_path):
                project_root = Path(__file__).resolve().parent.parent
                bm_path = str(project_root / bm_path)
            if os.path.isdir(bm_path):
                from glns.benchmark_loader import load_benchmark_instances as _load_bm

                benchmark_instances = _load_bm(bm_path)
                logger.info(
                    "External benchmark: %d instances from %s",
                    len(benchmark_instances),
                    bm_path,
                )
            else:
                logger.warning(
                    "Benchmark data dir not found: %s — skipping external benchmark",
                    bm_path,
                )

    tracking_instance_id = int(evo_instances[0]["instance_id"]) if evo_instances else 0

    # ----- Initialise archive, populations, LLM --------------------------
    archive = ParetoArchive(max_size=cfg.archive.max_size)
    if cfg.assignment_mode:
        _seed_archive_v2(
            evo_instances, archive, rng, sequencing_mode=cfg.sequencing_mode
        )
        logger.info(
            "ASSIGNMENT MODE: seeded archive with %d initial solutions (tracking instance %d has %d)",
            archive.size(),
            tracking_instance_id,
            archive.size(tracking_instance_id),
        )
    else:
        _seed_archive(evo_instances, archive, rng)
        logger.info(
            "Seeded archive with %d initial solutions (tracking instance %d has %d)",
            archive.size(),
            tracking_instance_id,
            archive.size(tracking_instance_id),
        )

    pop = PopulationManager(cfg.population)

    multi_keys = os.environ.get("GROQ_API_KEYS", "").strip()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    llm: Optional[GroqOperatorClient] = None
    if multi_keys:
        # Let the client parse GROQ_API_KEYS from the environment.
        llm = GroqOperatorClient(cfg.llm, api_key=None)
        logger.info("Using GROQ_API_KEYS for LLM (multi-key rotation enabled)")
    elif api_key:
        llm = GroqOperatorClient(cfg.llm, api_key=api_key)
    else:
        logger.warning(
            "Neither GROQ_API_KEY nor GROQ_API_KEYS set — running with seed operators only (no LLM evolution)"
        )

    if cfg.assignment_mode:
        _init_populations_v2(pop, llm, cfg)
    else:
        _init_populations(pop, llm, cfg)
    logger.info(
        "Populations initialised: %d destroy, %d repair",
        len(pop.destroy_pool),
        len(pop.repair_pool),
    )

    # Reference point for hypervolume logging (tracking instance).
    ref_cmax = max(inst["T"] for inst in evo_instances)
    ref_tec = float(
        ref_cmax
        * max(max(inst["e"]) for inst in evo_instances)
        * max(max(inst["ct"]) for inst in evo_instances)
    )

    # Per-instance reference points for aggregate HV (benchmark-style).
    # Each instance gets its own ref point: (T, max_e * max_ct * T).
    inst_ref: Dict[int, Tuple[int, float]] = {}
    for inst in evo_instances:
        iid = int(inst["instance_id"])
        inst_T = inst["T"]
        inst_max_e = max(inst["e"])
        inst_max_ct = max(inst["ct"])
        inst_ref[iid] = (inst_T, float(inst_T * inst_max_e * inst_max_ct))

    gen_log: List[Dict] = []
    prev_hv: Optional[float] = None
    best_hv: Optional[float] = None
    hv_stale_gens = 0

    # ----- Adaptive SA T0 state -------------------------------------------
    effective_sa_T0 = cfg.eval.sa_T0  # starts at config value (0.5)
    sa_T0_floor = 0.01  # never go below this (raised from 1e-4; with
    # instance-based normalization deltas are ~0.01-0.05, so
    # T0=0.01 gives exp(-0.02/0.01) ≈ 0.14 … meaningful rejection)
    sa_T0_ceiling = cfg.eval.sa_T0  # never exceed initial value
    sa_floor_stuck_gens = 0  # counts consecutive gens at floor with high accept

    # ----- Stagnation patience & wall-clock budget -----------------------
    stagnation_gens = 0  # consecutive gens with no HV improvement (for patience)
    t_run_start = time.perf_counter()  # wall-clock start for time_limit_hours

    # Allow Ctrl+C to stop the loop cleanly and still save results.
    _stop_requested = False
    _completed_gens = 0

    def _sigint_handler(signum, frame):  # noqa: ANN001
        nonlocal _stop_requested
        if not _stop_requested:
            _stop_requested = True
            logger.warning(
                "Ctrl+C received — will stop after the current generation and save results."
            )
        else:
            # Second Ctrl+C: hard exit.
            logger.error("Second interrupt — forcing exit without saving.")
            raise SystemExit(1)

    _prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    # ----- Main evolutionary loop ----------------------------------------
    for gen in range(cfg.evolution.G_max):
        t_gen = time.perf_counter()

        # ---- Wall-clock time limit check ---------------------------------
        if cfg.evolution.time_limit_hours > 0:
            elapsed_h = (t_gen - t_run_start) / 3600.0
            if elapsed_h >= cfg.evolution.time_limit_hours:
                logger.info(
                    "Time limit %.2f h reached after %d gens (elapsed %.2f h) — stopping.",
                    cfg.evolution.time_limit_hours,
                    gen,
                    elapsed_h,
                )
                _stop_requested = True
                break

        logger.info(
            "=== Generation %d/%d | inst=%d | archive=%d (total %d) | pools: D=%d R=%d | sa_T0_eff=%.4f | stale=%d ===",
            gen + 1,
            cfg.evolution.G_max,
            tracking_instance_id,
            archive.size(tracking_instance_id),
            archive.size(),
            len(pop.destroy_pool),
            len(pop.repair_pool),
            effective_sa_T0,
            stagnation_gens,
        )

        # Phase 1: Evaluate — with generation-adapted SA temperature.
        # Adaptive destroy ratio: sample uniformly in [min, max] each generation.
        dr_min = cfg.eval.destroy_ratio_min
        dr_max = cfg.eval.destroy_ratio_max
        if dr_min < dr_max:
            sampled_dr = dr_min + rng.random() * (dr_max - dr_min)
        else:
            sampled_dr = cfg.eval.destroy_ratio
        gen_eval_cfg = cfg.eval.model_copy(
            update={"sa_T0": effective_sa_T0, "destroy_ratio": sampled_dr}
        )
        if cfg.assignment_mode:
            F_d, F_r, synergy, eval_stats = run_evaluation_phase_v2(
                destroy_pool=list(pop.destroy_pool),
                repair_pool=list(pop.repair_pool),
                instances=evo_instances,
                archive=archive,
                eval_cfg=gen_eval_cfg,
                sandbox_cfg=cfg.sandbox,
                rng=rng,
                sequencing_mode=cfg.sequencing_mode,
                n_workers=cfg.max_workers,
            )
        else:
            F_d, F_r, synergy, eval_stats = run_evaluation_phase(
                destroy_pool=list(pop.destroy_pool),
                repair_pool=list(pop.repair_pool),
                instances=evo_instances,
                archive=archive,
                eval_cfg=gen_eval_cfg,
                sandbox_cfg=cfg.sandbox,
                rng=rng,
            )
        sa_rate = float(eval_stats.get("sa_rate", 0.0))

        # Adaptive SA T0 adjustment: prevent runaway acceptance.
        if cfg.eval.sa_adaptive:
            at_floor = effective_sa_T0 <= sa_T0_floor * 1.01
            if at_floor and sa_rate > 0.70:
                # Stuck at floor with high acceptance → reheat.
                # This triggers when normalisation-based deltas are still small
                # even after the instance-based fix (e.g. very tight fronts).
                sa_floor_stuck_gens += 1
                if sa_floor_stuck_gens >= 3:
                    reheat_target = min(sa_T0_ceiling, sa_T0_floor * 10.0)
                    effective_sa_T0 = reheat_target
                    sa_floor_stuck_gens = 0
                    logger.info(
                        "Gen %d: SA accept %.1f%% stuck at floor for 3 gens → REHEAT sa_T0_eff to %.4f",
                        gen + 1,
                        100.0 * sa_rate,
                        effective_sa_T0,
                    )
                else:
                    logger.info(
                        "Gen %d: SA accept %.1f%% at floor (stuck %d/3 gens)",
                        gen + 1,
                        100.0 * sa_rate,
                        sa_floor_stuck_gens,
                    )
            elif sa_rate > 0.70:
                # Way too hot — halve T0.
                effective_sa_T0 = max(sa_T0_floor, effective_sa_T0 * 0.5)
                sa_floor_stuck_gens = 0
                logger.info(
                    "Gen %d: SA accept %.1f%% (too high) → halved sa_T0_eff to %.4f",
                    gen + 1,
                    100.0 * sa_rate,
                    effective_sa_T0,
                )
            elif sa_rate > 0.50:
                # Mildly hot — apply normal per-generation decay.
                effective_sa_T0 = max(
                    sa_T0_floor, effective_sa_T0 * cfg.eval.sa_T0_gen_decay
                )
                sa_floor_stuck_gens = 0
                logger.info(
                    "Gen %d: SA accept %.1f%% (warm) → decayed sa_T0_eff to %.4f",
                    gen + 1,
                    100.0 * sa_rate,
                    effective_sa_T0,
                )
            elif sa_rate < 0.10:
                # Too cold — warm up a bit (but cap at initial value).
                effective_sa_T0 = min(sa_T0_ceiling, effective_sa_T0 * 1.3)
                sa_floor_stuck_gens = 0
                logger.info(
                    "Gen %d: SA accept %.1f%% (cold) → warmed sa_T0_eff to %.4f",
                    gen + 1,
                    100.0 * sa_rate,
                    effective_sa_T0,
                )
            else:
                # 10-50% is the healthy zone, keep T0 as is.
                sa_floor_stuck_gens = 0
        else:
            # No adaptive: just apply fixed per-generation decay.
            effective_sa_T0 = max(
                sa_T0_floor, effective_sa_T0 * cfg.eval.sa_T0_gen_decay
            )
            if sa_rate >= 0.97:
                logger.warning(
                    "Gen %d: SA acceptance is very high (%.1f%%).",
                    gen + 1,
                    100.0 * sa_rate,
                )
            elif sa_rate >= 0.92:
                logger.info(
                    "Gen %d: SA acceptance %.1f%%",
                    gen + 1,
                    100.0 * sa_rate,
                )
        pop.update_metrics(F_d, F_r, synergy)

        # Build concise search context for the LLM (helps diversify under stagnation).
        # Keep it short to avoid token burn.
        front = archive.front(instance_id=tracking_instance_id)
        hv_now = archive.hypervolume(
            ref_cmax, ref_tec, instance_id=tracking_instance_id
        )
        hv_delta_prev = (hv_now - prev_hv) if prev_hv is not None else None
        ctx_lines = [
            f"gen={gen+1} tracking_instance_id={tracking_instance_id}",
            f"HV={hv_now:.2f}"
            + (f" (Δprev={hv_delta_prev:+.2f})" if hv_delta_prev is not None else ""),
            f"hv_stale_gens={hv_stale_gens}",
            f"front[0]={front[0] if front else '?'} front[-1]={front[-1] if front else '?'}",
            f"archive_size_inst={archive.size(tracking_instance_id)} archive_size_total={archive.size()}",
            f"SA_accept={100.0*sa_rate:.1f}% sa_T0_eff={effective_sa_T0:.4f} worseΔmean={float(eval_stats.get('worse_delta_mean', 0.0)):.4f}",
            f"invalid={int(eval_stats.get('invalid_candidate', 0))} inf={int(eval_stats.get('inf_eval', 0))} destroy_fail={int(eval_stats.get('destroy_fail', 0))} repair_fail={int(eval_stats.get('repair_fail', 0))}",
        ]
        search_context = "\n".join(ctx_lines)

        # Phase 2: Prune.
        pruned = pop.rank_and_prune()

        # Phase 3+4: Evolve (one batched LLM call) + validate.
        if llm is not None and pruned > 0:
            outcome = evolve_generation(
                pop,
                llm,
                cfg.evolution,
                cfg.sandbox,
                gen,
                rng,
                search_context=(
                    search_context if cfg.llm.include_search_context else None
                ),
                assignment_mode=cfg.assignment_mode,
            )
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
                    assignment_mode=cfg.assignment_mode,
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
            prev_hv,
            inst_ref=inst_ref,
        )
        prev_hv = float(entry["hypervolume"])
        if best_hv is None or prev_hv > best_hv + 1e-9:
            best_hv = prev_hv
            hv_stale_gens = 0
            stagnation_gens = 0
        else:
            hv_stale_gens += 1
            stagnation_gens += 1
            if hv_stale_gens in (10, 25, 50):
                logger.warning(
                    "HV has not improved for %d generations (tracking inst %d).",
                    hv_stale_gens,
                    tracking_instance_id,
                )

        # ---- Stagnation patience: forced SA reheat -----------------------
        patience = cfg.evolution.stagnation_patience
        if patience > 0 and stagnation_gens >= patience:
            reheat_target = min(
                sa_T0_ceiling,
                effective_sa_T0 * cfg.evolution.stagnation_reheat_factor,
            )
            effective_sa_T0 = reheat_target
            stagnation_gens = 0
            logger.info(
                "Gen %d: STAGNATION patience=%d reached → FORCED REHEAT sa_T0_eff to %.4f",
                gen + 1,
                patience,
                reheat_target,
            )
        gen_log.append(entry)

        # ----- Periodic benchmark probe (dual-benchmark testing) ----------
        bm_every = cfg.eval.benchmark_eval_every
        if bm_every > 0 and (gen + 1) % bm_every == 0:
            # Probe on internal test instances from instances_90.json.
            if test_instances:
                _, int_hv, int_n = run_benchmark_probe(
                    pop, test_instances, cfg, rng, label="internal_test"
                )
                entry["probe_internal_hv"] = round(int_hv, 2)
                entry["probe_internal_n"] = int_n

            # Probe on external Wang2018 benchmark.
            if benchmark_instances:
                _, ext_hv, ext_n = run_benchmark_probe(
                    pop, benchmark_instances, cfg, rng, label="external_benchmark"
                )
                entry["probe_external_hv"] = round(ext_hv, 2)
                entry["probe_external_n"] = ext_n

        _completed_gens += 1
        if _stop_requested:
            logger.info(
                "Stopping evolutionary loop at generation %d (user request).", gen + 1
            )
            break

    # Restore the original SIGINT handler.
    signal.signal(signal.SIGINT, _prev_handler)

    # ----- Final test phase (skipped if interrupted) ----------------------
    _bm_eval_archive: Optional[ParetoArchive] = (
        None  # set below if benchmark_instances exist
    )
    if _stop_requested:
        logger.info(
            "Skipping final evaluation phase because run was interrupted after %d/%d generations.",
            _completed_gens,
            cfg.evolution.G_max,
        )
    if not _stop_requested:
        # ----- Final test phase -----------------------------------------------
        logger.info("=== FINAL EVALUATION ===")

        # Internal test (instances_90.json held-out).
        if test_instances:
            test_archive = run_test_phase(pop, test_instances, archive, cfg, rng)
            # Merge test results into main archive.
            for e in test_archive.entries:
                archive.add(copy.deepcopy(e))

        # External benchmark (Wang2018 original) / benchmark-adaptation eval set.
        final_ext_archive: Optional[ParetoArchive] = None
        if benchmark_instances:
            final_ext_archive, ext_hv, ext_n = run_benchmark_probe(
                pop, benchmark_instances, cfg, rng, label="final_external"
            )
            # Run heavier evaluation for final external benchmark.
            # T_final / K_final allow a separate, heavier budget for the last pass.
            final_ext_eval = cfg.eval.model_copy()
            final_ext_eval.K_episodes = (
                cfg.eval.K_final
                if cfg.eval.K_final > 0
                else max(10, cfg.eval.K_episodes)
            )
            final_ext_eval.T_iters = (
                cfg.eval.T_final if cfg.eval.T_final > 0 else cfg.eval.T_test
            )
            ext_test_archive = ParetoArchive(max_size=cfg.archive.max_size * 2)
            if final_ext_archive:
                for e_entry in final_ext_archive.entries:
                    ext_test_archive.add(copy.deepcopy(e_entry))
            if cfg.assignment_mode:
                run_evaluation_phase_v2(
                    destroy_pool=list(pop.destroy_pool),
                    repair_pool=list(pop.repair_pool),
                    instances=benchmark_instances,
                    archive=ext_test_archive,
                    eval_cfg=final_ext_eval,
                    sandbox_cfg=cfg.sandbox,
                    rng=rng,
                    sequencing_mode=cfg.sequencing_mode,
                    n_workers=cfg.max_workers,
                )
            else:
                run_evaluation_phase(
                    destroy_pool=list(pop.destroy_pool),
                    repair_pool=list(pop.repair_pool),
                    instances=benchmark_instances,
                    archive=ext_test_archive,
                    eval_cfg=final_ext_eval,
                    sandbox_cfg=cfg.sandbox,
                    rng=rng,
                )
            # Keep a reference so we can save it alongside archive_full.json.
            _bm_eval_archive = ext_test_archive
            # Compute final external HV.
            bm_ref_final: Dict[int, Tuple[int, float]] = {}
            for inst in benchmark_instances:
                iid = int(inst["instance_id"])
                bm_ref_final[iid] = (
                    inst["T"],
                    float(inst["T"] * max(inst["e"]) * max(inst["ct"])),
                )
            ext_mean_hv, ext_n_final, ext_per_inst = _compute_aggregate_hv(
                ext_test_archive, bm_ref_final
            )
            logger.info(
                "FINAL external benchmark: %d entries across %d instances | mean HV = %.2f",
                ext_test_archive.size(),
                ext_n_final,
                ext_mean_hv,
            )

    # ----- Save (always — even on interrupt) ------------------------------
    total_time = time.perf_counter() - t_start
    status = "interrupted" if _stop_requested else "completed"
    logger.info(
        "G-LNS %s in %.1f seconds (%d/%d generations) — saving results to %s",
        status,
        total_time,
        _completed_gens,
        cfg.evolution.G_max,
        cfg.output_dir,
    )
    _save_results(
        cfg.output_dir,
        archive,
        pop,
        gen_log,
        cfg,
        tracking_instance_id,
        benchmark_eval_archive=_bm_eval_archive,
    )

    return archive
