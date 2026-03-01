#!/usr/bin/env python3
"""CLI entry point for G-LNS.

Usage
-----
    # Basic run with defaults (needs GROQ_API_KEY env var):
    python run_glns.py

    # Custom seed and output directory:
    python run_glns.py --seed 123 --output-dir results/glns_s123

    # From a YAML config file:
    python run_glns.py --config configs/glns_small.yaml

    # Override specific options:
    python run_glns.py --generations 50 --K-episodes 5 --T-iters 50

    # Seed-only mode (no LLM, just for testing the harness):
    GROQ_API_KEY="" python run_glns.py --generations 5 --T-iters 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="G-LNS: Generative Large Neighborhood Search for bi-objective scheduling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file — overrides everything.
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON config file (overrides all CLI flags)",
    )

    # Core.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("results/glns"))
    p.add_argument(
        "--instances", type=Path, default=Path("New Benchmark/instances_90.json")
    )

    # ---- HPC convenience preset ------------------------------------------
    # Sets sensible high-resource defaults; individual flags below still
    # override the preset when supplied explicitly after --hpc.
    p.add_argument(
        "--hpc",
        action="store_true",
        default=False,
        help=(
            "HPC super-preset: sets generations=500, K-episodes=30, T-iters=200, "
            "T-test=1500, T-final=2000, K-final=20, pop-size=8, prune-count=3, "
            "sandbox-timeout=15, sandbox-memory=4096, archive-max-size=200, "
            "benchmark-eval-every=25, benchmark-eval-episodes=10, benchmark-eval-iters=100. "
            "Individual flags override the preset."
        ),
    )

    # LLM.
    p.add_argument("--model", type=str, default="moonshotai/Kimi-K2-Instruct-0905")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=8_000)

    # Evolution.
    p.add_argument("--generations", type=int, default=None, dest="G_max")
    p.add_argument("--pop-size", type=int, default=None, dest="N_pop")
    p.add_argument("--prune-count", type=int, default=None, dest="M_prune")

    # HPC: wall-clock budget & stagnation control.
    p.add_argument(
        "--time-limit-hours",
        type=float,
        default=None,
        dest="time_limit_hours",
        help="Stop after N hours of wall-clock time (0 = unlimited)",
    )
    p.add_argument(
        "--stagnation-patience",
        type=int,
        default=None,
        dest="stagnation_patience",
        help="Gens with no HV improvement before forced reheat (0=disabled, default=20)",
    )
    p.add_argument(
        "--stagnation-reheat-factor",
        type=float,
        default=None,
        dest="stagnation_reheat_factor",
        help="SA reheat multiplier on stagnation trigger (default=8.0)",
    )

    # Evaluation.
    p.add_argument("--K-episodes", type=int, default=None)
    p.add_argument("--T-iters", type=int, default=None)
    p.add_argument("--T-test", type=int, default=None)
    p.add_argument(
        "--T-final",
        type=int,
        default=None,
        help="ALNS iters for the very last benchmark eval (0 = use T-test)",
    )
    p.add_argument(
        "--K-final",
        type=int,
        default=None,
        help="Episodes for the very last benchmark eval (0 = use max(10,K-episodes))",
    )
    p.add_argument("--sa-T0", type=float, default=0.5, help="SA initial temperature")
    p.add_argument(
        "--sa-alpha", type=float, default=0.98, help="SA cooling rate per iteration"
    )
    p.add_argument(
        "--sa-T0-gen-decay",
        type=float,
        default=0.97,
        help="Per-generation multiplicative decay on effective T0",
    )
    p.add_argument(
        "--no-sa-adaptive",
        action="store_true",
        default=False,
        help="Disable adaptive SA T0 adjustment (use fixed decay only)",
    )
    p.add_argument(
        "--destroy-ratio",
        type=float,
        default=None,
        help="Fixed destruction ratio (overrides adaptive min/max if set alone)",
    )
    p.add_argument(
        "--destroy-ratio-min",
        type=float,
        default=None,
        dest="destroy_ratio_min",
        help="Min adaptive destruction ratio (default 0.10)",
    )
    p.add_argument(
        "--destroy-ratio-max",
        type=float,
        default=None,
        dest="destroy_ratio_max",
        help="Max adaptive destruction ratio (default 0.40)",
    )
    p.add_argument(
        "--lambda-smooth",
        type=float,
        default=None,
        dest="lambda_smooth",
        help="ALNS weight smoothing factor lambda (default 0.5)",
    )
    p.add_argument(
        "--sigma",
        type=str,
        default=None,
        help=(
            "Scoring vector as 4 comma-separated floats, e.g. '1.5,1.2,0.8,0.1'. "
            "sigma1=new Pareto, sigma2=better scalarised, sigma3=accepted, sigma4=rejected."
        ),
    )
    p.add_argument(
        "--n-reference-ops",
        type=int,
        default=None,
        dest="n_reference_ops",
        help="Number of top operators to include as reference in LLM prompt (default 2)",
    )
    p.add_argument(
        "--search-context-max-chars",
        type=int,
        default=None,
        dest="search_context_max_chars",
        help="Max chars of search-state context sent to LLM each generation (default 900)",
    )
    p.add_argument(
        "--benchmark-eval-every",
        type=int,
        default=None,
        help="Run dual-benchmark probe every N generations (0=disabled)",
    )
    p.add_argument(
        "--benchmark-eval-episodes",
        type=int,
        default=None,
        help="ALNS episodes per benchmark probe",
    )
    p.add_argument(
        "--benchmark-eval-iters",
        type=int,
        default=None,
        help="ALNS iterations per benchmark probe episode",
    )

    # Sandbox.
    p.add_argument("--sandbox-timeout", type=float, default=None)
    p.add_argument("--sandbox-memory-mb", type=int, default=None)
    p.add_argument(
        "--sandbox-start-method",
        type=str,
        default="auto",
        choices=["auto", "fork", "spawn", "forkserver"],
        help="Multiprocessing start method for sandbox workers",
    )

    # Archive.
    p.add_argument("--archive-max-size", type=int, default=None)

    # Benchmark data.
    p.add_argument(
        "--benchmark-data-dir",
        type=Path,
        default=Path("Benchmark/Data"),
        help="Path to original Wang2018 benchmark Data folder (set to '' to disable)",
    )

    # Benchmark adaptation mode.
    p.add_argument(
        "--benchmark-adaptation",
        action="store_true",
        default=False,
        help=(
            "Enable benchmark-adaptation (test-time adaptation) mode: "
            "evolve LLM operators directly on the Wang2018 benchmark instead of "
            "instances_90.json. The benchmark is split stratified-by-scale "
            "into evolution (~67%%) and held-out evaluation (~33%%) sets."
        ),
    )
    p.add_argument(
        "--benchmark-adaptation-evo-frac",
        type=float,
        default=2 / 3,
        dest="benchmark_adaptation_evo_frac",
        help=(
            "Fraction of each scale group used for evolution in benchmark-adaptation "
            "mode. Default 0.667 → 20 evolution + 10 evaluation per scale."
        ),
    )
    p.add_argument(
        "--benchmark-adaptation-split-seed",
        type=int,
        default=42,
        dest="benchmark_adaptation_split_seed",
        help="Random seed for the stratified evolution/evaluation split.",
    )

    # Assignment-only decomposition mode (v2).
    p.add_argument(
        "--assignment-mode",
        action="store_true",
        default=False,
        help=(
            "Use assignment-only operators + optimal DP sequencing. "
            "LLM operators work on List[int] assignments instead of List[List[int]] sequences. "
            "This is the recommended mode for the Wang2018 benchmark."
        ),
    )
    p.add_argument(
        "--sequencing-mode",
        type=str,
        default="auto",
        choices=["auto", "optimal", "heuristic"],
        dest="sequencing_mode",
        help=(
            "How to solve per-machine sequencing in assignment mode: "
            "'auto' (default) uses optimal DP when state space ≤ 500K, else heuristic."
        ),
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=0,
        dest="max_workers",
        help="Number of parallel workers for DP evaluation (0 = auto; use all CPU cores).",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Lazy imports so --help is fast.
    from glns.config import (
        ArchiveConfig,
        EvalConfig,
        EvolutionConfig,
        GLNSConfig,
        InstanceConfig,
        LLMConfig,
        PopulationConfig,
        SandboxConfig,
    )
    from glns.runner import run_glns

    if args.config is not None:
        # Load config from JSON file.
        with open(args.config) as f:
            raw = json.load(f)
        cfg = GLNSConfig(**raw)
    else:
        # ---- HPC preset: establish baseline values -------------------------
        # Start with the pydantic model defaults, then layer the preset on top,
        # then layer any explicit CLI overrides last (those arrive as non-None).
        _defaults = GLNSConfig()

        # Helper — picks the first non-None value in priority order.
        def _pick(*values, fallback=None):
            for v in values:
                if v is not None:
                    return v
            return fallback

        if args.hpc:
            hpc_G_max = 500
            hpc_K_ep = 30
            hpc_T_iters = 200
            hpc_T_test = 1500
            hpc_T_final = 2000
            hpc_K_final = 20
            hpc_N_pop = 8
            hpc_M_prune = 3
            hpc_sb_to = 15.0
            hpc_sb_mem = 4096
            hpc_arch = 200
            hpc_be_ev = 25
            hpc_be_ep = 10
            hpc_be_it = 100
        else:
            hpc_G_max = hpc_K_ep = hpc_T_iters = hpc_T_test = hpc_T_final = None
            hpc_K_final = hpc_N_pop = hpc_M_prune = hpc_sb_to = hpc_sb_mem = None
            hpc_arch = hpc_be_ev = hpc_be_ep = hpc_be_it = None

        # Parse --sigma string if provided.
        sigma_val = None
        if args.sigma is not None:
            try:
                parts = [float(x.strip()) for x in args.sigma.split(",")]
                if len(parts) != 4:
                    raise ValueError("need exactly 4 values")
                sigma_val = tuple(parts)
            except Exception as exc:
                sys.exit(f"ERROR: --sigma must be 4 comma-separated floats: {exc}")

        cfg = GLNSConfig(
            seed=args.seed,
            output_dir=args.output_dir,
            llm=LLMConfig(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                n_reference_ops=_pick(
                    args.n_reference_ops, fallback=_defaults.llm.n_reference_ops
                ),
                search_context_max_chars=_pick(
                    args.search_context_max_chars,
                    fallback=_defaults.llm.search_context_max_chars,
                ),
            ),
            population=PopulationConfig(
                N=_pick(args.N_pop, hpc_N_pop, fallback=_defaults.population.N),
                M=_pick(args.M_prune, hpc_M_prune, fallback=_defaults.population.M),
            ),
            evolution=EvolutionConfig(
                G_max=_pick(args.G_max, hpc_G_max, fallback=_defaults.evolution.G_max),
                time_limit_hours=_pick(
                    args.time_limit_hours, fallback=_defaults.evolution.time_limit_hours
                ),
                stagnation_patience=_pick(
                    args.stagnation_patience,
                    fallback=_defaults.evolution.stagnation_patience,
                ),
                stagnation_reheat_factor=_pick(
                    args.stagnation_reheat_factor,
                    fallback=_defaults.evolution.stagnation_reheat_factor,
                ),
            ),
            eval=EvalConfig(
                K_episodes=_pick(
                    args.K_episodes, hpc_K_ep, fallback=_defaults.eval.K_episodes
                ),
                T_iters=_pick(
                    args.T_iters, hpc_T_iters, fallback=_defaults.eval.T_iters
                ),
                T_test=_pick(args.T_test, hpc_T_test, fallback=_defaults.eval.T_test),
                T_final=_pick(
                    args.T_final, hpc_T_final, fallback=_defaults.eval.T_final
                ),
                K_final=_pick(
                    args.K_final, hpc_K_final, fallback=_defaults.eval.K_final
                ),
                sa_T0=args.sa_T0,
                sa_alpha=args.sa_alpha,
                sa_T0_gen_decay=args.sa_T0_gen_decay,
                sa_adaptive=not args.no_sa_adaptive,
                destroy_ratio=_pick(
                    args.destroy_ratio, fallback=_defaults.eval.destroy_ratio
                ),
                destroy_ratio_min=_pick(
                    args.destroy_ratio_min, fallback=_defaults.eval.destroy_ratio_min
                ),
                destroy_ratio_max=_pick(
                    args.destroy_ratio_max, fallback=_defaults.eval.destroy_ratio_max
                ),
                lambda_smooth=_pick(
                    args.lambda_smooth, fallback=_defaults.eval.lambda_smooth
                ),
                sigma=sigma_val or _defaults.eval.sigma,
                benchmark_eval_every=_pick(
                    args.benchmark_eval_every,
                    hpc_be_ev,
                    fallback=_defaults.eval.benchmark_eval_every,
                ),
                benchmark_eval_episodes=_pick(
                    args.benchmark_eval_episodes,
                    hpc_be_ep,
                    fallback=_defaults.eval.benchmark_eval_episodes,
                ),
                benchmark_eval_iters=_pick(
                    args.benchmark_eval_iters,
                    hpc_be_it,
                    fallback=_defaults.eval.benchmark_eval_iters,
                ),
            ),
            archive=ArchiveConfig(
                max_size=_pick(
                    args.archive_max_size, hpc_arch, fallback=_defaults.archive.max_size
                ),
            ),
            sandbox=SandboxConfig(
                start_method=args.sandbox_start_method,
                timeout_sec=_pick(
                    args.sandbox_timeout,
                    hpc_sb_to,
                    fallback=_defaults.sandbox.timeout_sec,
                ),
                max_memory_mb=_pick(
                    args.sandbox_memory_mb,
                    hpc_sb_mem,
                    fallback=_defaults.sandbox.max_memory_mb,
                ),
            ),
            instances=InstanceConfig(
                instances_json=args.instances,
                benchmark_data_dir=(
                    args.benchmark_data_dir if str(args.benchmark_data_dir) else None
                ),
                benchmark_adaptation=args.benchmark_adaptation,
                benchmark_adaptation_evo_frac=args.benchmark_adaptation_evo_frac,
                benchmark_adaptation_split_seed=args.benchmark_adaptation_split_seed,
            ),
            assignment_mode=args.assignment_mode,
            sequencing_mode=args.sequencing_mode,
            max_workers=args.max_workers,
        )

    # Print config summary.
    hpc_tag = " [HPC preset]" if (not args.config and args.hpc) else ""
    print(f"G-LNS config{hpc_tag}:")
    print(f"  seed             = {cfg.seed}")
    print(f"  generations      = {cfg.evolution.G_max}")
    if cfg.evolution.time_limit_hours > 0:
        print(f"  time_limit       = {cfg.evolution.time_limit_hours:.1f} h")
    print(
        f"  stagnation_pat   = {cfg.evolution.stagnation_patience} gens (reheat x{cfg.evolution.stagnation_reheat_factor})"
    )
    print(f"  population       = {cfg.population.N} (prune {cfg.population.M})")
    print(f"  K_episodes       = {cfg.eval.K_episodes}")
    print(f"  T_iters          = {cfg.eval.T_iters}")
    print(f"  T_test           = {cfg.eval.T_test}")
    t_fin = cfg.eval.T_final or cfg.eval.T_test
    k_fin = cfg.eval.K_final or max(10, cfg.eval.K_episodes)
    print(f"  T_final/K_final  = {t_fin} / {k_fin}")
    print(
        f"  destroy_ratio    = [{cfg.eval.destroy_ratio_min:.2f}, {cfg.eval.destroy_ratio_max:.2f}] (fixed={cfg.eval.destroy_ratio:.2f})"
    )
    print(f"  lambda_smooth    = {cfg.eval.lambda_smooth}")
    print(f"  sigma            = {cfg.eval.sigma}")
    print(f"  sa_T0            = {cfg.eval.sa_T0}")
    print(f"  sa_alpha         = {cfg.eval.sa_alpha}")
    print(f"  sa_T0_decay      = {cfg.eval.sa_T0_gen_decay}")
    print(f"  sa_adaptive      = {cfg.eval.sa_adaptive}")
    print(f"  model            = {cfg.llm.model}")
    print(f"  max_tokens       = {cfg.llm.max_tokens}")
    print(f"  n_ref_ops        = {cfg.llm.n_reference_ops}")
    print(f"  output_dir       = {cfg.output_dir}")
    print(f"  instances        = {cfg.instances.instances_json}")
    print(f"  benchmark        = {cfg.instances.benchmark_data_dir or 'disabled'}")
    print(f"  bm_adapt_mode    = {cfg.instances.benchmark_adaptation}")
    if cfg.instances.benchmark_adaptation:
        print(
            f"  bm_evo_frac      = {cfg.instances.benchmark_adaptation_evo_frac:.2f} (seed={cfg.instances.benchmark_adaptation_split_seed})"
        )
    print(f"  bm_eval_every    = {cfg.eval.benchmark_eval_every}")
    print(
        f"  bm_eval_ep/iter  = {cfg.eval.benchmark_eval_episodes} / {cfg.eval.benchmark_eval_iters}"
    )
    print(f"  archive_max      = {cfg.archive.max_size}")
    print(
        f"  sandbox          = timeout={cfg.sandbox.timeout_sec}s mem={cfg.sandbox.max_memory_mb}MB start={cfg.sandbox.start_method}"
    )
    print(f"  assign_mode      = {cfg.assignment_mode}")
    if cfg.assignment_mode:
        print(f"  seq_mode         = {cfg.sequencing_mode}")
        print(f"  max_workers      = {cfg.max_workers or 'auto'}")

    multi_keys = os.environ.get("GROQ_API_KEYS", "").strip()
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if multi_keys:
        # Don't print secrets; just show how many keys we detected.
        n_keys = len([k for k in multi_keys.replace("\n", ",").split(",") if k.strip()])
        print(f"  GROQ_API_KEYS = {n_keys} key(s) configured")
    elif api_key:
        print(f"  GROQ_API_KEY  = {'*' * (len(api_key) - 4) + api_key[-4:]}")
    else:
        print(
            "\n  WARNING: Neither GROQ_API_KEY nor GROQ_API_KEYS set. Running in seed-operator-only mode.\n"
        )

    archive = run_glns(cfg)

    # Print final summary.
    instance_ids = sorted({e.instance_id for e in archive.entries})
    shown_instance_id = instance_ids[0] if instance_ids else 0
    front = archive.front(instance_id=shown_instance_id)

    # Compute aggregate HV across all instances present in archive.
    from glns.runner import _compute_aggregate_hv

    # Build per-instance ref points from the instance data.
    from glns.evaluation import load_instances_from_json
    import os as _os

    inst_path = str(cfg.instances.instances_json)
    if not _os.path.isabs(inst_path):
        inst_path = str(Path(__file__).resolve().parent / inst_path)
    all_inst = load_instances_from_json(inst_path)
    inst_ref = {}
    for inst in all_inst:
        iid = int(inst["instance_id"])
        inst_ref[iid] = (inst["T"], float(inst["T"] * max(inst["e"]) * max(inst["ct"])))
    agg_hv, agg_n, per_inst = _compute_aggregate_hv(archive, inst_ref)

    print(f"\n{'='*60}")
    print(
        f"G-LNS COMPLETE — archive has {archive.size()} total entries "
        f"across {len(instance_ids)} instance(s)"
    )
    print(f"Aggregate HV (mean over {agg_n} instances with entries): {agg_hv:.2f}")
    print(
        f"Pareto front for instance_id={shown_instance_id} "
        f"({len(front)} points, archive entries={archive.size(shown_instance_id)}):"
    )
    for cmax, energy in front[:10]:
        print(f"  Cmax={cmax:6d}  TEC={energy:.2f}")
    if len(front) > 10:
        print(f"  ... and {len(front) - 10} more")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
