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

    # LLM.
    p.add_argument("--model", type=str, default="moonshotai/Kimi-K2-Instruct-0905")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=4_500)

    # Evolution.
    p.add_argument("--generations", type=int, default=200, dest="G_max")
    p.add_argument("--pop-size", type=int, default=5, dest="N_pop")
    p.add_argument("--prune-count", type=int, default=2, dest="M_prune")

    # Evaluation.
    p.add_argument("--K-episodes", type=int, default=20)
    p.add_argument("--T-iters", type=int, default=100)
    p.add_argument("--T-test", type=int, default=500)
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
    p.add_argument("--destroy-ratio", type=float, default=0.2)
    p.add_argument(
        "--benchmark-eval-every",
        type=int,
        default=10,
        help="Run dual-benchmark probe every N generations (0=disabled)",
    )
    p.add_argument(
        "--benchmark-eval-episodes",
        type=int,
        default=5,
        help="ALNS episodes per benchmark probe",
    )
    p.add_argument(
        "--benchmark-eval-iters",
        type=int,
        default=50,
        help="ALNS iterations per benchmark probe episode",
    )

    # Sandbox.
    p.add_argument("--sandbox-timeout", type=float, default=5.0)
    p.add_argument("--sandbox-memory-mb", type=int, default=512)
    p.add_argument(
        "--sandbox-start-method",
        type=str,
        default="auto",
        choices=["auto", "fork", "spawn", "forkserver"],
        help="Multiprocessing start method for sandbox workers",
    )

    # Archive.
    p.add_argument("--archive-max-size", type=int, default=100)

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
        # Build config from CLI arguments.
        cfg = GLNSConfig(
            seed=args.seed,
            output_dir=args.output_dir,
            llm=LLMConfig(
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ),
            population=PopulationConfig(
                N=args.N_pop,
                M=args.M_prune,
            ),
            evolution=EvolutionConfig(
                G_max=args.G_max,
            ),
            eval=EvalConfig(
                K_episodes=args.K_episodes,
                T_iters=args.T_iters,
                T_test=args.T_test,
                sa_T0=args.sa_T0,
                sa_alpha=args.sa_alpha,
                sa_T0_gen_decay=args.sa_T0_gen_decay,
                sa_adaptive=not args.no_sa_adaptive,
                destroy_ratio=args.destroy_ratio,
                benchmark_eval_every=args.benchmark_eval_every,
                benchmark_eval_episodes=args.benchmark_eval_episodes,
                benchmark_eval_iters=args.benchmark_eval_iters,
            ),
            archive=ArchiveConfig(
                max_size=args.archive_max_size,
            ),
            sandbox=SandboxConfig(
                start_method=args.sandbox_start_method,
                timeout_sec=args.sandbox_timeout,
                max_memory_mb=args.sandbox_memory_mb,
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
        )

    # Print config summary.
    print(f"G-LNS config:")
    print(f"  seed          = {cfg.seed}")
    print(f"  generations   = {cfg.evolution.G_max}")
    print(f"  population    = {cfg.population.N} (prune {cfg.population.M})")
    print(f"  K_episodes    = {cfg.eval.K_episodes}")
    print(f"  T_iters       = {cfg.eval.T_iters}")
    print(f"  T_test        = {cfg.eval.T_test}")
    print(f"  sa_T0         = {cfg.eval.sa_T0}")
    print(f"  sa_alpha      = {cfg.eval.sa_alpha}")
    print(f"  sa_T0_decay   = {cfg.eval.sa_T0_gen_decay}")
    print(f"  sa_adaptive   = {cfg.eval.sa_adaptive}")
    print(f"  model         = {cfg.llm.model}")
    print(f"  max_tokens    = {cfg.llm.max_tokens}")
    print(f"  output_dir    = {cfg.output_dir}")
    print(f"  instances     = {cfg.instances.instances_json}")
    print(f"  benchmark     = {cfg.instances.benchmark_data_dir or 'disabled'}")
    print(f"  bm_adapt_mode = {cfg.instances.benchmark_adaptation}")
    if cfg.instances.benchmark_adaptation:
        print(
            f"  bm_evo_frac   = {cfg.instances.benchmark_adaptation_evo_frac:.2f} (seed={cfg.instances.benchmark_adaptation_split_seed})"
        )
    print(f"  bm_eval_every = {cfg.eval.benchmark_eval_every}")
    print(f"  sandbox_start = {cfg.sandbox.start_method}")

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
