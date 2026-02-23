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
    p.add_argument("--max-tokens", type=int, default=3_000)

    # Evolution.
    p.add_argument("--generations", type=int, default=200, dest="G_max")
    p.add_argument("--pop-size", type=int, default=5, dest="N_pop")
    p.add_argument("--prune-count", type=int, default=2, dest="M_prune")

    # Evaluation.
    p.add_argument("--K-episodes", type=int, default=10)
    p.add_argument("--T-iters", type=int, default=100)
    p.add_argument("--T-test", type=int, default=500)
    p.add_argument("--destroy-ratio", type=float, default=0.2)

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
                destroy_ratio=args.destroy_ratio,
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
    print(f"  model         = {cfg.llm.model}")
    print(f"  max_tokens    = {cfg.llm.max_tokens}")
    print(f"  output_dir    = {cfg.output_dir}")
    print(f"  instances     = {cfg.instances.instances_json}")
    print(f"  sandbox_start = {cfg.sandbox.start_method}")

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print(
            "\n  WARNING: GROQ_API_KEY not set. Running in seed-operator-only mode.\n"
        )
    else:
        print(f"  GROQ_API_KEY  = {'*' * (len(api_key) - 4) + api_key[-4:]}")

    archive = run_glns(cfg)

    # Print final summary.
    instance_ids = sorted({e.instance_id for e in archive.entries})
    shown_instance_id = instance_ids[0] if instance_ids else 0
    front = archive.front(instance_id=shown_instance_id)
    print(f"\n{'='*60}")
    print(
        f"G-LNS COMPLETE — archive has {archive.size()} total entries "
        f"across {len(instance_ids)} instance(s)"
    )
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
