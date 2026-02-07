#!/usr/bin/env python3
"""
Learnability Test — Verify all NeuroLS variants can learn on the scheduling task.

This script runs a short training session for each variant and checks whether
the learned policy outperforms a random baseline.  The idea is to catch
fundamental bugs (NaN, frozen gradients, wrong reward signal) quickly.

Design:
  • Small instances  : n=10 jobs, m=2 machines, K=40
  • Short training   : 500 episodes, 200 steps/episode
  • Serial (1 worker): simpler debugging, still CPU-fast
  • Pass criterion   : greedy improvement > random improvement by ≥ 0.5 pp
                       AND loss decreased (end avg < start avg)
  • Variants tested  : 9 (6 bipartite + 3 tripartite), no "none" price mode

Expected runtime: ~10-15 min per variant → ~1.5-2.5 h total on a single GPU.
Use MAX_PARALLEL=2 to cut wall-clock time in half if GPU memory allows.

Usage:
    python -m PaST.scripts.learnability_test                     # all variants
    python -m PaST.scripts.learnability_test --variant AANP_zprice  # single
    python -m PaST.scripts.learnability_test --device cpu         # force CPU
    python -m PaST.scripts.learnability_test --episodes 300       # override
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Variant definitions ───────────────────────────────────────────────────────


@dataclass
class VariantSpec:
    """Specification for one training variant."""

    name: str
    action_space: str
    price_mode: str
    graph_type: str = "bipartite"


VARIANTS: Dict[str, VariantSpec] = {
    # Bipartite — z_price
    "AA_zprice": VariantSpec("AA_zprice", "AA", "z_price"),
    "AAN_zprice": VariantSpec("AAN_zprice", "AAN", "z_price"),
    "AANP_zprice": VariantSpec("AANP_zprice", "AANP", "z_price"),
    # Bipartite — full
    "AA_full": VariantSpec("AA_full", "AA", "full"),
    "AAN_full": VariantSpec("AAN_full", "AAN", "full"),
    "AANP_full": VariantSpec("AANP_full", "AANP", "full"),
    # Tripartite-A — AANP
    "triA_zprice": VariantSpec("triA_zprice", "AANP", "z_price", "tripartite"),
    "triA_full": VariantSpec("triA_full", "AANP", "full", "tripartite"),
    # Tripartite-B — AANPD (learned destroy)
    "triB_full": VariantSpec("triB_full", "AANPD", "full", "tripartite"),
}

# ── Test configuration ────────────────────────────────────────────────────────


@dataclass
class TestConfig:
    """Hyper-params for the quick learnability test."""

    # Instance size (small for speed)
    n_jobs: int = 10
    n_machines: int = 2
    K_fixed: int = 40

    # Training budget
    n_episodes: int = 500
    max_steps: int = 200
    stagnation_limit: int = 50

    # Model (smaller for speed)
    d_emb: int = 64
    n_layers_static: int = 2
    n_layers_dynamic: int = 1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3  # Larger LR for faster learning signal
    buffer_size: int = 4000
    min_buffer_size: int = 200
    gamma: float = 0.99
    n_step: int = 3
    epsilon_start: float = 0.9
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.99  # Faster epsilon decay

    # Eval
    eval_interval: int = 50
    n_eval_episodes: int = 5
    n_instances: int = 30

    # Target net
    target_update_freq: int = 200
    use_soft_update: bool = False

    # IQN
    use_iqn: bool = True
    n_quantile_samples: int = 16
    n_target_quantile_samples: int = 16

    # Thresholds for pass/fail
    min_advantage_pp: float = 0.5  # Greedy must beat random by ≥ 0.5 pp
    loss_decrease_ratio: float = 0.8  # Avg loss in last 25% < first 25%

    # Misc
    seed: int = 42
    device: str = "cuda"


# ── Core test runner ──────────────────────────────────────────────────────────


def run_single_variant(
    variant: VariantSpec,
    test_cfg: TestConfig,
) -> Dict:
    """Train one variant for a short while and return diagnostic metrics.

    Returns dict with:
        - name, passed, reason
        - final_advantage_pp, final_greedy_imp, final_random_imp
        - avg_loss_first, avg_loss_last
        - avg_grad_norm, any_nan
        - wall_time_s, n_episodes_run
    """
    import numpy as np
    import random as py_random

    # Lazy imports to keep startup fast
    from PaST.neurols.train import TrainConfig, NeuroLSTrainer
    from PaST.neurols.env import NeuroLSEnv, EnvConfig
    from PaST.neurols.action_space import get_action_space
    from PaST.data.sm_benchmark_data import generate_raw_instance
    from PaST.config import DataConfig

    print(f"\n{'━'*60}")
    print(f"  VARIANT: {variant.name}")
    print(
        f"  action_space={variant.action_space}  price_mode={variant.price_mode}  "
        f"graph_type={variant.graph_type}"
    )
    print(f"{'━'*60}")

    result = {
        "name": variant.name,
        "action_space": variant.action_space,
        "price_mode": variant.price_mode,
        "graph_type": variant.graph_type,
        "passed": False,
        "reason": "",
    }

    t0 = time.time()

    try:
        # Build TrainConfig
        train_cfg = TrainConfig(
            exp_name=f"learntest_{variant.name}",
            seed=test_cfg.seed,
            device=test_cfg.device,
            log_dir="logs/learnability_test",
            checkpoint_dir="checkpoints/learnability_test",
            n_jobs_train=[test_cfg.n_jobs],
            n_machines_train=[test_cfg.n_machines],
            K_fixed=test_cfg.K_fixed,
            n_instances_per_reset=test_cfg.n_instances,
            max_steps=test_cfg.max_steps,
            stagnation_limit=test_cfg.stagnation_limit,
            action_space=variant.action_space,
            reward_mode="improvement",
            graph_type=variant.graph_type,
            d_emb=test_cfg.d_emb,
            n_layers_static=test_cfg.n_layers_static,
            n_layers_dynamic=test_cfg.n_layers_dynamic,
            use_iqn=test_cfg.use_iqn,
            use_dueling=False,
            price_mode=variant.price_mode,
            dropout=0.05,
            n_episodes=test_cfg.n_episodes,
            batch_size=test_cfg.batch_size,
            learning_rate=test_cfg.learning_rate,
            weight_decay=1e-5,
            gamma=test_cfg.gamma,
            n_step=test_cfg.n_step,
            epsilon_start=test_cfg.epsilon_start,
            epsilon_end=test_cfg.epsilon_end,
            epsilon_decay=test_cfg.epsilon_decay,
            buffer_size=test_cfg.buffer_size,
            min_buffer_size=test_cfg.min_buffer_size,
            target_update_freq=test_cfg.target_update_freq,
            use_soft_update=test_cfg.use_soft_update,
            tau=0.005,
            n_quantile_samples=test_cfg.n_quantile_samples,
            n_target_quantile_samples=test_cfg.n_target_quantile_samples,
            kappa=1.0,
            log_interval=25,
            eval_interval=test_cfg.eval_interval,
            save_interval=99999,  # Don't save checkpoints
            n_eval_episodes=test_cfg.n_eval_episodes,
            n_parallel_envs=1,  # Serial for simplicity
        )

        # Create trainer
        trainer = NeuroLSTrainer(train_cfg)

        # Monkey-patch to collect loss/grad history
        loss_history: List[float] = []
        grad_history: List[float] = []
        any_nan = False
        eval_history: List[Dict] = []

        original_train_step = trainer.train_step

        def patched_train_step():
            nonlocal any_nan
            loss, grad = original_train_step()
            if loss > 0:
                loss_history.append(loss)
                grad_history.append(grad)
            if not np.isfinite(loss) or not np.isfinite(grad):
                any_nan = True
            return loss, grad

        trainer.train_step = patched_train_step

        # Monkey-patch evaluate to capture results
        original_evaluate = trainer.evaluate

        def patched_evaluate(env, instances):
            metrics = original_evaluate(env, instances)
            eval_history.append(dict(metrics))
            return metrics

        trainer.evaluate = patched_evaluate

        # Run training
        trainer.train()

        # ── Analyze results ──
        wall_time = time.time() - t0
        result["wall_time_s"] = round(wall_time, 1)
        result["n_episodes_run"] = test_cfg.n_episodes
        result["any_nan"] = any_nan

        # Loss analysis: compare first 25% vs last 25%
        if loss_history:
            n = len(loss_history)
            q1 = max(1, n // 4)
            avg_first = float(np.mean(loss_history[:q1]))
            avg_last = float(np.mean(loss_history[-q1:]))
            result["avg_loss_first"] = round(avg_first, 6)
            result["avg_loss_last"] = round(avg_last, 6)
            result["loss_decreased"] = (
                avg_last < avg_first * test_cfg.loss_decrease_ratio
            )
        else:
            result["avg_loss_first"] = None
            result["avg_loss_last"] = None
            result["loss_decreased"] = False

        # Gradient norm
        if grad_history:
            result["avg_grad_norm"] = round(float(np.mean(grad_history)), 4)
        else:
            result["avg_grad_norm"] = None

        # Eval analysis: use last eval
        if eval_history:
            last_eval = eval_history[-1]
            greedy_imp = last_eval.get("improvement", 0.0)
            random_imp = last_eval.get("random_improvement", 0.0)
            advantage = last_eval.get("advantage_over_random", greedy_imp - random_imp)
            result["final_greedy_imp"] = round(greedy_imp, 3)
            result["final_random_imp"] = round(random_imp, 3)
            result["final_advantage_pp"] = round(advantage, 3)
            result["final_avg_q"] = round(last_eval.get("avg_q", 0.0), 4)

            # Also track best advantage across all evals
            best_adv = max(
                e.get(
                    "advantage_over_random",
                    e.get("improvement", 0) - e.get("random_improvement", 0),
                )
                for e in eval_history
            )
            result["best_advantage_pp"] = round(best_adv, 3)
        else:
            result["final_greedy_imp"] = None
            result["final_random_imp"] = None
            result["final_advantage_pp"] = None
            result["final_avg_q"] = None
            result["best_advantage_pp"] = None

        # ── Pass/Fail decision ──
        reasons = []

        if any_nan:
            reasons.append("NaN/Inf detected in loss or gradients")

        if result["final_advantage_pp"] is not None:
            if result["best_advantage_pp"] >= test_cfg.min_advantage_pp:
                pass  # OK
            else:
                reasons.append(
                    f"Best advantage {result['best_advantage_pp']:.2f}pp < "
                    f"{test_cfg.min_advantage_pp:.1f}pp threshold"
                )

        if not result.get("loss_decreased", False):
            reasons.append("Loss did not decrease (end avg >= start avg)")

        if not reasons:
            result["passed"] = True
            result["reason"] = "PASS"
        else:
            result["passed"] = False
            result["reason"] = "; ".join(reasons)

    except Exception as exc:
        import traceback

        result["passed"] = False
        result["reason"] = f"EXCEPTION: {exc}"
        result["traceback"] = traceback.format_exc()
        result["wall_time_s"] = round(time.time() - t0, 1)

    # Print summary
    status = "✅ PASS" if result["passed"] else "❌ FAIL"
    print(f"\n  {status}: {variant.name}")
    if result.get("final_advantage_pp") is not None:
        print(
            f"    Advantage: {result['final_advantage_pp']:+.2f}pp "
            f"(best: {result.get('best_advantage_pp', 'N/A')}pp)"
        )
    if result.get("avg_loss_first") is not None:
        print(
            f"    Loss: {result['avg_loss_first']:.4f} → {result['avg_loss_last']:.4f} "
            f"({'↓' if result.get('loss_decreased') else '↑'})"
        )
    if not result["passed"]:
        print(f"    Reason: {result['reason']}")
    print(f"    Time: {result.get('wall_time_s', '?')}s")

    return result


# ── Main entry point ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="NeuroLS Learnability Test — verify all variants can learn"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Run only this variant (e.g. AANP_zprice). Default: run all.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="learnability_results.json",
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Min advantage (pp) to pass. Default: 0.5",
    )
    args = parser.parse_args()

    test_cfg = TestConfig()
    if args.device:
        test_cfg.device = args.device
    if args.episodes:
        test_cfg.n_episodes = args.episodes
    if args.seed:
        test_cfg.seed = args.seed
    if args.threshold is not None:
        test_cfg.min_advantage_pp = args.threshold

    # Select variants
    if args.variant:
        if args.variant not in VARIANTS:
            print(
                f"Unknown variant '{args.variant}'. Available: {list(VARIANTS.keys())}"
            )
            sys.exit(1)
        variants_to_run = {args.variant: VARIANTS[args.variant]}
    else:
        variants_to_run = VARIANTS

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            NeuroLS Learnability Test                       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(
        f"║  Variants : {len(variants_to_run):>3}                                        ║"
    )
    print(
        f"║  Episodes : {test_cfg.n_episodes:>5}  │  Steps/ep: {test_cfg.max_steps:>4}              ║"
    )
    print(
        f"║  Instance : n={test_cfg.n_jobs}, m={test_cfg.n_machines}, K={test_cfg.K_fixed}"
        f"{'':>24}║"
    )
    print(f"║  Device   : {test_cfg.device:<10}                                 ║")
    print(
        f"║  Pass thr : advantage ≥ {test_cfg.min_advantage_pp:.1f}pp                         ║"
    )
    print("╚══════════════════════════════════════════════════════════════╝")

    all_results: List[Dict] = []
    t_total = time.time()

    for key in variants_to_run:
        result = run_single_variant(variants_to_run[key], test_cfg)
        all_results.append(result)

    total_time = time.time() - t_total

    # ── Summary ──
    n_passed = sum(1 for r in all_results if r["passed"])
    n_total = len(all_results)

    print(f"\n{'='*60}")
    print(f"  LEARNABILITY TEST SUMMARY")
    print(f"{'='*60}")
    print(
        f"  {'Variant':<20} {'Status':<8} {'Advantage':>10} {'Loss ↓':>8} {'Time':>8}"
    )
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8} {'─'*8}")

    for r in all_results:
        status = "PASS" if r["passed"] else "FAIL"
        adv = (
            f"{r.get('best_advantage_pp', 0):+.2f}pp"
            if r.get("best_advantage_pp") is not None
            else "N/A"
        )
        loss_ok = "yes" if r.get("loss_decreased") else "no"
        t = f"{r.get('wall_time_s', 0):.0f}s"
        print(f"  {r['name']:<20} {status:<8} {adv:>10} {loss_ok:>8} {t:>8}")

    print(
        f"\n  Result: {n_passed}/{n_total} passed  |  Total time: {total_time/60:.1f} min"
    )

    if n_passed < n_total:
        failed = [r for r in all_results if not r["passed"]]
        print(f"\n  Failed variants:")
        for r in failed:
            print(f"    • {r['name']}: {r['reason']}")

    print(f"{'='*60}\n")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": asdict(test_cfg),
                "variants": all_results,
                "summary": {
                    "n_passed": n_passed,
                    "n_total": n_total,
                    "total_time_s": round(total_time, 1),
                },
            },
            f,
            indent=2,
            default=str,
        )
    print(f"  Results saved to {output_path}")

    # Exit code
    sys.exit(0 if n_passed == n_total else 1)


if __name__ == "__main__":
    main()
