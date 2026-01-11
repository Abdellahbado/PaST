"""
Evaluation module for PaST-SM.

Provides:
- Deterministic evaluation (argmax actions)
- Benchmark evaluation on fixed instance sets
- Metrics collection for ablation reporting
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class EvalResult:
    """Evaluation result container."""

    # Energy statistics
    energy_mean: float
    energy_std: float
    energy_min: float
    energy_max: float

    # Feasibility
    infeasible_count: int
    infeasible_rate: float

    # Timing
    makespan_mean: float
    makespan_std: float

    # Episode statistics
    num_episodes: int
    steps_mean: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "eval/energy_mean": self.energy_mean,
            "eval/energy_std": self.energy_std,
            "eval/energy_min": self.energy_min,
            "eval/energy_max": self.energy_max,
            "eval/infeasible_count": self.infeasible_count,
            "eval/infeasible_rate": self.infeasible_rate,
            "eval/makespan_mean": self.makespan_mean,
            "eval/makespan_std": self.makespan_std,
            "eval/num_episodes": self.num_episodes,
            "eval/steps_mean": self.steps_mean,
        }


class Evaluator:
    """
    Evaluator for PaST-SM models.

    Runs deterministic evaluation (argmax actions) on fixed instance sets.
    """

    def __init__(
        self,
        model: nn.Module,
        env,  # GPUBatchSingleMachinePeriodEnv or similar
        device: torch.device,
        max_steps_per_episode: int = 100,
    ):
        """
        Initialize evaluator.

        Args:
            model: PaSTSMNet model
            env: Batched environment (GPU or CPU)
            device: Device for tensors
            max_steps_per_episode: Safety limit for episode length
        """
        self.model = model
        self.env = env
        self.device = device
        self.max_steps = max_steps_per_episode

    @torch.no_grad()
    def evaluate(
        self,
        batch_data: Dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> Tuple[EvalResult, Dict[str, np.ndarray]]:
        """
        Evaluate model on a batch of instances.

        Args:
            batch_data: Batch of evaluation instances
            deterministic: If True, use argmax actions; else sample

        Returns:
            result: EvalResult with statistics
            per_instance: Dictionary of per-instance arrays for aggregation
        """
        self.model.eval()

        # Reset environment with evaluation data
        obs = self.env.reset(batch_data)
        batch_size = self.env.batch_size

        # Track metrics per instance
        total_energies = torch.zeros(batch_size, device=self.device)
        total_steps = torch.zeros(batch_size, device=self.device)
        final_times = torch.zeros(batch_size, device=self.device)
        done_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        infeasible_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step in range(self.max_steps):
            if done_mask.all():
                break

            # Get action logits from model
            logits, _ = self.model(
                jobs=obs["jobs"],
                periods_local=obs["periods"],
                ctx=obs["ctx"],
                job_mask=obs.get("job_mask"),
                period_mask=obs.get("period_mask"),
                periods_full=obs.get("periods_full"),
                period_full_mask=obs.get("period_full_mask"),
            )

            # Apply action mask
            if "action_mask" in obs:
                action_mask = obs["action_mask"]
                logits = logits.masked_fill(action_mask == 0, float("-inf"))

            # Check for infeasible states (all actions masked)
            all_masked = ~torch.isfinite(logits).any(dim=-1)
            newly_infeasible = all_masked & ~done_mask & ~infeasible_mask
            infeasible_mask = infeasible_mask | newly_infeasible

            # For infeasible states, mark as done (cannot continue)
            # and record current time as final time
            if newly_infeasible.any() and hasattr(self.env, "t"):
                final_times = torch.where(
                    newly_infeasible,
                    self.env.t.float(),
                    final_times,
                )
            done_mask = done_mask | newly_infeasible

            # CRITICAL: Propagate done_mask to env so it won't update done instances
            # This prevents infeasible/done instances from being advanced by env.step()
            if hasattr(self.env, "done_mask"):
                self.env.done_mask = self.env.done_mask | done_mask

            # Select action (only for non-done, non-infeasible instances)
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                # Handle all-masked by using uniform over action space
                safe_logits = torch.where(
                    all_masked.unsqueeze(-1),
                    torch.zeros_like(logits),
                    logits,
                )
                dist = torch.distributions.Categorical(logits=safe_logits)
                actions = dist.sample()

            # Skip env step for instances that are done or infeasible
            # Use action 0 as placeholder (won't affect done instances)
            actions = actions.masked_fill(done_mask, 0)

            # Environment step
            next_obs, rewards, dones, info = self.env.step(actions)

            # Track which envs just finished (BEFORE updating done_mask)
            newly_done = dones & ~done_mask

            # Update tracking (only for active, non-infeasible instances)
            active = ~done_mask
            total_energies[active] += -rewards[active]  # rewards are negative energy
            total_steps[active] += 1

            # Track final times for newly done instances
            if hasattr(self.env, "t"):
                final_times = torch.where(
                    newly_done,
                    self.env.t.float(),
                    final_times,
                )

            # Update done mask AFTER tracking newly_done
            done_mask = done_mask | dones

            obs = next_obs

        # For unfinished episodes, use current time as makespan
        unfinished = ~done_mask
        if unfinished.any() and hasattr(self.env, "t"):
            final_times = torch.where(
                unfinished,
                self.env.t.float(),
                final_times,
            )

        # Use env.total_energy as the source of truth for energy, but only for
        # instances that weren't marked infeasible (infeasible ones may have been
        # stepped with arbitrary actions before we could propagate done_mask)
        if hasattr(self.env, "total_energy"):
            # Keep our accumulated energy for infeasible instances, use env's for others
            total_energies = torch.where(
                infeasible_mask,
                total_energies,
                self.env.total_energy,
            )

        # Move to CPU for statistics
        energies_np = total_energies.cpu().numpy()
        steps_np = total_steps.cpu().numpy()
        infeasible_np = infeasible_mask.cpu().numpy()
        makespans_np = final_times.cpu().numpy().astype(float)
        done_np = done_mask.cpu().numpy()

        self.model.train()

        # Per-instance data for proper multi-batch aggregation
        per_instance = {
            "energies": energies_np,
            "makespans": makespans_np,
            "steps": steps_np,
            "infeasible": infeasible_np,
            "done": done_np,
        }

        result = EvalResult(
            energy_mean=float(np.mean(energies_np)),
            energy_std=float(np.std(energies_np)),
            energy_min=float(np.min(energies_np)),
            energy_max=float(np.max(energies_np)),
            infeasible_count=int(np.sum(infeasible_np)),
            infeasible_rate=float(np.mean(infeasible_np)),
            makespan_mean=float(np.mean(makespans_np)),
            makespan_std=float(np.std(makespans_np)),
            num_episodes=batch_size,
            steps_mean=float(np.mean(steps_np)),
        )

        return result, per_instance

    @torch.no_grad()
    def evaluate_multiple_batches(
        self,
        batch_generator,
        num_batches: int,
        deterministic: bool = True,
    ) -> EvalResult:
        """
        Evaluate over multiple batches and aggregate results.

        Args:
            batch_generator: Callable that returns batch_data
            num_batches: Number of batches to evaluate
            deterministic: If True, use argmax actions

        Returns:
            Aggregated EvalResult with correct statistics
        """
        # Collect per-instance data from all batches
        all_energies = []
        all_makespans = []
        all_steps = []
        all_infeasible = []

        for _ in range(num_batches):
            batch_data = batch_generator()
            result, per_instance = self.evaluate(
                batch_data, deterministic=deterministic
            )

            # Collect actual per-instance arrays
            all_energies.append(per_instance["energies"])
            all_makespans.append(per_instance["makespans"])
            all_steps.append(per_instance["steps"])
            all_infeasible.append(per_instance["infeasible"])

        # Concatenate all per-instance data
        energies_arr = np.concatenate(all_energies)
        makespans_arr = np.concatenate(all_makespans)
        steps_arr = np.concatenate(all_steps)
        infeasible_arr = np.concatenate(all_infeasible)

        total_episodes = len(energies_arr)
        total_infeasible = int(np.sum(infeasible_arr))

        return EvalResult(
            energy_mean=float(np.mean(energies_arr)),
            energy_std=float(np.std(energies_arr)),
            energy_min=float(np.min(energies_arr)),
            energy_max=float(np.max(energies_arr)),
            infeasible_count=total_infeasible,
            infeasible_rate=(
                float(total_infeasible / total_episodes) if total_episodes > 0 else 0.0
            ),
            makespan_mean=float(np.mean(makespans_arr)),
            makespan_std=float(np.std(makespans_arr)),
            num_episodes=total_episodes,
            steps_mean=float(np.mean(steps_arr)),
        )


def save_eval_results(
    results: Dict[str, EvalResult],
    path: Path,
    extra_info: Optional[Dict[str, Any]] = None,
):
    """
    Save evaluation results to JSON file.

    Args:
        results: Dictionary of variant_id -> EvalResult
        path: Output path
        extra_info: Additional metadata to include
    """
    data = {
        "results": {name: result.to_dict() for name, result in results.items()},
    }
    if extra_info:
        data["info"] = extra_info

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_eval_results(path: Path) -> Dict[str, Dict]:
    """
    Load evaluation results from JSON file.

    Args:
        path: Input path

    Returns:
        Dictionary of results
    """
    with open(path, "r") as f:
        return json.load(f)


def compare_variants(
    results: Dict[str, EvalResult],
    baseline_variant: str = "ppo_short_base",
) -> str:
    """
    Generate comparison table for ablation study.

    Args:
        results: Dictionary of variant_id -> EvalResult
        baseline_variant: Variant to use as baseline for comparison

    Returns:
        Formatted table string
    """
    if baseline_variant not in results:
        baseline_variant = list(results.keys())[0]

    baseline = results[baseline_variant]

    lines = []
    lines.append("=" * 80)
    lines.append("ABLATION COMPARISON")
    lines.append("=" * 80)
    lines.append(
        f"{'Variant':<25} {'Energy↓':>12} {'Δ%':>8} {'Infeas%':>10} {'Steps':>8}"
    )
    lines.append("-" * 80)

    for name, result in sorted(results.items()):
        delta_pct = (
            (result.energy_mean - baseline.energy_mean) / baseline.energy_mean * 100
            if baseline.energy_mean != 0
            else 0
        )
        delta_str = f"{delta_pct:+.2f}%" if name != baseline_variant else "baseline"

        lines.append(
            f"{name:<25} {result.energy_mean:>12.2f} {delta_str:>8} "
            f"{result.infeasible_rate*100:>9.1f}% {result.steps_mean:>8.1f}"
        )

    lines.append("=" * 80)

    return "\n".join(lines)
