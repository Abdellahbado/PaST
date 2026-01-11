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
    ) -> EvalResult:
        """
        Evaluate model on a batch of instances.

        Args:
            batch_data: Batch of evaluation instances
            deterministic: If True, use argmax actions; else sample

        Returns:
            EvalResult with statistics
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
            newly_infeasible = all_masked & ~done_mask
            infeasible_mask = infeasible_mask | newly_infeasible

            # Select action
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()

            # For infeasible states, use action 0 (will be ignored)
            actions = actions.masked_fill(all_masked, 0)

            # Environment step
            next_obs, rewards, dones, info = self.env.step(actions)

            # Update tracking (only for non-done instances)
            active = ~done_mask
            total_energies[active] += -rewards[active]  # rewards are negative energy
            total_steps[active] += 1

            # Update done mask
            done_mask = done_mask | dones

            # Track final times
            if hasattr(self.env, "t"):
                final_times = torch.where(
                    dones & ~done_mask,  # newly done
                    self.env.t.float(),
                    final_times,
                )

            obs = next_obs

        # Get final energies from environment if available
        if hasattr(self.env, "total_energy"):
            total_energies = self.env.total_energy

        # Move to CPU for statistics
        energies_np = total_energies.cpu().numpy()
        steps_np = total_steps.cpu().numpy()
        infeasible_np = infeasible_mask.cpu().numpy()

        # Handle makespans
        if hasattr(self.env, "t"):
            makespans_np = self.env.t.cpu().numpy().astype(float)
        else:
            makespans_np = np.zeros(batch_size)

        self.model.train()

        return EvalResult(
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
            Aggregated EvalResult
        """
        all_energies = []
        all_makespans = []
        all_steps = []
        total_infeasible = 0
        total_episodes = 0

        for _ in range(num_batches):
            batch_data = batch_generator()
            result = self.evaluate(batch_data, deterministic=deterministic)

            # Would need actual per-instance data for proper aggregation
            # For now, use weighted statistics
            n = result.num_episodes
            all_energies.extend([result.energy_mean] * n)
            all_makespans.extend([result.makespan_mean] * n)
            all_steps.extend([result.steps_mean] * n)
            total_infeasible += result.infeasible_count
            total_episodes += n

        energies_arr = np.array(all_energies)
        makespans_arr = np.array(all_makespans)
        steps_arr = np.array(all_steps)

        return EvalResult(
            energy_mean=float(np.mean(energies_arr)),
            energy_std=float(np.std(energies_arr)),
            energy_min=float(np.min(energies_arr)),
            energy_max=float(np.max(energies_arr)),
            infeasible_count=total_infeasible,
            infeasible_rate=(
                total_infeasible / total_episodes if total_episodes > 0 else 0.0
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
