"""
Full Parallel Machine Scheduling Pipeline with Epsilon-Constraint Optimization.

This script combines:
1. CWP (Cheapest-Window Placement) for job-to-machine assignment
2. Q-Sequence model for job ordering on each machine
3. DP (Dynamic Programming) for optimal timing given a job sequence
4. Accelerated epsilon-constraint for Pareto frontier exploration

Epsilon-Constraint Acceleration:
- Start with epsilon = T_max (full horizon)
- Solve instance, get makespan = Cmax
- Next epsilon = Cmax (skip redundant values)
- Continue until infeasible or Cmax = T_min

Usage:
    conda activate new-ml-env
    PYTHONPATH=/Users/mac/Documents/Study/PFE python -m PaST.cli.run_full_pipeline \\
        --checkpoint "models/ensia hpc/q seq cwe/Checkpoint 85.pt" \\
        --scale small \\
        --num_instances 10 \\
        --epsilon_constraint
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from PaST.config import DataConfig, VariantID, get_variant_config
from PaST.sm_benchmark_data import generate_raw_instance, RawInstance
from PaST.solvers.cwp_solver import construct_schedule_CWP, get_machine_job_sets, CWPResult
from PaST.solvers.baselines_sequence_dp import (
    DPResult,
    dp_schedule_for_job_sequence,
    spt_sequence,
    _dp_schedule_fixed_order,
)
from PaST.q_sequence_model import build_q_model, QSequenceNet
from PaST.sequence_env import GPUBatchSequenceEnv


@dataclass
class MachineSchedule:
    """Schedule for a single machine."""
    machine_idx: int
    job_indices: List[int]  # Original job indices assigned to this machine
    job_sequence: List[int]  # Order of jobs (local indices in job_indices)
    start_times: List[int]   # Start times for each job in sequence
    total_energy: float
    makespan: int  # Completion time of last job


@dataclass
class FullScheduleResult:
    """Complete parallel machine schedule."""
    total_energy: float
    makespan: int  # Max completion time across all machines
    machine_schedules: List[MachineSchedule]
    solve_time_ms: float
    feasible: bool


@dataclass
class EpsilonConstraintResult:
    """Result of epsilon-constraint optimization."""
    pareto_points: List[Tuple[float, int]]  # (energy, makespan) pairs
    schedules: List[FullScheduleResult]
    total_time_ms: float
    num_iterations: int


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


def _extract_q_model_state(ckpt: Any) -> Dict[str, Any]:
    """Extract model state dict from checkpoint."""
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict")
    if "model_state" in ckpt:
        return ckpt["model_state"]
    sample_keys = list(ckpt.keys())[:5]
    if any("." in k for k in sample_keys):
        return ckpt
    raise KeyError("Could not find model state in checkpoint")


def load_q_sequence_model(
    checkpoint_path: Path,
    variant_id: str = "q_sequence_cwe_ctx13",  # Use ctx13 variant for checkpoint compatibility
    device: torch.device = None,
) -> Tuple[QSequenceNet, Any]:
    """Load Q-Sequence model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    variant_config = get_variant_config(VariantID(variant_id))
    model = build_q_model(variant_config).to(device)
    
    ckpt = _load_checkpoint(checkpoint_path, device)
    state_dict = _extract_q_model_state(ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, variant_config


def make_single_machine_batch(
    raw_instance: RawInstance,
    machine_idx: int,
    job_indices: List[int],
    T_limit: int,
    N_job_pad: int = 50,
    K_period_pad: int = 250,
    T_max_pad: int = 500,
) -> Dict[str, np.ndarray]:
    """Create a single-machine batch dict for Q-Sequence model inference."""
    n_jobs = len(job_indices)
    p_subset = np.array([raw_instance.p[j] for j in job_indices], dtype=np.int32)
    e_single = raw_instance.e[machine_idx]
    
    batch = {
        "p_subset": np.zeros((1, N_job_pad), dtype=np.int32),
        "n_jobs": np.array([n_jobs], dtype=np.int32),
        "job_mask": np.zeros((1, N_job_pad), dtype=np.float32),
        "T_max": np.array([raw_instance.T_max], dtype=np.int32),
        "T_limit": np.array([T_limit], dtype=np.int32),
        "T_min": np.array([int(np.sum(p_subset))], dtype=np.int32),
        "ct": np.zeros((1, T_max_pad), dtype=np.int32),
        "Tk": np.zeros((1, K_period_pad), dtype=np.int32),
        "ck": np.zeros((1, K_period_pad), dtype=np.int32),
        "period_starts": np.zeros((1, K_period_pad), dtype=np.int32),
        "K": np.array([len(raw_instance.Tk)], dtype=np.int32),
        "period_mask": np.zeros((1, K_period_pad), dtype=np.float32),
        "e_single": np.array([e_single], dtype=np.int32),
        "price_q": np.zeros((1, 3), dtype=np.float32),
    }
    
    # Fill job data
    n = min(n_jobs, N_job_pad)
    if n > 0:
        batch["p_subset"][0, :n] = p_subset[:n]
        batch["job_mask"][0, :n] = 1.0
    
    # Fill period data
    k = min(len(raw_instance.Tk), K_period_pad)
    t_max = min(raw_instance.T_max, T_max_pad)
    
    if k > 0:
        batch["Tk"][0, :k] = np.array(raw_instance.Tk[:k], dtype=np.int32)
        batch["ck"][0, :k] = np.array(raw_instance.ck[:k], dtype=np.int32)
        batch["period_starts"][0, :k] = np.array(raw_instance.period_starts[:k], dtype=np.int32)
        batch["period_mask"][0, :k] = 1.0
    
    if t_max > 0:
        batch["ct"][0, :t_max] = np.array(raw_instance.ct[:t_max], dtype=np.int32)
        ct_valid = np.array(raw_instance.ct[:t_max])
        q25, q50, q75 = np.quantile(ct_valid, [0.25, 0.5, 0.75])
        batch["price_q"][0] = [q25, q50, q75]
    
    return batch


def q_sequence_order_for_machine(
    model: QSequenceNet,
    variant_config,
    batch: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[List[int], DPResult]:
    """Use Q-Sequence model to determine job order, then DP for scheduling."""
    env_config = variant_config.env
    n = int(batch["n_jobs"][0])
    
    if n == 0:
        return [], DPResult(
            total_energy=0.0,
            total_return=0.0,
            job_sequence=[],
            start_times=[],
        )
    
    env = GPUBatchSequenceEnv(
        batch_size=1,
        env_config=env_config,
        device=device,
    )
    obs = env.reset(batch)
    
    sequence = []
    
    model.eval()
    with torch.no_grad():
        for step in range(n):
            jobs_t = obs["jobs"]
            periods_t = obs["periods"]
            ctx_t = obs["ctx"]
            
            job_mask_float = obs.get("job_mask", env.job_available)
            if job_mask_float.dtype != torch.bool:
                job_mask = job_mask_float < 0.5
            else:
                job_mask = ~job_mask_float
            
            q_values = model(jobs_t, periods_t, ctx_t, job_mask)
            q_values = q_values.masked_fill(job_mask, float("inf"))
            
            action = q_values.argmin(dim=-1)
            sequence.append(int(action.item()))
            
            obs, _, done, _ = env.step(action)
            if done.all():
                break
    
    # Use DP to schedule the sequence
    result = dp_schedule_for_job_sequence(batch, sequence)
    
    return sequence, result


def spt_order_for_machine(
    batch: Dict[str, np.ndarray],
) -> Tuple[List[int], DPResult]:
    """Use SPT ordering + DP for scheduling."""
    n = int(batch["n_jobs"][0])
    if n == 0:
        return [], DPResult(
            total_energy=0.0,
            total_return=0.0,
            job_sequence=[],
            start_times=[],
        )
    
    p_subset = batch["p_subset"][0]
    sequence = spt_sequence(p_subset, n)
    result = dp_schedule_for_job_sequence(batch, sequence)
    
    return sequence, result


def solve_parallel_machine_instance(
    raw_instance: RawInstance,
    model: Optional[QSequenceNet],
    variant_config,
    device: torch.device,
    epsilon: Optional[int] = None,  # Makespan constraint
    use_model: bool = True,  # Use Q-Sequence model vs SPT
) -> FullScheduleResult:
    """Solve a parallel machine instance with CWP + Q-Sequence + DP.
    
    Args:
        raw_instance: Full parallel machine instance
        model: Q-Sequence model (optional, uses SPT if None)
        variant_config: Model variant configuration
        device: Torch device
        epsilon: Optional makespan constraint (deadline)
        use_model: Whether to use model for sequencing (vs SPT baseline)
        
    Returns:
        FullScheduleResult with complete schedule
    """
    start_time = time.perf_counter()
    
    m = raw_instance.m
    n = raw_instance.n
    T_max = raw_instance.T_max
    
    if epsilon is None:
        epsilon = T_max
    
    # Step 1: CWP for assignment
    cwp_result = construct_schedule_CWP(
        processing_times=raw_instance.p,
        energy_rates=[float(e) for e in raw_instance.e],
        ct=np.array(raw_instance.ct, dtype=np.float64),
        epsilon=epsilon,
        top_k=80,
    )
    
    job_sets = get_machine_job_sets(cwp_result, m)
    
    # Step 2: For each machine, use Q-Sequence (or SPT) + DP
    machine_schedules = []
    total_energy = 0.0
    overall_makespan = 0
    feasible = True
    
    for i in range(m):
        job_indices = job_sets[i]
        
        if len(job_indices) == 0:
            machine_schedules.append(MachineSchedule(
                machine_idx=i,
                job_indices=[],
                job_sequence=[],
                start_times=[],
                total_energy=0.0,
                makespan=0,
            ))
            continue
        
        # Compute T_limit for this machine (use epsilon as deadline)
        T_limit = epsilon
        
        # Create single-machine batch
        batch = make_single_machine_batch(
            raw_instance=raw_instance,
            machine_idx=i,
            job_indices=job_indices,
            T_limit=T_limit,
        )
        
        # Get job sequence and DP result
        if use_model and model is not None:
            sequence, dp_result = q_sequence_order_for_machine(
                model, variant_config, batch, device
            )
        else:
            sequence, dp_result = spt_order_for_machine(batch)
        
        # Check feasibility
        if not np.isfinite(dp_result.total_energy):
            feasible = False
        
        # Compute makespan for this machine
        if dp_result.start_times:
            last_job_local = sequence[-1]
            last_start = dp_result.start_times[-1]
            last_duration = raw_instance.p[job_indices[last_job_local]]
            machine_makespan = last_start + last_duration
        else:
            machine_makespan = 0
        
        machine_schedules.append(MachineSchedule(
            machine_idx=i,
            job_indices=job_indices,
            job_sequence=sequence,
            start_times=dp_result.start_times,
            total_energy=dp_result.total_energy,
            makespan=machine_makespan,
        ))
        
        total_energy += dp_result.total_energy
        overall_makespan = max(overall_makespan, machine_makespan)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return FullScheduleResult(
        total_energy=total_energy,
        makespan=overall_makespan,
        machine_schedules=machine_schedules,
        solve_time_ms=elapsed_ms,
        feasible=feasible,
    )


def epsilon_constraint_solve(
    raw_instance: RawInstance,
    model: Optional[QSequenceNet],
    variant_config,
    device: torch.device,
    use_model: bool = True,
) -> EpsilonConstraintResult:
    """Solve using accelerated epsilon-constraint method.
    
    1. Start with epsilon = T_max
    2. Solve, get makespan = Cmax
    3. Next epsilon = Cmax - 1 (tighten constraint)
    4. Continue until infeasible
    """
    start_time = time.perf_counter()
    
    T_max = raw_instance.T_max
    m = raw_instance.m
    # For parallel machines, T_min is approximately total processing / machines
    # Add some slack for imbalanced assignments
    T_min = max(1, int(sum(raw_instance.p) / m) - 10)
    
    pareto_points = []
    schedules = []
    num_iterations = 0
    
    epsilon = T_max
    
    while epsilon >= T_min:
        num_iterations += 1
        
        result = solve_parallel_machine_instance(
            raw_instance=raw_instance,
            model=model,
            variant_config=variant_config,
            device=device,
            epsilon=epsilon,
            use_model=use_model,
        )
        
        if not result.feasible or result.total_energy == float('inf'):
            break
        
        pareto_points.append((result.total_energy, result.makespan))
        schedules.append(result)
        
        # Accelerated epsilon update:
        # Next epsilon = actual makespan (skip redundant values)
        new_epsilon = result.makespan - 1
        
        if new_epsilon >= epsilon:
            # No improvement possible
            break
        
        epsilon = new_epsilon
    
    total_time_ms = (time.perf_counter() - start_time) * 1000
    
    return EpsilonConstraintResult(
        pareto_points=pareto_points,
        schedules=schedules,
        total_time_ms=total_time_ms,
        num_iterations=num_iterations,
    )


def run_benchmark(
    checkpoint_path: Optional[Path],
    scale: str,
    num_instances: int,
    seed: int,
    use_epsilon_constraint: bool,
    use_model: bool = True,
) -> None:
    """Run benchmark on instances of a given scale."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model if provided
    model = None
    variant_config = None
    if checkpoint_path is not None and use_model:
        print(f"Loading model from: {checkpoint_path}")
        model, variant_config = load_q_sequence_model(checkpoint_path, device=device)
        print("Model loaded successfully")
    else:
        variant_config = get_variant_config(VariantID.Q_SEQUENCE_CWE)
        print("Using SPT baseline (no model)")
    
    # Set up data config
    config = DataConfig()
    rng = random.Random(seed)
    
    # Set T_max based on scale
    if scale == "small":
        T_max_choices = [50, 80]
    elif scale in ("medium", "mls"):
        T_max_choices = [100, 300, 350]
    else:  # vls
        T_max_choices = [500]
    
    print(f"\n{'='*70}")
    print(f"Parallel Machine Scheduling Benchmark - Scale: {scale.upper()}")
    print(f"{'='*70}")
    print(f"Instances: {num_instances}, Seed: {seed}")
    print(f"Epsilon Constraint: {'Yes' if use_epsilon_constraint else 'No'}")
    print(f"Method: {'Q-Sequence + DP' if use_model else 'SPT + DP'}")
    
    results = []
    
    for i in range(num_instances):
        T_max = rng.choice(T_max_choices)
        instance = generate_raw_instance(config, rng, instance_id=i, T_max=T_max)
        
        if use_epsilon_constraint:
            ec_result = epsilon_constraint_solve(
                raw_instance=instance,
                model=model,
                variant_config=variant_config,
                device=device,
                use_model=use_model,
            )
            
            # Take the best energy point (first in Pareto set)
            if ec_result.pareto_points:
                best_energy, best_makespan = ec_result.pareto_points[0]
                results.append({
                    "instance": i,
                    "m": instance.m,
                    "n": instance.n,
                    "T_max": instance.T_max,
                    "energy": best_energy,
                    "makespan": best_makespan,
                    "time_ms": ec_result.total_time_ms,
                    "pareto_size": len(ec_result.pareto_points),
                    "ec_iterations": ec_result.num_iterations,
                    "feasible": True,
                })
            else:
                results.append({
                    "instance": i,
                    "m": instance.m,
                    "n": instance.n,
                    "T_max": instance.T_max,
                    "energy": float('inf'),
                    "makespan": -1,
                    "time_ms": ec_result.total_time_ms,
                    "pareto_size": 0,
                    "ec_iterations": ec_result.num_iterations,
                    "feasible": False,
                })
        else:
            result = solve_parallel_machine_instance(
                raw_instance=instance,
                model=model,
                variant_config=variant_config,
                device=device,
                epsilon=None,  # Use T_max
                use_model=use_model,
            )
            
            results.append({
                "instance": i,
                "m": instance.m,
                "n": instance.n,
                "T_max": instance.T_max,
                "energy": result.total_energy,
                "makespan": result.makespan,
                "time_ms": result.solve_time_ms,
                "pareto_size": 1,
                "ec_iterations": 1,
                "feasible": result.feasible,
            })
    
    # Print results
    print(f"\n{'ID':>3} {'m':>3} {'n':>4} {'T':>4} {'Energy':>10} {'Cmax':>6} {'Time(ms)':>10} {'Pareto':>7} {'Iters':>6}")
    print("-" * 70)
    
    for r in results:
        energy_str = f"{r['energy']:.2f}" if r['feasible'] else "INFEAS"
        print(f"{r['instance']:>3} {r['m']:>3} {r['n']:>4} {r['T_max']:>4} {energy_str:>10} {r['makespan']:>6} {r['time_ms']:>10.2f} {r['pareto_size']:>7} {r['ec_iterations']:>6}")
    
    # Summary statistics
    feasible_results = [r for r in results if r['feasible']]
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total instances: {len(results)}")
    print(f"Feasible: {len(feasible_results)} ({100*len(feasible_results)/len(results):.1f}%)")
    
    if feasible_results:
        avg_energy = np.mean([r['energy'] for r in feasible_results])
        avg_makespan = np.mean([r['makespan'] for r in feasible_results])
        avg_time = np.mean([r['time_ms'] for r in feasible_results])
        total_time = sum(r['time_ms'] for r in feasible_results)
        
        print(f"Average energy: {avg_energy:.2f}")
        print(f"Average makespan: {avg_makespan:.1f}")
        print(f"Average solve time: {avg_time:.2f} ms")
        print(f"Total solve time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
        
        if use_epsilon_constraint:
            avg_pareto = np.mean([r['pareto_size'] for r in feasible_results])
            avg_iters = np.mean([r['ec_iterations'] for r in feasible_results])
            print(f"Average Pareto size: {avg_pareto:.1f}")
            print(f"Average EC iterations: {avg_iters:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Full Parallel Machine Scheduling Pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Q-Sequence model checkpoint",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="small",
        choices=["small", "medium", "mls", "large", "vls"],
        help="Instance scale",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=10,
        help="Number of instances to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--epsilon_constraint",
        action="store_true",
        help="Use accelerated epsilon-constraint optimization",
    )
    parser.add_argument(
        "--spt_baseline",
        action="store_true",
        help="Use SPT baseline instead of Q-Sequence model",
    )
    
    args = parser.parse_args()
    
    checkpoint_path = None
    if args.checkpoint and not args.spt_baseline:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            # Try relative to PaST directory
            checkpoint_path = Path("/Users/mac/Documents/Study/PFE/PaST") / args.checkpoint
    
    run_benchmark(
        checkpoint_path=checkpoint_path,
        scale=args.scale,
        num_instances=args.num_instances,
        seed=args.seed,
        use_epsilon_constraint=args.epsilon_constraint,
        use_model=not args.spt_baseline,
    )


if __name__ == "__main__":
    main()
