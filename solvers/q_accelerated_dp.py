"""Q-Accelerated Dynamic Programming for Job Scheduling.

Compares:
1. Strong BnB (Baseline) - Rigorous BnB with relaxation bounds & bin packing
2. Q-Accel BnB (Root-Only) - Uses Q-model ONCE at root to order ALL branches
3. SGBS - Simulation-Guided Beam Search (Approximate)
4. Greedy - Simple argmin Q
5. Baselines - SPT/LPT

The key insight: calling the model at every BnB node is too expensive. Instead,
we use the model ONCE to compute a global job ordering, then use that fixed
ordering for all branching decisions (like a learned variable selection heuristic).

Example:
    python -m PaST.q_accelerated_dp \
        --checkpoint runs/best.pt \
        --num_instances 8 --scale small
"""

from __future__ import annotations

import argparse
import time
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from PaST.baselines_sequence_dp import (
    dp_schedule_for_job_sequence,
    spt_sequence,
    lpt_sequence,
)
from PaST.bnb_solver_custom import BranchAndBoundSolver, Instance
from PaST.config import VariantID, get_variant_config
from PaST.q_sequence_model import build_q_model, QSequenceNet
from PaST.sequence_env import GPUBatchSequenceEnv
from PaST.sm_benchmark_data import generate_episode_batch


@dataclass
class ProfileStats:
    model_forwards: int = 0
    model_time_s: float = 0.0
    dp_calls: int = 0
    dp_time_s: float = 0.0

    def add_model(self, dt: float):
        self.model_forwards += 1
        self.model_time_s += float(dt)

    def add_dp(self, dt: float):
        self.dp_calls += 1
        self.dp_time_s += float(dt)


# =============================================================================
# Q-Accelerated BnB: ROOT-ONLY Q-ordering (Efficient)
# =============================================================================


class QGuidedRootOrderingSolver(BranchAndBoundSolver):
    """
    BnB solver that uses Q-model ONCE at the root to compute a global job ordering.

    This is the correct way to use learned heuristics in BnB:
    - Compute Q-values at empty state (root)
    - Sort jobs by Q-value (best first)
    - Use this FIXED ordering for all branching decisions

    This gives BnB a "learned variable selection" heuristic while keeping
    the overhead minimal (one model forward pass total).
    """

    def __init__(
        self,
        instance: Instance,
        model: QSequenceNet,
        env_config,
        device: torch.device,
        time_limit: float = 300,
    ):
        super().__init__(instance, time_limit)
        self.model = model
        self.env_config = env_config
        self.device = device
        self.global_job_order: Optional[List[int]] = None  # Computed once at root

    def solve(self) -> Tuple[List[int], float]:
        """Override solve to compute Q-ordering at root before BnB."""
        # First compute the Q-guided ordering at the root (empty state)
        if hasattr(self, "single_data") and self.single_data is not None:
            self.global_job_order = self._compute_root_ordering()
        else:
            # Fallback to SPT ordering
            self.global_job_order = self._spt_heuristic()

        # Now run standard solve which uses our overridden _branch_and_bound_dfs
        return super().solve()

    def _compute_root_ordering(self) -> List[int]:
        """Compute job ordering using Q-model at empty state (ONE forward pass)."""
        n = self.instance.n_jobs

        env = GPUBatchSequenceEnv(
            batch_size=1, env_config=self.env_config, device=self.device
        )
        obs = env.reset(self.single_data)

        with torch.no_grad():
            jobs_t = obs["jobs"]
            periods_t = obs["periods"]
            ctx_t = obs["ctx"]

            # Get Q-values for all jobs
            q = self.model(jobs_t, periods_t, ctx_t)  # (1, N_pad)
            q = q[0].cpu().numpy()  # (N_pad,)

        # Sort valid jobs by Q-value (lower Q = better = earlier in search)
        job_q_pairs = [(j, q[j]) for j in range(n)]
        job_q_pairs.sort(key=lambda x: x[1])

        return [j for j, _ in job_q_pairs]

    def _branch_and_bound_dfs(
        self, partial_sequence: List[int], remaining_jobs: set, start_time: float
    ):
        if time.time() - start_time > self.time_limit:
            return

        self.nodes_explored += 1

        if not remaining_jobs:
            cost = self._evaluate_sequence(partial_sequence)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_sequence = list(partial_sequence)
            return

        # 1. Rigorous Lower Bound
        lb, blocks, _ = self._compute_lower_bound_with_blocks(
            partial_sequence, remaining_jobs
        )
        if lb >= self.best_cost:
            return

        # 2. Bin Packing Prune
        if blocks:
            self.binpack_attempts += 1
            packed = self._try_bin_packing(remaining_jobs, blocks)
            if packed:
                candidate = partial_sequence + packed
                self.best_cost = lb
                self.best_sequence = candidate
                self.pruned_by_binpack += 1
                return

        # 3. Q-Guided Branching using PRECOMPUTED global ordering
        # Filter remaining jobs, keep them in the precomputed order
        sorted_jobs = [j for j in self.global_job_order if j in remaining_jobs]

        for job in sorted_jobs:
            self._branch_and_bound_dfs(
                partial_sequence + [job], remaining_jobs - {job}, start_time
            )
            if self.best_cost <= lb:
                break


# =============================================================================
# SGBS (Simulation-Guided Beam Search) - FIXED VERSION
# =============================================================================


def sgbs_q_sequence_v2(
    model: QSequenceNet,
    variant_config,
    single: Dict[str, Any],
    device: torch.device,
    beta: int = 4,
    gamma: int = 4,
    use_dp_in_simulation: bool = True,
    prof: Optional[ProfileStats] = None,
) -> Dict[str, Any]:
    """
    Paper-faithful SGBS implementation with proper tracking.

    SGBS Algorithm:
    1. EXPANSION: For each beam state, expand top-gamma candidates using Q-model
    2. SIMULATION: Complete each candidate greedily, evaluate with DP
    3. PRUNING: Keep top-beta candidates by simulated cost

        Returns dict with energy, time, nodes_expanded, sequence.

        Notes:
        - If use_dp_in_simulation=True: paper-style, but expensive (many DP calls).
        - If use_dp_in_simulation=False: DP-free variant that ranks candidates by the
            Q-values already computed during expansion (fast, but approximate).
    """
    t0 = time.perf_counter()
    env_config = variant_config.env
    n = int(single["n_jobs"][0])

    # Beam: list of partial sequences
    beam: List[List[int]] = [[]]

    # Track statistics
    total_expansions = 0
    total_simulations = 0

    model.eval()
    with torch.no_grad():
        for step in range(n):
            candidates = []

            # 1. EXPANSION: For each beam state, expand top-gamma candidates
            for partial in beam:
                remaining = set(range(n)) - set(partial)
                if not remaining:
                    # Already complete
                    candidates.append(partial)
                    continue

                # Create env and replay to current state
                env = GPUBatchSequenceEnv(
                    batch_size=1, env_config=env_config, device=device
                )
                obs = env.reset(single)
                for a in partial:
                    obs, _, _, _ = env.step(torch.tensor([a], device=device))

                # Get Q-values for all jobs
                t_model0 = time.perf_counter()
                q_vals_t = model(obs["jobs"], obs["periods"], obs["ctx"])
                t_model1 = time.perf_counter()
                if prof is not None:
                    prof.add_model(t_model1 - t_model0)

                q_vals = q_vals_t[0].cpu().numpy()  # (N_pad,)

                # Get top-gamma from remaining jobs (lower Q = better)
                pairs = [(j, q_vals[j]) for j in remaining]
                pairs.sort(key=lambda x: x[1])

                for j, _ in pairs[:gamma]:
                    # Cache the Q(s,j) score at expansion time.
                    candidates.append((partial + [j], float(q_vals[j])))
                    total_expansions += 1

            if not candidates:
                break

            scored_candidates = []

            if not use_dp_in_simulation:
                # DP-free: rank by cached Q(s,j) from expansion.
                for cand, q_score in candidates:
                    scored_candidates.append((cand, q_score))
                total_simulations += len(scored_candidates)
            else:
                # 2. SIMULATION: Complete each candidate greedily and evaluate with DP
                for cand, _q_score in candidates:
                    remaining = set(range(n)) - set(cand)

                    # Build complete sequence
                    full_seq = list(cand)

                    if remaining:
                        # Create fresh env and replay to candidate state
                        env = GPUBatchSequenceEnv(
                            batch_size=1, env_config=env_config, device=device
                        )
                        obs = env.reset(single)
                        for a in cand:
                            obs, _, _, _ = env.step(torch.tensor([a], device=device))

                        # Greedy completion: pick best Q among remaining at each step
                        current_remaining = set(remaining)
                        while current_remaining:
                            t_model0 = time.perf_counter()
                            q_t = model(obs["jobs"], obs["periods"], obs["ctx"])
                            t_model1 = time.perf_counter()
                            if prof is not None:
                                prof.add_model(t_model1 - t_model0)

                            q = q_t[0].cpu().numpy()  # (N_pad,)

                            best_j = min(current_remaining, key=lambda j: q[j])
                            full_seq.append(best_j)
                            current_remaining.remove(best_j)

                            if current_remaining:
                                obs, _, _, _ = env.step(
                                    torch.tensor([best_j], device=device)
                                )

                    # Evaluate complete sequence with DP
                    t_dp0 = time.perf_counter()
                    res = dp_schedule_for_job_sequence(single, full_seq)
                    t_dp1 = time.perf_counter()
                    if prof is not None:
                        prof.add_dp(t_dp1 - t_dp0)

                    scored_candidates.append((cand, res.total_energy))
                    total_simulations += 1

            # 3. PRUNING: Keep top-beta candidates by simulated cost
            scored_candidates.sort(key=lambda x: x[1])
            beam = [c for c, _ in scored_candidates[:beta]]

    # Final: best beam entry should be complete (or nearly complete)
    best_seq = beam[0]
    if len(best_seq) < n:
        # Complete if somehow incomplete
        remaining = set(range(n)) - set(best_seq)
        best_seq = list(best_seq) + sorted(remaining)

    # Final exact DP (even in DP-free simulation mode)
    t_dp0 = time.perf_counter()
    res = dp_schedule_for_job_sequence(single, best_seq)
    t_dp1 = time.perf_counter()
    if prof is not None:
        prof.add_dp(t_dp1 - t_dp0)

    elapsed = time.perf_counter() - t0
    return {
        "energy": res.total_energy,
        "time": elapsed,
        "nodes": total_expansions,
        "simulations": total_simulations,
        "sequence": best_seq,
    }


# =============================================================================
# Helper: Greedy Decode
# =============================================================================


def greedy_decode_single(model, variant_config, single, device):
    """Greedy decoding: always pick job with lowest Q-value."""
    n = int(single["n_jobs"][0])
    env = GPUBatchSequenceEnv(
        batch_size=1, env_config=variant_config.env, device=device
    )
    obs = env.reset(single)
    seq = []

    with torch.no_grad():
        remaining = set(range(n))
        while remaining:
            q = model(obs["jobs"], obs["periods"], obs["ctx"])
            q = q[0].cpu().numpy()  # (N_pad,)

            # Find best among remaining
            best_j = min(remaining, key=lambda j: q[j])
            seq.append(best_j)
            remaining.remove(best_j)

            if remaining:
                obs, _, _, _ = env.step(torch.tensor([best_j], device=device))

    return seq


# =============================================================================
# Visualization
# =============================================================================


def plot_schedules(
    inst: Instance, methods_res: Dict[str, Any], title: str, output_path: str
):
    """Generate Gantt chart for multiple methods."""
    import matplotlib.pyplot as plt

    methods = list(methods_res.keys())
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(12, 2 + n_methods))

    # Colors for jobs
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(inst.n_jobs)]

    y_ticks = []
    y_labels = []

    for i, method in enumerate(methods):
        result = methods_res[method]
        if result is None:
            continue

        y_pos = i * 10
        y_ticks.append(y_pos)
        y_labels.append(f"{method}\nE={result.total_energy:.1f}")

        start_times = result.start_times
        seq = result.job_sequence

        for j_idx, start in zip(seq, start_times):
            duration = inst.processing_times[j_idx]
            ax.broken_barh(
                [(start, duration)],
                (y_pos - 4, 8),
                facecolors=colors[j_idx],
                edgecolors="k",
                alpha=0.8,
            )
            if duration > 2:
                ax.text(
                    start + duration / 2,
                    y_pos,
                    str(j_idx),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.axvline(x=inst.T, color="r", linestyle="--", label="Deadline")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


# =============================================================================
# Comparison Logic
# =============================================================================


def run_comparison(
    checkpoint,
    num_instances,
    seed,
    scale="small",
    device_str="cpu",
    beta=4,
    gamma=4,
    time_limit=30.0,
    plot_num=3,
    use_dp_in_sgbs: bool = True,
    profile: bool = False,
    no_plot: bool = False,
):
    device = torch.device(device_str)

    # Load Model
    vid = VariantID("q_sequence")
    vcfg = get_variant_config(vid)
    model = build_q_model(vcfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    data_cfg = vcfg.data
    if scale == "small":
        data_cfg.T_max_choices = [t for t in data_cfg.T_max_choices if t <= 100]
    elif scale == "medium":
        data_cfg.T_max_choices = [t for t in data_cfg.T_max_choices if 100 < t <= 350]

    batch = generate_episode_batch(
        num_instances, data_cfg, seed, vcfg.env.N_job_pad, 250, 500
    )

    results = {
        "Strong BnB": {"e": [], "t": [], "n": []},
        "Q-Accel BnB": {"e": [], "t": [], "n": []},
        f"SGBS (b{beta}g{gamma})": {"e": [], "t": [], "n": []},
        "Greedy": {"e": [], "t": [], "n": []},
        "SPT": {"e": [], "t": [], "n": []},
    }

    print(f"Comparing on {num_instances} {scale} instances...")

    for i in range(num_instances):
        single = _slice_single_instance(batch, i)

        prof = ProfileStats() if profile else None

        # Instance Setup
        n_jobs = int(single["n_jobs"][0])
        if torch.is_tensor(single["p_subset"]):
            pts = single["p_subset"][0, :n_jobs].cpu().numpy()
        else:
            pts = single["p_subset"][0, :n_jobs]

        T = int(single["T_limit"][0])
        if torch.is_tensor(single["ct"]):
            ct = single["ct"][0].cpu().numpy()
        else:
            ct = single["ct"][0]

        total_T = T
        if ct.ndim == 1:
            energy_costs = ct[:total_T]
        else:
            energy_costs = np.zeros(total_T)
            for k in range(len(ct)):
                dur = int(ct[k, 0])
                price = ct[k, 1]
                start = int(ct[k, 2])
                if start >= total_T:
                    continue
                end = min(start + dur, total_T)
                energy_costs[start:end] = price

        inst = Instance(n_jobs, pts, T, energy_costs)

        methods_plot_data = {}

        # --- Strong BnB ---
        t0 = time.perf_counter()
        solver = BranchAndBoundSolver(inst, time_limit=time_limit)
        seq, _ = solver.solve()
        t_bnb = time.perf_counter() - t0

        if seq is None:
            seq = spt_sequence(pts, n_jobs)
        t_dp0 = time.perf_counter()
        res_bnb = dp_schedule_for_job_sequence(single, seq)
        t_dp1 = time.perf_counter()
        if prof is not None:
            prof.add_dp(t_dp1 - t_dp0)

        results["Strong BnB"]["e"].append(res_bnb.total_energy)
        results["Strong BnB"]["t"].append(t_bnb)
        results["Strong BnB"]["n"].append(solver.nodes_explored)
        methods_plot_data["Strong BnB"] = res_bnb

        # --- Q-Accel BnB (Root-Only Ordering) ---
        t0 = time.perf_counter()
        q_solver = QGuidedRootOrderingSolver(
            inst, model, vcfg.env, device, time_limit=time_limit
        )
        q_solver.single_data = single
        seq, _ = q_solver.solve()
        t_qbnb = time.perf_counter() - t0

        if seq is None:
            seq = spt_sequence(pts, n_jobs)
        t_dp0 = time.perf_counter()
        res_qbnb = dp_schedule_for_job_sequence(single, seq)
        t_dp1 = time.perf_counter()
        if prof is not None:
            prof.add_dp(t_dp1 - t_dp0)

        results["Q-Accel BnB"]["e"].append(res_qbnb.total_energy)
        results["Q-Accel BnB"]["t"].append(t_qbnb)
        results["Q-Accel BnB"]["n"].append(q_solver.nodes_explored)
        methods_plot_data["Q-Accel BnB"] = res_qbnb

        # --- SGBS ---
        res_sgbs_dict = sgbs_q_sequence_v2(
            model,
            vcfg,
            single,
            device,
            beta=beta,
            gamma=gamma,
            use_dp_in_simulation=use_dp_in_sgbs,
            prof=prof,
        )
        res_sgbs = dp_schedule_for_job_sequence(single, res_sgbs_dict["sequence"])

        results[f"SGBS (b{beta}g{gamma})"]["e"].append(res_sgbs.total_energy)
        results[f"SGBS (b{beta}g{gamma})"]["t"].append(res_sgbs_dict["time"])
        results[f"SGBS (b{beta}g{gamma})"]["n"].append(res_sgbs_dict["nodes"])
        methods_plot_data[f"SGBS (b{beta}g{gamma})"] = res_sgbs

        # --- Greedy ---
        t0 = time.perf_counter()
        seq_greedy = greedy_decode_single(model, vcfg, single, device)
        t_greedy = time.perf_counter() - t0
        res_greedy = dp_schedule_for_job_sequence(single, seq_greedy)

        results["Greedy"]["e"].append(res_greedy.total_energy)
        results["Greedy"]["t"].append(t_greedy)
        results["Greedy"]["n"].append(0)
        methods_plot_data["Greedy"] = res_greedy

        # --- SPT ---
        spt = spt_sequence(pts, n_jobs)
        res_spt = dp_schedule_for_job_sequence(single, spt)
        results["SPT"]["e"].append(res_spt.total_energy)
        results["SPT"]["t"].append(0.0)
        results["SPT"]["n"].append(0)
        methods_plot_data["SPT"] = res_spt

        # Plotting
        if (not no_plot) and (i < plot_num):
            out_path = f"PaST/plots/comparison/instance_{i}.png"
            plot_schedules(
                inst, methods_plot_data, f"Instance {i} (N={n_jobs}, T={T})", out_path
            )

        if prof is not None:
            print(
                f"[profile inst {i}] model_forwards={prof.model_forwards} model_time={prof.model_time_s:.4f}s "
                f"dp_calls={prof.dp_calls} dp_time={prof.dp_time_s:.4f}s use_dp_in_sgbs={use_dp_in_sgbs}"
            )

    # Print Summary
    print("\n" + "=" * 90)
    print("Q-ACCELERATED DP COMPARISON")
    print("=" * 90)
    print(
        f"{'Method':<20} | {'Energy':<10} | {'Time (s)':<10} | {'Nodes':<10} | {'Runs (n)':<8}"
    )
    print("-" * 90)

    for method, data in results.items():
        energies = np.array(data["e"])
        times = np.array(data["t"])
        nodes = np.array(data["n"])

        valid = ~np.isnan(energies)
        if not np.any(valid):
            print(f"{method:<20} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {0:<8}")
            continue

        e_mean = np.mean(energies[valid])
        t_mean = np.mean(times[valid])
        n_mean = np.mean(nodes[valid]) if len(nodes) > 0 else 0
        n_runs = np.sum(valid)

        print(
            f"{method:<20} | {e_mean:<10.2f} | {t_mean:<10.4f} | {n_mean:<10.0f} | {n_runs:<8}"
        )
    print("=" * 90)


def _slice_single_instance(batch, i):
    """Extract single instance from batch."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[i : i + 1]
        elif isinstance(v, np.ndarray):
            out[k] = v[i : i + 1]
        else:
            out[k] = v[i : i + 1]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scale", default="small")
    parser.add_argument("--num_instances", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=int, default=4, help="Beam width for SGBS")
    parser.add_argument(
        "--gamma", type=int, default=4, help="Expansion factor for SGBS"
    )
    parser.add_argument(
        "--time_limit", type=float, default=30.0, help="Time limit for BnB solvers"
    )
    parser.add_argument(
        "--plot_num", type=int, default=3, help="Number of instances to plot"
    )
    parser.add_argument(
        "--sgbs_no_dp",
        action="store_true",
        help="Use DP-free SGBS ranking (approximate, faster)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-instance timing breakdown for model vs DP",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable plotting (avoids matplotlib dependency)",
    )
    args = parser.parse_args()

    run_comparison(
        args.checkpoint,
        args.num_instances,
        args.seed,
        args.scale,
        beta=args.beta,
        gamma=args.gamma,
        time_limit=args.time_limit,
        plot_num=args.plot_num,
        use_dp_in_sgbs=(not args.sgbs_no_dp),
        profile=args.profile,
        no_plot=args.no_plot,
    )
