"""RL Environment for NeuroLS learned local search.

This module implements a gym-like environment that wraps the local search
process as an MDP following the NeuroLS paper.

Episode structure:
1. reset(instance, K): Initialize with a scheduling instance and horizon K
2. step(action): Execute decoded action (accept/reject, operator, perturbation)
3. Episode terminates when:
   - Time limit reached (max_steps)
   - No improvement for too long (stagnation)
   - Temperature annealing complete

State: NeuroLSState containing solution + LS statistics
Action: Integer from ActionSpace (e.g., AANP has 14 actions)
Reward: Improvement in best cost (energy minimization)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, TYPE_CHECKING
import numpy as np
import random

if TYPE_CHECKING:
    from PaST.data.sm_benchmark_data import RawInstance

from PaST.neurols.solution import PMALNSSolution
from PaST.neurols.state import NeuroLSState, StateBuilder
from PaST.neurols.operators import OPERATORS, OPERATOR_BY_ID, OperatorID
from PaST.neurols.perturbations import PERTURBATION_BY_ID, PerturbationID
from PaST.neurols.action_space import ActionSpace, ActionType, AANP_SPACE, DecodedAction
from PaST.neurols.candidate_generator import CandidateGenerator, CandidateConfig
from PaST.neurols.move_evaluator import MoveEvaluator, FullEvaluation, MoveEvaluation
from PaST.neurols.price_embedding import PriceFeatureExtractor


def _per_machine_energy_and_makespan(full_eval: FullEvaluation):
    per_machine_energy = [float(me.energy) for me in full_eval.per_machine]
    per_machine_makespan = [int(me.makespan) for me in full_eval.per_machine]
    return per_machine_energy, per_machine_makespan


@dataclass
class EnvConfig:
    """Configuration for NeuroLS environment."""

    # Episode limits
    max_steps: int = 1000
    stagnation_limit: int = 200  # Terminate if no improve for this many steps

    # Acceptance probability (Metropolis style)
    use_acceptance: bool = True
    initial_temp: float = 1.0
    final_temp: float = 0.01
    cooling_rate: float = 0.995  # temp *= cooling_rate each step

    # Action space
    action_space: ActionSpace = field(default_factory=lambda: AANP_SPACE)

    # Candidate generation
    top_k: int = 10  # Top-K moves per operator
    use_proxy: bool = True  # Use proxy ranking for large instances

    # Reward shaping
    reward_mode: str = "improvement"  # improvement, normalized, potential
    improvement_scale: float = 1.0
    step_penalty: float = 0.0  # Small penalty per step to encourage faster convergence

    # Solution initialization
    init_mode: str = "load_balanced"  # load_balanced, random, assignment

    # Deterministic mode (for evaluation — disables SA acceptance)
    deterministic: bool = False

    def __post_init__(self):
        # Ensure action_space is ActionSpace instance
        if isinstance(self.action_space, str):
            from PaST.neurols.action_space import AA_SPACE, AAN_SPACE, AANP_SPACE

            space_map = {"AA": AA_SPACE, "AAN": AAN_SPACE, "AANP": AANP_SPACE}
            self.action_space = space_map.get(self.action_space, AANP_SPACE)


class NeuroLSEnv:
    """RL environment for learned local search.

    Implements gym-like interface for NeuroLS training.
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()

        # Will be set in reset()
        self.instance: Optional[RawInstance] = None
        self.K: int = 0
        self.state: Optional[NeuroLSState] = None
        self.temperature: float = self.config.initial_temp

        # Helpers - initialized lazily
        self._evaluator: Optional[MoveEvaluator] = None
        self._candidate_gen: Optional[CandidateGenerator] = None
        self._current_eval = None  # FullEvaluation from evaluator

        # Episode tracking
        self._step_count: int = 0
        self._no_improve_count: int = 0
        self._best_cost_episode: float = float("inf")
        self._initial_cost: float = float("inf")

        # Cached instance data (set in reset)
        self._processing_times: Optional[np.ndarray] = None
        self._ct: Optional[np.ndarray] = None
        self._machine_energy_rates: Optional[np.ndarray] = None
        self._price_per_hour: Optional[np.ndarray] = None  # (hours_per_day, 5)

        # RNG for perturbations
        self._rng = np.random.default_rng()

    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
        random.seed(seed)

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions."""
        return self.config.action_space.n_actions

    def reset(
        self,
        instance: RawInstance,
        K: int,
        initial_solution: Optional[PMALNSSolution] = None,
    ) -> NeuroLSState:
        """Reset environment with new instance.

        Args:
            instance: Scheduling instance
            K: Horizon constraint (makespan == K)
            initial_solution: Optional starting solution

        Returns:
            Initial NeuroLSState
        """
        self.instance = instance
        self.K = K

        # Extract instance data as numpy arrays
        # RawInstance fields: .p (processing times), .ct (per-slot prices), .e (energy rates)
        self._processing_times = np.asarray(instance.p, dtype=np.int64)
        # Truncate to K so the price feature has consistent length K.
        self._ct = np.asarray(instance.ct[:K], dtype=np.float64)
        self._machine_energy_rates = np.asarray(instance.e, dtype=np.float64)
        n = len(self._processing_times)
        m = instance.m

        # Cache per-hour price features for the CNN encoder (constant per episode)
        _price_extractor = PriceFeatureExtractor(self._ct, K)
        self._price_per_hour = _price_extractor.get_per_hour_features()  # (20, 5)

        # Create evaluator
        self._evaluator = MoveEvaluator(
            processing_times=self._processing_times,
            ct=self._ct,
            machine_energy_rates=self._machine_energy_rates,
            K=K,
            n_machines=m,
        )

        # Create candidate generator
        # EnvConfig.top_k controls how many moves we evaluate in large neighborhoods.
        cand_config = CandidateConfig(
            max_moves_topk=self.config.top_k,
            use_proxy_ranking=self.config.use_proxy,
        )
        self._candidate_gen = CandidateGenerator(
            evaluator=self._evaluator,
            config=cand_config,
        )

        # Initialize solution
        if initial_solution is not None:
            solution = initial_solution
        else:
            if self.config.init_mode == "load_balanced":
                solution = PMALNSSolution.from_load_balanced(
                    n_jobs=n,
                    n_machines=m,
                    processing_times=self._processing_times,
                )
            else:
                # Random assignment
                assignment = [int(self._rng.integers(m)) for _ in range(n)]
                solution = PMALNSSolution.from_assignment(
                    assignment=assignment,
                    n_machines=m,
                    processing_times=self._processing_times,
                    sort_by="spt",
                )

        # Evaluate initial solution
        self._current_eval = self._evaluator.evaluate_solution(solution)
        initial_cost = float(self._current_eval.total_energy)

        self._initial_cost = initial_cost
        self._best_cost_episode = initial_cost

        # Initialize state using StateBuilder
        self.state = StateBuilder.from_instance(
            processing_times=self._processing_times,
            machine_energy_rates=self._machine_energy_rates,
            ct=self._ct,
            K=K,
            n_machines=m,
            initial_solution=solution,
            max_iterations=self.config.max_steps,
        )
        # Set costs from evaluation
        self.state.current_cost = initial_cost
        self.state.best_cost = initial_cost
        per_machine_energy, per_machine_makespan = _per_machine_energy_and_makespan(
            self._current_eval
        )
        self.state.per_machine_energy = per_machine_energy
        self.state.per_machine_makespan = per_machine_makespan
        self.state.makespan = int(self._current_eval.makespan)

        # Reset episode tracking
        self._step_count = 0
        self._no_improve_count = 0
        self.temperature = self.config.initial_temp

        return self.state

    def step(self, action: int) -> Tuple[NeuroLSState, float, bool, Dict[str, Any]]:
        """Execute one step of local search.

        Args:
            action: Integer action from action space

        Returns:
            (next_state, reward, done, info)
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Decode action
        decoded = self.config.action_space.decode_action(action)

        # Execute action
        reward, info = self._execute_action(decoded)

        # Update episode tracking
        self._step_count += 1

        # Check termination
        done = self._is_done()

        return self.state, reward, done, info

    def _execute_action(self, decoded: DecodedAction) -> Tuple[float, Dict[str, Any]]:
        """Execute decoded action and compute reward.

        Args:
            decoded: DecodedAction(accept, action_type, operator_id, perturbation_id)

        Returns:
            (reward, info_dict)
        """
        info = {
            "accept_action": decoded.accept,
            "operator_id": decoded.operator_id,
            "perturbation_id": decoded.perturbation_id,
            "cost_before": self.state.current_cost,
            "accepted": False,
            "improved": False,
        }

        # Handle operator actions
        if (
            decoded.action_type == ActionType.OPERATOR
            and decoded.operator_id is not None
        ):
            move_result = self._apply_operator(decoded.operator_id)
            info["move_result"] = move_result

            if move_result is not None:
                move, new_cost, new_eval = move_result
                info["new_cost"] = new_cost

                # Decision: accept or reject
                accepted = self._make_acceptance_decision(
                    decoded.accept,
                    self.state.current_cost,
                    new_cost,
                )

                if accepted:
                    # Apply move to a cloned solution
                    new_solution = self.state.solution.clone()
                    OPERATOR_BY_ID[move.operator].apply_move(new_solution, move)

                    # Update evaluator state (FullEvaluation) using incremental results
                    if isinstance(new_eval, MoveEvaluation):
                        updated_per_machine = list(self._current_eval.per_machine)
                        for mi, machine_eval in new_eval.affected_machine_evals.items():
                            updated_per_machine[int(mi)] = machine_eval
                        self._current_eval = FullEvaluation(
                            total_energy=float(new_cost),
                            makespan=int(new_eval.new_makespan),
                            feasible=bool(new_eval.feasible),
                            per_machine=updated_per_machine,
                            per_job_costs=None,
                        )

                    per_machine_energy, per_machine_makespan = (
                        _per_machine_energy_and_makespan(self._current_eval)
                    )

                    # Update state via update_from_move
                    self.state.update_from_move(
                        new_solution=new_solution,
                        new_cost=new_cost,
                        new_per_machine_energy=per_machine_energy,
                        new_per_machine_makespan=per_machine_makespan,
                        new_makespan=int(self._current_eval.makespan),
                        accepted=True,
                        operator=decoded.operator_id,
                    )
                    info["accepted"] = True

                    # Check if improved best
                    if new_cost < self._best_cost_episode:
                        self._best_cost_episode = new_cost
                        self._no_improve_count = 0
                        info["improved"] = True
                    else:
                        self._no_improve_count += 1
                else:
                    # Move rejected — update state (no solution change)
                    self.state.update_from_move(
                        new_solution=self.state.solution,
                        new_cost=self.state.current_cost,
                        new_per_machine_energy=self.state.per_machine_energy,
                        new_per_machine_makespan=self.state.per_machine_makespan,
                        new_makespan=self.state.makespan,
                        accepted=False,
                        operator=decoded.operator_id,
                    )
                    self._no_improve_count += 1
            else:
                # No valid move found
                self._no_improve_count += 1
                self.state.step_t += 1

        # Handle perturbation actions
        if (
            decoded.action_type == ActionType.PERTURBATION
            and decoded.perturbation_id is not None
            and decoded.perturbation_id != PerturbationID.NONE
        ):
            perturbation = PERTURBATION_BY_ID[decoded.perturbation_id]

            new_solution = perturbation.apply(
                solution=self.state.solution,
                evaluation=self._current_eval,
                processing_times=self._processing_times,
                ct=self._ct,
                machine_energy_rates=self._machine_energy_rates,
                K=self.K,
            )

            # Evaluate perturbed solution
            new_eval = self._evaluator.evaluate_solution(new_solution)
            new_cost = float(new_eval.total_energy)

            per_machine_energy, per_machine_makespan = _per_machine_energy_and_makespan(
                new_eval
            )

            # Update state via update_from_perturbation
            self.state.update_from_perturbation(
                new_solution=new_solution,
                new_cost=new_cost,
                new_per_machine_energy=per_machine_energy,
                new_per_machine_makespan=per_machine_makespan,
                new_makespan=int(new_eval.makespan),
            )
            self._current_eval = new_eval

            info["perturbed"] = True
            info["cost_after_perturbation"] = new_cost
            self._no_improve_count = 0

        elif decoded.action_type == ActionType.PERTURBATION and (
            decoded.perturbation_id is None
            or decoded.perturbation_id == PerturbationID.NONE
        ):
            # NONE perturbation — just advance step
            self.state.step_t += 1

        # Compute reward
        reward = self._compute_reward(info)
        info["reward"] = reward

        # Update temperature
        self.temperature *= self.config.cooling_rate
        self.temperature = max(self.temperature, self.config.final_temp)

        return reward, info

    def _apply_operator(self, operator_id: OperatorID):
        """Generate and select best move for operator.

        Returns:
            (move, new_cost, move_eval) or None if no valid move
        """
        operator = OPERATOR_BY_ID[operator_id]

        # Use candidate generator to find best move
        best_result = self._candidate_gen.generate_best_move(
            solution=self.state.solution,
            current_eval=self._current_eval,
            operator=operator,
        )

        if best_result is None:
            return None

        # best_result is a MoveEvaluation with .move, .new_cost, etc.
        return best_result.move, best_result.new_cost, best_result

    def _make_acceptance_decision(
        self,
        action_accept: bool,
        current_cost: float,
        new_cost: float,
    ) -> bool:
        """Decide whether to accept move based on action and costs.

        Args:
            action_accept: True = always accept if improving or use SA
                           False = only accept if strictly improving
            current_cost: Current solution cost
            new_cost: New solution cost after move

        Returns:
            Whether to accept the move
        """
        # Always accept improvements
        if new_cost < current_cost:
            return True

        # For worsening moves, check action
        if not action_accept:
            return False

        # In deterministic mode (evaluation), never accept worsening moves
        if self.config.deterministic:
            return False

        # Action says accept -> use Metropolis criterion
        if not self.config.use_acceptance:
            return True

        delta = new_cost - current_cost
        if self.temperature > 0:
            prob = np.exp(-delta / self.temperature)
            return float(self._rng.random()) < prob
        else:
            return False

    def _compute_reward(self, info: Dict[str, Any]) -> float:
        """Compute reward for step.

        Args:
            info: Step info dictionary

        Returns:
            Reward value
        """
        mode = self.config.reward_mode

        if mode == "improvement":
            # Reward = improvement in best cost, clamped at 0
            # Paper eq. (4): r_t = max(f(s_hat_t) - f(s_{t+1}), 0)
            old_best = info["cost_before"]
            new_best = self.state.best_cost
            improvement = old_best - new_best
            reward = max(improvement, 0.0) * self.config.improvement_scale

        elif mode == "normalized":
            # Normalize by initial cost
            old_best = info["cost_before"]
            new_best = self.state.best_cost
            improvement = old_best - new_best
            reward = (
                improvement
                / (self._initial_cost + 1e-8)
                * self.config.improvement_scale
            )

        elif mode == "potential":
            # Potential-based shaping: phi(s') - phi(s)
            # phi(s) = -current_cost (so improvement gives positive reward)
            old_cost = info["cost_before"]
            new_cost = self.state.current_cost
            reward = (old_cost - new_cost) * self.config.improvement_scale

        else:
            reward = 0.0

        # Step penalty
        reward -= self.config.step_penalty

        return reward

    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        # Max steps reached
        if self._step_count >= self.config.max_steps:
            return True

        # Stagnation
        if self._no_improve_count >= self.config.stagnation_limit:
            return True

        return False

    def get_state_features(self) -> Dict[str, np.ndarray]:
        """Get state features for neural network.

        Returns dict with keys:
            - job_features: (n_jobs, d_job)
            - machine_features: (m, d_machine)
            - state_features: (d_state,)
            - job_to_machine: (n_jobs,) machine assignment per job
            - assignment_edges: (2, n_jobs) raw assignment (job_i -> machine_m)
            - static_edge_index: (2, E_static) full bipartite edges with node offsets
            - dynamic_edge_index: (2, E_dynamic) current assignment edges with node offsets
            - prices: (T,)
        """
        if self.state is None or self.instance is None:
            raise RuntimeError("Environment not initialized")

        n = len(self._processing_times)
        m = self.instance.m

        # Use individual feature extraction methods from NeuroLSState
        src, dst = self.state.get_assignment_edges()
        assignment_edges = np.stack([src, dst], axis=0)  # (2, n_jobs)
        job_to_machine = np.asarray(dst, dtype=np.int64)  # (n_jobs,)

        # Build static bipartite edges (all jobs can go to all machines)
        # Node indices: jobs [0..n-1], machines [n..n+m-1]
        static_src = []
        static_dst = []
        for j in range(n):
            for mi in range(m):
                static_src.append(j)
                static_dst.append(n + mi)
                static_src.append(n + mi)
                static_dst.append(j)
        static_edge_index = np.array([static_src, static_dst], dtype=np.int64)

        # Build dynamic edges from current assignment (with node offset)
        dyn_src = []
        dyn_dst = []
        for j in range(n):
            mi = dst[j]  # machine index for job j
            dyn_src.append(j)
            dyn_dst.append(n + mi)
            dyn_src.append(n + mi)
            dyn_dst.append(j)
        dynamic_edge_index = np.array([dyn_src, dyn_dst], dtype=np.int64)

        return {
            "job_features": self.state.get_job_features(),
            "machine_features": self.state.get_machine_features(),
            "state_features": self.state.get_scalar_features(),
            "job_to_machine": job_to_machine,
            "assignment_edges": assignment_edges,
            "static_edge_index": static_edge_index,
            "dynamic_edge_index": dynamic_edge_index,
            "prices": self._ct.astype(np.float32),
            "price_per_hour": self._price_per_hour,  # (hours_per_day, 5)
        }

    def get_torch_features(self, device: str = "cpu"):
        """Get state features as torch tensors.

        Returns dict with torch tensors ready for neural network.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch required for torch features")

        features = self.get_state_features()

        return {
            "job_features": torch.tensor(
                features["job_features"], dtype=torch.float32, device=device
            ),
            "machine_features": torch.tensor(
                features["machine_features"], dtype=torch.float32, device=device
            ),
            "state_features": torch.tensor(
                features["state_features"], dtype=torch.float32, device=device
            ),
            "job_to_machine": torch.tensor(
                features["job_to_machine"], dtype=torch.long, device=device
            ),
            "assignment_edges": torch.tensor(
                features["assignment_edges"], dtype=torch.long, device=device
            ),
            "static_edge_index": torch.tensor(
                features["static_edge_index"], dtype=torch.long, device=device
            ),
            "dynamic_edge_index": torch.tensor(
                features["dynamic_edge_index"], dtype=torch.long, device=device
            ),
            "prices": torch.tensor(
                features["prices"], dtype=torch.float32, device=device
            ),
            "price_per_hour": torch.tensor(
                features["price_per_hour"], dtype=torch.float32, device=device
            ),
        }

    def render(self, mode: str = "text") -> Optional[str]:
        """Render current state.

        Args:
            mode: "text" for string representation

        Returns:
            String representation if mode="text"
        """
        if self.state is None:
            return "Environment not initialized"

        lines = [
            f"Step: {self._step_count}/{self.config.max_steps}",
            f"Current cost: {self.state.current_cost:.2f}",
            f"Best cost: {self.state.best_cost:.2f}",
            f"Initial cost: {self._initial_cost:.2f}",
            f"Improvement: {(self._initial_cost - self.state.best_cost) / self._initial_cost * 100:.2f}%",
            f"Temperature: {self.temperature:.4f}",
            f"No-improve steps: {self._no_improve_count}",
            f"Last accepted: {self.state.last_acceptance}",
            f"Last operator: {self.state.last_operator}",
        ]

        if mode == "text":
            return "\n".join(lines)
        else:
            print("\n".join(lines))
            return None


class VectorizedNeuroLSEnv:
    """Vectorized environment for parallel training.

    Runs multiple NeuroLS environments in parallel.
    """

    def __init__(
        self,
        n_envs: int,
        config: Optional[EnvConfig] = None,
    ):
        self.n_envs = n_envs
        self.envs = [NeuroLSEnv(config) for _ in range(n_envs)]

    def seed(self, seed: int):
        """Set seeds for all environments."""
        for i, env in enumerate(self.envs):
            env.seed(seed + i)

    @property
    def action_space_size(self) -> int:
        return self.envs[0].action_space_size

    def reset(
        self,
        instances: List[RawInstance],
        Ks: List[int],
    ) -> List[NeuroLSState]:
        """Reset all environments.

        Args:
            instances: List of n_envs instances
            Ks: List of n_envs horizon values

        Returns:
            List of initial states
        """
        if len(instances) != self.n_envs or len(Ks) != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} instances and Ks")

        states = []
        for env, instance, K in zip(self.envs, instances, Ks):
            state = env.reset(instance, K)
            states.append(state)

        return states

    def step(
        self,
        actions: List[int],
    ) -> Tuple[List[NeuroLSState], List[float], List[bool], List[Dict]]:
        """Step all environments.

        Args:
            actions: List of n_envs actions

        Returns:
            (states, rewards, dones, infos)
        """
        states, rewards, dones, infos = [], [], [], []

        for env, action in zip(self.envs, actions):
            state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return states, rewards, dones, infos

    def get_batch_features(self, device: str = "cpu"):
        """Get batched features for all environments.

        Returns dict with batched torch tensors.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch required")

        all_features = [env.get_torch_features(device) for env in self.envs]

        # Stack features that have same dimensions
        return {
            "job_features": torch.stack([f["job_features"] for f in all_features]),
            "machine_features": torch.stack(
                [f["machine_features"] for f in all_features]
            ),
            "state_features": torch.stack([f["state_features"] for f in all_features]),
            "job_to_machine": torch.stack([f["job_to_machine"] for f in all_features]),
            "assignment_edges": torch.stack(
                [f["assignment_edges"] for f in all_features]
            ),
            "static_edge_index": torch.stack(
                [f["static_edge_index"] for f in all_features]
            ),
            "dynamic_edge_index": torch.stack(
                [f["dynamic_edge_index"] for f in all_features]
            ),
            "prices": torch.stack([f["prices"] for f in all_features]),
            "price_per_hour": torch.stack([f["price_per_hour"] for f in all_features]),
        }
