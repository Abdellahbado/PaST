# NeuroLS in PaST: Learned Local Search Controller for Parallel-Machine Assignment + Sequencing

This folder contains our **NeuroLS-style approach** for the parallel machine setting.

The goal of this workstream is to learn a **controller over local search** that improves a current solution by choosing:

1) whether to **accept/reject** a candidate move,
2) which **neighborhood operator** to apply (relocate/swap/insert/block), and
3) when to apply **perturbations** (shake/restart) to escape local optima.

Importantly for *our* problem: we focus on **(i) assigning jobs to machines** and **(ii) sequencing jobs per machine**.
We explicitly **do not learn continuous scheduling/timing** decisions; instead, given an assignment+sequence, the **timing is computed by a deterministic dynamic program (DP)** that finds the cheapest placement under the time-of-use (TOU) price timeline and a horizon constraint.

This is intended for communicating the approach to supervisors: *what the reference paper did* and *how we adapt it to our case*.

---

## 1) Reference paper: what NeuroLS did (and why it’s relevant)

**Paper:** *Learning to Control Local Search for Combinatorial Optimization* (Falkner et al.), arXiv:2206.13181.

### Key idea

Many strong heuristics for combinatorial optimization are variants of **local search (LS)** wrapped by **meta-heuristics** (e.g., simulated annealing / iterated local search / variable neighborhood methods). A core difficulty is that selecting *which move family to try*, *whether to accept a proposed move*, and *when/how to perturb* is traditionally hand-tuned.

NeuroLS formalizes those intervention points as a **Markov Decision Process (MDP)** and learns a policy (in the paper: a Q-learning style controller) that *steers* a generic LS procedure.

### “Three independent aspects” controlled by NeuroLS

The paper identifies and learns decisions over:

- **Acceptance rule**: accept/reject proposed moves (including sometimes accepting worse moves, like simulated annealing).
- **Neighborhood selection**: choose which operator/neighborhood to explore at each step.
- **Perturbation / restart**: decide when to apply larger changes to escape local optima.

These are exposed via action spaces named **AA / AAN / AANP**:

- **AA**: acceptance only (binary accept/reject), with a fixed operator.
- **AAN**: acceptance + neighborhood/operator selection.
- **AANP**: acceptance + operator selection + perturbation selection.

### Model architecture

NeuroLS uses a **Graph Neural Network (GNN)** encoder to represent the state of the current solution and search trajectory, then predicts action-values.
At a high level (matching the paper’s description):

- Build a graph representation of the current solution.
- Run multiple GNN message-passing phases (static structure → dynamic edges → consolidation).
- Pool to obtain a compact representation.
- Use an MLP head to output Q-values for the discrete action set.

### Why this paper is “in the spirit” of our problem

Even though the paper demonstrates on problems like CVRP and job shop scheduling, the **structural pattern matches our setting**:

- Our objective is a combinatorial objective defined over a solution space (assignment + sequencing decisions).
- We already rely on an **iterative improvement** routine (local search) with multiple neighborhoods.
- We face the same question: *which neighborhood to try next, when to accept a non-improving move, and when to perturb/restart.*

So the paper’s problem class and our problem are aligned at the “algorithm-control” level: both can be phrased as **learning to control local search**.

---

## 2) Our problem framing (what we control vs what we solve exactly)

### What we learn/control

We learn a policy that acts on:

- **Assignment**: which machine each job belongs to.
- **Sequencing**: the order of jobs on each machine.

These are represented by a solution object consisting of per-machine job lists.

### What we do *not* learn (and why)

We do **not** directly learn the fine-grained schedule start times.
Given a per-machine job sequence and a TOU price horizon, the timing problem is solved with **DP** to obtain an optimal placement (minimum energy cost) under the horizon constraint $K$.

This makes the RL problem cleaner:

- The agent searches over **assignment+permutation structure**.
- The environment computes objective values using **exact timing** (DP) for each candidate structure.

---

## 3) High-level algorithm: learned controller over a deterministic LS core

At each LS step:

1. The current solution induces a state representation (graph + features).
2. For each available operator, we deterministically generate the “best candidate move” in that neighborhood.
3. The agent selects an action of the form:
   - accept/reject + operator, or
   - accept/reject + perturbation (shake/restart).
4. If accepted, the environment updates the solution; otherwise it remains.
5. Reward is computed from **improvement in best-found cost**, with optional shaping.

This matches the NeuroLS paper’s spirit (controller chooses *how to run LS*), but adapts evaluation to our domain by using **DP-based cost evaluation**.

---

## 4) Implementation map (what lives in this folder)

The key modules in this folder are:

- `train.py`: training loop (Double DQN + optional IQN, n-step replay, target network, parallel collectors, logging/checkpoints).
- `env.py`: gym-like environment wrapping the LS process as an MDP.
- `state.py`: `NeuroLSState` + feature extraction for model inputs.
- `solution.py`: solution representation = assignment + per-machine sequences.
- `operators.py`: deterministic local-search neighborhoods (relocate/swap/insert/block).
- `perturbations.py`: deterministic perturbations (shake/restart), used to escape local optima.
- `candidate_generator.py`: enumerates neighborhoods deterministically; selects best move (full or Top-K proxy for large instances).
- `move_evaluator.py`: evaluates solutions and moves; uses incremental DP for fast recomputation.
- `gnn_encoder.py`: bipartite GNN encoder (+ a tripartite ablation).
- `decoder.py`: policy model and heads (standard Q head, dueling option, IQN distributional option).
- `price_embedding.py`: TOU-aware embeddings and per-machine exposure features.
- `parallel_collector.py`: CPU-side parallel episode collection; GPU trains on replay.
- `smoke_test.py`: quick end-to-end check (reset → features → forward → backward).

---

## 5) State representation: how we encode “where the search is”

NeuroLS works when the policy can “see”:

- What the **current solution structure** is (assignment + sequences).
- How good it is (current/best cost).
- Where we are in the search trajectory (steps since improvement, last operator, etc.).
- Domain-specific signals (TOU price structure, machine rates, load imbalance).

In our implementation, the observation is a dict of tensors returned by `NeuroLSEnv.get_state_features()`.

### 5.1 Scalar / global features (`state_features`)

Computed in `NeuroLSState.get_scalar_features()` and include (conceptually):

- normalized current cost and best cost,
- gap to best,
- last acceptance and last operator,
- progress and stagnation indicators,
- slack to horizon $K$,
- load imbalance and per-machine energy dispersion statistics.

These are designed to let the controller recognize patterns like:

- “we are stuck” (long no-improve streak),
- “the schedule is tight” (low slack),
- “one machine is overloaded/expensive” (imbalance/exposure).

### 5.2 Per-job features (`job_features`)

Computed in `NeuroLSState.get_job_features()` and include:

- normalized processing time,
- job position and relative position within its machine,
- normalized machine assignment,
- per-job view of machine load.

Optional extensions can append job price/exposure features (see `EnvConfig.job_price_features`).

### 5.3 Per-machine features (`machine_features`)

Machines include signals such as energy rate and load/exposure statistics (see environment/state builder).

### 5.4 Graph structure (edges)

We encode solution structure through edges:

- **Static edges**: job↔machine compatibility (in identical machines this is a full bipartite graph).
- **Dynamic edges**: current assignment job↔assigned-machine.

This mirrors the NeuroLS paper’s split between “static problem structure” and “dynamic solution edges”.

---

## 6) Encoders: GNN + TOU price representations

### 6.1 Bipartite GNN encoder (mainline)

`BipartiteGNNEncoder` in `gnn_encoder.py` implements the paper-style three-stage encoder:

- **Stage 1 (static):** message passing over the static bipartite structure.
- **Stage 2 (dynamic):** message passing over current assignment edges.
- **Stage 3 (consolidation):** one more pass to consolidate embeddings.

`NeuroLSEncoder` then applies **group pooling by machine** (max + mean over jobs assigned to each machine), producing:

- $\omega_{node}$: pooled job/node embedding,
- $\omega_{group}$: pooled machine/group embedding,
- $\omega_{feat}$: projected scalar feature embedding (plus price embedding if enabled).

These are concatenated and fed to the Q-head.

### 6.2 Price embeddings (TOU awareness)

`price_embedding.py` provides TOU-aware features:

- **Per-hour tokenization** of the daily TOU pattern (one-hot level + normalized price + positional encoding).
- **CNN-based embedding** to produce a compact global price profile vector (referred to as `z_price`).
- **Per-machine exposure stats** (how much workload overlaps off-peak/shoulder/peak and the average paid price).

We support ablations:

- `price_mode = none`: no price embedding.
- `price_mode = z_price`: global embedding only.
- `price_mode = full`: use richer price/exposure signals.

### 6.3 Tripartite graph encoder (ablation)

In addition to the bipartite graph, `gnn_encoder.py` contains a **tripartite** variant:

- Nodes: jobs + machines + “period” nodes.
- Edges: job↔machine and job/period/machine relations.

This is used as an ablation to test whether explicitly modeling time/period structure as nodes helps beyond `z_price`.

---

## 7) Operators and perturbations (what actions actually do)

### 7.1 Local-search operators (neighborhoods)

Defined in `operators.py` as deterministic neighborhoods over assignment+sequence:

- `RELOCATE_1`: move one job between machines and/or positions.
- `SWAP_1`: swap two jobs.
- `INTRA_INSERT`: reorder within one machine.
- `BLOCK_RELOCATE`: move a block of 2–3 jobs.

The **candidate generator** (`candidate_generator.py`) enumerates moves in a fixed order and chooses the **best-improvement** move for that operator (full enumeration for small instances, deterministic Top-K proxy for larger ones).

### 7.2 Perturbations (escape mechanisms)

Defined in `perturbations.py` and applied directly (not “best-of-neighborhood”):

- `SHAKE_SMALL`: relocate a few “worst-exposed” jobs (based on current schedule cost contributions when available).
- `SHAKE_PEAK`: relocate jobs with high peak-slot exposure.
- `RESTART`: rebuild a solution with a deterministic construction heuristic.

This aligns with the paper’s “learned perturbation decision” component.

---

## 8) Evaluation: why we can score moves accurately and fast

### 8.1 DP timing is the key domain adaptation

Given a per-machine sequence, the best timing under TOU prices and horizon $K$ is computed by DP.

This is a core difference from many generic LS benchmarks: our move evaluation is *not* just a local delta in objective; it depends on where jobs can be placed on the price timeline.

### 8.2 Incremental DP for speed

`move_evaluator.py` uses an `IncrementalDPSolver` with checkpoints so that when a move modifies only one or two machines, we recompute only what changed.

This makes it feasible to:

- evaluate many candidate moves per step,
- run LS steps inside an RL loop,
- and do parallel episode collection on CPU.

---

## 9) Training: what we actually run

Training is implemented in `train.py` and follows the NeuroLS paper’s Section 4.4-inspired recipe:

- **Double DQN** targets (reduce overestimation),
- **n-step returns** (better credit assignment),
- optional **IQN (Implicit Quantile Networks)** for distributional Q-learning,
- epsilon-greedy exploration with decay,
- target network updates (hard or soft),
- replay buffer.

### Parallel collection

The environment step is CPU-heavy (move evaluation uses DP), so we optionally use `parallel_collector.py`:

- Workers: run full episodes on CPU with a lightweight model copy.
- Main process: trains on GPU using replay.

---

## 10) Learnability tests (status: ongoing, early signals promising)

We are still running **learnability tests** to validate that:

- the MDP has actionable signal, and
- the observation features are informative enough to predict “good” next actions.

The main script is:

- `PaST/scripts/task_learnability_test.py`

It performs (per variant/config):

1) short random rollouts to collect states,
2) a **one-step oracle** that evaluates all actions under deterministic acceptance,
3) margin statistics (how much better the best action is),
4) a small supervised **probe** trained to predict the oracle action from a pooled feature vector.

So far, results are **somewhat promising** (i.e., the oracle margins and probe predictability suggest the task is *hopefully learnable* from the current inputs), but this is still exploratory and we are iterating on feature sets and variants.

---

## 11) How to run (minimal commands)

### Smoke test

Run a quick sanity check of env + model forward/backward:

```bash
python -m PaST.neurols.smoke_test
```

### Learnability test

```bash
python -m PaST.scripts.task_learnability_test --variants AAN_zprice AAN_full
```

### Training

Configs live in `PaST/configs/` (examples: `neurols_AANP_full.yaml`, `neurols_AAN_zprice.yaml`, tripartite configs).

```bash
python -m PaST.neurols.train --config PaST/configs/neurols_AANP_full.yaml
```

You can override common knobs via CLI flags (see `train.py`).

---

## 12) “Paper vs our adaptation” (quick mapping)

- **Learned LS controller (paper):** controls acceptance / neighborhood / perturbation.
  - **Ours:** same control points, with AA/AAN/AANP-style discrete actions.

- **Graph encoder (paper):** static + dynamic edges, pooled embedding.
  - **Ours:** bipartite jobs↔machines GNN with the same staged message passing; optional tripartite ablation.

- **Objective evaluation (paper):** problem-dependent evaluation of candidate solutions.
  - **Ours:** *domain-specific* DP computes optimal timing cost given assignment+sequence under TOU and horizon $K$.

- **Meta-heuristic flavor (paper):** acceptance of worse moves + perturbations to escape local minima.
  - **Ours:** optional simulated-annealing-style acceptance in `env.py` (disabled for deterministic evaluation), plus deterministic shake/restart perturbations.

---

## 13) Current positioning for the supervisors

If you need a one-slide summary:

- We are adopting NeuroLS’s **“learn to control local search”** view.
- Our controller learns decisions over **assignment + sequencing** on parallel machines.
- Scheduling/timing is handled by a **DP subroutine**, so the learned part focuses on the combinatorial structure.
- We are actively validating **learnability**; early signals look promising but results are not final.
