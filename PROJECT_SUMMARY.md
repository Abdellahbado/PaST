# PaST (Period-aware Scheduler Transformer) Project Summary

## 1. Project Overview
**PaST** is a Deep Reinforcement Learning framework built for **Single-Machine Parallel Machine Scheduling**, focusing on **Energy Cost Minimization** under Time-of-Use (TOU) pricing with **Deadline Constraints**.

The project treats scheduling as a sequential decision problem where an agent constructs a schedule step-by-step. The key innovation is "Period-Awareness"—explicitly modeling the timeline as a sequence of discrete price/duration tokens, allowing the Transformer to "see" optimal windows.

**Repository**: `PaST` (Period-aware Scheduler Transformer)
**Core Object**: `SingleMachinePeriodEnv` (Gym environment)

---

## 2. Problem Formulation

### Objective Function
Minimize total Energy Cost subject to completion by a deadline $T_{\text{limit}}$.
$$ J = \sum_{j} \text{Energy}(j) $$
Where $\text{Energy}(j) = \text{Rate}_m \times \int_{S_j}^{C_j} \text{Price}(t) dt$.
*   $S_j, C_j$: Start and completion times of job $j$.
*   $\text{Rate}_m$: Energy consumption rate of the machine.
*   $\text{Price}(t)$: Time-varying electricity price.

### Constraints
*   **Non-preemptive**: Once started, a job runs to completion.
*   **Machine Availability**: One job at a time.
*   **Deadline**: All jobs must finish by $T_{\text{limit}}$. If the agent takes an action that violates this, it is masked out (invalid).

### Data format
*   **Time Horizon ($T$)**: Discretized into variable-length periods (e.g., 1 hour at high price, 30 mins at low price).
*   **Jobs**: Defined by Processing Time ($p_j$).
*   **Machines**: Defined by Energy Rate ($e_m$).

---

## 3. Data Architectures & Inputs

### Global Inputs (Environment State)
The agent observes the state $S_t$ composed of three primary tensors:

#### 1. Period Features (The "Timeline")
Tensor shape: `[K_period, 4]`
Where `K_period` is the lookahead window size (e.g., 48 or 250).
*   **Feature 0: Duration** (`Tk`): Length of the period in time slots.
*   **Feature 1: Price** (`ck`): Electricity price per unit time.
*   **Feature 2: Start Offset**: Absolute start time of the period.
*   **Feature 3: Is Current**: Binary flag (1.0 if this is the period containing current time $t$, else 0.0).

#### 2. Job Features (The "Backlog")
Tensor shape: `[M_job_bins, 2]`
Jobs are aggregated into bins to handle variable numbers of jobs.
*   **Feature 0: Processing Time**: Duration of the job.
*   **Feature 1: Count**: (Legacy/Unused) Number of such jobs remaining. Currently, each row is typically one job.
*   **Mask**: A separate boolean mask indicates which job bins are empty/filled.

#### 3. Context Features (Global State)
Tensor shape: `[F_ctx]` (6, 13, or 18 dims depending on variant).

**Base Context (6 dims)**:
1.  **Current Time** ($t$)
2.  **Deadline** ($T_{\text{limit}}$)
3.  **Remaining Work** ($\sum p_j$ of unscheduled jobs)
4.  **Energy Rate** ($e_m$)
5.  **Avg Price Beyond Window**: Mean price of periods not visible in the local lookahead.
6.  **Min Price Beyond Window**: Min price of periods not visible.

**Ctx13 (+7 dims)** - *Improved Price Awareness*:
7-9. **Price Quantiles**: $Q_{25}, Q_{50}, Q_{75}$ of valid slot prices in the episode.
10-13. **Family Deltas**: Distance (time steps) to the next feasible start slot for each price family (0=Cheap, ..., 3=Exp).

**Ctx18 (+5 dims)** - *Capacity Awareness*:
14-17. **Family Capacity**: Fraction of remaining horizon belonging to each family.
18. **Cheap Capacity Deficit**: Ratio of (Remaining Work - Cheap Slots) / Remaining Work.

---

## 4. Model Architectures

The project uses a unified **Encoder-Decoder** design pattern with interchangeable backbones.

### A. PaSTEncoder (Transformer Backbone)
*   **Philosophy**: Full attention between jobs and the pricing timeline.
*   **Structure**: `[Job Embed]`, `[Period Embed]`, `[Ctx Embed]`.
*   **Blocks**: $L$ layers of:
    1.  **Cross-Attention**: Jobs (Query) $\leftarrow$ Periods (Key/Value). *("Where does this job fit best?")*
    2.  **Self-Attention**: Jobs $\leftarrow$ Jobs. *("Which job should I pick over others?")*
    3.  **FFN**: Feed-forward network.
*   **Config**: Pre-LN (default) or Post-LN.

### B. CNNDeepSetsEncoder (Lightweight Backbone)
*   **Philosophy**: Faster inference for CPU/Simulators.
*   **Period Branch**: 1D CNN (`Conv1d`, kernel=3/5) slides over period tokens to detect "valleys" (cheap intervals) and "cliffs" (price jumps). Pooled to a single vector.
*   **Job Branch**: DeepSets (Per-job MLP $\phi$ + Sum Pooling $\rho$).
*   **Fusion**: Concatenates Context + Pooled Periods + Job Embeddings.

### C. CandidateWindowSparseEncoder (CWE)
*   **Philosophy**: Structure-aware efficiency.
*   **Anchors**: Instead of attending to *all* period tokens, the model identifies "Candidate Windows" (multi-scale intervals with low energy cost).
*   **Sparse Attention**: Jobs attend only to these top-$K$ anchor windows.

---

## 5. Model Variants (Agents)

The system is designed as a modular ablation study with 3 main categories of agents.

### Category 1: Joint Assignment & Timing (PPO)
*   **Action**: Select `(Job, Slack)`.
*   **Meaning**: "Wait `Slack` units of time, then start `Job`".
*   **Reward**: $-(\text{Energy Consumed})$.

| Variant | Slack Space | Details |
| :--- | :--- | :--- |
| **PPO Short** | Discrete Array | e.g., `[0, 1, 2, 3, 5, 8, 13...]`. Good for fine-grained local adjustments. |
| **PPO C2F** | Coarse-to-Fine | Hierarchical: Pick `[0-5]`, then `3` inside. Reduces action space size. |
| **PPO Full** | Full Horizon | Can target *any* future period. Requires global horizon embedding. |

### Category 2: Price Families (PPO) - *Most Promising*
*   **Action**: Select `(Job, FamilyID)`.
*   **Meaning**: "Schedule `Job` in the next available slot belonging to `FamilyID`".
*   **Families**: Defined by price quartiles (0=Cheap, 3=Expensive).
*   **Decoders**:
    *   **Earliest**: First slot where $Price(t) \in Family$.
    *   **BestStart** (`_beststart`): Scans the family to find the absolute lowest window energy cost $\int_{t}^{t+p} c(\tau)d\tau$.
*   **Duration-Aware** (`_duration_aware_family`):
    *   Families are calculated based on **Window Cost** ($W(t, p)/p$) rather than instantaneous price $c(t)$. This prevents scheduling a long job in a short 5-min cheap slot that spills into a 2-hour expensive peak.

### Category 3: Sequence-Only (PPO / Q-Learning)
*   **Action**: Select `Job` only.
*   **Timing**: Solved optimally by **Batch DP**.
*   **State**: Only needs to track set of remaining jobs and current time $t=0$ (since DP handles the timeline).

#### Algorithm: Batch Sequence DP (`batch_dp_solver.py`)
Computes $V(S) = \min \text{Cost}$ for a fixed job sequence.
*   **State**: `min_prev[t]` = min cost to complete first $k$ jobs finishing at time $t$.
*   **Transition**: `min_prev[t] = min(min_prev[t-p] + Energy(t-p, t))`.
*   **Complexity**: $O(N \cdot T_{\text{max}})$. Vectorized on GPU.

#### Training: Q-Sequence (`q_sequence`)
Uses **DAgger (Dataset Aggregation)** to train a Q-function $Q(S, j) \approx \text{DP\_Cost}(S + j + \text{Rest})$.
1.  **Rollout**: Execute policy to get trajectory.
2.  **Counterfactuals**: At each step $S_t$, speculatively try *every* available job $j$.
3.  **Completion**: Finish the sequence using a baseline (SPT/LPT) or the model itself.
4.  **Labeling**: Compute exact DP cost for each full sequence.
5.  **Loss**: Huber Loss between predicted $Q(S, j)$ and calculated Cost.
6.  **Architecture**: Dueling Q-Head.
    *   $Q(s, j) = V(s) + A(s, j) - \text{mean}(A(s, \cdot))$.

---

## 6. Training & Evaluation Details

### DAgger Loop (Q-Sequence)
*   **Warmup**: First $N$ rounds use SPT (Shortest Processing Time) to complete sequences. Prevents model from learning garbage early on.
*   **Mixture**: Later rounds mix Model and Heuristic completions.
*   **Exploration**: Epsilon-greedy during data collection.

### PPO Loop
*   **Algorithm**: Standard PPO-Clip.
*   **Value Function**: Shared encoder with Policy, separate head.
*   **Advantage**: GAE (Generalized Advantage Estimation).
*   **Entropy**: Small coefficient (0.01) to encourage exploration.

### Evaluation Metrics
1.  **Optimality Gap**: $( \text{AgentCost} - \text{OptimalCost} ) / \text{OptimalCost} \%$.
2.  **VS_SPT**: Improvement over Shortest-Processing-Time heuristic.
3.  **SGBS (Stochastic Greedy Beam Search)**:
    *   Instead of taking $\arg\max Q$, sample from Softmax($-Q/T$).
    *   Maintain beam of width $\beta$ (e.g., 4).
    *   Used for final inference to boost performance.
4.  **EAS (Efficient Active Search)**:
    *   Test-time fine-tuning.
    *   Given a specific test instance, run gradient descent on the Q-network to minimize the cost of valid sequences found during sampling.

---

## 7. Code Structure Map

*   **`config.py`**: The "Bible" of the project. Defines all variants, hyperparameters, and data structures.
*   **`sm_env.py`**: `SingleMachinePeriodEnv`. Handles state transitions, masking, and reward calculation.
*   **`past_sm_model.py`**: PyTorch code for PaST Encoder, Pre/Post-LN blocks, Attention.
*   **`models/q_sequence_model.py`**: Dueling Q-Hads, `QSequenceNet`, `CNNDeepSets` variants.
*   **`train_q_sequence.py`**: DAgger training loop, multiprocessing for DP, dataset management.
*   **`solvers/batch_dp_solver.py`**: The parallelized exact timing solver.
*   **`cli/`**: Command line interface entries for training and evaluation.
