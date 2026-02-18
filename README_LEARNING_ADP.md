# Learning-Accelerated Dynamic Programming for Single-Machine Scheduling

This document describes the learning-based approach to accelerate dynamic programming for single-machine scheduling under Time-of-Use (TOU) pricing constraints. The system learns value function approximations from solved instances to guide beam search within the DP solver, reducing computational overhead while maintaining near-optimal solution quality.

## Problem Statement

The system addresses single-machine scheduling with the following characteristics:

- Jobs with varying processing times must be scheduled on a single machine
- Time-of-Use pricing creates variable energy costs across the scheduling horizon
- Each instance has a deadline constraint that must be satisfied
- The objective is to minimize total energy cost: sum of price over all time slots where the machine operates
- Idle time is permitted and incurs no cost
- Jobs are identical if they share the same processing time (multiset representation)

## Core Approach

The implementation combines exact dynamic programming with learned heuristics to accelerate solution finding. The key insight is that while exact DP provides optimal solutions, it can be computationally expensive for large instances. By learning from previously solved instances, the system develops heuristics that guide search toward high-quality solutions more efficiently.

### Dynamic Programming Foundation

The base solver uses sparse dynamic programming on a state-space DAG:

- State representation: (t, u) where t is current time and u is a vector of used job counts per processing time length
- State encoding: Mixed-radix integer encoding for O(1) state transitions and compact storage
- Transitions: At time t, either idle (advance to t+1) or start a job of length L (advance to t+L and pay interval cost)
- Optimality: Shortest path computation on an acyclic graph where time only moves forward

**Mixed-Radix State Encoding**: Since jobs with the same processing time are identical, the state only needs to track counts of used jobs per length class. For K distinct processing times with totals [n₁, n₂, ..., nₖ], a state u = [u₁, u₂, ..., uₖ] where 0 ≤ uᵢ ≤ nᵢ is encoded as a single integer:

```
state = u₁ × mult₁ + u₂ × mult₂ + ... + uₖ × multₖ
```

where `mult[i] = ∏(j<i) (totals[j] + 1)` are the radix multipliers. This encoding enables:
- O(1) state transitions by adding/subtracting mult[i]
- Compact integer keys for dictionary-based sparse DP
- Efficient state caching and lookup

The algorithm sweeps through time maintaining:
- `best[u]` = minimum cost to reach state u at current time t
- `finish_at[t_end][u2]` = minimum cost to finish exactly at time t_end in state u2

Complexity: O(T × S × K) where T is horizon length, S is number of reachable states, and K is number of distinct processing times.

### Learning-Based Acceleration

The system learns a value function approximation V̂(t, u) that estimates the cost-to-go from state (t, u). This learned function guides beam search within the DP:

**Beam-Pruned DP with Learned Heuristic**:
1. At each time layer t, if the number of states exceeds beam_width × prune_factor, prune to keep only beam_width states
2. States are ranked by g(u) + V̂(t, u) where g(u) is the cost-so-far and V̂(t, u) is the learned cost-to-go estimate
3. The mixed-radix encoded state must be decoded to extract the count vector u for feature computation
4. Preserve idle transitions as first-class moves to maintain feasibility
5. Continue DP forward pass with pruned state set

This approach provides:
- Significant speedup by reducing the number of states explored
- Near-optimal solutions when V̂ provides good guidance
- Guaranteed feasibility (never prunes away all paths to completion)

## Features for Learning

The system uses structured features designed to capture scheduling-relevant information while remaining efficient to compute within the DP loop. Features are extracted from the decoded state (t, u) where u is the vector of used job counts.

### State Features

**Temporal Features**:
- Current time t and remaining horizon R = T - t
- Hour within daily cycle h = t mod H (for repeating TOU patterns)
- Time-of-day regime (off-peak, shoulder, peak) as one-hot encoding
- Distance to next off-peak period within the daily cycle
- Distance to next cheap period (off-peak or shoulder) within the cycle

**Workload Features**:
- N = number of remaining jobs (computed from u and totals)
- W = total remaining work (sum of remaining processing times)
- S = slack capacity (remaining horizon minus remaining work)
- Short jobs count (processing time ≤ 2)
- Long jobs count (processing time ≥ median or 3)

**Price-Aware Features**:
- Count of off-peak slots remaining in horizon
- Count of peak slots remaining in horizon
- Pressure ratio: W / (off-peak capacity + 1)
- Cheap pressure ratio: W / (off-peak + shoulder capacity + 1)

**Interaction Features**:
- Slack × off-peak regime indicator
- Slack × peak regime indicator

**Optional Extended Features**:
- Per-class job counts (when K is small and fixed)
- Per-class immediate scheduling cost
- Fixed-length histogram of remaining job processing times (for varying K)
- Metadata features: log(T), log(N), log(W), utilization ratio, slack ratio

### Feature Normalization

Features can be normalized by horizon length T to create scale-invariant ratios. This enables:
- Transfer learning across different instance sizes
- Consistent feature magnitudes regardless of problem scale
- Better generalization when training on small instances and evaluating on larger ones

## Training Methodology

### Data Collection

Training data consists of (state, cost-to-go) pairs collected from solved instances:

**Optimal Path Labeling** (recommended):
1. Solve the full instance once with exact DP
2. Extract states along the optimal schedule
3. Label each state with its exact cost-to-go (remaining cost along the optimal path)
4. Yields O(T) labeled states per instance with one DP solve
5. Can extract from multiple optimal schedules using different tie-breaking rules

**Subproblem Labeling** (alternative):
1. Sample random feasible states (t, u)
2. For each state, solve the remaining subproblem with exact DP
3. Label the state with the subproblem's optimal cost
4. More expensive but provides labels for diverse states

**Label Normalization**:
- Labels can be normalized by the remaining price budget sum(prices[t:])
- Normalized labels represent cost as a fraction of remaining budget
- Essential for cross-size transfer learning

### Model Types

Multiple model architectures are supported:

**Linear Ridge Regression**:
- Fast training and inference
- Closed-form solution
- L2 regularization for stability
- Works well with engineered features

**Polynomial Ridge Regression**:
- Degree-2 polynomial expansion of features
- Captures feature interactions
- Still has closed-form solution
- Higher capacity than linear

**Multi-Layer Perceptron (MLP)**:
- Small neural network (2-3 hidden layers)
- Trained with early stopping
- More expressive than polynomial
- Slower inference than linear models

**Gradient Boosted Trees (LightGBM)**:
- Ensemble of decision trees
- Handles non-linear relationships well
- Automatic feature interaction discovery
- Good performance with minimal tuning

### Pooled Training

The system supports pooled cross-instance training:

1. Generate multiple training instances with varying characteristics
2. Collect labeled states from all training instances
3. Pool all (state, label) pairs into a single dataset
4. Train one shared model on the pooled data
5. Evaluate the frozen model on held-out test instances

This approach enables:
- Learning from diverse problem structures
- Better generalization to unseen instances
- Amortized training cost across many evaluations

### Curriculum and Transfer Learning

**Curriculum Training**:
- Train on small instances first
- Gradually increase instance size
- Use learned model as warm-start for next stage
- Quadratic prior around previous weights for fine-tuning

**Cross-Size Transfer**:
- Train on one instance size, evaluate on another
- Requires transferable features (fixed dimension regardless of K)
- Requires feature and label normalization
- Enables scaling to larger instances without retraining

## Repeating Price Profiles

The system is designed for repeating Time-of-Use price patterns, which reflect realistic pricing structures.

### Price Structure

Instances use repeating daily TOU patterns:
- Horizon divided into days of H hours (typically H=20)
- Each day follows the same price pattern
- Daily pattern divided into periods with different price levels
- Common structure: off-peak (cheap), shoulder (moderate), peak (expensive)

Example daily pattern:
- Hours 0-3: Off-peak (price = 1)
- Hours 4-11: Shoulder (price = 2)
- Hours 12-15: Peak (price = 4)
- Hours 16-19: Shoulder (price = 2)

### Why Repeating Profiles Work Best

Repeating patterns enable effective learning because:

**Temporal Structure**: The learned value function can exploit the periodic structure. Features like "distance to next cheap period" and "hour within cycle" capture when it's beneficial to wait versus schedule immediately.

**Consistent Semantics**: Price regimes (off-peak, shoulder, peak) have consistent meaning across instances. The model learns that scheduling during off-peak periods is generally preferable, and this knowledge transfers.

**Feature Efficiency**: Precomputed daily pattern features (regime counts, window costs) can be reused across the horizon, making feature extraction efficient within the DP loop.

**Generalization**: Models trained on instances with one repeating pattern can generalize to instances with different repeating patterns, as long as the features capture the relative price structure rather than absolute values.

**Realistic Modeling**: Real-world TOU pricing follows daily cycles (e.g., cheaper electricity at night, expensive during peak demand hours). The repeating pattern assumption aligns with practical applications.

For non-repeating or random price sequences, the learned heuristics are less effective because:
- Temporal features lose predictive power
- No consistent pattern to exploit
- Each time slot must be treated independently
- The value function cannot leverage periodicity

## Implementation Details

### Sparse DP with Early Pruning

The exact DP solver includes several optimizations:

**Feasibility Pruning**: At each state, check if remaining work exceeds remaining horizon. Prune infeasible states immediately.

**Lower Bound Pruning**: Compute a lower bound on cost-to-go by selecting the cheapest remaining slots. Prune states whose cost + lower bound exceeds the best known solution.

**Memory Management**: Free old time layers after processing to reduce memory footprint. Only keep layers within max_job_length of current time.

**Timeout Handling**: Support time limits with greedy completion fallback. If DP times out, complete the partial schedule greedily and return a feasible (but possibly suboptimal) solution.

### Beam Search Integration

The guided beam DP integrates the learned heuristic:

**Scoring Function**: States scored by g(u) + V̂(t, u) where g is cost-so-far and V̂ is learned cost-to-go.

**State Decoding for Features**: The mixed-radix encoded state integer must be decoded back to the count vector u to extract features. A cache is maintained to avoid repeated decoding of the same state:

```python
used_cache: Dict[int, Tuple[int, ...]] = {}

def decode_state(state: int, radices: np.ndarray) -> Tuple[int, ...]:
    cached = used_cache.get(state)
    if cached is not None:
        return cached
    u = [0] * K
    x = state
    for i in range(K):
        u[i] = x % radices[i]
        x //= radices[i]
    used_cache[state] = tuple(u)
    return tuple(u)
```

**Lexicographic Tie-Breaking**: When costs are equal, use secondary penalty (e.g., sum of start times for "early" tie-breaking).

**Prune Threshold**: Only prune when layer size exceeds beam_width × prune_factor to avoid premature pruning on small layers.

**Idle Preservation**: Always include idle transitions to ensure feasibility. Never prune away the ability to wait.

### Parallel Data Collection

Training data collection is parallelized across CPU cores:

**Worker Pool**: Use multiprocessing.Pool to distribute instance generation and labeling across workers.

**Memory Management**: Support on-disk memmaps for pooled features to reduce peak RAM usage with large worker counts.

**Fault Tolerance**: Workers catch exceptions and return empty results rather than crashing the entire collection process.

**Streaming Fit**: For very large pooled datasets, fit models in chunks to avoid loading the full dataset into memory.

## Repository Structure

```
├── solvers/
│   ├── optimal_benchmark_dp.py           # Exact sparse DP solver with radix encoding
│   ├── optimal_benchmark_dp_numba.py     # Numba-accelerated DP core
│   ├── vhat_linear.py                    # Linear/polynomial value models
│   ├── vhat_models.py                    # MLP and LightGBM value models
│   └── vhat_tou_features.py              # TOU feature extraction
├── sandbox/
│   ├── train_eval_vhat_beam_dp.py        # Single-instance training and evaluation
│   ├── eval_pooled_vhat.py               # Pooled cross-instance training
│   └── eval_guided_vhat_holdout.py       # Holdout evaluation with baselines
└── ADP/
    └── logs/                              # Experimental results
```

## Usage

### Single-Instance Training and Evaluation

Train a value function on one instance and evaluate with beam search:

```bash
python sandbox/train_eval_vhat_beam_dp.py \
    --D 3 --N 40 --pmax 8 \
    --samples 2000 \
    --beam 2000 \
    --transferable-features \
    --save-model models/vhat_single.npz
```

### Pooled Cross-Instance Training

Train one model on multiple instances and evaluate on held-out instances:

```bash
python sandbox/eval_pooled_vhat.py \
    --D 6 --N 30 --pmax 3 \
    --train-seeds 0-19 \
    --samples-per-instance 2000 \
    --eval-seeds 100-129 \
    --beams 2,3,5 \
    --model-type poly \
    --transferable-features \
    --normalize --normalize-labels \
    --label-mode optimal_path \
    --save-model models/vhat_pooled_poly.npz \
    --out-csv logs/pooled_poly.csv
```

### Holdout Evaluation with Baselines

Evaluate learned heuristic against baselines:

```bash
python sandbox/eval_guided_vhat_holdout.py \
    --seed-start 30 --seed-end 59 \
    --D 3 --N 18 --pmax 5 \
    --samples 4500 \
    --beams 50,100,200,400 \
    --transferable-features --normalize \
    --load-model models/vhat_pooled.npz \
    --out-csv logs/holdout_eval.csv
```

### Cross-Size Transfer

Train on small instances, evaluate on larger instances:

```bash
# Train on small
python sandbox/eval_pooled_vhat.py \
    --D 3 --N 20 --pmax 3 \
    --train-seeds 0-49 \
    --samples-per-instance 1000 \
    --model-type linear \
    --transferable-features --normalize --normalize-labels \
    --save-model models/vhat_small.npz

# Evaluate on medium (no retraining)
python sandbox/eval_pooled_vhat.py \
    --D 6 --N 60 --pmax 3 \
    --eval-seeds 100-129 \
    --beams 5,10,20 \
    --transferable-features --normalize --normalize-labels \
    --load-model models/vhat_small.npz \
    --out-csv logs/transfer_small_to_medium.csv
```

## Performance Characteristics

### Speedup vs Optimality Trade-off

The beam width controls the speedup-optimality trade-off:
- Larger beam: closer to optimal, slower
- Smaller beam: faster, potentially larger optimality gap
- Typical beam widths: 2-20 for small instances, 50-500 for medium instances

### Learned Heuristic vs Baselines

Compared to baseline heuristics:
- Zero heuristic (V̂ = 0): no guidance, explores states uniformly
- Random heuristic: random noise, no useful signal
- Price-aware heuristic: remaining work × mean future price

The learned heuristic typically achieves:
- Lower optimality gaps than baselines at the same beam width
- Faster convergence to near-optimal solutions
- Better exploitation of TOU price structure

### Computational Complexity

Training:
- Optimal path labeling: O(instances × T_dp) where T_dp is time for one exact DP solve
- Subproblem labeling: O(instances × samples × T_dp)
- Model fitting: O(samples × features²) for linear/poly, O(samples × features × trees) for LightGBM

Inference:
- State decoding: O(K) per state (cached)
- Feature extraction: O(K) per state
- Model prediction: O(features) for linear, O(features × depth × trees) for LightGBM
- Beam DP: O(T × beam_width × K × (K + feature_cost))

## Limitations and Considerations

**Repeating Pattern Dependency**: The approach is most effective when training and test instances share similar repeating price structures. Performance degrades on random or non-repeating price sequences.

**State Space Growth**: For instances with many distinct processing times (large K), the state space can grow exponentially. The sparse DP and beam pruning mitigate this but may still struggle with K > 12.

**Label Quality**: Training requires exact DP solutions for labeling. For very large instances where exact DP is intractable, labels become approximate (greedy-completed), reducing model quality.

**Feature Engineering**: The effectiveness of learned heuristics depends on feature quality. Domain knowledge about TOU pricing and scheduling is encoded in the feature design.

**Generalization Limits**: Models trained on one problem distribution may not transfer well to significantly different distributions (e.g., different pmax ranges, different utilization levels).

## Future Directions

Potential extensions include:
- Graph neural networks for learning on the DP state graph structure
- Reinforcement learning to directly optimize beam search decisions
- Adaptive beam width based on learned confidence estimates
- Multi-machine scheduling with job assignment decisions
- Online learning from deployment experience
- Integration with mixed-integer programming solvers as warm-start heuristics
