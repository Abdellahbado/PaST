# Plan: Supervised Move Ranking

Date: 2026-04-20

## 1. Why this is a new research thread

The predecessor thread:

- `research/dp_heuristic_oracle_20260419/`

was an empirical search for a strong handcrafted DP-centered heuristic.

That thread has now done its job:

- it produced a credible handcrafted base solver
- it identified the exact bottleneck
- it produced the diagnostic evidence needed to define a learning task

The next objective is different enough to justify a new thread:

- design a learning-based method whose novelty is the move-ranking component, not purely the handcrafted heuristic performance

## 2. What should be preserved from the old thread

Do not duplicate or rewrite the old thread's reports.

Preserve them as precursor evidence and baseline source.

The most important carried-forward artifacts are:

- Phase H:
  - `research/dp_heuristic_oracle_20260419/phaseH_vnd_exact_oracle_results.md`
- Phase I:
  - `research/dp_heuristic_oracle_20260419/phaseI_noscreen_diagnostic_results.md`
- Phase J:
  - `research/dp_heuristic_oracle_20260419/phaseJ_insert_screening_redesign_results.md`
- Phase K:
  - `research/dp_heuristic_oracle_20260419/phaseK_insert_efficiency_pass_results.md`

These establish:

1. improving `insert_inter` moves exist
2. handcrafted screening can recover them
3. handcrafted screening now saturates
4. the best base method for learning handoff is the insert-focused diverse family, especially the trimmed version as a cleaner design

## 3. Current best handcrafted base

Current recommended base for learning handoff:

- variant family:
  - `vnd_exact_dp_insert_rank_diverse_trimmed`
- implementation home:
  - `solvers/cpp/parallel_heuristic_compare.cpp`

Current fixed-point benchmark anchor:

- instance `61`
- epsilon `347`

Current reference numbers at that point:

- `greedy_dp = 7088`
- `greedy_dp_local_search_relocate_only = 7081`
- Phase H `vnd_exact_dp = 6944`
- Phase I no-screen best = `6920`
- Phase J / K best handcrafted = `6884`
- paper EHS = `6710`
- reference exact = `6643`

## 4. Recommended folder structure

Keep the old handcrafted thread as-is.

Do not move its phase reports.

Use this new thread for:

- learning-specific planning
- data extraction design
- feature design
- model design
- evaluation design
- learning-branch experiments

Recommended files in this thread:

- `OVERVIEW.md`
- `LITERATURE.md`
- `LOG.md`
- `ACTIVE.md`
- thread-level plans:
  - `PLAN_supervised_move_ranking.md`
  - later, if needed:
    - `PLAN_feature_schema.md`
    - `PLAN_offline_dataset.md`
    - `PLAN_online_integration.md`

Iteration folders should then separate:

1. data logging / dataset extraction
2. offline ranking-model experiments
3. online integration into the solver
4. broader evaluation

## 5. First method choice

Start with:

- supervised **ranking** of `insert_inter` moves

Do not start with:

- RL
- transformers
- GNNs
- end-to-end schedule prediction
- value-function ADP

## 6. First model family

Primary first model:

- gradient-boosted trees
  - XGBoost / LightGBM / CatBoost class of models

Optional small secondary comparison:

- tiny tabular MLP

Reason:

- current inputs are tabular
- data size will initially be modest
- interpretability matters
- fast inference matters
- low engineering overhead matters

## 7. Learning task definition

### 7.1 Target decision

The model should help answer:

- which candidate `insert_inter` moves deserve exact DP evaluation?

### 7.2 Preferred formulation

First formulation:

- ranking / scoring

Avoid pure hard classification as the first integrated method because:

- improving moves are rare
- false negatives are costly
- the actual solver decision is top-k selection under a DP budget

### 7.3 Online pipeline

Desired eventual pipeline:

1. generate candidate `insert_inter` moves
2. compute cheap analytical features
3. learned model assigns a move score
4. keep top-k or top-k-per-source candidates
5. exact DP evaluates only that shortlisted set
6. accept only exact-DP-improving moves

Safety design:

- keep an analytical fallback shortlist
- optionally keep a small exploration slice
- exact DP remains the acceptance oracle

## 8. Feature design plan

Start with simple, interpretable, cheap features.

### 8.1 Job features

- processing time
- job type id or type-level signature
- relative size versus source load
- relative size versus target slack

### 8.2 Source machine features

- current exact cost
- current relaxed LB
- exact-minus-LB gap
- current load
- utilization `load / epsilon`
- machine rate class / rate
- number of jobs

### 8.3 Target machine features

- current exact cost
- current relaxed LB
- exact-minus-LB gap
- current load
- utilization `load / epsilon`
- residual slack before insertion
- machine rate class / rate
- number of jobs

### 8.4 Move-combination features

- source-to-target rate difference
- projected target load ratio after insertion
- projected source load ratio after removal
- cheap lower-bound delta terms
- source cost density vs target cost density
- whether move targets one of the top expensive machines

### 8.5 Search-state features

- accepted improving moves so far in the current search
- current TEC minus starting TEC
- local-search pass / iteration index
- remaining exact-evaluation budget or current cap tier
- whether the source / target machine has already participated in accepted moves

Do not start with huge feature engineering.

## 9. Dataset plan

### 9.1 What data to log

For each generated `insert_inter` candidate:

- instance id
- epsilon
- current incumbent TEC
- move id
- job id / type / size
- source machine id / rate / cost / LB / load
- target machine id / rate / cost / LB / load
- all engineered features
- whether exact DP evaluation was performed
- exact before/after touched-machine cost
- exact total delta
- improving or not
- accepted or not

### 9.2 Two useful label levels

Store both:

1. binary improving label
2. exact improvement magnitude

This allows:

- ranking
- classification
- ablation on label choice

### 9.3 Two data streams

The dataset should explicitly separate:

1. **dense exact-labeled runs**
   - bounded no-screen or weak-screen exact evaluation
   - smaller volume
   - high-value labels for learning

2. **broad candidate logs**
   - all generated candidates with cheap features
   - much larger volume
   - useful for context, calibration, and later joining with exact-labeled subsets

The first learning experiments should be trained primarily on exact-labeled examples.

### 9.4 Data collection policy

Use the handcrafted base to generate data first.

Collect first from:

- the current anchor row `61/347`
- multiple multistart seeds on the same point, not just one run

Recommended first collection policy:

- 10 to 20 deterministic multistart seeds / start states
- for each seed:
  - log broad candidate features
  - exact-label a bounded subset of `insert_inter` moves

Only after this pipeline is clean should data collection widen to a few nearby or representative rows.

Initial goal is not big data.
Initial goal is a correct, analyzable dataset with enough exact-labeled positive moves.

## 10. Evaluation plan

### 10.1 Offline evaluation first

Before online integration, test:

- whether the model ranks improving moves above non-improving ones
- top-k hit rate
- recall of improving moves under a fixed exact-eval budget
- precision among top-ranked moves

Key metric:

- can learned top-k recover the improving moves better than handcrafted screening at the same exact-DP budget?

### 10.2 Online evaluation second

Integrate the model into the solver only after offline evidence is acceptable.

Online comparison should be against:

- handcrafted best base (`vnd_exact_dp_insert_rank_diverse_trimmed`)
- Phase J best
- no-screen diagnostic

Main online metrics:

- TEC
- exact DP calls
- total runtime
- memory
- accepted insert moves

## 11. Concrete staged plan

### Stage L1: data logging branch

Goal:

- produce a high-quality move-level dataset from the current handcrafted base

Tasks:

- add structured move logging for `insert_inter`
- export one dataset format under `temp/`
- create a data dictionary markdown
- run on `61/347`
- collect across multiple multistart seeds, not just one incumbent
- separate:
  - broad candidate logs
  - exact-labeled move logs
- optionally add 1-2 nearby or contrast rows only after the first anchor dataset is clean

Success condition:

- clean labeled dataset with enough positive improving examples to support offline ranking experiments
- clear accounting of:
  - total generated candidates
  - total exact-labeled candidates
  - positive-rate among exact-labeled moves

### Stage L2: offline supervised ranking

Goal:

- train and evaluate a small ranking model offline

Tasks:

- define train/validation split
- start with tabular boosted trees
- compare against handcrafted ranking heuristics offline
- measure top-k improving-move recovery
- compare at fixed exact-eval budgets such as:
  - `k = 10`
  - `k = 25`
  - `k = 50`
  - `k = 100`

Success condition:

- learned ranking beats handcrafted ranking offline under the same evaluation budget
- concrete target:
  - at fixed budget `k`, the learned ranker finds materially more improving moves than the handcrafted ranker
  - or reaches a materially better best exact delta within the same budget

### Stage L3: online learned-shortlist integration

Goal:

- replace or augment handcrafted shortlist ordering with the learned ranker

Tasks:

- keep exact DP acceptance unchanged
- use learned score to order candidates
- retain analytical fallback / exploration
- compare online on `61/347`

Success condition:

- similar or better TEC with fewer DP calls or lower time
  OR
- better TEC under the same bounded DP budget

### Stage L4: small generalization study

Goal:

- test whether the learned screening generalizes beyond the anchor point

Tasks:

- select 3-5 additional rows, not the full benchmark
- compare to handcrafted base

Success condition:

- method shows stable benefit beyond one single point

## 12. When to stop

Stop the learning branch early if:

- offline ranking cannot beat handcrafted ordering even on the anchor row
- labels are too sparse / unstable to support a reliable model
- online integration degrades quality materially without reducing DP cost

If that happens, fall back to the handcrafted paper/thesis story.

## 13. Why this is a good thesis direction

This direction does not require state-of-the-art heuristic performance to be publishable.

Its novelty is methodological:

- exact DP oracle
- analytical lower-bound features
- learned move ranking
- exact verification

That is a much cleaner contribution than trying to learn the whole schedule end-to-end.
