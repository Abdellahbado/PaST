# Component Ablation Study

## Goal

Use the smallest ablation that still answers the paper-facing questions clearly:

1. How much does exact relaxed-block packing help over our default Step 1 packer?
2. How much does smart reconstruction help after Step 1?
3. How much faster is the full pipeline than going straight to exact DP?
4. For every solved instance, which component actually closed it?

## Recommended Configurations

Run exactly these four configurations:

| Config | Meaning | Why it matters |
|---|---|---|
| `full_default` | Full pipeline with current default Step 1 packer | Reference version of our method |
| `full_exact_pack` | Same pipeline, but Step 1 uses exact relaxed-block packing via `constraint` | Direct comparison to the paper's exact block-fitting idea |
| `no_smart_recon` | Same as `full_exact_pack`, but without smart reconstruction | Isolates the contribution of our count-aware bridge |
| `exact_only` | Banded SPACES + exact DP only | Measures total value of the non-exact-DP layers |

This is intentionally lean:

- It keeps the focus on the parts that affect the hard instances.
- It avoids spending benchmark time on bounds that already showed little value on the current benchmark.
- It makes the "paper comparison" explicit without mixing it with unrelated engineering ablations.

By default, `exact_only` should **not** be run on the full benchmark. Run it only on the **survivor set**:

- instances that are **not** closed at `fwd_relax` in at least one of the main non-exact configurations

This keeps the baseline meaningful while avoiding a large amount of unnecessary exact-DP time on trivially solved instances.

## Visibility Requirements

Each solved instance should record:

- `step_reached`: the first major stage that closes the gap
- `winner_detail`: finer attribution, especially inside Step 1
- `fwd_pack_method`: which Step 1 packer found the relaxed-block realization
- `fwd_pack_outcome`: whether Step 1 packing was feasible, infeasible, or skipped
- `fwd_pack_solver`: `default`, `constraint`, `ortools`, or `z3`
- `t_*`: runtime of every major stage
- `t_fwd_pack_*`: runtime split inside Step 1

This makes it possible to answer both:

- "Which part solved this instance?"
- "What runtime should we expect before that part usually closes the gap?"

## Expected Tables / Figures

The study should report:

1. Solver-path census
   - Count of instances solved by `fwd_relax`, `heuristic_ub`, `local_search`, `smart_recon`, `exact`
   - Same table per configuration

2. Step 1 submethod census
   - Count of `external_constraint`, `ffd`, `bfd`, `ffi`, `bfi`, `random_*`, `dfs_exact`, `block_dp_exact`
   - Only for instances closed at Step 1

3. Runtime summary
   - Average / median / max runtime per configuration
   - Average runtime conditional on winning stage

4. Exact-pack delta
   - Instances newly closed at Step 1 by `full_exact_pack` vs `full_default`
   - Instances for which exact DP is avoided by `full_exact_pack`

5. Smart reconstruction delta
   - Instances solved by `smart_recon`
   - Instances for which exact DP is avoided by enabling `smart_recon`

6. End-to-end speedup
   - `full_exact_pack` vs `exact_only`

## Run Command

```bash
bash /Users/mac/Documents/Study/PFE/PaST/hpc/run_component_ablation.sh
```

The script uses the survivor-only exact baseline by default. To force the old expensive version:

```bash
bash /Users/mac/Documents/Study/PFE/PaST/hpc/run_component_ablation.sh \
  --exact-only-policy all
```

For a quick smoke test:

```bash
bash /Users/mac/Documents/Study/PFE/PaST/hpc/run_component_ablation.sh \
  --section 1 \
  --config full_exact_pack \
  --max-instances 4
```

## Interpretation

- If `full_exact_pack` improves over `full_default`, that is direct evidence that exact relaxed-block realization is worth studying.
- If `smart_recon` avoids exact DP on nontrivial instances, that highlights a contribution that the paper does not have in the same form.
- If `full_exact_pack` still leaves some instances to `exact`, that is expected: exact packing only certifies the specific relaxed block pattern, not all possible block patterns.
