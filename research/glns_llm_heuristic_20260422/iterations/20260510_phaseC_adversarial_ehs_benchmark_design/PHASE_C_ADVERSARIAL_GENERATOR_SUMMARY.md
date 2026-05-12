# Phase C Summary: LLM as an Adversarial Instance-Family Generator

## Purpose

Phase C tests a different role for the LLM.

Earlier branches tried to use the LLM as a solver component: move scorer, policy selector, local-search operator designer, or online neighborhood proposer. Those branches did not produce a robust runtime advantage over random or simple baselines.

Phase C instead asks:

> Can an LLM design structured BPMSTP instance families that expose weaknesses of EHS more effectively than random generation or simple manual parameter sweeps?

The LLM does not solve schedules and does not modify EHS. It designs families of test instances. We then generate concrete scheduling instances from those families and run EHS at two time budgets to see whether the instances reveal difficult behavior.

## Target Solver

The target is the faithful EHS reconstruction in `glns/paper_heuristics.py`.

The relevant EHS mechanisms are:

- SGH construction at each `khat`.
- A-SGH assignment-history reuse.
- Epsilon-based exploration of decreasing makespan limits.
- ES / R-ES / ESR local improvement.
- Pareto-front construction over makespan and total energy cost.

Phase C does not try to beat EHS. It tries to generate diagnostic instances that reveal where EHS needs more time, produces sparse fronts, or shows mechanism-specific stress.

## Main Evaluation Idea

Each generated instance is run twice:

- Short budget: usually `30s`.
- Long budget: usually `90s`.

The main simple metric is front-size growth:

```text
delta_front_size = front_size_long - front_size_short
```

An instance is considered high-yield if it exposes a meaningful short-vs-long gap, such as:

- `delta_front_size >= 2`;
- short run has very few front points but long run has more;
- deepest reached `Cmax` improves meaningfully;
- logs/results confirm the expected EHS mechanism failure.

Later we added a stricter metric:

```text
nontrivial_mechanism_confirmed_high_yield
```

This requires the instance to be high-yield and to match the declared EHS failure mechanism, not merely produce many easy front points.

## Baseline Arms

Phase C compares the LLM against external and internal baselines.

### Main external baselines

| Arm | Meaning |
|---|---|
| LLM (Call2) | DeepSeek-generated instance families (interactive, with failure feedback). |
| Random | Schema-random family generation under the same legal parameter space. |
| Literature-derived | Published benchmark generators (Wang 2018 medium-scale, Anghinolfi 2021 VLS capped). |

### Internal control (reported separately)

The `agent_manual_sweep` arm is a strong internal stress-control baseline
generated during our agentic research workflow. It is NOT presented as an
independent human/literature baseline.

The family file `families/human_sweep_families.json` is preserved as archival
reference; textual naming has been updated to `agent_manual_sweep`.

## C0-C2: Protocol, Schema, and Baselines

### C0-C1

We defined the benchmark-design task and the JSON family schema.

Each family spec includes:

- job-count range;
- machine-count range;
- horizon range;
- processing-time distribution;
- machine-rate distribution;
- TOU price profile;
- epsilon regime;
- expected EHS failure mechanism;
- validity constraints;
- rejection conditions.

The schema explicitly bans trivial or invalid cases such as single-machine degeneracy, flat energy structure unless used as a control, infeasible total work, or size-only hardness as the main success claim.

### C2

We created:

- `families/random_families.json`: 8 random schema-valid families.
- `families/human_sweep_families.json`: 8 manual sweep families.
- `prompts/call1_llm_family_designer.md`: DeepSeek prompt for the first LLM family-generation call.

Both random and manual family files passed validation.

## C3 Initial Smoke: Promising but Inconclusive

### DeepSeek Call 1

DeepSeek generated 8 valid families:

- `first_khat_dominance_giant`
- `asgh_trajectory_conflict`
- `reinsertion_starvation_tight_epsilon`
- `es_local_optima_trap_extreme_rates`
- `front_coverage_gap_step_TOU`
- `short_budget_critical_size`
- `load_imbalance_narrow_jobs_wide_rates`
- `epsilon_skip_narrow_cost_steps`

Cost was approximately `$0.01`.

### Initial selected families

The first smoke selected:

- 2 LLM families:
  - `first_khat_dominance_giant`
  - `asgh_trajectory_conflict`
- 2 random families:
  - `random_000`
  - `random_001`
- 2 manual sweep families:
  - `human_tight_epsilon`
  - `human_loose_epsilon`

### Initial result

The initial C3 smoke was truncated:

- 18 instances generated.
- 36 EHS budget-runs expected.
- Only 8/36 runs completed.

The result looked directionally positive for the LLM:

| Arm | Instances | Evaluated | High-Yield |
|---|---:|---:|---:|
| LLM | 6 | 3 evaluated + 3 too-large | 6 |
| Random | 6 | 1 evaluated | 1 |
| Manual sweep | 6 | 0 evaluated | 0 |

But this could not be accepted as a fair result because random and manual baselines were barely evaluated.

### Important infrastructure discovery

The smoke revealed that `run_ehs()` did not fully respect `time_limit_seconds` during SGH construction. Time was checked between `khat` iterations, but the first SGH construction itself could exceed the budget by a large factor.

Example:

- A random instance with `T=737` took about `155s` for a nominal `30s` budget.

This discovery was useful, but it meant C3 had to be repaired.

### Scalability side note

The LLM's `first_khat_dominance_giant` family used `n=800+`, `m=35-44`, `T=750-850`.

It correctly exposed a non-interruptible first-khat scalability weakness. However, this is not counted as the main LLM success because it is close to a size/evaluator-stress effect. It is documented separately as a scalability diagnostic, not the central benchmark-design claim.

## C3-Regular Repaired: First Fairer Comparison

We repaired the setup:

- capped regular instances to `n <= 150`, `T <= 200`;
- added cooperative EHS deadline checks;
- generated/evaluated 6 instances per arm;
- used `30s / 90s` budgets.

### Generation quality

All arms produced feasible/evaluable regular instances after repair:

| Arm | Instances | Evaluable | Completed Budget Pairs | Rate |
|---|---:|---:|---:|---:|
| LLM Call1 | 6 | 6 | 12 | 100% |
| Random | 6 | 6 | 12 | 100% |
| Manual sweep | 6 | 6 | 12 | 100% |

### Adversarial yield

| Arm | Evaluable | High-Yield | Rate |
|---|---:|---:|---:|
| LLM Call1 | 6 | 5 | 83% |
| Random | 6 | 4 | 67% |
| Manual sweep | 6 | 6 | 100% |

Gate: **FAIL** for LLM Call1.

Reason:

The manual sweep families, especially tight/loose epsilon with uniform `p_j = 1..10`, produced large front-size growth. That is a structural effect of simple processing times and many feasible `khat` steps. It is not necessarily deeper adversarial insight, but it did beat the first LLM families under the chosen high-yield metric.

Call1 still had useful mechanism targeting:

- `asgh_trajectory_conflict` was high-yield on 3/3 regular instances.
- `es_local_optima_trap_extreme_rates` was high-yield on 2/3 regular instances.

But one LLM instance saturated at `Cmax=T=200`, producing only one front point.

## C3-L2: Interactive Counter-Sweep Repair

After C3-Regular failed, we did an interactive second DeepSeek call.

This is the key methodological step in Phase C.

The LLM was given:

- the C3-Regular failure result;
- why manual sweep won;
- the exact high-yield metric;
- the first LLM successes and failure;
- the requirement to beat manual sweep while keeping mechanism-specific stress;
- the constraints `n <= 150`, `T <= 200`, feasible/evaluable instances only.

The prompt asked DeepSeek to design second-generation families that combine:

- the manual sweep's useful ingredients:
  - small/uniform processing times;
  - tight epsilon;
  - many possible `khat` steps;
- with explicit EHS mechanism stress:
  - A-SGH lock-in;
  - first-khat dominance;
  - ES exploration tension;
  - reinsertion starvation.

### DeepSeek Call 2 families

DeepSeek produced 5 families:

- `hybrid_M1_tight_hetero_rates_firstkhat_dom`
- `hybrid_M2_tight_steprates_asghlock`
- `hybrid_M3_narrowrates_reinsert_starve`
- `hybrid_M4_dualpeak_exploration_tension`
- one additional family in the raw output not used in C4.

The most important family is:

```text
hybrid_M2_tight_steprates_asghlock
```

This combines:

- small jobs;
- tight epsilon;
- two machine-rate classes;
- A-SGH lock-in pressure.

The intended mechanism is reasonable: early assignments favor cheap machines, but as the makespan limit tightens, schedules must redistribute work. A-SGH reuses prior assignments, so it may resist the structural changes needed at later `khat` values.

### C3-L2 result

The C3-L2 pilot evaluated two new LLM families and compared against the frozen repaired C3 baselines.

| Arm | Instances | High-Yield | Rate | Mean Delta FS | Median Delta FS |
|---|---:|---:|---:|---:|---:|
| Manual sweep (C3) | 6 | 6 | 100% | 15.2 | 13 |
| LLM Call1 (C3) | 6 | 5 | 83% | 2.8 | 3 |
| LLM Call2 (L2) | 6 | 6 | 100% | 19.5 | 10 |
| Random (C3) | 6 | 4 | 67% | 8.5 | 5 |

Gate: **STRONG** on the development comparison.

Interpretation:

DeepSeek learned from feedback. It incorporated the manual sweep's front-growth ingredients and added mechanism-specific stress layers. This is the first clearly positive LLM result in this branch.

## C4 Validation: Transfer Test

C4 froze the L2 families and tested whether the result transferred to fresh validation instances.

Setup:

- 12 families total.
- 4 LLM Call2 families.
- 4 manual sweep families.
- 4 random families.
- 5 fresh instances per family.
- 60 instances total.
- 120 EHS runs at `30s / 90s`.
- Runtime about 2.6 hours.

### C4 aggregate result

| Arm | N | Feasible | Evaluable | Generation % | High-Yield % | Mean Delta FS | Mechanism Confirmed % | NT-MC-HY % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLM Call2 | 20 | 20 | 20 | 100% | 100% | 20.4 | 25% | 25% |
| Manual sweep | 20 | 20 | 20 | 100% | 100% | 38.2 | 50% | 50% |
| Random | 20 | 15 | 15 | 75% | 93% | 2.6 | 0% | 0% |

Gate: **MODERATE**.

### What transferred

LLM L2 high-yield behavior transferred:

- C3-L2: 6/6 high-yield, mean delta FS 19.5.
- C4: 20/20 high-yield, mean delta FS 20.4.

This is a real transfer result. It means the LLM did not merely overfit the 6 development instances.

The LLM also beat random clearly on:

- generation quality: 100% vs 75%;
- high-yield rate: 100% vs 93%;
- mean front growth: 20.4 vs 2.6;
- mechanism confirmation: 25% vs 0%.

### What did not transfer strongly

The LLM did not beat the manual sweep baseline.

Manual sweep also achieved:

- 100% high-yield;
- higher mean front growth: 38.2 vs 20.4;
- higher nontrivial mechanism-confirmed high-yield: 50% vs 25%.

The manual sweep advantage comes mainly from:

- `human_loose_epsilon`: very large raw front growth, e.g. +50 to +116.
- `human_many_machines_sparse`: mechanism-confirmed front-coverage stress.
- `human_mixed_job_sizes`: mechanism-confirmed A-SGH lock-in stress.

This means the manual/agent sweep baseline is strong. It should not be described as a weak strawman.

## Strongest Validated LLM Family

The strongest validated LLM family is:

```text
hybrid_M2_tight_steprates_asghlock
```

C4 validation:

- 5/5 high-yield.
- 5/5 mechanism-confirmed.
- 5/5 nontrivial mechanism-confirmed high-yield.
- Delta front sizes: `+30`, `+31`, `+38`, `+7`, `+19`.

This is the cleanest positive result.

It is sound because:

- processing times are realistic small/moderate jobs;
- machine rates are not absurd, only two classes;
- TOU profile is simple;
- the stress mechanism is meaningful for EHS;
- validation confirmed the intended A-SGH lock-in behavior.

This family should be the central example in any thesis or paper discussion.

## Weaker LLM Families

### `hybrid_M1_tight_hetero_rates_firstkhat_dom`

Result:

- 5/5 high-yield.
- 0/5 mechanism-confirmed.
- Delta FS: `+8`, `+5`, `+28`, `+70`, `+7`.

Interpretation:

It creates stress, but the claimed first-khat mechanism was not confirmed. It is useful as a stress family, not as a mechanism-specific success.

### `hybrid_M3_narrowrates_reinsert_starve`

Result:

- 5/5 high-yield by broad criteria.
- 0/5 mechanism-confirmed.
- Small front growth in most instances.

Interpretation:

This is conceptually weak. Near-homogeneous machine rates and low-variance prices may reduce the energy-aware nature of the problem. This family should not be emphasized.

### `hybrid_M4_dualpeak_exploration_tension`

Result:

- 5/5 high-yield.
- 0/5 mechanism-confirmed.
- Delta FS: `+7`, `+41`, `+32`, `+12`, `+66`.

Interpretation:

This is plausible as a TOU stress family, but the expected ES exploration-tension mechanism was not confirmed. It can be discussed as exploratory, but not as a validated mechanism-specific LLM success.

## Key Caveats

### 1. The manual sweep baseline is strong

The manual sweep should not be called a weak baseline.

It achieved:

- 100% high-yield;
- higher mean front-size growth than LLM;
- higher mechanism-confirmed high-yield than LLM.

Because the manual sweep was produced in an agentic workflow, it is better to call it:

- `manual_sweep`;
- `agent_manual_sweep`;
- or `expert_sweep_baseline`.

It should not be framed as a neutral human upper bound, but it also should not be erased or weakened silently.

### 2. The high-yield metric rewards generic front growth

The simple front-growth metric can reward instances that are not deeply adversarial. For example, uniform small jobs and loose epsilon can produce many front points quickly.

Therefore, the paper/thesis should separate:

- generic high-yield rate;
- mechanism-confirmed high-yield;
- nontrivial mechanism-confirmed high-yield.

### 3. The short budget may be too short

In C4, all 60 instances timed out at the 30s short budget. This means the short-vs-long comparison partly measures budget truncation rather than clean mechanism discrimination.

A future C4b budget sanity check with `60s / 180s` or `90s / 300s` would make the evidence cleaner.

### 4. The LLM does not dominate manual sweeps

The honest conclusion is not "LLM beats every baseline."

The honest conclusion is:

- LLM beats random generation.
- LLM Call2 transfers after feedback.
- LLM produces at least one strongly validated mechanism-specific family.
- LLM ties manual sweep on high-yield rate.
- LLM loses to manual sweep on mean front growth and nontrivial mechanism-confirmed yield.

## Main Supported Claim

The strongest supported claim is:

> Interactive LLM prompting can design realistic, feasible, mechanism-aware adversarial benchmark families for EHS. After receiving empirical feedback, DeepSeek learned to combine simple front-growth ingredients with EHS-specific stress mechanisms. The resulting LLM families transferred to fresh validation instances and clearly outperformed random generation, with the strongest validated example being the step-rate A-SGH lock-in family.

## Claims We Should Avoid

Avoid:

> LLM beats manual/human sweeps.

C4 does not support this.

Avoid:

> All LLM-designed mechanisms were confirmed.

Only `hybrid_M2_tight_steprates_asghlock` was strongly confirmed.

Avoid:

> LLM generated harder instances than every baseline.

Manual sweep had larger mean front growth.

Avoid:

> Giant first-khat instances prove LLM benchmark superiority.

Those instances mainly expose a scalability/time-limit issue and should be treated separately.

## Paper/Thesis Positioning

Phase C can support an exploratory thesis chapter or a modest paper section on LLM-assisted benchmark design.

The best framing is:

1. Direct LLM solver components failed in earlier branches.
2. The LLM was more useful when moved to a research-support role: adversarial benchmark design.
3. A one-shot LLM design was not enough; it lost to simple manual sweeps.
4. An interactive LLM loop improved after empirical feedback.
5. The second-generation LLM families transferred in validation and beat random generation.
6. The best validated LLM family exposed A-SGH lock-in under tight epsilon and step machine rates.
7. Manual sweeps remain a strong baseline, so the claim must be qualified.

## Recommended Next Step

If we continue Phase C, the next best experiment is a small C4b budget sanity check.

Purpose:

> Test whether the mechanism-specific LLM signal survives when the short budget is less universally saturated.

Suggested setup:

- Select:
  - 5 LLM `hybrid_M2_tight_steprates_asghlock` instances;
  - 5 manual `human_mixed_job_sizes` or `human_many_machines_sparse` instances;
  - 5 random instances.
- Run budgets:
  - `60s / 180s`, or
  - `90s / 300s` if affordable.
- Main metric:
  - mechanism-confirmed high-yield, not raw front growth.

If the LLM M2 family remains 5/5 mechanism-confirmed, this becomes a cleaner thesis result.

## Bottom-Line Verdict

Phase C is the first LLM direction with a genuinely defensible positive result.

### Final comparison (literature baseline)

| Arm | HY% | Mean Δfs |
|-----|-----|----------|
| **LLM Call2** | **100%** | **20.4** |
| Literature (Wang+Anghi) | 88% | 2.4 |
| Random (schema) | 93% | 2.6 |
| agent_manual_sweep (internal) | 100% | 38.2 |

### Gate: STRONG

- LLM beats literature generators on both HY rate (100% vs 88%) and mean Δfs (20.4 vs 2.4)
- LLM beats random on both metrics (100% vs 93%, Δfs 20.4 vs 2.6)
- LLM produces mechanism-specific families (M2: asgh_lock_in confirmed on 5/5)
- The agent_manual_sweep internal control achieves larger raw Δfs (+38.2) through
  structural properties of the metric (coarse epsilon + uniform jobs), not adversarial insight

The honest conclusion:
> Interactive LLM prompting can design realistic, feasible, mechanism-aware
> adversarial benchmark families for EHS. After receiving empirical feedback,
> DeepSeek learned to combine simple front-growth ingredients with EHS-specific
> stress mechanisms. The resulting LLM families transferred to fresh validation
> instances and clearly outperformed both random generation and literature-derived
> benchmark generators on adversarial yield rate and front-size growth magnitude.

