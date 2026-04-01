# Why Semigroup Was Enough For Large-`n`, But Backups Were Needed For Structural Extensions

Date: 2026-04-01

## 1. Purpose

This note is only for explanation and discussion.

Its goal is to explain clearly:

1. why semigroup was enough when we scaled only the instance size,
2. why we needed stronger backup relaxations when we changed other parameters,
3. exactly which benchmark parameters were scaled and to which values,
4. what the theoretical weakness of semigroup is,
5. how `R_feas` mitigates that weakness.

This is not meant as a final paper section. It is an internal explanation note
for supervisor discussions.

---

## 2. One-Sentence Summary

When we scaled only `n` on the paper's hard `{8,10}` family, the main difficulty
was computational and disappeared after fixing the banding/gap-cap logic, so the
default semigroup method was enough. When we changed the benchmark structurally
by adding more job types, scarcity, and alternative arithmetic combinations,
the difficulty became modeling-related: the semigroup state could no longer
distinguish different hidden inventories, and then `R_feas` became useful.

---

## 3. The Main Distinction

There are two different ways to make the benchmark harder:

1. **Size scaling**
   Increase the number of jobs while keeping the same structural family.

2. **Structural scaling**
   Change the arithmetic or combinatorial structure of the jobs and the horizon:
   more types, scarcity, larger processing times, larger slack, different size
   interactions.

These two directions stress different parts of the method.

| Extension type | Main question | What it stresses |
|---|---|---|
| Scale only `n` | Does the default production method still scale? | runtime, state-space volume, certification cost |
| Change structure | Is the semigroup relaxation still modeling the real instance tightly? | bounded-count ambiguity, hidden count collisions, packability |

This distinction is the key to understanding the results.

---

## 4. What We Mean By "Scaling `M`"

If by "scaling `M`" we mean scaling the instance size, then in our code and
benchmark generator this is effectively scaling:

- `n`: the total number of jobs.

This is the main extension that follows the supervisor's suggestion:

- keep the hard paper family,
- mainly extend `n`,
- check whether the method still solves the instances in reasonable time.

In our implementation, that family is:

- `A_nscale_8_10`

with:

- processing times fixed to `{8,10}`,
- machine fixed to `TWOSBY`,
- horizon multiplier fixed to `lambda = 1.3`,
- only `n` changed.

The actual implemented values are:

- `n = 300, 400, 500, 600, 750, 1000`

as defined in
[generate_extended_stress_benchmark.py](/Users/mac/Documents/Study/PFE/PaST/scripts/generate_extended_stress_benchmark.py#L111).

---

## 5. Exact Benchmark Parameters We Scaled

The exploratory generator extends the benchmark along several axes.

### 5.1 Large-`n` paper family

| Family | Fixed parameters | Scaled parameter | Actual values |
|---|---|---|---|
| `A_nscale_8_10` | `p={8,10}`, `lambda=1.3`, `TWOSBY` | `n` | `300, 400, 500, 600, 750, 1000` |

Purpose:

- pure scalability extension of the paper's hard family.

### 5.2 Horizon-slack scaling on the same pair

| Family | Fixed parameters | Scaled parameter | Actual values |
|---|---|---|---|
| `C_lambda_8_10` | `p={8,10}`, `n=200`, `TWOSBY` | `lambda` | `1.3, 1.6, 2.0, 2.5, 3.0` |

Purpose:

- more temporal freedom,
- more candidate block placements,
- wider DP search region.

### 5.3 Alternative 2-type arithmetic structure

| Family | Fixed/varied parameters | Actual values |
|---|---|---|
| `D_adv_semigroup` | `p` varied, `lambda=1.3`, `n` varied | `p in {(7,9),(8,9),(9,10),(7,11)}`, `n in {200,300}` |

Purpose:

- probe whether different two-type arithmetic structures expose semigroup
  weakness more clearly than `{8,10}`.

### 5.4 Bounded-count 3-type structural extensions

| Family | Processing-time groups | `n_total` values | `lambda` values | Scarcity pattern |
|---|---|---|---|---|
| `E_3type_adversarial` | `(4,6,10)`, `(8,10,14)`, `(3,5,7)`, `(5,7,11)` | `30, 40, 50, 60, 80` depending on group | `1.3, 1.8` | one or two scarce types fixed explicitly |

Examples actually used:

- `(4,6,10), n=60, scarce={0:1, 2:3}`
- `(8,10,14), n=50, scarce={0:1, 2:2}`
- `(5,7,11), n=60, scarce={0:2, 2:3}`

Purpose:

- create bounded-count ambiguity,
- create multiple hidden count decompositions for the same remaining work,
- test whether the backup lower bound is needed.

### 5.5 Larger 3-type slack scaling

| Family | Processing-time groups | `n_total` values | `lambda` values | Scarcity pattern |
|---|---|---|---|---|
| `F_3type_high_lambda` | `(4,6,10)`, `(8,10,14)`, `(5,7,11)`, `(9,13,17)` | `90, 100, 120` depending on group | `2.2, 3.0` | one scarce endpoint and one moderately scarce endpoint |

Purpose:

- same count-collision mechanism as Family E,
- but with much larger temporal freedom.

### 5.6 4-type dual-scarcity and larger processing times

| Family | Main change | Actual values |
|---|---|---|
| `G_4type_dual_scarcity` | add a fourth type and two scarce types | groups like `(4,6,10,14)`, `(5,7,11,13)` with `lambda in {1.4,2.0}` |
| `H_large_processing_times` | scale the absolute processing times upward | groups like `(12,16,20)`, `(11,17,23)` with `lambda in {1.6,2.4}` |

Purpose:

- exploratory stress testing,
- more packing rigidity,
- larger semigroup holes,
- more complex bounded-count structure.

---

## 6. Why Semigroup Was Enough When We Scaled Only `n`

### 6.1 What happened empirically

After the gap-cap/banding fix, the default semigroup-based production path
solved representative large-`n` `{8,10}` instances up to `n=1000` without
needing the backup relaxations.

Representative results:

| `n` | Runtime | Winning stage |
|---:|---:|---|
| 300 | `1.17s` | `fwd_relax:random_ff` |
| 400 | `1.94s` | `fwd_relax:bfd` |
| 500 | `2.99s` | `fwd_relax:block_dp_exact` |
| 600 | `4.19s` | `fwd_relax:block_dp_exact` |
| 750 | `6.71s` | `fwd_relax:block_dp_exact` |
| 1000 | `12.36s` | `fwd_relax:block_dp_exact` |

The important point is that these were solved by the default method:

1. semigroup relaxation,
2. relaxed-profile recovery,
3. heuristic packing or exact fixed-profile block certification.

The stronger backup lower bounds were not what made these instances solvable.

### 6.2 Why that happened

Scaling only `n` on the same `{8,10}` family changes mainly:

- total workload,
- number of DP states visited,
- size of the recovered packing problem.

But it does **not** change the fundamental structure of the relaxed model:

- still only two job sizes,
- still no complicated bounded-count collision structure,
- still no hidden ambiguity caused by several scarce interacting types,
- still the same narrow low-variability family.

So semigroup remained a good structural model of the instance.

### 6.3 What the real bottleneck was

The earlier large-`n` slowdown was not mainly a lower-bound weakness.

It came from a computational issue in the banded forward DP:

- the old `auto_max_gap` logic used a very loose bound,
- on the real price traces this often pushed `max_gap` to the full horizon,
- so the "banded" DP behaved almost like an unbanded DP.

After we replaced that coarse estimate by a sharper safe bound, the forward
stage became fast again and semigroup was sufficient.

So in this regime the problem was:

- **implementation scalability**

not:

- **semigroup modeling weakness**

---

## 7. Why Structural Extensions Needed Stronger Backup Relaxations

### 7.1 What changed in the 3-type bounded-count families

When we moved from:

- two-type low-variability families such as `{8,10}`

to families such as:

- `(4,6,10)` with one very scarce short type,
- `(8,10,14)` with one scarce endpoint,
- `(5,7,11)` with a scarce short type and a moderately scarce long type,

we changed the structure in a way that semigroup cannot fully represent.

The difficulty is no longer just "there are many jobs."

The difficulty becomes:

- several distinct count vectors can produce the same total remaining work,
- but they do not allow the same future transitions,
- and some relaxed-optimal block profiles become unpackable with the real job
  multiset.

### 7.2 The key phenomenon: hidden count collisions

Semigroup tracks a state of the form:

- `(t_end, rw)`

where:

- `t_end` is the end time of the current block pattern,
- `rw` is the remaining total processing work.

This is compact and powerful, but it forgets:

- how many jobs of each size were already used,
- how many remain of each size.

That is harmless when the instance behaves almost like a purely arithmetic
workload balance problem.

It becomes dangerous when:

- one or two sizes are scarce,
- and different hidden decompositions of the same `rw` lead to different
  remaining inventories.

This is exactly what happens in the realistic 3-type scarcity/collision family.

### 7.3 Why this can hurt even if the lower bound looks good

There are two failure modes:

1. the semigroup lower bound can become numerically too optimistic,
2. the semigroup can recover a block profile that is optimal in the relaxation
   but unpackable with the real bounded multiset of jobs.

The second failure mode is especially important for us, because the practical
pipeline needs a recovered relaxed profile that can actually be realized.

So in these families the question is not only:

- "Is the lower bound tight?"

but also:

- "Is the recovered relaxed solution packable?"

---

## 8. How `R_feas` Mitigates The Weakness Of Semigroup

### 8.1 Core idea

`R_feas` keeps the same compact state:

- `(t_end, rw)`

but it does not allow every semigroup-valid transition.

Instead, it adds bounded-count feasibility checks on the transition itself.

In words:

- before placing a job type `j`, it checks whether the already placed work can
  still be explained by a bounded multiset that leaves one copy of `j`
  available,
- and whether the remaining suffix after taking `j` is still compatible with
  the bounded inventory.

So semigroup says:

- "this total work is arithmetically representable"

while `R_feas` says:

- "this step is arithmetically representable **and** consistent with the finite
  job inventory."

### 8.2 What it repairs

`R_feas` directly repairs semigroup's main blind spot:

- multiplicity blindness.

Semigroup may treat several hidden inventory states as equivalent because they
share the same `rw`.

`R_feas` does not store the full inventory, but it removes transitions that
cannot possibly be supported by any bounded explanation of the prefix and suffix.

So it is still a relaxation, but a much safer one in scarcity-driven families.

### 8.3 Why it helped exactly where semigroup failed

On the realistic rescue rows in the `(5,7,11)` family:

- semigroup produced a weaker value and an unpackable recovered profile,
- `R_feas` produced a stronger value and a packable recovered profile,
- and that stronger value matched the exact optimum.

The three clean rescue rows were:

- `0360_famE_3type_p5_7_11_n60_l1.8_sc0c2_2c3_s0`
- `0364_famE_3type_p5_7_11_n60_l1.8_sc0c2_2c3_s4`
- `0368_famE_3type_p5_7_11_n60_l1.8_sc0c2_2c3_s8`

On each of them:

| Method | Value | Packable? |
|---|---:|---|
| `R_semi` | `4,993,458` | No |
| `R_feas` | `4,993,562` | Yes |
| Exact optimum | `4,993,562` | Yes |

This is the cleanest evidence for the backup method:

- semigroup was close, but not safe enough,
- `R_feas` repaired the exact structural weakness that mattered.

---

## 9. Why We Did Not Need The Other Relaxations On Large-`n`

This point is important for presentation.

After the scalability fix:

- large-`n` `{8,10}` was solved by the default semigroup pipeline,
- the backup methods were not needed there,
- because the main issue was speed of the default forward stage, not lack of
  lower-bound strength.

So the right interpretation is:

- **large-`n` `{8,10}`** demonstrates scalability of the default method,
- **bounded-count 3-type families** demonstrate the need for the backup method.

These are complementary benchmark roles.

---

## 10. Clean Supervisor-Level Message

If this needs to be explained briefly in a meeting, the clean version is:

1. We extended the benchmark in two directions.
2. When we extended only `n` on the hard `{8,10}` family, semigroup remained
   enough once we fixed the forward-DP scalability issue.
3. When we changed the structural parameters by adding more types, scarcity, and
   arithmetic alternatives, semigroup could no longer distinguish all relevant
   bounded inventories.
4. In that second regime, `R_feas` mattered because it filters out transitions
   that are semigroup-valid but impossible with the real job counts.

That is the main reason the two benchmark extensions behaved differently.

---

## 11. Practical Takeaway

For the current method, the clean benchmark split is:

### A. Scalability benchmark

Use:

- `A_nscale_8_10`

Purpose:

- show that the default semigroup method now scales on the paper's hardest
  family.

### B. Backup-method benchmark

Use:

- bounded-count realistic 3-type scarcity/collision families,
- especially the `(5,7,11)` family with one scarce short type and one
  moderately scarce long type.

Purpose:

- show where semigroup can still return an unpackable relaxed profile,
- and where `R_feas` repairs that failure.

This is the clearest and most defensible way to present the benchmark logic.
