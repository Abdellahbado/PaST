# Ideas

Considered efficiency-focused ideas:

1. **Idea A: tighter candidate-pool structure**
   - lower `source_top_k`
   - lower per-source keep
   - per-target quota while selecting candidates from each source
   - lower shortlist and exact-eval caps

2. **Idea B: staged exact-eval budget**
   - laddered exact-eval limits (small first stage, expand only if warranted)
   - stop expansion early when score quality decays strongly

3. **Idea C: source-machine focusing refinement**
   - keep source prioritization by exact-minus-LB gap and exact cost
   - in trimmed mode, keep only top-priority sources by threshold

4. **Idea D: tiny secondary neighborhood budget (optional)**
   - considered as a potential add-on after insert saturation
   - not selected in this pass to avoid confounding insert-screening efficiency signal.

Selected for implementation now:

- `vnd_exact_dp_insert_rank_diverse_trimmed`: A + C
- `vnd_exact_dp_insert_rank_diverse_budgeted`: A + B + C

Rationale:

- they are direct continuations of Phase J diverse signal,
- they change only screening/budget structure, not method family,
- they provide clean evidence for “last non-ML” decision.
