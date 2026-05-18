# Results

Current accepted benchmark facts relevant to this iteration:

- `g3567` is exact through `n=6000`
- `g24` is exact through `n=10000`
- `g12357` is exact through `n=8000`
- `g246810` is exact through `n=6000`
- `g810` is exact through `n=5000`
- `g12345678910` is exact through `n=3500` and times out from `n=5000`
- `g37` old ledger is exact only through `n=600`, but PLAN13 reroute later
  closes tested rows through `n=5000`

Important correction:

- Old `g37` later rows entered sparse exact / Step 4 on some runs, but did not
  close because they were not routed through the intended K=2 Step-3 exact path.
- Current corrected evidence is `csv/plan13/PLAN13_g37_k2_reroute.csv`:
  `selector_reason=k2_exact_default`, `step3_mode=exact`,
  `fwd_pack_method=profile_realization_dp_exact`, exact through `n=5000`.

Primary evidence:

- [PAPER_GROUPS_PLAN05_n_extension.csv](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/plan05/PAPER_GROUPS_PLAN05_n_extension.csv)
- [PAPER_GROUPS_PLAN11_n_extension.csv](/Users/mac/Documents/Study/PFE/PaST/research/k_vs_arithmetic_axes_20260412/csv/plan11/PAPER_GROUPS_PLAN11_n_extension.csv)
