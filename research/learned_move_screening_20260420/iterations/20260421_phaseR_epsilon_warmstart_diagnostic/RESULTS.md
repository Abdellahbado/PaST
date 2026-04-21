# Results

Not run yet.

This iteration starts from these grounded findings:

- Phase I no-screen diagnostic on `paper_61` at `epsilon=347` found a true improving insert-inter move and improved TEC from `6944` to `6920`.
- Phase J/K screening improvements reduced the same instance further to `6884` with `vnd_exact_dp_insert_rank_diverse`.
- Phase Q synthetic-only offline learning showed only marginal gains over handcrafted `screen_score_s2`, so the next branch should test an algorithmic mechanism rather than continue fine-grained learned screening.

Primary evidence files:

- `temp/phaseI_noscreen_diagnostic/run_61_347_noscreen_cap256.csv`
- `temp/phaseJ_insert_screening_redesign/run_61_347_insert_rank_diverse.csv`
- `temp/phaseK_insert_efficiency_pass/run_61_347_insert_rank_diverse.csv`
- `research/learned_move_screening_20260420/iterations/20260421_phaseQ_synthetic_offline_training/RESULTS.md`
