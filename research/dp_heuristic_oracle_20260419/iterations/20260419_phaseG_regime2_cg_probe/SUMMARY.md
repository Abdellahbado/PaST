# Summary

Phase G executed a bounded regime-2 restricted-CG probe at `61/347`, then a bounded correction rerun focused on the duplicate-stop bug.

Delivered:

- reduced-cost design note with explicit LP-dual mapping and caveats
- restricted master + pricing-style loop implementation
- machine-readable artifacts under `temp/phaseG_regime2_cg/`

Main result:

- feasible and optimal restricted-master IP solution found: TEC `7040`
- better than one-shot heuristics (`7088`, `7081`) but still far from paper/reference (`6710`, `6643`)
- after duplicate-stop correction, loop added multiple genuinely new columns (`259 -> 271`) but LP bound and TEC stayed unchanged

Signal:

- decomposition path is technically runnable in regime 2,
- prior duplicate-stop was a real logic bug and is now corrected,
- corrected rerun shows the branch remains non-competitive in current bounded form (not just duplicate-stop limited).

Next action recommendation:

- stop this regime-2 restricted-CG branch in current bounded form.
- only reopen if we explicitly choose one tightly bounded quality-moving mechanism (for example, best-of-k insertion with LP-impact screening) and re-test again on the same `61/347` point.
