# Problem

Date: 2026-04-20

Phase J established a real insert-screening branch at `61/347`:

- old screened `vnd_exact_dp`: `6944`
- no-screen diagnostic best: `6920`
- Phase J best (`vnd_exact_dp_insert_rank_diverse`): `6884`

The remaining issue is efficiency rather than signal recovery:

- insert candidate screening volume remained high (`29160` screened insert candidates)
- runtime and memory overhead became significant in diverse mode.

Phase K objective:

- run one bounded non-ML pass to tighten candidate-pool/screening structure,
- preserve or improve TEC around `6884`,
- reduce screening and exact-evaluation overhead if possible,
- keep strict single-point scope (`instance 61`, `epsilon 347`).
