# Problem

Date: 2026-04-20

Phase G regime-2 decomposition is stopped in current form.

New branch hypothesis:

- richer VND-style assignment neighborhoods, combined with exact per-machine DP evaluation, can materially improve fixed-epsilon TEC quality versus the current greedy/relocate branch.

Bounded test scope:

- single-point viability test only: instance `61`, epsilon `347`
- no full EOA / no epsilon oscillation / no frontier run.

Decision target:

- determine whether this method family is worth continuation based on quality-vs-runtime signal at `61/347`.
