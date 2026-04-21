# Summary

Phase J targets the concrete Phase H failure mode at `61/347`: insert screening/ranking was too aggressive and blocked move-level improvement.

What was implemented:

- `vnd_exact_dp_insert_rank_v1` (dual-pressure analytical ranking)
- `vnd_exact_dp_insert_rank_diverse` (per-source diversity + two-stage rerank)
- both keep exact DP acceptance only on touched machines and remain bounded single-point runs.

Key outcome:

- old screened baseline: `6944`
- no-screen best: `6920`
- rank_v1: `6908`
- rank_diverse: `6884`

Decision signal:

- branch is worth continuing: redesigned analytical screening recovers and exceeds no-screen improvement signal.
- next immediate work should reduce screening overhead (especially candidate pool memory/time in diverse variant) while preserving quality.
