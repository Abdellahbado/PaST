# Blockers

Current blockers:

- none for regime-1 correctness on instance `46`; probe solved and matched exact at required epsilon.

Remaining technical caution:

- direct transition to true column generation is not yet validated.
- `solve_pricing_dp` usage for exact reduced-cost pricing still needs explicit dual/sign/empty-pattern mapping validation before branch-and-price claims.

Scaling limits observed:

- pricing all enumerated configurations is already the dominant cost in regime 1.
- regime-2 (12 job types) likely needs restricted pools or true CG due to configuration explosion.
