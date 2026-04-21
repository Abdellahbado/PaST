# Ideas

Analytical screening/ranking families considered:

1. **Idea A (dual pressure ranking):** score insert moves using source pressure (exact cost / exact-LB gap), target fullness/headroom, and job size interaction.
2. **Idea B (diversity-preserving shortlist):** keep top-k candidates per promising source machine instead of one global shortlist.
3. **Idea C (two-stage screening):** permissive stage-1 score, then stricter stage-2 rerank before exact DP.
4. **Idea D (gap-aware source priority):** prioritize sources by exact cost and exact-minus-LB gap to focus where quality loss concentrates.

Selected for implementation now:

- Variant 1: Idea A + Idea D (`vnd_exact_dp_insert_rank_v1`)
- Variant 2: Idea B + Idea C + Idea D (`vnd_exact_dp_insert_rank_diverse`)
