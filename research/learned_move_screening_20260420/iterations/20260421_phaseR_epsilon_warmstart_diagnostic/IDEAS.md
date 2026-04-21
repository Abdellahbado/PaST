# Ideas

1. Reuse the strongest available Phase J/K insert-based VND variant as the base solver rather than creating a new heuristic family.
2. Start from benchmark instance `61`, because this instance already has dense evidence across Phases I/J/K and is the clearest place to test the warm-start hypothesis.
3. Use a short descending epsilon ladder around the known hard region near `347`, for example `380, 370, 360, 350, 347, 340`.
4. At each epsilon, compare:
   - fresh-start VND,
   - warm-start initial incumbent inherited from previous epsilon,
   - warm-start final result after local search.
5. Record both carryover value and new local improvement:
   - carryover gain = `fresh_final_tec - warm_final_tec`
   - local improvement from seed = `warm_initial_tec - warm_final_tec`
6. Keep the first implementation single-instance and diagnostic-first. Do not broaden to full benchmark sweep until the mechanism is validated.
