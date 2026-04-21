# Problem

The handcrafted insert-focused exact-DP heuristic has reached a credible but saturating point:

- best handcrafted TEC at `61/347`: `6884`
- paper EHS at `61/347`: `6710`

The remaining bottleneck is not existence of improving moves.
It is selection:

- there are many candidate `insert_inter` moves
- only a few exact DP evaluations can be afforded
- handcrafted screening now has diminishing return

The new problem is to design a learning-based move-ranking component that helps choose which moves receive exact DP evaluation.
