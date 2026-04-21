# Problem

Stage L1.5 produced enough exact-labeled rows for a development-stage learnability check, but dataset/protocol cleanliness is not yet paper-grade.

Stage L2 must:

- clean and document a modeling dataset from L1/L1.5 artifacts,
- run a dev-only offline ranking probe (no solver integration),
- compare boosted-tree ranking against handcrafted `screen_score_s2`,
- avoid benchmark-generalization claims.
