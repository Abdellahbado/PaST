# Blockers

Current blockers after Phase I diagnostic:

- `swap_inter` signal remains limited in this pass because improving `insert_inter` is found and accepted early under bounded first-improvement logic.
- no-screen exact evaluation is costlier than screened VND, so full-scale usage is not suitable without better ranking/staging.

Implication:

- next branch should redesign screening/ranking to preserve improving move discovery while retaining bounded cost.
